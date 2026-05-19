import time
from dataclasses import dataclass

from paper.semantic_scholar import arxiv_id_of, get_paper, get_references, get_related


@dataclass
class RankedPaper:
    rank: int
    reason: str   # "B_direct" | "D_lineage" | "C_related"
    title: str
    authors: str
    year: int | None
    arxiv_id: str | None
    ss_id: str | None

    def __str__(self) -> str:
        year_str = str(self.year) if self.year else "n/a"
        aid = f"arxiv:{self.arxiv_id}" if self.arxiv_id else ""
        label = {"B_direct": "B:direct", "D_lineage": "D:lineage", "C_related": "C:related"}[self.reason]
        return f"[{self.rank}] [{label}] {self.title} ({year_str}) -- {self.authors} {aid}"

def build_reading_list(arxiv_id: str, top_k: int = 10) -> list[RankedPaper]:
    """
    B → D → C ranked reading list.

    B: papers this paper directly cites (most recent first — what it builds on)
    D: 2-hop references (what B papers build on) sorted by year ASC (foundational)
    C: Semantic Scholar recommendations (parallel/related work)
    """
    paper = get_paper(arxiv_id)
    if not paper:
        return []
    ss_id = paper["paperId"]

    # B: direct references
    b_raw = get_references(ss_id, limit=50)
    b_ids = {p["paperId"] for p in b_raw if p.get("paperId")}

    # D: 2-hop from top-3 most recent B papers
    seen_ids = set(b_ids)
    d_raw: list[dict] = []
    top_b = sorted(b_raw, key=lambda x: x.get("year") or 0, reverse=True)[:3]
    for bp in top_b:
        if bp.get("paperId"):
            time.sleep(3)
            refs2 = get_references(bp["paperId"], limit=20)
            for p in refs2:
                pid = p.get("paperId")
                if pid and pid not in seen_ids:
                    d_raw.append(p)
                    seen_ids.add(pid)

    # C: semantic recommendations not in B or D
    time.sleep(3)
    related = get_related(ss_id, limit=20)
    c_papers = [p for p in related if p.get("paperId") not in seen_ids]

    # Allocate top_k slots: 50% B, 30% D, 20% C (minimum 1 each)
    b_n = max(1, top_k // 2)
    d_n = max(1, top_k * 3 // 10)
    c_n = max(1, top_k - b_n - d_n)

    b_sorted = sorted(b_raw, key=lambda x: x.get("year") or 0, reverse=True)
    d_sorted = sorted(d_raw, key=lambda x: x.get("year") or 9999)

    result: list[RankedPaper] = []
    rank = 1

    for p in b_sorted[:b_n]:
        result.append(_to_ranked(p, "B_direct", rank)); rank += 1
    for p in d_sorted[:d_n]:
        result.append(_to_ranked(p, "D_lineage", rank)); rank += 1
    for p in c_papers[:c_n]:
        result.append(_to_ranked(p, "C_related", rank)); rank += 1

    return result

def _to_ranked(paper: dict, reason: str, rank: int) -> RankedPaper:
    authors = paper.get("authors") or []
    names = [a.get("name", "") for a in authors[:3]]
    author_str = ", ".join(names) + (" et al." if len(authors) > 3 else "")
    return RankedPaper(
        rank=rank,
        reason=reason,
        title=paper.get("title") or "Unknown",
        authors=author_str,
        year=paper.get("year"),
        arxiv_id=arxiv_id_of(paper),
        ss_id=paper.get("paperId"),
    )