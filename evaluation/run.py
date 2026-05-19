"""Evaluation runner. Usage: python -m evaluation.run"""
import json
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def _print_search_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("SEARCH EVALUATION")
    print("=" * 60)
    print(f"  Recall@{r['k']}  : {r['recall_at_k']:.3f}  ({r['hits']}/{r['total']} hits)")
    print(f"  Hit Rate   : {r['hit_rate']:.3f}")
    print("\n  Per-query results:")
    for d in r["details"]:
        status = "✓" if d["hit"] else "✗"
        print(f"  {status} [{d['id']}] {d['query'][:55]}")
        if not d["hit"]:
            print(f"      expected : {d['expected']}")
            print(f"      retrieved: {d['retrieved']}")


def _print_architecture_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("ARCHITECTURE EVALUATION")
    print("=" * 60)
    print(f"  Field Completeness : {r['mean_field_completeness']:.3f}")
    print(f"  Component Coverage : {r['mean_component_coverage']:.3f}")
    print(f"  LLM Judge Score    : {r['mean_judge_score']:.1f} / 5.0")
    print("\n  Per-paper results:")
    for d in r["details"]:
        if "error" in d:
            print(f"  ✗ [{d['id']}] {d['title']} — ERROR: {d['error']}")
            continue
        print(f"  [{d['id']}] {d['title']}")
        print(f"      completeness={d['field_completeness']:.2f}  "
              f"coverage={d['component_coverage']:.2f}  "
              f"judge={d['judge_score']}/5")
        if d["judge_issues"]:
            for issue in d["judge_issues"]:
                print(f"      ⚠ {issue}")
        print(f"      {d['judge_reasoning']}")


def main() -> None:
    print("RAGToySys Evaluation Harness")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    from evaluation.metrics import run_search_eval, run_architecture_eval

    print("\n[1/2] Running search evaluation...")
    search_results = run_search_eval(top_k=4)

    print("\n[2/2] Running architecture evaluation...")
    arch_results = run_architecture_eval()

    _print_search_report(search_results)
    _print_architecture_report(arch_results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "search": search_results,
        "architecture": arch_results,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
