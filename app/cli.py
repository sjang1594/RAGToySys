import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

def cmd_ingest(args):
    if args.arxiv:
        from paper.ingestor import ingest_arxiv
        count = ingest_arxiv(args.arxiv)
    else:
        from ingestion.pipeline import ingest
        path = Path(args.input)
        if not path.exists():
            print(f"[error] File not found: {path}")
            sys.exit(1)
        count = ingest(path)
    print(f"Done. {count} chunks stored.")

def cmd_query(args):
    from generation.chain import ask
    question = " ".join(args.question)
    result = ask(question, top_k=args.top_k)
    print("\n" + result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in set(result["sources"]):
            print(f"  - {s}")

def cmd_list(args):
    from retrieval.vectorstore import list_sources, count
    sources = list_sources()
    total = count()
    if not sources:
        print("No documents ingested yet.")
        return
    print(f"{total} chunks across {len(sources)} document(s):")
    for s in sources:
        print(f"  - {s}")

def cmd_remove(args):
    from retrieval.vectorstore import delete_source
    deleted = delete_source(args.filename)
    if deleted == 0:
        print(f"[error] '{args.filename}' not found in knowledge base.")
    else:
        print(f"Removed '{args.filename}' ({deleted} chunks deleted).")

def cmd_agent(args):
    from generation.agent import run
    task = " ".join(args.task)
    print(f"\nTask: {task}\n{'─' * 60}")
    answer = run(task, verbose=not args.quiet)
    print(f"\n{'─' * 60}\n{answer}")

def main():
    parser = argparse.ArgumentParser(prog="rag", description="Personal RAG System")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest an image, PDF, or arXiv paper")
    p_ingest_src = p_ingest.add_mutually_exclusive_group(required=True)
    p_ingest_src.add_argument("--input", help="Path to image or PDF file")
    p_ingest_src.add_argument("--arxiv", help="arXiv ID or URL (e.g. 2301.00001)")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", nargs="+", help="Your question")
    p_query.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    p_query.set_defaults(func=cmd_query)

    # list
    p_list = sub.add_parser("list", help="List all ingested documents")
    p_list.set_defaults(func=cmd_list)

    # remove
    p_remove = sub.add_parser("remove", help="Remove a document from the knowledge base")
    p_remove.add_argument("filename", help="Filename to remove (e.g. sample.jpg)")
    p_remove.set_defaults(func=cmd_remove)

    # agent
    p_agent = sub.add_parser("agent", help="Run the ReAct agent on a task")
    p_agent.add_argument("task", nargs="+", help="Task description")
    p_agent.add_argument("--quiet", action="store_true", help="Suppress tool call output")
    p_agent.set_defaults(func=cmd_agent)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()