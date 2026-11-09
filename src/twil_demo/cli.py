"""Convenient entry point; each individual module also runs with python -m."""

import argparse
import importlib
import sys


def refresh():
    from twil_demo import build_report, charts, ingest, summarize, trends

    for module in (ingest, summarize, trends, charts, build_report):
        print(f"\n--- {module.__name__.rsplit('.', 1)[-1]} ---", flush=True)
        old_argv = sys.argv
        try:
            sys.argv = [module.__name__]
            module.main()
        finally:
            sys.argv = old_argv
    print("\nData and report refreshed. To update semantic search: python -m twil_demo.rag_app index")


def main():
    parser = argparse.ArgumentParser(description="Local Garmin tables, reports, and TwIL RAG.", add_help=False)
    parser.add_argument("command", choices=["refresh", "ingest", "summarize", "trends", "charts", "build_report", "chat", "rag", "ask", "coach", "dashboard"])
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    args, rest = parser.parse_known_args()
    try:
        if args.command == "refresh":
            if rest == ["--help"] or rest == ["-h"]:
                print("Refresh clean tables, summaries, trends, charts, and report without model calls. Takes no arguments.")
                return
            if rest:
                parser.error("refresh takes no additional arguments")
            refresh()
        elif args.command == "dashboard":
            import subprocess

            from twil_demo.paths import ROOT

            try:
                import streamlit  # noqa: F401
            except ImportError:
                parser.exit(1, "Install the optional dashboard first: uv sync --extra dashboard\n")
            raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "src/twil_demo/dashboard.py"), "--server.address", "127.0.0.1", *rest]))
        else:
            module_name = {"rag": "rag_app", "ask": "ask_garmin"}.get(args.command, args.command)
            sys.argv = [f"twil-demo {args.command}", *rest]
            importlib.import_module(f"twil_demo.{module_name}").main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
