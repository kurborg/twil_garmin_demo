"""Talk directly to TwIL without retrieving Garmin documents."""

import argparse

from twil_demo.ollama import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--prompt")
    args = parser.parse_args()
    try:
        if prompt := args.prompt or args.question:
            print(generate(prompt))
            return
        print("Direct TwIL chat. No Garmin retrieval. Type quit to exit.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if question:
                print(generate(question))
    except (RuntimeError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, RuntimeError):
            parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
