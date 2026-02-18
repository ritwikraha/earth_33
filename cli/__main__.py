"""CLI entry point: dispatches to subcommands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cli <command>")
        print("Commands: run_episode, evaluate, replay")
        sys.exit(1)

    cmd = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module works
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "run_episode":
        from cli.run_episode import main
        main()
    elif cmd == "evaluate":
        from cli.evaluate import main
        main()
    elif cmd == "replay":
        from cli.replay import main
        main()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
