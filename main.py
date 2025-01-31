import sys

import lightning as L

import parser
import command


def main():
    """Main function of the program."""
    L.seed_everything(seed=42)
    args = parser.args()
    sys.exit(getattr(command, args.commands)(args))
    return 0


if __name__ == "__main__":
    main()
