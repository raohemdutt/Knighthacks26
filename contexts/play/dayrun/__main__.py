import sys

from contexts.play.dayrun.demo import main

if __name__ == "__main__":
    if sys.argv[1:2] != ["demo"]:
        sys.exit("usage: python -m contexts.play.dayrun demo [scenario]")
    main(*sys.argv[2:3])
