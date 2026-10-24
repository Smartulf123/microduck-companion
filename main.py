"""Say a word, the duck does it.

    python main.py            # needs the sim running (see README)
    python main.py --debug
"""
import argparse
import sys

from duck_adapter import Duck
from follow import Follower, SimEyes
from voice import Ears


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mic", type=int, default=None)
    args = ap.parse_args()

    duck = Duck()
    follower = Follower(duck, SimEyes(), say=lambda text: print(f"  duck: {text}"))

    def on_command(command, text):
        print(f"heard: {command}   ({text})")
        follower.cancel()  # any new command ends following, stop most of all
        if command == "stop":
            duck.stop()
        elif command in ("follow", "come"):
            follower.start(command)
        else:
            result = duck.do(command)
            if result != "ok":
                print(f"  duck says no: {result}")

    try:
        Ears(on_command, args.debug, args.mic).run()
    except KeyboardInterrupt:
        duck.stop()


if __name__ == "__main__":
    main()
