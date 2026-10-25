"""Say a word (or press a button), the duck does it.

    python main.py            # needs the sim running (see README)
    python main.py --debug
"""
import argparse
import sys

from duck_adapter import Duck
from follow import Follower, SimEyes
from gamepad import Gamepad
from skates import Skates
from voice import Ears


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--eyes", choices=("sim", "webcam"), default="sim",
                    help="who the duck follows: the sim's person, or you, seen by a webcam")
    ap.add_argument("--camera", type=int, default=1)
    args = ap.parse_args()

    duck = Duck()
    if args.eyes == "webcam":
        from eyes import CameraEyes
        eyes = CameraEyes(args.camera)
    else:
        eyes = SimEyes()
    follower = Follower(duck, eyes, say=lambda text: print(f"  duck: {text}"))
    skates = Skates(duck, follower, say=lambda text: print(f"  duck: {text}"))
    Gamepad(duck, follower, skates, say=lambda text: print(f"  {text}")).start()  # if a controller is plugged in

    def on_command(command, text):
        print(f"heard: {command}   ({text})")
        follower.cancel()  # any new command ends following, stop most of all
        if command == "stop":
            duck.stop()
        elif command == "skates":
            skates.toggle()
        elif command in ("follow", "come") and skates.on:
            print("  duck: no following on skates")
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
