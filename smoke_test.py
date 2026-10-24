"""Walks the duck through every behaviour. Run after pulling a new Pollen version.

    python smoke_test.py
"""
import time

from duck_adapter import Duck

duck = Duck()
p = duck.call("robot.policies")
print("skills:", p.get("skills"), "| sitting:", p.get("sitting"))


def step(name, action, wait):
    result = action()
    time.sleep(wait)
    sitting = duck.call("robot.policies").get("sitting")
    print(f"{name:7} -> {result!s:30} sitting={sitting}")


step("sit", lambda: duck.do("sit"), 5)
step("sit", lambda: duck.do("sit"), 1)      # already sitting: no-op
step("stand", lambda: duck.do("stand"), 5)
step("kick", lambda: duck.do("kick"), 3)
step("walk", lambda: duck.walk(0.35, 0, 0), 3)  # velstand in the sim barely steps below ~0.25 m/s
step("stop", lambda: duck.stop(), 1)
step("drop", lambda: duck.do("drop"), 3)
step("stand", lambda: duck.do("stand"), 8)      # gets up from the floor
step("stop", lambda: duck.stop(), 2)
step("roll", lambda: duck.do("roll"), 4)
step("pick", lambda: duck.do("pick"), 5)
print("healthy:", duck.call("robot.health").get("healthy"))
