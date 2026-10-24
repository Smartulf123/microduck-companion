"""Roller skates on/off.

On the real duck you clip the wheels on and hold it while it switches; in the sim, sim_world swaps
its feet for wheels. Either way the duck then runs Pollen's roller policy: fast (≈0.6 m/s against
≈0.17 walking), but it was trained for straight lines, so steering is loose and braking tips it over.
"""
import json
import socket
import threading
import time

MAX_VX, MAX_YAW = 0.6, 0.3  # Pollen's own roller pad limits (forward only: braking falls)
SIM_WORLD = ("127.0.0.1", 7820)


def swap_feet(on: bool, timeout=30.0) -> None:
    """Sim only: ask sim_world for wheels (or feet), and wait until the new world is up."""
    def ask(request):
        with socket.create_connection(SIM_WORLD, timeout=2) as s:
            f = s.makefile("rw")
            f.write(json.dumps(request) + "\n")
            f.flush()
            return json.loads(f.readline())

    if not ask({"skates": on}).get("restarting"):
        return
    time.sleep(1.0)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if ask({}).get("skates") == on:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError("the sim did not come back")


class Skates:
    def __init__(self, duck, follower, say=print, sim=True):
        self.duck, self.follower, self.say, self.sim = duck, follower, say, sim
        self.on = duck.mode() == "roller"
        self.busy = threading.Lock()

    def toggle(self):
        if self.busy.locked():
            return
        threading.Thread(target=self._switch, args=(not self.on,), daemon=True).start()

    def _switch(self, on):
        with self.busy:
            self.follower.cancel()
            mode = "roller" if on else "walk"
            self.say("skates on — hold tight" if on else "skates off")
            if not self.sim:
                self.say("put the wheels on and hold the duck" if on else "take the wheels off and hold the duck")
            self.duck.set_mode(mode)
            self._wait(lambda: self.duck.mode() == mode)
            if self.sim:
                swap_feet(on)
                self._wait(lambda: self.duck.call("robot.health").get("healthy"))
                self.duck.call("robot.relax")  # robotd thinks the old body is still powered;
                time.sleep(0.5)                # off and on makes it take the new one
            self.duck.do("stand")
            self.on = on
            self.say("ready — push forward!" if on else "back on my feet")

    @staticmethod
    def _wait(done, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not done():
            time.sleep(0.3)
        time.sleep(0.5)
