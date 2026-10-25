"""Follow-me: steer the duck toward a person it can see.

Eyes answer look() with (distance m, bearing rad, positive = to the left) or None when nobody is in
view. SimEyes reads the simulator; eyes.CameraEyes finds and locks on to you in a camera image.
"""
import json
import math
import socket
import threading
import time

# Measured in the sim with velstand (the real duck's walk policy): below ~0.25 m/s it does not step,
# turning on the spot is unreliable, and |yaw| above 0.6 rad/s turns the wrong way. So it walks in
# arcs. Retune these on the real robot.
MIN_VX, MAX_VX, MAX_YAW = 0.25, 0.45, 0.6
KEEP = {"follow": 0.7, "come": 0.5}  # metres to stop short of the person
SEARCH_S = 6.0                        # after losing sight: arc toward where they went, then wait


class SimEyes:
    """A pretend camera on the duck: sees the sim's person only in front and within range."""

    HALF_FOV = math.radians(45)
    RANGE = 5.0

    def __init__(self, address=("127.0.0.1", 7820)):
        self.address, self.file = address, None

    def look(self):
        try:
            if self.file is None:  # (re)connect: the sim restarts when the skates go on or off
                self.file = socket.create_connection(self.address, timeout=2).makefile("rw")
            self.file.write("{}\n")
            self.file.flush()
            truth = json.loads(self.file.readline())
        except (OSError, ValueError):
            self.file = None
            return None
        (px, py), (dx, dy, yaw) = truth["person"], truth["duck"]
        x, y = px - dx, py - dy
        ahead = x * math.cos(yaw) + y * math.sin(yaw)
        left = -x * math.sin(yaw) + y * math.cos(yaw)
        distance, bearing = math.hypot(ahead, left), math.atan2(left, ahead)
        return (distance, bearing) if distance < self.RANGE and abs(bearing) < self.HALF_FOV else None


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class Follower:
    def __init__(self, duck, eyes, say=print):
        self.duck, self.eyes, self.say = duck, eyes, say
        self.mode = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, mode: str):
        """"follow" keeps going; "come" stops once it has arrived. Eyes that can lock on
        (eyes.CameraEyes) lock on to whoever is in front first."""
        lock = getattr(self.eyes, "lock", None)
        if lock is not None and not lock():
            self.say("I can't see anyone to follow")
            return
        with self.lock:
            self.mode = mode

    def cancel(self):
        with self.lock:
            self.mode = None
        self.duck.walk(0, 0, 0)

    def _loop(self):
        last_bearing, lost_at, waiting = 0.0, None, False
        while True:
            time.sleep(0.1)
            with self.lock:
                mode = self.mode
            if mode is None:
                lost_at, waiting = None, False
                continue
            seen = self.eyes.look()
            if seen is None:
                lost_at = lost_at or time.time()
                if time.time() - lost_at < SEARCH_S:
                    self.duck.walk(MIN_VX, 0, math.copysign(MAX_YAW, last_bearing))
                elif not waiting:
                    self.duck.walk(0, 0, 0)
                    self.say("lost you — waiting")
                    waiting = True
                continue
            if waiting:
                self.say("there you are")
            lost_at, waiting = None, False
            distance, bearing = seen
            last_bearing = bearing
            keep = KEEP[mode]
            if distance > keep:
                vx = clip(MIN_VX + 0.3 * (distance - keep), MIN_VX, MAX_VX)
                self.duck.walk(vx, 0, clip(1.2 * bearing, -MAX_YAW, MAX_YAW))
            else:
                self.duck.walk(0, 0, 0)
                if mode == "come":
                    with self.lock:
                        if self.mode == "come":
                            self.mode = None
                    self.say("here I am")
