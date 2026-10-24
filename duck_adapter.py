"""The ONLY place that talks to the duck.

    duck = Duck()               # sim: robotd's socket, bridged to TCP 127.0.0.1:7810
    duck.do("sit"); duck.walk(0.15, 0, 0); duck.stop()

It speaks robotd's own JSON-RPC (one object per line, the same calls the gamepad and the console
send), so the real robot is the same code at a different address — an ssh tunnel to its
/run/robotd.sock. Methods and params: ref/microduck/duck-ipc-proto/src/lib.rs.
"""
import itertools
import json
import socket
import threading
import time

ADDRESS = ("127.0.0.1", 7810)

# Our behaviour names -> robotd skill names (robot.do). sit/stand are special: one toggle.
SKILLS = {"kick": "kick_left", "pick": "ground_pick", "roll": "roulade"}

MOVE_HZ = 20  # robotd stops the duck if robot.move is quiet for 500 ms (deadman)


class Duck:
    def __init__(self, address=ADDRESS):
        self.sock = socket.create_connection(address, timeout=5)
        self.file = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.velocity = (0.0, 0.0, 0.0)
        threading.Thread(target=self._keep_moving, daemon=True).start()
        self.call("robot.enable", {"on": True})  # hand the robot to the policy; it stands itself up

    # ── the interface ────────────────────────────────────────────────────────

    def do(self, behaviour: str) -> str:
        """sit, stand, kick, pick, roll. Returns "ok" or why the duck refused."""
        if behaviour in ("sit", "stand"):
            sitting = self.call("robot.policies").get("sitting")
            if sitting == (behaviour == "sit"):
                return "ok"  # already there
            self.velocity = (0.0, 0.0, 0.0)
            result = self.call("robot.do", {"skill": "sit_toggle"})
        elif behaviour in SKILLS:
            self.velocity = (0.0, 0.0, 0.0)
            result = self.call("robot.do", {"skill": SKILLS[behaviour]})
        else:
            return f"unknown behaviour {behaviour!r}"
        return "ok" if result.get("accepted") else result.get("reason", "refused")

    def walk(self, vx: float, vy: float, yaw_rate: float) -> None:
        """m/s forward, m/s left, rad/s (positive turns left). Held until changed or stop()."""
        self.velocity = (vx, vy, yaw_rate)

    def stop(self) -> None:
        """Stop walking, now. A skill already running (a 0.5 s kick, a 1 s roll) finishes."""
        self.velocity = (0.0, 0.0, 0.0)
        self.call("robot.stop")

    # ── wire ─────────────────────────────────────────────────────────────────

    def call(self, method: str, params: dict | None = None) -> dict:
        """A request: send it, wait for the answer with the same id."""
        msg = {"jsonrpc": "2.0", "id": next(self.ids), "method": method}
        if params is not None:
            msg["params"] = params
        with self.lock:
            self.file.write(json.dumps(msg) + "\n")
            self.file.flush()
            while True:
                answer = json.loads(self.file.readline())
                if answer.get("id") == msg["id"]:
                    break
        if "error" in answer:
            raise RuntimeError(f"{method}: {answer['error'].get('message')}")
        return answer.get("result") or {}

    def _notify(self, method: str, params: dict) -> None:
        with self.lock:
            self.file.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
            self.file.flush()

    def _keep_moving(self):
        while True:
            vx, vy, vyaw = self.velocity
            if (vx, vy, vyaw) != (0.0, 0.0, 0.0):
                self._notify("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})
            time.sleep(1 / MOVE_HZ)
