"""Pollen's simulated duck body, plus a person to follow, plus roller skates.

    python -m sim_world --host <ip> --port 7801      # same arguments as Pollen's body_server
    python -m sim_world ... --skates                 # the duck on wheels

Move the person with the arrow keys in the 3D window (click the window first). The duck sees the
person through a pretend camera: sim_world serves the true positions on 127.0.0.1:7820, and
follow.SimEyes turns them into what a camera would give — a distance and a bearing, only when the
person is in front. This file wraps Pollen's body_server without editing it.

MuJoCo cannot add wheels to a running world, so skates.swap_feet() restarts this process with or without
--skates; robotd reconnects on its own but must be powered off and on to take the new body. On
skates a "hand" holds the duck still for a few seconds after it powers up: robotd ramps the legs
to the home pose with nothing balancing, which on rolling wheels tips it over (on the real robot,
you are that hand).
"""
import argparse
import json
import math
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from mjlab_microduck.sim import body_server as bs

START = (1.2, 0.0)  # metres in front of the duck
STEP = 0.1          # per arrow key press
ARROWS = {265: (STEP, 0.0), 264: (-STEP, 0.0), 263: (0.0, STEP), 262: (0.0, -STEP)}  # up down left right
TRUTH = ("127.0.0.1", 7820)
HOLD_S = 3.0        # the steadying hand, on skates

world = None


def add_person(spec: mujoco.MjSpec) -> None:
    person = spec.worldbody.add_body(name="person", mocap=True, pos=[*START, 0.0])
    skin, shirt, pants = [0.9, 0.75, 0.6, 1], [0.2, 0.45, 0.8, 1], [0.15, 0.15, 0.2, 1]
    capsule = mujoco.mjtGeom.mjGEOM_CAPSULE

    def part(kind, size, rgba, fromto=None, pos=None):
        geom = person.add_geom(type=kind, size=size, rgba=rgba, contype=0, conaffinity=0)
        if fromto is not None:
            geom.fromto = fromto
        if pos is not None:
            geom.pos = pos

    for y in (-0.1, 0.1):
        part(capsule, [0.065, 0, 0], pants, fromto=[0, y, 0.06, 0, y, 0.85])          # legs
        part(capsule, [0.05, 0, 0], shirt, fromto=[0, y * 2.1, 1.38, 0, y * 2.4, 0.95])  # arms
    part(capsule, [0.17, 0, 0], shirt, fromto=[0, 0, 0.95, 0, 0, 1.35])                # torso
    part(mujoco.mjtGeom.mjGEOM_SPHERE, [0.11, 0, 0], skin, pos=[0, 0, 1.6])           # head


def build_world(scene, count):
    if count != 1:
        raise SystemExit("sim_world supports one duck")
    spec = mujoco.MjSpec.from_file(str(scene))
    add_person(spec)
    return spec.compile()


_world_init = bs.World.__init__


def world_init(self, scene, count=1):
    global world
    _world_init(self, scene, count)
    world = self


_launch_passive = mujoco.viewer.launch_passive


def launch_passive(model, data, **kwargs):
    def on_key(key):
        if key in ARROWS:
            dx, dy = ARROWS[key]
            with world.lock:
                world.data.mocap_pos[0][:2] += (dx, dy)

    viewer = _launch_passive(model, data, key_callback=on_key, **kwargs)

    def aim():  # behind the duck, looking the way it faces: arrow keys match the screen
        with viewer.lock():
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.2, -18, 0
            viewer.cam.lookat[:] = (1.0, 0.0, 0.6)

    aim()
    threading.Timer(1.5, aim).start()  # the viewer sometimes resets its camera as it opens
    return viewer


# ── the steadying hand (skates only) ─────────────────────────────────────────

_holds = 0


def hold(body):
    global _holds
    with body.world.lock:
        body.remember()
        body.released = False
        _holds += 1
        mine = _holds

    def let_go():
        if _holds == mine:
            body.released = True

    threading.Timer(HOLD_S, let_go).start()


_set_torque = bs.Body.set_torque


def set_torque(self, on):
    """Pollen's body already waits, frozen, until powered; on skates the hand keeps holding
    through the power-up ramp too."""
    _set_torque(self, on)
    if on:
        hold(self)


# ── truth, and the restart that swaps the feet ───────────────────────────────

def positions() -> dict:
    data = world.data
    with world.lock:
        person = data.mocap_pos[0][:2].tolist()
        trunk = world.bodies[0].trunk
        x, y = data.qpos[trunk : trunk + 2].tolist()
        w, qx, qy, qz = data.qpos[trunk + 3 : trunk + 7].tolist()
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return {"person": person, "duck": [x, y, yaw], "skates": SKATES}


def restart(skates: bool):
    """Start the other world, then leave so it can take the ports."""
    args = [a for a in ARGS if a != "--skates"] + (["--skates"] if skates else [])
    subprocess.Popen([sys.executable, "-m", "sim_world", *args], cwd=os.getcwd())
    threading.Timer(0.3, lambda: os._exit(0)).start()


class Truth(socketserver.StreamRequestHandler):
    """One JSON line in, one out. {} -> positions; {"person": [x, y]} moves the person;
    {"skates": true|false} restarts with or without wheels."""

    def handle(self):
        for raw in self.rfile:
            request = json.loads(raw)
            if "person" in request:
                with world.lock:
                    world.data.mocap_pos[0][:2] = request["person"]
            answer = positions()
            if "skates" in request and request["skates"] != SKATES:
                restart(request["skates"])
                answer["restarting"] = True
            self.wfile.write((json.dumps(answer) + "\n").encode())
            self.wfile.flush()


class TruthServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def wait_for_ports(*addresses):
    """The previous world is still letting go of its ports for a moment after a swap."""
    for address in addresses:
        for _ in range(100):
            with socket.socket() as probe:
                try:
                    probe.bind(address)
                    break
                except OSError:
                    time.sleep(0.1)


if __name__ == "__main__":
    ours = argparse.ArgumentParser(add_help=False)
    ours.add_argument("--skates", action="store_true")
    ours.add_argument("--host", default="127.0.0.1")
    ours.add_argument("--port", type=int, default=7801)
    mine, rest = ours.parse_known_args()
    ARGS, SKATES = sys.argv[1:], mine.skates
    sys.argv = [sys.argv[0], *rest, "--host", mine.host, "--port", str(mine.port)]
    if SKATES:
        sys.argv += ["--scene", str(Path(bs.SCENES) / "scene_rollers.xml"), "--keyframe", "STAND"]
        bs.Body.set_torque = set_torque
    wait_for_ports((mine.host, mine.port), TRUTH)
    bs.build_world = build_world
    bs.World.__init__ = world_init
    mujoco.viewer.launch_passive = launch_passive
    threading.Thread(target=TruthServer(TRUTH, Truth).serve_forever, daemon=True).start()
    bs.main()
