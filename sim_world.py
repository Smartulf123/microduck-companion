"""Pollen's simulated duck body, plus a person to follow.

    python -m sim_world --host <ip> --port 7801      # same arguments as Pollen's body_server

Move the person with the arrow keys in the 3D window (click the window first). The duck sees the
person through a pretend camera: sim_world serves the true positions on 127.0.0.1:7820, and
follow.SimEyes turns them into what a camera would give — a distance and a bearing, only when the
person is in front. This file wraps Pollen's body_server without editing it.
"""
import json
import math
import socketserver
import threading

import mujoco
import mujoco.viewer
from mjlab_microduck.sim import body_server as bs

START = (1.2, 0.0)  # metres in front of the duck
STEP = 0.1          # per arrow key press
ARROWS = {265: (STEP, 0.0), 264: (-STEP, 0.0), 263: (0.0, STEP), 262: (0.0, -STEP)}  # up down left right

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


def positions() -> dict:
    data = world.data
    with world.lock:
        person = data.mocap_pos[0][:2].tolist()
        trunk = world.bodies[0].trunk
        x, y = data.qpos[trunk : trunk + 2].tolist()
        w, qx, qy, qz = data.qpos[trunk + 3 : trunk + 7].tolist()
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return {"person": person, "duck": [x, y, yaw]}


class Truth(socketserver.StreamRequestHandler):
    """One JSON line in, one out. {} -> positions; {"person": [x, y]} also moves the person."""

    def handle(self):
        for raw in self.rfile:
            request = json.loads(raw)
            if "person" in request:
                with world.lock:
                    world.data.mocap_pos[0][:2] = request["person"]
            self.wfile.write((json.dumps(positions()) + "\n").encode())
            self.wfile.flush()


class TruthServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    bs.build_world = build_world
    bs.World.__init__ = world_init
    mujoco.viewer.launch_passive = launch_passive
    threading.Thread(target=TruthServer(("127.0.0.1", 7820), Truth).serve_forever, daemon=True).start()
    bs.main()
