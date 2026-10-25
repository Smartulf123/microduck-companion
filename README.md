# microduck-companion

Voice commands (in Norwegian) and, later, follow-me for
[Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck). Offline, CPU only.
A community project, not affiliated with Pollen Robotics. Apache-2.0.

| Say | Duck does |
|---|---|
| sitt / sett deg | sit |
| stå / stå opp | stand |
| spark | kick (left foot) |
| plukk / plukk opp | beak grab |
| rull | forward roll |
| fall / sov / besvim | motors off: drops on its face (stå or våkn opp gets it up) |
| følg / følg meg | follows the person, keeping ~0.7 m |
| kom / kom hit | walks to the person and stops ~0.5 m away |
| rulleskøyter / skøyter | roller skates on/off (≈0.6 m/s; forward only, steering is loose) |
| stopp | stop (always wins) |

In the 3D window, move the person with the arrow keys (click the window first).

A PS5 controller (USB or Bluetooth) works beside voice — mapping at the top of `gamepad.py`:
left stick walks, ✕ sit/stand, ○ fall, △ roll, □ beak grab, L1/R1 kicks, Options get up,
R2 follow on/off, L2 come here, D-pad up skates on/off, Create stop. Needs `pip install pygame`.

Real eyes (`eyes.py`, needs `pip install opencv-python onnxruntime` and
`models/yolox_nano.onnx` from github.com/Megvii-BaseDetection/YOLOX releases 0.1.1rc0): finds
people with YOLOX-Nano, locks on to the one in front, and keeps following that one by how they look.

```
.venv\Scripts\python eyes.py --height 0.2 --pitch 15   # BRIO on the floor, tilted up: the duck's view
.venv\Scripts\python main.py --eyes webcam             # "følg" locks on to you through the webcam
```

Skates: in the sim the 3D window restarts with a wheeled duck (`sim_world.py --skates`); on the
real duck you clip the wheels on and hold it while it switches to Pollen's roller mode.

## Setup (Windows, once)

```
python -m venv .venv
.venv\Scripts\pip install pywhispercpp sounddevice numpy
curl -L -o models/nb-whisper-base-q5_0.bin https://huggingface.co/NbAiLab/nb-whisper-base/resolve/main/ggml-model-q5_0.bin
```

Simulator (WSL2 Ubuntu 24.04): Rust, `build-essential pkg-config libudev-dev libasound2-dev
libdbus-1-dev socat`, Pollen's repos cloned into `~/Pollen/`, and `~/Pollen/microduck_rl/.venv`
with `mujoco numpy onnxruntime` plus a `.pth` pointing at `microduck_rl/src`.

## Run

```
powershell -ExecutionPolicy Bypass -File start.ps1    # 3D duck + robot software + voice, all at once
```

The 3D view runs on Windows (`pip install mujoco` in `.venv`) because WSL only has software
OpenGL here; Pollen's robot software runs in WSL (`sim.sh`) and connects to it. `sim_world.py`
wraps Pollen's body server to add the person; `follow.py` steers toward whatever `Eyes` sees —
`SimEyes` in the sim, a camera on the real duck.

```
.venv\Scripts\python voice.py --debug   # voice only, no duck
.venv\Scripts\python smoke_test.py      # every behaviour, after pulling a new Pollen version
```

`duck_adapter.py` is the only file that talks to the duck. Pollen's repos go in `ref/`
(read-only, not tracked). Tested against microduck `942e58e`, microduck_rl `b02fdea`.
