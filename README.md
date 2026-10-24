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
| stopp | stop (always wins) |

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
OpenGL here; Pollen's robot software runs in WSL (`sim.sh`) and connects to it.

```
.venv\Scripts\python voice.py --debug   # voice only, no duck
.venv\Scripts\python smoke_test.py      # every behaviour, after pulling a new Pollen version
```

`duck_adapter.py` is the only file that talks to the duck. Pollen's repos go in `ref/`
(read-only, not tracked). Tested against microduck `942e58e`, microduck_rl `b02fdea`.
