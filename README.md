# microduck-companion

Talk to your [Microduck](https://github.com/pollen-robotics/microduck) in Norwegian, drive it with a
PS5 controller, and have it find you — by your face — and follow you around. Everything runs
offline on a laptop CPU: no cloud, no GPU, no paid APIs.

A community project, not affiliated with Pollen Robotics. Apache-2.0.

**Status:** built and tested against Pollen's own robot software driving a simulated duck, before
the real robots ship. The code talks to the duck exactly the way the real one is driven, so the
move to hardware is a different address in one file — but it has not run on a real duck yet.

## What it does

| Say (Norwegian) | Controller | The duck |
|---|---|---|
| sitt / sett deg | ✕ | sits (✕ again stands) |
| stå / stå opp / våkn opp | Options | stands up, also after a fall |
| spark | L1 / R1 | kicks (left / right) |
| plukk / plukk opp | □ | beak grab |
| rull | △ | forward roll |
| fall / sov / besvim | ○ | motors off: drops on its face |
| følg / følg meg | R2 | looks up, finds you, follows ~0.7 m behind |
| kom / kom hit | L2 | looks up, finds you, walks over and stops |
| rulleskøyter / skøyter | D-pad up | roller skates on/off (fast, forward only, loose steering) |
| stopp | Create | stop — always wins |
| | left stick | walk and turn |

The words are one table at the top of `voice.py`; put in your own language's.

**Følg, step by step:** the duck stops, gives a questioning quack, looks up where a standing
person's face would be (ahead, left, right), and once it recognises your face it quacks hello and
locks on to the person under it. While you walk away it keeps track of you by your clothes; if it
loses you, it searches, then waits — and seeing your face again picks you back up. Without an
enrolled face it follows whoever is in front.

## How it fits together

| File | Does |
|---|---|
| `duck_adapter.py` | The only file that talks to the duck: Pollen's own JSON-RPC (`robot.do`, `robot.move`, …) |
| `voice.py` | Mic → [NB-Whisper](https://huggingface.co/NbAiLab/nb-whisper-base) (Norwegian Whisper, whisper.cpp) → command |
| `gamepad.py` | PS5 / any SDL controller → commands, with rumble |
| `follow.py` | Steers toward whatever the eyes see; the følg ritual |
| `eyes.py` | Camera → people (YOLOX-Nano) → lock on to you by appearance → distance and direction |
| `faces.py` | Face recognition (YuNet + SFace): enrol once, then "is this you?" |
| `skates.py` | Roller skates on/off: Pollen's roller mode, plus wheels in the sim |
| `sim_world.py` | Pollen's simulated body (MuJoCo), plus a person to follow and skates |
| `sim.sh` | Pollen's robot software (`robotd`) in WSL, driving that body |
| `start.ps1` | Starts all of it |

## Setup

Tested on Windows 11 with WSL2, Python 3.12, no NVIDIA GPU.

**1. This repo and Python**

```
git clone https://github.com/Smartulf123/microduck-companion
cd microduck-companion
python -m venv .venv
.venv\Scripts\pip install pywhispercpp sounddevice numpy pygame opencv-python onnxruntime mujoco
```

**2. Models** (into `models/`, not in git)

```
mkdir models
curl -L -o models/nb-whisper-base-q5_0.bin https://huggingface.co/NbAiLab/nb-whisper-base/resolve/main/ggml-model-q5_0.bin
curl -L -o models/yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

**3. Pollen's code**, read-only, for the simulator's body (Windows side):

```
git clone https://github.com/pollen-robotics/microduck_rl ref/microduck_rl
```

**4. The robot software in WSL** (`wsl --install Ubuntu-24.04`, then inside Ubuntu):

```
sudo apt install build-essential pkg-config libudev-dev libasound2-dev libdbus-1-dev socat python3-venv
curl -sSf https://sh.rustup.rs | sh -s -- -y
mkdir -p ~/Pollen && cd ~/Pollen
git clone https://github.com/pollen-robotics/microduck && git clone https://github.com/pollen-robotics/microduck_rl
cd microduck && cargo build -p robotd -p tof -p sounds
cd ../microduck_rl && python3 -m venv .venv && .venv/bin/pip install onnxruntime
```

`sim.sh` downloads Pollen's official policy set on first run.

## Run

```
powershell -ExecutionPolicy Bypass -File start.ps1                 # sim duck + voice + controller
powershell -ExecutionPolicy Bypass -File start.ps1 --eyes webcam   # follow YOU through the webcam
```

In the 3D window, the arrow keys move the simulated person (click the window first).

```
.venv\Scripts\python faces.py enroll   # once: look at the webcam for a few seconds
.venv\Scripts\python faces.py          # live check: green = you
.venv\Scripts\python eyes.py           # what the eyes see; L lock, F find my face
.venv\Scripts\python voice.py --debug  # voice only
.venv\Scripts\python smoke_test.py     # every behaviour, after pulling a new Pollen version
```

Not the first webcam? `setx DUCK_CAMERA 1` (or pass `--camera 1`). For the duck's view, put the
webcam on the floor tilted up: `eyes.py --height 0.2 --pitch 15`.

**Privacy:** your face is stored as a fingerprint in `me/face.npy` on your own machine, ignored by
git. Nothing is sent anywhere.

## Things worth knowing

- **The sim duck walks slower than the real one will.** Pollen's simulated body tracks about a
  third of the trained 0.4 m/s, and only turns while walking. The speeds in `follow.py` and
  `gamepad.py` are tuned to the sim; retune on a real duck.
- **The sim's 3D view runs on Windows**, not in WSL, because WSL here has only software OpenGL.
- **Skates:** Pollen's roller policy was trained for straight lines, so steering is loose and
  braking tips it over. The sim restarts its world to put wheels on; on a real duck you clip them
  on and hold it while it switches.
- **Your clothes keep you tracked**, your face tells the duck it is you. Two people dressed alike
  can still swap places in its eyes.

Tested against microduck `942e58e` and microduck_rl `b02fdea`.

## Credits

[Pollen Robotics](https://github.com/pollen-robotics) — the Microduck, its robot software, policies
and simulator (Apache-2.0) · [NB-Whisper](https://huggingface.co/NbAiLab) by the National Library of
Norway (Apache-2.0) · [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (MIT) ·
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) ·
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) YuNet (MIT) and SFace (Apache-2.0) ·
[MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
