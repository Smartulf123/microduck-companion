# microduck-companion

Voice commands (in Norwegian) and, later, follow-me for
[Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck). Offline, CPU only.
A community project, not affiliated with Pollen Robotics. Apache-2.0.

## Setup (once)

```
python -m venv .venv
.venv\Scripts\pip install pywhispercpp sounddevice numpy
curl -L -o models/nb-whisper-base-q5_0.bin https://huggingface.co/NbAiLab/nb-whisper-base/resolve/main/ggml-model-q5_0.bin
```

## Run

```
.venv\Scripts\python voice.py            # say: sitt, stå, spark, plukk, rull, stopp
.venv\Scripts\python voice.py --debug    # also shows what Whisper heard, and how long it took
.venv\Scripts\python voice.py --list-mics
.venv\Scripts\python voice.py --mic 3
```

Words live in `WORDS` at the top of `voice.py`.

Pollen's repos go in `ref/` (read-only, not tracked). Tested against microduck `942e58e`,
microduck_rl `b02fdea`.
