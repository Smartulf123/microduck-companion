"""Laptop mic -> Norwegian speech (NB-Whisper, offline) -> command string.

    python voice.py            # prints "heard: sit" as you speak
    python voice.py --debug    # also prints raw transcripts and mic level
"""
import argparse
import difflib
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from pywhispercpp.model import Model

MODEL = Path(__file__).parent / "models" / "nb-whisper-base-q5_0.bin"
RATE = 16000
BLOCK = 480  # 30 ms

# What you say -> what the duck does. The only Norwegian in the code.
WORDS = {
    "stopp": "stop", "stop": "stop", "stans": "stop",
    "sitt": "sit", "sett deg": "sit", "sitt ned": "sit",
    "stå": "stand", "stå opp": "stand", "reis deg": "stand",
    "spark": "kick", "sparke": "kick",
    "plukk": "pick", "plukk opp": "pick",
    "rull": "roll", "rulle": "roll",
    "fall": "drop", "besvim": "drop", "sov": "drop",
    "våkn": "stand", "våkn opp": "stand",
    "følg": "follow", "følg meg": "follow",
    "kom": "come", "kom hit": "come", "kom her": "come",
}
MAX_WORDS = 4  # commands are short; a word inside a longer sentence is not a command (stop excepted)


def match(text: str) -> str | None:
    """The command in a transcript, or None. "stop" wins over anything else said with it."""
    text = re.sub(r"[^\wæøå ]", " ", text.lower())
    tokens = text.split()
    found = []
    for i, tok in enumerate(tokens):
        pair = " ".join(tokens[i:i + 2])
        if pair in WORDS:
            found.append(WORDS[pair])
            continue
        close = difflib.get_close_matches(tok, WORDS, n=1, cutoff=0.85)
        if close:
            found.append(WORDS[close[0]])
    if "stop" in found:
        return "stop"
    if len(tokens) > MAX_WORDS:
        return None
    # More than two commands in one breath is Whisper making things up, not you.
    return found[0] if 0 < len(found) <= 2 else None


class Ears:
    """Cuts the mic into utterances by loudness, transcribes each, calls on_command."""

    def __init__(self, on_command, debug=False, device=None):
        self.on_command = on_command
        self.debug = debug
        self.device = device
        self.model = Model(str(MODEL), redirect_whispercpp_logs_to=None,
                           n_threads=6, language="no", no_context=True, single_segment=True,
                           print_progress=False)
        self.clips: queue.Queue = queue.Queue(maxsize=3)

    def transcribe(self, audio: np.ndarray) -> str:
        return " ".join(s.text for s in self.model.transcribe(audio)).strip()

    def _worker(self):
        while True:
            clip = self.clips.get()
            t0 = time.perf_counter()
            text = self.transcribe(clip)
            command = match(text)
            if self.debug:
                print(f"  [{len(clip) / RATE:.1f}s audio, {time.perf_counter() - t0:.2f}s] {text!r}")
            if command:
                self.on_command(command, text)

    def run(self):
        threading.Thread(target=self._worker, daemon=True).start()
        floor, speech, silent, pre = 0.003, [], 0, []
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=BLOCK, device=self.device) as mic:
            print("listening — sitt, stå, spark, plukk, rull, fall, følg, kom hit, stopp (Ctrl-C quits)")
            while True:
                block = mic.read(BLOCK)[0][:, 0].copy()
                rms = float(np.sqrt(np.mean(block ** 2)))
                loud = rms > max(floor * 3, 0.008)
                if not speech:
                    floor = 0.95 * floor + 0.05 * rms if not loud else floor
                    pre = (pre + [block])[-7:]  # ~200 ms before the word starts
                    if loud:
                        speech, silent = pre[:], 0
                    continue
                speech.append(block)
                silent = 0 if loud else silent + 1
                length = len(speech) * BLOCK / RATE
                if silent >= 13 or length > 3.0:  # ~400 ms quiet ends it
                    if length > 0.35:
                        clip = np.concatenate(speech)
                        if self.clips.full():  # behind: drop the oldest, keep up
                            self.clips.get_nowait()
                        self.clips.put(clip)
                    speech, pre = [], []


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # æøå on a Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mic", type=int, default=None, help="input device index (see --list-mics)")
    ap.add_argument("--list-mics", action="store_true")
    args = ap.parse_args()
    if args.list_mics:
        print(sd.query_devices())
        return
    ears = Ears(lambda cmd, text: print(f"heard: {cmd}   ({text})"), args.debug, args.mic)
    try:
        ears.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
