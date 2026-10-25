"""Camera eyes: find people, lock on to one, and keep following that one.

    python eyes.py                      # webcam with a preview window; L locks, F finds your face, U unlocks
    python eyes.py --height 0.2 --pitch 15   # camera on the floor, tilted up: the duck's view

look() gives (distance m, bearing rad, positive = left) of the locked person, like follow.SimEyes.

Lock-on: when told to lock (on "følg"), it takes the person most in front and remembers how they
look — colour histograms of the upper and lower half of their box (from the duck's height, mostly
trousers and shoes). Each frame it picks the person who looks most like that and is near where they
were. Two people dressed alike can still fool it; then it stops and waits, and "følg" re-locks.

Distance comes from where the feet are in the image (the floor is flat, the camera height known);
bearing from how far off-centre the person is. Both need the camera's field of view — on the real
duck, calibrate it once.
"""
import argparse
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).parent / "models" / "yolox_nano.onnx"  # YOLOX-Nano, Apache-2.0, COCO class 0 = person
CAMERA = int(os.environ.get("DUCK_CAMERA", 0))  # which webcam; set DUCK_CAMERA if yours is not the first
SIZE = 416
MATCH = 0.45      # appearance distance (0 same .. 1 nothing alike) above which it is not "you"
FRESH_S = 0.5     # a sighting older than this counts as lost


class Detector:
    def __init__(self):
        self.session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = SIZE // stride
            ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grids, self.strides = np.concatenate(grids), np.concatenate(strides)

    def people(self, frame, threshold=0.45):
        """[(x1, y1, x2, y2, score)] in frame pixels."""
        h, w = frame.shape[:2]
        scale = min(SIZE / h, SIZE / w)
        padded = np.full((SIZE, SIZE, 3), 114, np.uint8)
        padded[: int(h * scale), : int(w * scale)] = cv2.resize(frame, (int(w * scale), int(h * scale)))
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {"images": blob})[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        score = out[:, 4] * out[:, 5]  # objectness x person
        keep = score > threshold
        boxes = np.concatenate([xy - wh / 2, wh], 1)[keep] / scale
        score = score[keep]
        picked = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), threshold, 0.45)
        found = []
        for i in np.array(picked).flatten():
            x, y, bw, bh = boxes[i]
            if bw > 8 and bh > 8:
                found.append((max(0, x), max(0, y), min(w, x + bw), min(h, y + bh), float(score[i])))
        return found


def signature(frame, box):
    """How a person looks: hue/saturation histograms of the top and bottom half of their box."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    halves = []
    for part in (crop[: len(crop) // 2], crop[len(crop) // 2 :]):
        hist = cv2.calcHist([part], [0, 1], None, [16, 8], [0, 180, 0, 256])
        halves.append(cv2.normalize(hist, hist).flatten())
    return halves


def unlike(a, b):
    return sum(cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)) / 2


class CameraEyes:
    def __init__(self, camera=None, hfov=70.0, height=0.8, pitch=0.0, show=True):
        """hfov: horizontal field of view, degrees. height: camera above the floor, m.
        pitch: degrees the camera is tilted UP (negative = looking down)."""
        self.hfov, self.height, self.pitch = math.radians(hfov), height, math.radians(pitch)
        camera = CAMERA if camera is None else camera
        self.capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise SystemExit(f"no camera {camera}")
        self.detector, self.show = Detector(), show
        self.quit_on_close = False  # eyes.py on its own exits when its window is closed
        self.me, self.box, self.seen_at = None, None, 0.0
        self.sighting = None
        self.want_lock = threading.Event()
        self.lock_done = threading.Event()
        self.want_face = threading.Event()
        self.face_done = threading.Event()
        self.faces = None
        try:
            from faces import Faces
            faces = Faces()
            self.faces = faces if faces.known() else None  # not enrolled: lock on whoever is in front
        except Exception:  # models missing: same
            pass
        self.face_boxes = []
        threading.Thread(target=self._run, daemon=True).start()

    # ── what follow.py uses ──────────────────────────────────────────────────

    def look(self):
        if self.sighting and time.time() - self.seen_at < FRESH_S:
            return self.sighting
        return None

    def lock(self, timeout=2.0) -> bool:
        """Lock on to the person most in front. False if nobody is in view."""
        self.lock_done.clear()
        self.want_lock.set()
        self.lock_done.wait(timeout)
        return self.me is not None

    def find_me(self, timeout=1.5) -> bool:
        """Look for your face for up to `timeout` s; on a match, lock on to the body under it.
        Without an enrolled face this is lock(): whoever is in front."""
        if self.faces is None:
            return self.lock(timeout)
        self.face_done.clear()
        self.want_face.set()
        found = self.face_done.wait(timeout)
        self.want_face.clear()
        return found

    def unlock(self):
        self.me, self.box, self.sighting = None, None, None

    # ── the loop ─────────────────────────────────────────────────────────────

    def _run(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.1)
                continue
            people = self.detector.people(frame)
            if self.want_lock.is_set():
                self._lock(frame, people)
            target = self._find(frame, people) if self.me is not None else None
            lost = self.me is not None and target is None and time.time() - self.seen_at > 1.0
            self.face_boxes = []
            if self.faces is not None and (self.want_face.is_set() or lost):
                # asked to find you, or you got lost and your face might bring you back
                self.face_boxes = self.faces.find(frame)
                body = self._body_under_my_face(frame, people)
                if body is not None:
                    self.me, self.box, self.seen_at = signature(frame, body), body, time.time()
                    target = body
                    self.face_done.set()
            if target is not None:
                self.box, self.seen_at = target, time.time()
                self.sighting = self._where(frame, target)
            if self.show:
                self._draw(frame, people, target)

    def _lock(self, frame, people):
        w = frame.shape[1]
        if people:  # biggest and most central: the one standing in front
            best = max(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * (1 - abs((b[0] + b[2]) / 2 / w - 0.5)))
            self.me, self.box, self.seen_at = signature(frame, best), best, time.time()
        else:
            self.unlock()
        self.want_lock.clear()
        self.lock_done.set()

    def _body_under_my_face(self, frame, people):
        from faces import SAME
        mine = [f for f in self.face_boxes if f[4] >= SAME]
        if not mine:
            return None
        x, y, w, h, _ = max(mine, key=lambda f: f[4])
        cx, cy = x + w / 2, y + h / 2
        for box in people:  # the person whose box has this face in its top part
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[1] + 0.5 * (box[3] - box[1]):
                return box
        H, W = frame.shape[:2]  # a face but no body box (crouching close): take what is below the face
        return (max(0, cx - 1.5 * w), y, min(W, cx + 1.5 * w), min(H, y + 6 * h), 1.0)

    def _find(self, frame, people):
        w = frame.shape[1]
        best, best_score = None, 1e9
        for box in people:
            looks = unlike(self.me, signature(frame, box))
            if looks > MATCH:
                continue
            moved = 0.0
            if self.box is not None and time.time() - self.seen_at < 2.0:
                moved = abs((box[0] + box[2]) - (self.box[0] + self.box[2])) / 2 / w  # 0..1 of the frame
            score = looks + 0.5 * moved
            if score < best_score:
                best, best_score = box, score
        if best is not None and best_score < 0.3:  # sure it is you: let the memory follow lighting changes
            now = signature(frame, best)
            self.me = [0.9 * a + 0.1 * b for a, b in zip(self.me, now)]
        return best

    def _where(self, frame, box):
        h, w = frame.shape[:2]
        f = (w / 2) / math.tan(self.hfov / 2)
        bearing = -math.atan(((box[0] + box[2]) / 2 - w / 2) / f)
        if box[3] < h - 3:  # feet in view: the floor gives the distance
            below = self.pitch * -1 + math.atan((box[3] - h / 2) / f)
            distance = self.height / math.tan(below) if below > math.radians(1) else 6.0
        else:               # feet cut off by the bottom of the image: very close
            distance = 0.3
        return min(distance, 6.0), bearing

    def _draw(self, frame, people, target):
        from faces import SAME
        for x, y, w, h, likeness in self.face_boxes:
            you = likeness >= SAME
            colour = (0, 220, 0) if you else (0, 0, 230)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), colour, 2)
            cv2.putText(frame, f"{'you' if you else '?'} {likeness:.2f}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        for box in people:
            mine = target is not None and box is target
            colour = (0, 220, 0) if mine else (160, 160, 160)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, 3 if mine else 1)
        if self.me is None:
            text = "not locked - L: whoever is in front, F: find my face"
        elif target is None:
            text = "locked, but can't see you"
        else:
            d, b = self.sighting
            text = f"YOU: {d:.1f} m, {math.degrees(b):+.0f} deg"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("duck eyes", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("duck eyes", cv2.WND_PROP_VISIBLE) < 1:
            self.show = False  # closed: stop drawing (the eyes keep working)
            try:
                cv2.destroyWindow("duck eyes")
            except cv2.error:  # already gone: the close button did it
                pass
            if self.quit_on_close:
                os._exit(0)
            return
        if key == ord("l"):
            self.want_lock.set()
        elif key == ord("f"):
            threading.Thread(target=self.find_me, args=(3.0,), daemon=True).start()
        elif key == ord("u"):
            self.unlock()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=CAMERA, help="webcam number (default: $DUCK_CAMERA, else 0)")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--height", type=float, default=0.8, help="camera height above the floor, m")
    ap.add_argument("--pitch", type=float, default=0.0, help="degrees tilted up")
    a = ap.parse_args()
    eyes = CameraEyes(a.camera, a.hfov, a.height, a.pitch)
    eyes.quit_on_close = True
    while True:
        time.sleep(1)
        seen = eyes.look()
        print("you:", f"{seen[0]:.2f} m, {math.degrees(seen[1]):+.0f} deg" if seen else "-")
