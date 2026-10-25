"""Face recognition: is this you? YuNet finds faces, SFace compares them with yours. CPU, offline.

    python faces.py enroll        # look at the webcam for a few seconds; saves me/face.npy
    python faces.py               # live test: green = you, red = someone else

Your face is stored as a fingerprint in me/face.npy on this laptop only (me/ is not in git).
Models (opencv_zoo): YuNet (MIT) and SFace (Apache-2.0), in models/.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
HERE = Path(__file__).parent
DETECTOR = HERE / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER = HERE / "models" / "face_recognition_sface_2021dec.onnx"
ME = HERE / "me" / "face.npy"
SAME = 0.363  # SFace's own cosine threshold for "same person"


class Faces:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR), "", (640, 480), 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER), "")
        self.me = np.load(ME) if ME.exists() else None

    def known(self) -> bool:
        return self.me is not None

    def find(self, frame):
        """[(x, y, w, h, likeness)] for every face; likeness >= SAME means it is you (0 if not enrolled)."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        found = []
        for face in faces if faces is not None else []:
            likeness = 0.0
            if self.me is not None:
                likeness = float(self.recognizer.match(self._feature(frame, face), self.me,
                                                       cv2.FaceRecognizerSF_FR_COSINE))
            found.append((*face[:4].tolist(), likeness))
        return found

    def _feature(self, frame, face):
        return self.recognizer.feature(self.recognizer.alignCrop(frame, face))


def enroll(camera: int, seconds: float = 6.0):
    faces, capture = Faces(), cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    ready = time.time() + 3
    while time.time() < ready:  # a moment to sit in front of the camera
        ok, frame = capture.read()
        if ok:
            cv2.putText(frame, f"get ready... {ready - time.time():.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("enroll", frame)
            cv2.waitKey(1)
    features, start = [], time.time()
    while time.time() - start < seconds:
        ok, frame = capture.read()
        if not ok:
            continue
        faces.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, found = faces.detector.detect(frame)
        if found is not None and len(found) == 1:  # only you in the picture
            features.append(faces._feature(frame, found[0]))
            x, y, w, h = found[0][:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        left = seconds - (time.time() - start)
        cv2.putText(frame, f"look at the camera, turn your head a little... {left:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("enroll", frame)
        cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()
    if len(features) < 10:
        sys.exit(f"only {len(features)} good frames — more light, only you in view, and try again")
    me = np.mean(features, axis=0)
    ME.parent.mkdir(exist_ok=True)
    np.save(ME, me / np.linalg.norm(me))
    print(f"enrolled from {len(features)} frames -> {ME}")


def live(camera: int):
    faces, capture = Faces(), cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not faces.known():
        sys.exit("not enrolled yet: python faces.py enroll")
    while True:
        ok, frame = capture.read()
        if not ok:
            continue
        for x, y, w, h, likeness in faces.find(frame):
            you = likeness >= SAME
            colour = (0, 220, 0) if you else (0, 0, 230)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), colour, 2)
            cv2.putText(frame, f"{'you' if you else '?'} {likeness:.2f}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow("faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("faces", cv2.WND_PROP_VISIBLE) < 1:
            break  # Q, or the window's close button
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="live", choices=("live", "enroll"))
    ap.add_argument("--camera", type=int, default=int(os.environ.get("DUCK_CAMERA", 0)))
    a = ap.parse_args()
    enroll(a.camera) if a.action == "enroll" else live(a.camera)
