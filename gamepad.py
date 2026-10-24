"""Drive the duck with a PS5 (or any SDL-supported) controller. Started by main.py beside voice.

    left stick       walk (up/down) and turn (left/right)
    ✕ cross          sit ⇄ stand          △ triangle   forward roll
    ○ circle         FALL (faceplant)     □ square     beak grab
    L1 / R1          kick left / right    Options      get up
    R2               follow on/off        L2           come here
    D-pad up         skates on/off        Create       stop everything
"""
import os
import threading
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame  # noqa: E402
from pygame._sdl2 import controller  # noqa: E402

import skates  # noqa: E402

DEADZONE = 0.15
B = pygame  # button and axis constants live on the pygame module
BUTTONS = {
    B.CONTROLLER_BUTTON_A: "sit_stand",   # cross
    B.CONTROLLER_BUTTON_B: "drop",        # circle
    B.CONTROLLER_BUTTON_Y: "roll",        # triangle
    B.CONTROLLER_BUTTON_X: "pick",        # square
    B.CONTROLLER_BUTTON_LEFTSHOULDER: "kick",
    B.CONTROLLER_BUTTON_RIGHTSHOULDER: "kick_right",
    B.CONTROLLER_BUTTON_START: "stand",   # options
    B.CONTROLLER_BUTTON_BACK: "stop",     # create
    B.CONTROLLER_BUTTON_DPAD_UP: "skates",
}
RUMBLE = {"drop": (1.0, 1.0, 600), "kick": (0.3, 0.8, 150), "kick_right": (0.3, 0.8, 150),
          "roll": (0.6, 0.4, 400), "skates": (0.2, 0.2, 800)}


def axis(pad, which):
    value = pad.get_axis(which) / 32767
    return 0.0 if abs(value) < DEADZONE else max(-1.0, min(1.0, value))


# Measured in the sim (velstand): forward only happens between 0.3 and 0.45, backward needs -0.45,
# and turning only works while walking forward, up to ±0.6. So every stick push lands in the range
# that actually moves the duck, instead of most of the stick doing nothing. Retune on the real duck.
WALK_MIN, WALK_MAX, BACK, ARC_YAW = 0.3, 0.45, -0.45, 0.6


def stick_to_walk(forward, turn):
    """Stick (each -1..1, deadzone already removed) -> (vx, vy, vyaw) the sim duck responds to."""
    if forward < -0.3:
        return BACK, 0, 0
    if forward < 0 and not turn:
        return 0, 0, 0
    vx = WALK_MIN + (WALK_MAX - WALK_MIN) * max(forward, 0)  # turning alone still walks a gentle arc
    return vx, 0, turn * ARC_YAW


class Gamepad:
    def __init__(self, duck, follower, skates, say=print):
        self.duck, self.follower, self.skates, self.say = duck, follower, skates, say
        pygame.init()
        controller.init()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _find(self):
        pygame.joystick.quit()  # rescan, so a controller plugged in later is found
        pygame.joystick.init()
        for i in range(pygame.joystick.get_count()):
            if controller.is_controller(i):
                self.say(f"controller: {controller.name_forindex(i)}")
                return controller.Controller(i)
        return None

    def _run(self):
        pad, held, pulled, driving, following = None, set(), set(), False, False
        while True:
            time.sleep(1 / 30)
            pygame.event.pump()
            if pad is None or not pad.attached():
                pad = self._find()
                if pad is None:
                    time.sleep(2)
                    continue

            now = {b for b in BUTTONS if pad.get_button(b)}
            for button in now - held:  # pressed this frame
                following = False
                self._press(pad, BUTTONS[button])
            held = now

            triggers = {t for t in (B.CONTROLLER_AXIS_TRIGGERRIGHT, B.CONTROLLER_AXIS_TRIGGERLEFT)
                        if pad.get_axis(t) > 16000}
            for trigger in triggers - pulled:
                if self.skates.on:
                    self.say("pad: no following on skates")
                elif trigger == B.CONTROLLER_AXIS_TRIGGERLEFT:
                    following = False
                    self.say("pad: come here")
                    self.follower.start("come")
                elif following:
                    following = False
                    self.say("pad: follow off")
                    self.follower.cancel()
                else:
                    following = True
                    self.say("pad: follow")
                    self.follower.start("follow")
            pulled = triggers

            forward = -axis(pad, B.CONTROLLER_AXIS_LEFTY)
            turn = -axis(pad, B.CONTROLLER_AXIS_LEFTX)
            if forward or turn:
                if not driving:
                    self.follower.cancel()  # taking the sticks takes the duck back
                    following = False
                if self.skates.on:  # forward only: braking on skates tips it over
                    self.duck.walk(max(forward, 0) * skates.MAX_VX, 0, turn * skates.MAX_YAW)
                else:
                    self.duck.walk(*stick_to_walk(forward, turn))
                driving = True
            elif driving:
                self.duck.walk(0, 0, 0)
                driving = False

    def _press(self, pad, action):
        self.say(f"pad: {action}")
        self.follower.cancel()
        if action == "stop":
            self.duck.stop()
            return
        if action == "skates":
            pad.rumble(*RUMBLE[action])
            self.skates.toggle()
            return
        if action == "sit_stand":
            action = "stand" if self.duck.call("robot.policies").get("sitting") else "sit"
        if action in RUMBLE:
            pad.rumble(*RUMBLE[action])
        result = self.duck.do(action)
        if "not driving" in result:  # lying on the floor after a fall: any button gets it up first
            result = self.duck.do("stand")
        if result != "ok":
            self.say(f"duck says no: {result}")
