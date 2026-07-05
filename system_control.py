"""
system_control.py
Gesture-driven computer control module.
Maps hand gestures to mouse movements, clicks, scrolling,
keyboard shortcuts, and browser/page navigation.
"""

import math
import time
import platform
import threading

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    PYAUTOGUI_AVAILABLE = True
except (ImportError, SystemExit):
    PYAUTOGUI_AVAILABLE = False

import numpy as np


class SystemController:
    ACTION_STOP = "stop"
    ACTION_CLICK = "click"
    ACTION_RIGHT_CLICK = "right_click"
    ACTION_DOUBLE_CLICK = "double_click"
    ACTION_SCROLL_UP = "scroll_up"
    ACTION_SCROLL_DOWN = "scroll_down"
    ACTION_PLAY_PAUSE = "play_pause"
    ACTION_PAGE_BACK = "page_back"
    ACTION_PAGE_FORWARD = "page_forward"
    ACTION_TAB_SWITCH = "tab_switch"
    ACTION_FULLSCREEN = "fullscreen"
    ACTION_COPY = "copy"
    ACTION_PASTE = "paste"
    ACTION_VOLUME_UP = "volume_up"
    ACTION_VOLUME_DOWN = "volume_down"
    ACTION_REFRESH = "refresh"
    ACTION_LOCK = "lock"

    def __init__(self, screen_width=None, screen_height=None):
        if not PYAUTOGUI_AVAILABLE:
            raise ImportError(
                "pyautogui is required for system control. "
                "Install it with: pip install pyautogui"
            )
        self.screen_width = screen_width or pyautogui.size().width
        self.screen_height = screen_height or pyautogui.size().height

        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._last_action = None
        self._last_action_time = 0.0
        self._prev_scroll_y = None

        self._open_hand_cooldown = 0.12
        self._pinch_cooldown = 0.4
        self._key_cooldown = 0.5

        # Scroll state
        self._scroll_pointer_cooldown = 0.065
        self._smooth_scroll_intensity = 0.0

        # Hold-to-confirm state for one-shot gestures
        self._held_gesture = None
        self._hold_start_time = 0.0
        self._hold_fired = False
        self._hold_threshold = 0.25

        # Continuous gesture stability (prevents idle↔scroll loops)
        self._stable_gesture = None
        self._stable_change_time = 0.0
        self._stable_lock = 0.12

    def _get_system_shortcuts(self):
        system = platform.system()
        if system == "Darwin":
            return {
                self.ACTION_PLAY_PAUSE:  ["space"],
                self.ACTION_PAGE_BACK:   ["alt", "left"],
                self.ACTION_PAGE_FORWARD:["alt", "right"],
                self.ACTION_TAB_SWITCH:  ["command", "tab"],
                self.ACTION_FULLSCREEN:  ["ctrl", "command", "f"],
                self.ACTION_COPY:        ["command", "c"],
                self.ACTION_PASTE:       ["command", "v"],
                self.ACTION_REFRESH:     ["command", "r"],
                self.ACTION_VOLUME_UP:   ["volumeup"],
                self.ACTION_VOLUME_DOWN: ["volumedown"],
                self.ACTION_LOCK:        ["cmd", "ctrl", "q"],
            }
        elif system == "Linux":
            return {
                self.ACTION_PLAY_PAUSE:  ["space"],
                self.ACTION_PAGE_BACK:   ["alt", "left"],
                self.ACTION_PAGE_FORWARD:["alt", "right"],
                self.ACTION_TAB_SWITCH:  ["alt", "tab"],
                self.ACTION_FULLSCREEN:  ["f11"],
                self.ACTION_COPY:        ["ctrl", "c"],
                self.ACTION_PASTE:       ["ctrl", "v"],
                self.ACTION_REFRESH:     ["ctrl", "r"],
                self.ACTION_VOLUME_UP:   ["volumeup"],
                self.ACTION_VOLUME_DOWN: ["volumedown"],
            }
        else:
            return {
                self.ACTION_PLAY_PAUSE:  ["space"],
                self.ACTION_PAGE_BACK:   ["alt", "left"],
                self.ACTION_PAGE_FORWARD:["alt", "right"],
                self.ACTION_TAB_SWITCH:  ["alt", "tab"],
                self.ACTION_FULLSCREEN:  ["f11"],
                self.ACTION_COPY:        ["ctrl", "c"],
                self.ACTION_PASTE:       ["ctrl", "v"],
                self.ACTION_REFRESH:     ["ctrl", "r"],
                self.ACTION_VOLUME_UP:   ["volumeup"],
                self.ACTION_VOLUME_DOWN: ["volumedown"],
            }

    def recognize(self, landmarks, handedness="Right"):
        gestures = []

        if landmarks and len(landmarks) >= 21:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_dist = math.hypot(thumb_tip.x - index_tip.x,
                                    thumb_tip.y - index_tip.y)

            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            hand_size = max(0.01, math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y))

            if pinch_dist < hand_size * 0.3 and pinch_dist < 0.12:
                gestures.append("pinch")

            extended = []
            for i in range(1, 5):
                tip = landmarks[4 + i * 4]
                mcp = landmarks[5 + (i - 1) * 4]
                pip = landmarks[6 + (i - 1) * 4]
                tip_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
                pip_mcp = math.hypot(pip.x - mcp.x, pip.y - mcp.y)
                extended.append(tip_mcp > pip_mcp * 1.05)
            if all(extended):
                gestures.append("open_hand")

            thumb, index, middle, ring, pinky = [], [], [], [], []
            for i in range(1, 5):
                tip = landmarks[4 + i * 4]
                pip = landmarks[6 + (i - 1) * 4]
                mcp = landmarks[5 + (i - 1) * 4]
                extended = (math.hypot(tip.x - mcp.x, tip.y - mcp.y) >
                            math.hypot(pip.x - mcp.x, pip.y - mcp.y) * 1.05)
                if i == 1:
                    index.append(extended)
                elif i == 2:
                    middle.append(extended)
                elif i == 3:
                    ring.append(extended)
                else:
                    pinky.append(extended)

            index_up = index[0] if index else False
            middle_up = middle[0] if middle else False
            ring_up = ring[0] if ring else False
            pinky_up = pinky[0] if pinky else False

            if index_up and middle_up and not ring_up and not pinky_up:
                gestures.append("peace")
            if middle_up and ring_up and not index_up and not pinky_up:
                gestures.append("middle_ring")
            if index_up and middle_up and ring_up and not pinky_up:
                gestures.append("three_fingers")
            if not index_up and not middle_up and ring_up and pinky_up:
                gestures.append("pinky_ring")
            if index_up and not middle_up and not ring_up and not pinky_up:
                gestures.append("pointing")
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            thumb_index_dist = math.hypot(thumb_tip.x - index_tip.x,
                                          thumb_tip.y - index_tip.y)
            if thumb_index_dist < hand_size * 0.18:
                gestures.append("ok")

            if not index_up and not middle_up and not ring_up and not pinky_up:
                thumb_ip = landmarks[3]
                thumb_extended = (math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y) >
                                  math.hypot(thumb_ip.x - wrist.x, thumb_ip.y - wrist.y) * 1.1)
                if not thumb_extended:
                    gestures.append("fist")

        return gestures

    def process(self, landmarks, handedness="Right"):
        if not PYAUTOGUI_AVAILABLE:
            return self.ACTION_STOP

        now = time.time()
        gestures = self.recognize(landmarks, handedness)
        action = self.ACTION_STOP
        hand_present = landmarks is not None and len(landmarks) >= 21

        # ---------------------------------------------------------------
        # Continuous gesture stability (debounce to prevent idle↔scroll loops)
        # ---------------------------------------------------------------
        target = None
        if hand_present:
            if "pointing" in gestures:
                target = "pointing"
            elif "fist" in gestures:
                target = "fist"
            elif "open_hand" in gestures:
                target = "open_hand"

        if target != self._stable_gesture:
            if (now - self._stable_change_time) > self._stable_lock:
                self._stable_gesture = target
                self._stable_change_time = now
                self._held_gesture = None
                self._hold_start_time = 0.0
                self._hold_fired = False

        state = self._stable_gesture if hand_present else None

        # -------------------------------------------------------------------
        # PRIORITY 1: Pointing = continuous proportional scroll
        # -------------------------------------------------------------------
        if state == "pointing" and "pointing" in gestures:
            if landmarks and len(landmarks) >= 8:
                index_tip = landmarks[8]
                y = index_tip.y
                deadzone_top = 0.40
                deadzone_bottom = 0.60
                max_scroll_zone = 0.12

                if now - self._last_action_time > self._scroll_pointer_cooldown:
                    scroll_amount = 0
                    direction = 0
                    if y < deadzone_top:
                        raw_intensity = max(0.0, min(1.0, (deadzone_top - y) / max_scroll_zone))
                        direction = 1
                    elif y > deadzone_bottom:
                        raw_intensity = max(0.0, min(1.0, (y - deadzone_bottom) / max_scroll_zone))
                        direction = -1

                    if direction != 0:
                        alpha = 0.4
                        if (self._prev_scroll_y is not None and
                                ((y < deadzone_top and self._prev_scroll_y >= deadzone_top) or
                                 (y > deadzone_bottom and self._prev_scroll_y <= deadzone_bottom))):
                            self._smooth_scroll_intensity = 0.0
                        else:
                            self._smooth_scroll_intensity = (
                                (1.0 - alpha) * self._smooth_scroll_intensity + alpha * raw_intensity)
                        intensity = self._smooth_scroll_intensity
                        scroll_amount = max(1, int(round(intensity * 5)))
                        scroll_amount = min(scroll_amount, 5) * direction
                        with self._lock:
                            pyautogui.scroll(scroll_amount, _pause=False)
                        self._last_action = (self.ACTION_SCROLL_UP if direction > 0
                                             else self.ACTION_SCROLL_DOWN)
                        self._last_action_time = now
                        action = self._last_action
                        self._prev_scroll_y = y

        # -------------------------------------------------------------------
        # PRIORITY 2: Fist = stable idle (explicit no-op)
        # -------------------------------------------------------------------
        elif state == "fist":
            action = self.ACTION_STOP
            self._prev_scroll_y = None
            self._smooth_scroll_intensity = 0.0

        # -------------------------------------------------------------------
        # PRIORITY 3: Open hand = hold-to-confirm play/pause
        # -------------------------------------------------------------------
        elif state == "open_hand" and "open_hand" in gestures:
            if self._held_gesture != "open_hand":
                self._held_gesture = "open_hand"
                self._hold_start_time = now
                self._hold_fired = False
            elif not self._hold_fired:
                if (now - self._hold_start_time) > self._hold_threshold:
                    if now - self._last_action_time > self._key_cooldown:
                        action = self._fire_one_shot("open_hand", now)
                        self._hold_fired = True

        # -------------------------------------------------------------------
        # PRIORITY 4: One-shot gestures (pinch, ok, middle_ring, three_fingers)
        # -------------------------------------------------------------------
        elif hand_present:
            self._prev_scroll_y = None
            self._smooth_scroll_intensity = 0.0
            current_one_shot = None
            for g in ["pinch", "ok", "middle_ring", "three_fingers"]:
                if g in gestures:
                    current_one_shot = g
                    break

            if current_one_shot is not None:
                if current_one_shot == self._held_gesture and not self._hold_fired:
                    hold_duration = now - self._hold_start_time
                    if (hold_duration >= self._hold_threshold and
                            now - self._last_action_time > self._key_cooldown):
                        action = self._fire_one_shot(current_one_shot, now)
                        self._hold_fired = True
                elif current_one_shot != self._held_gesture:
                    if self._held_gesture is None or (now - self._hold_start_time) > 0.5:
                        self._held_gesture = current_one_shot
                        self._hold_start_time = now
                        self._hold_fired = False

            if current_one_shot is None and self._held_gesture not in ("open_hand",):
                self._held_gesture = None
                self._hold_start_time = 0.0
                self._hold_fired = False
                if self._last_action not in (None, self.ACTION_STOP):
                    self._last_action = None

        # -------------------------------------------------------------------
        # Loss of hand: full reset
        # -------------------------------------------------------------------
        if not hand_present:
            self._stable_gesture = None
            self._held_gesture = None
            self._hold_start_time = 0.0
            self._hold_fired = False
            self._last_action = None
            self._prev_scroll_y = None
            self._smooth_scroll_intensity = 0.0

        return action

    def _fire_one_shot(self, gesture_name, now):
        if gesture_name == "pinch":
            with self._lock:
                pyautogui.click(_pause=False)
            self._last_action = self.ACTION_CLICK
            self._last_action_time = now
            return self.ACTION_CLICK
        elif gesture_name == "ok":
            self._press_key(self.ACTION_PLAY_PAUSE)
            self._last_action = self.ACTION_PLAY_PAUSE
            self._last_action_time = now
            return self.ACTION_PLAY_PAUSE
        elif gesture_name == "open_hand":
            self._press_key(self.ACTION_PLAY_PAUSE)
            self._last_action = self.ACTION_PLAY_PAUSE
            self._last_action_time = now
            return self.ACTION_PLAY_PAUSE
        elif gesture_name == "middle_ring":
            self._press_key(self.ACTION_PAGE_FORWARD)
            self._last_action = self.ACTION_PAGE_FORWARD
            self._last_action_time = now
            return self.ACTION_PAGE_FORWARD
        elif gesture_name == "three_fingers":
            self._press_key(self.ACTION_TAB_SWITCH)
            self._last_action = self.ACTION_TAB_SWITCH
            self._last_action_time = now
            return self.ACTION_TAB_SWITCH
        return self.ACTION_STOP

    def _press_key(self, action):
        shortcuts = self._get_system_shortcuts()
        keys = shortcuts.get(action, [])
        with self._lock:
            if not keys:
                return
            for k in keys:
                pyautogui.keyDown(k, _pause=False)
            for k in reversed(keys):
                pyautogui.keyUp(k, _pause=False)

    def get_action_label(self, action):
        labels = {
            self.ACTION_STOP:           "Idle",
            self.ACTION_CLICK:          "Click",
            self.ACTION_RIGHT_CLICK:    "Right Click",
            self.ACTION_DOUBLE_CLICK:   "Double Click",
            self.ACTION_SCROLL_UP:      "Scroll Up",
            self.ACTION_SCROLL_DOWN:    "Scroll Down",
            self.ACTION_PLAY_PAUSE:     "Play/Pause",
            self.ACTION_PAGE_BACK:      "Back Page",
            self.ACTION_PAGE_FORWARD:   "Forward Page",
            self.ACTION_TAB_SWITCH:     "Tab Switch",
            self.ACTION_FULLSCREEN:     "Fullscreen",
            self.ACTION_COPY:           "Copy",
            self.ACTION_PASTE:          "Paste",
            self.ACTION_REFRESH:        "Refresh",
            self.ACTION_LOCK:           "Lock Screen",
        }
        return labels.get(action, action)

    def reset(self):
        self._last_action = None
        self._last_action_time = 0.0
        self._prev_scroll_y = None
        self._held_gesture = None
        self._hold_start_time = 0.0
        self._hold_fired = False
