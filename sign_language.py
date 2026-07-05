"""
sign_language.py
Multi-language sign language display: Indonesian (Bahasa Isyarat), ASL, and Numbers.

Supports toggling between sign languages with the L key.
When two hands are detected, each hand's sign is displayed independently at its wrist.
"""

import cv2
import numpy as np

SIGN_LANGUAGES = ["indonesian", "asl_en", "numbers"]

INDONESIAN_MAP = {
    "open":      "HALO",
    "peace":     "SALAM",
    "thumbs_up": "BAGUS!",
    "pointing":  "APA KABAR?",
    "fist":      "TIDAK",
    "ok":        "OKE / SETUJU",
    "unknown":   "",
}

ASL_EN_MAP = {
    "open":      "A-N OPEN",
    "fist":      "A-N FIST",
    "peace":     "V",
    "thumbs_up": "THUMBS UP",
    "pointing":  "POINT",
    "ok":        "F / OK",
    "sign_a":    "A",
    "sign_b":    "B",
    "sign_l":    "L",
    "sign_y":    "Y",
    "sign_f":    "F",
    "sign_d":    "D",
    "sign_k":    "K",
    "sign_w_stub": "W",
    "num_1":     "1",
    "num_2":     "2",
    "num_3":     "3",
    "num_4":     "4",
    "num_5":     "5",
    "num_6":     "6",
    "num_7_stub": "7",
    "num_8":     "8",
    "unknown":   "?",
}

NUMBERS_MAP = {
    "open":      "5",
    "fist":      "0",
    "peace":     "2",
    "thumbs_up": "10",
    "pointing":  "1",
    "ok":        "6",
    "sign_a":    "",
    "sign_b":    "",
    "sign_l":    "",
    "sign_y":    "",
    "sign_f":    "",
    "sign_d":    "",
    "sign_k":    "",
    "sign_w_stub": "",
    "num_1":     "1",
    "num_2":     "2",
    "num_3":     "3",
    "num_4":     "4",
    "num_5":     "5",
    "num_6":     "6",
    "num_8":     "8",
    "unknown":   "",
}

LANG_ACCENT = {
    "indonesian": (255, 200, 100),
    "asl_en":     (150, 255, 200),
    "numbers":    (200, 150, 255),
}

LANG_LABELS = {
    "indonesian": "Bahasa Isyarat",
    "asl_en":     "ASL",
    "numbers":    "Numbers",
}


class _HandSignState:
    def __init__(self):
        self.current_gesture = "unknown"
        self.hold_timer = 0.0
        self.fade_timer = 0.0
        self.display_text = ""
        self.current_color = (180, 180, 180)
        self.wrist_px = None


class SignLanguageDisplay:
    HOLD_DURATION = 0.35
    FADE_DURATION = 1.2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.85
    FONT_THICKNESS = 2

    def __init__(self, language="indonesian"):
        self.language = language
        self.hand_states = [_HandSignState() for _ in range(2)]

    def set_language(self, language):
        self.language = language
        for hs in self.hand_states:
            hs.current_color = LANG_ACCENT.get(language, (180, 180, 180))
            hs.display_text = ""
            hs.fade_timer = 0.0
            hs.hold_timer = 0.0

    def set_language_index(self, idx):
        language = SIGN_LANGUAGES[idx % len(SIGN_LANGUAGES)]
        self.set_language(language)

    def _get_translation(self, gesture):
        lang = self.language
        if lang == "indonesian":
            return INDONESIAN_MAP.get(gesture, "")
        elif lang == "asl_en":
            return ASL_EN_MAP.get(gesture, "?")
        elif lang == "numbers":
            return NUMBERS_MAP.get(gesture, "")
        return ""

    def _update_hand_state(self, dt, hand_idx, gesture, landmarks, width, height):
        hs = self.hand_states[hand_idx]
        if gesture == "unknown" or not landmarks:
            if hs.display_text:
                hs.fade_timer += dt
                hs.hold_timer = 0.0
                if hs.fade_timer >= self.FADE_DURATION:
                    hs.display_text = ""
                    hs.fade_timer = 0.0
                    hs.wrist_px = None
            return

        if gesture == hs.current_gesture:
            hs.hold_timer += dt
        else:
            hs.current_gesture = gesture
            hs.hold_timer = 0.0
            hs.fade_timer = 0.0

        if hs.hold_timer >= self.HOLD_DURATION:
            text = self._get_translation(gesture)
            if text and text != hs.display_text:
                hs.display_text = text
                hs.fade_timer = 0.0

        if landmarks and len(landmarks) >= 21:
            wrist = landmarks[0]
            hs.wrist_px = (int(wrist.x * width), int(wrist.y * height))

    def update(self, dt, gesture, landmarks, width, height):
        self._update_hand_state(dt, 0, gesture, landmarks, width, height)

    def update_multi(self, dt, hands, width, height):
        for i, hand in enumerate(hands):
            if i >= len(self.hand_states):
                self.hand_states.append(_HandSignState())
            g = hand.get("gesture", "unknown")
            lm = hand.get("landmarks")
            self._update_hand_state(dt, i, g, lm, width, height)

    def _render_bubble(self, img, text, wrist_px, color, alpha):
        if not text or not wrist_px:
            return img
        x, y = wrist_px
        (tw, th), baseline = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)

        pad_x, pad_y = 18, 12
        bx1 = x - tw // 2 - pad_x
        by1 = y - th - pad_y - 10
        bx2 = x + tw // 2 + pad_x
        by2 = y + baseline + pad_y - 10

        h, w = img.shape[:2]
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        by2 = min(h, by2)

        ov = img.copy()
        cv2.rectangle(ov, (bx1, by1), (bx2, by2), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(ov, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha * 0.75, img, 1 - alpha * 0.75, 0, img)

        cv2.putText(img, text, (x - tw // 2, y - 8), self.FONT, self.FONT_SCALE, color, self.FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(img, text, (x - tw // 2, y - 8), self.FONT, self.FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def render(self, img):
        hs = self.hand_states[0]
        if not hs.display_text or not hs.wrist_px:
            return img
        alpha = 1.0 if hs.fade_timer <= 0 else max(0.0, 1.0 - hs.fade_timer / self.FADE_DURATION)
        return self._render_bubble(img, hs.display_text, hs.wrist_px, hs.current_color, alpha)

    def render_multi(self, img, hands, width, height):
        result = img
        for i, hand in enumerate(hands):
            if i >= len(self.hand_states):
                break
            hs = self.hand_states[i]
            lm = hand.get("landmarks")
            if not lm or len(lm) < 21 or not hs.display_text or not hs.wrist_px:
                continue
            alpha = 1.0 if hs.fade_timer <= 0 else max(0.0, 1.0 - hs.fade_timer / self.FADE_DURATION)
            result = self._render_bubble(result, hs.display_text, hs.wrist_px, hs.current_color, alpha)
        return result
