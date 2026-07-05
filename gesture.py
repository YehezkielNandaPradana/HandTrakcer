"""
gesture.py
Extended gesture recognition supporting Indonesian and ASL sign languages.
"""

import numpy as np
import math


class GestureRecognizer:
    def __init__(self):
        self.TIPS = [4, 8, 12, 16, 20]
        self.PIPS = [2, 6, 10, 14, 18]
        self.MCPS = [1, 5, 9, 13, 17]

    def _distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return math.degrees(math.acos(max(-1, min(1, cos))))

    def _is_finger_extended(self, landmarks, finger_idx, is_thumb=False):
        wrist = landmarks[0]
        tip = landmarks[self.TIPS[finger_idx]]
        pip = landmarks[self.PIPS[finger_idx]]
        mcp = landmarks[self.MCPS[finger_idx]]
        if is_thumb:
            ip_joint = landmarks[3]
            tip_dist = self._distance(tip, wrist)
            ip_dist = self._distance(ip_joint, wrist)
            return tip_dist > ip_dist * 1.1
        else:
            tip_dist = self._distance(tip, wrist)
            pip_dist = self._distance(pip, wrist)
            return tip_dist > pip_dist * 1.05

    def recognize(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return "unknown"

        thumb, index, middle, ring, pinky = [self._is_finger_extended(landmarks, i) for i in range(5)]
        thumb_is_extended = thumb
        index_is_extended = index
        middle_is_extended = middle
        ring_is_extended = ring
        pinky_is_extended = pinky

        extended_count = sum([thumb_is_extended, index_is_extended, middle_is_extended, ring_is_extended, pinky_is_extended])

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]

        thumb_index_dist = self._distance(thumb_tip, index_tip)
        wrist_to_middle_mcp = self._distance(wrist, middle_mcp)

        thumb_ok = thumb_index_dist < wrist_to_middle_mcp * 0.18

        if thumb_ok and middle_is_extended and ring_is_extended and pinky_is_extended:
            return "ok"

        thumb_side = self._distance(landmarks[4], landmarks[2]) > self._distance(landmarks[4], landmarks[3]) * 1.2

        index_angle = self._angle(landmarks[5], landmarks[6], landmarks[7])
        middle_tip_to_middle_mcp = self._distance(middle_tip, middle_mcp)
        index_tip_to_index_mcp = self._distance(index_tip, index_mcp)

        thumb_pinky = thumb_is_extended and pinky_is_extended and not index_is_extended and not middle_is_extended and not ring_is_extended
        thumb_index_l = thumb_is_extended and index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended
        only_index = index_is_extended and not thumb_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended
        only_pinky = pinky_is_extended and not thumb_is_extended and not index_is_extended and not middle_is_extended and not ring_is_extended
        three_center = index_is_extended and middle_is_extended and ring_is_extended and not thumb_is_extended and not pinky_is_extended
        three_thumb = thumb_is_extended and index_is_extended and middle_is_extended and not ring_is_extended and not pinky_is_extended
        four = not thumb_is_extended and index_is_extended and middle_is_extended and ring_is_extended and pinky_is_extended
        thumb_three = thumb_is_extended and index_is_extended and middle_is_extended and ring_is_extended and not pinky_is_extended
        fingers_together = self._distance(index_mcp, index_tip) < self._distance(middle_mcp, middle_tip) * 1.5

        if only_pinky and thumb_is_extended:
            return "sign_y"
        if thumb_pinky:
            return "sign_y"
        if only_pinky and not thumb_is_extended:
            return "num_6"
        if thumb_side and not index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "sign_a"
        if not thumb_is_extended and index_is_extended and middle_is_extended and ring_is_extended and pinky_is_extended:
            return "open"
        if not thumb_is_extended and not index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "fist"
        if thumb_is_extended and not index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "thumbs_up"
        if not thumb_is_extended and index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "pointing"
        if not thumb_is_extended and index_is_extended and middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "peace"
        if thumb_is_extended and index_is_extended and not middle_is_extended and not ring_is_extended and not pinky_is_extended:
            return "sign_l"
        if three_center:
            return "sign_w_stub"
        if thumb_index_l and index_tip_to_index_mcp > wrist_to_middle_mcp * 0.3 and middle_tip_to_middle_mcp < wrist_to_middle_mcp * 0.3:
            return "sign_k"
        if thumb_three:
            return "num_3"
        if four:
            return "num_4"
        if thumb_side and middle_is_extended and not index_is_extended and not ring_is_extended and not pinky_is_extended:
            return "sign_d"
        if thumb_ok and not middle_is_extended:
            return "ok"

        if thumb_is_extended and index_is_extended and middle_is_extended and ring_is_extended and not pinky_is_extended:
            return "num_8"

        if thumb_index_dist < wrist_to_middle_mcp * 0.1:
            return "sign_f"

        return "unknown"

    def gesture_to_effect(self, gesture):
        mapping = {
            "open":      "portal",
            "fist":      "dragon",
            "peace":     "lightning",
            "thumbs_up": "golden",
            "pointing":  "laser",
            "ok":        "energy",
            "unknown":   "plasma",
            "sign_a":    "aura",
            "sign_b":    "glow",
            "sign_l":    "trail",
            "sign_y":    "spark",
            "sign_f":    "fire",
            "sign_d":    "ice",
            "sign_w_stub": "neon_wave",
            "sign_k":    "shield",
            "num_1":     "laser",
            "num_2":     "lightning",
            "num_3":     "golden",
            "num_4":     "plasma",
            "num_5":     "portal",
            "num_6":     "spark",
            "num_8":     "aura",
        }
        return mapping.get(gesture, "plasma")
