"""
hand_tracker.py
MediaPipe Hands wrapper for real-time hand landmark detection.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects hands and extracts 21 landmarks using MediaPipe."""

    # Landmark indices for fingertips
    TIP_IDS = [4, 8, 12, 16, 20]
    # PIP (proximal interphalangeal) indices for fingers
    PIP_IDS = [2, 6, 10, 14, 18]
    # MCP indices
    MCP_IDS = [1, 5, 9, 13, 17]

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """
        Process a BGR frame and return hand landmarks.
        Returns list of hand results, each a dict with:
            - landmarks: list of 21 normalized landmarks
            - handedness: 'Left' or 'Right'
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands = []
        h, w = frame.shape[:2]

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm)
                handedness = results.multi_handedness[idx].classification[0].label if results.multi_handedness else "Unknown"
                hands.append({
                    "landmarks": landmarks,
                    "handedness": handedness,
                    "landmarks_raw": hand_landmarks,
                })
        return hands

    def get_fingertips(self, landmarks, width, height):
        """Return pixel coordinates of all fingertips."""
        return [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in self.TIP_IDS]

    def get_landmarks_pixels(self, landmarks, width, height):
        """Return all landmarks in pixel coordinates."""
        return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()