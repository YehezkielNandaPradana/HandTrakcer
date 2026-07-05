"""
main.py
Real-time hand tracking application with visual effects, multi-language sign support,
and gesture-driven computer control (mouse, keyboard, scrolling).
Supports up to 2 hands simultaneously.
"""

import os
import sys
import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from gesture import GestureRecognizer
from effects import EffectsManager
from sign_language import SignLanguageDisplay, SIGN_LANGUAGES, LANG_LABELS
from utils import landmark_to_pixel
from system_control import SystemController


class HandMagicApp:
    def __init__(self, camera_index=0, width=1280, height=720):
        self.width = width
        self.height = height
        self.camera_index = camera_index
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.gesture_recognizer = GestureRecognizer()
        self.hand_tracker = HandTracker(
            max_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.7
        )

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        os.makedirs(assets_dir, exist_ok=True)
        self.effects_manager = EffectsManager(width, height, assets_dir)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.show_help = True
        self.screenshot_count = 0   
        self.show_sign_language = True
        self.sign_language_display = SignLanguageDisplay(language="indonesian")

        self.control_mode = False
        try:
            self.system_controller = SystemController()
            self.control_available = True
        except ImportError:
            self.control_available = False
            self.system_controller = None
            print("Warning: pyautogui not installed. System control disabled. Run: pip install pyautogui")

    def _calculate_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.fps * 0.8 + instant_fps * 0.1
        return dt

    def _draw_ui(self, img, num_hands, effect_name):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 90), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.4, overlay, 0.6, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_color = (255, 255, 255)

        cv2.putText(img, f"FPS: {self.fps:.1f}", (20, 35), font, 0.8, line_color, 2, cv2.LINE_AA)
        cv2.putText(img, f"Hands: {num_hands}", (200, 35), font, 0.8, line_color, 2, cv2.LINE_AA)

        effect_text = f"Effect: {effect_name.replace('_', ' ').title()}"
        color = (0, 255, 255) if self.effects_manager.manual_override else (200, 200, 200)
        cv2.putText(img, effect_text, (340, 35), font, 0.8, color, 2, cv2.LINE_AA)

        lang = self.sign_language_display.language
        sign_text = f"Language: {LANG_LABELS.get(lang, lang)}"
        cv2.putText(img, sign_text, (680, 35), font, 0.7, (180, 220, 255), 2, cv2.LINE_AA)

        sign_status = "ON" if self.show_sign_language else "OFF"
        sign_color = (0, 255, 150) if self.show_sign_language else (100, 100, 100)
        cv2.putText(img, f"Sign: {sign_status}", (950, 35), font, 0.7, sign_color, 2, cv2.LINE_AA)

        if self.control_available:
            ctrl_text = "CTRL: ON" if self.control_mode else "CTRL: OFF"
            ctrl_color = (0, 200, 255) if self.control_mode else (100, 100, 100)
            cv2.putText(img, ctrl_text, (1100, 35), font, 0.7, ctrl_color, 2, cv2.LINE_AA)

        if self.show_help:
            help_text = ("E: Effect | R: Reset | S: Screenshot | L: Language | "
                         "K: Sign | C: Control Mode | H: Help | ESC: Exit")
            cv2.putText(img, help_text, (20, 75), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_control_overlay(self, img, enriched_hands):
        if not self.system_controller:
            return
        overlay = img.copy()
        cv2.rectangle(overlay, (0, self.height - 60), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.4, overlay, 0.6, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        action = enriched_hands[0].get("control_action", SystemController.ACTION_STOP) if enriched_hands else SystemController.ACTION_STOP
        action_label = self.system_controller.get_action_label(action)

        cv2.putText(img, f"[CONTROL MODE] Action: {action_label}",
                    (20, self.height - 35), font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

        if enriched_hands:
            landmarks = enriched_hands[0].get("landmarks", [])
            if landmarks and len(landmarks) >= 8:
                index_tip = landmarks[8]
                px = int(index_tip.x * self.width)
                py = int(index_tip.y * self.height)
                cv2.circle(img, (px, py), 8, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 16, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"({px},{py})", (px + 20, py - 10),
                            font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        guide_lines = [
            "Gestures: Point=Scroll | Fist=Idle (stable) | Open Hand=Hold=Play/Pause | Pinch=Click",
            "           OK=Play/Pause | Middle+Ring=Forward | 3 Fingers=Tab",
        ]
        for i, line in enumerate(guide_lines):
            cv2.putText(img, line, (20, self.height - 18 + i * 15),
                        font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    def _save_screenshot(self, frame):
        screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        filename = f"screenshot_{self.screenshot_count:04d}.png"
        path = os.path.join(screenshots_dir, filename)
        cv2.imwrite(path, frame)
        self.screenshot_count += 1
        print(f"Screenshot saved: {path}")

    def _handle_input(self, key):
        if key == 27:
            self.running = False
        elif key == ord('e') or key == ord('E'):
            if not self.control_mode:
                self.effects_manager.next_effect()
                print(f"Switched to effect: {self.effects_manager.active_effect}")
        elif key == ord('q') or key == ord('Q'):
            if not self.control_mode:
                self.effects_manager.prev_effect()
                print(f"Switched to effect: {self.effects_manager.active_effect}")
        elif key == ord('r') or key == ord('R'):
            if not self.control_mode:
                self.effects_manager.reset()
                print("Particles reset")
        elif key == ord('s') or key == ord('S'):
            self._save_screenshot(self.current_frame)
        elif key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
        elif key == ord('k') or key == ord('K'):
            self.show_sign_language = not self.show_sign_language
            print(f"Sign Language Display: {'ON' if self.show_sign_language else 'OFF'}")
        elif key == ord('l') or key == ord('L'):
            if not self.control_mode:
                idx = SIGN_LANGUAGES.index(self.sign_language_display.language)
                self.sign_language_display.set_language_index(idx + 1)
                lang = self.sign_language_display.language
                print(f"Sign Language: {LANG_LABELS.get(lang, lang)}")
        elif key == ord('c') or key == ord('C'):
            if self.control_available:
                self.control_mode = not self.control_mode
                status = "ON" if self.control_mode else "OFF"
                print(f"System Control Mode: {status}")
                if not self.control_mode and self.system_controller:
                    self.system_controller.reset()
            else:
                print("System control not available. Install pyautogui: pip install pyautogui")

    def run(self):
        print("=" * 55)
        print("  Hand Magic  -  Hand Tracking + System Control")
        print("=" * 55)
        print("Controls:")
        print("  E / Q     - Next / Previous effect")
        print("  R         - Reset particles")
        print("  S         - Save screenshot")
        print("  L         - Cycle sign language")
        print("  K         - Toggle sign display")
        print("  C         - Toggle System Control Mode")
        print("  H         - Toggle help text")
        print("  ESC       - Exit")
        print("=" * 55)
        print("")
        print("Control Mode Gestures:")
        print("  Point Up/Down       -> Scroll (.42-.58 deadzone, proportional)")
        print("  Fist (closed grip)  -> Idle (stable)")
        print("  Open Hand (hold)    -> Play / Pause video")
        print("  Pinch (hold .25s)   -> Left click")
        print("  OK sign             -> Play / Pause video")
        print("  Middle+Ring (hold)  -> Forward page")
        print("  3 fingers (hold)    -> Switch tab")
        print("  NOTE: Peace/Back disabled to prevent accidental navigation")
        print("=" * 55)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera frame capture failed. Retrying...")
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self.current_frame = frame.copy()
            dt = self._calculate_fps()

            hands = self.hand_tracker.process(frame)
            num_hands = len(hands)

            enriched_hands = []
            for hand in hands:
                landmarks = hand.get("landmarks", [])
                gesture = self.gesture_recognizer.recognize(landmarks) if len(landmarks) >= 21 else "unknown"
                enriched_hands.append({
                    "landmarks": landmarks,
                    "handedness": hand.get("handedness", "Unknown"),
                    "gesture": gesture,
                })

            if self.control_available and self.control_mode:
                control_action = SystemController.ACTION_STOP
                for hand in enriched_hands:
                    landmarks = hand.get("landmarks", [])
                    handedness = hand.get("handedness", "Right")
                    if landmarks and len(landmarks) >= 21:
                        control_action = self.system_controller.process(landmarks, handedness)
                    hand["control_action"] = control_action

            self.effects_manager.update(dt, enriched_hands)

            if self.show_sign_language and not self.control_mode:
                self.sign_language_display.update_multi(dt, enriched_hands, self.width, self.height)

            self.effects_manager.render(frame, enriched_hands)

            if self.show_sign_language and not self.control_mode:
                frame = self.sign_language_display.render_multi(frame, enriched_hands, self.width, self.height)

            if self.control_available and self.control_mode:
                self._draw_control_overlay(frame, enriched_hands)

            gesture = enriched_hands[0]["gesture"] if enriched_hands else "unknown"
            effect_name = self.effects_manager.active_effect
            if self.control_mode:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, 90), (20, 20, 50), -1)
                cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                cv2.putText(frame, "[ CONTROL MODE ]", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Hands: {num_hands}", (300, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Press C to exit | ESC to quit", (500, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            else:
                self._draw_ui(frame, num_hands, effect_name)

            cv2.imshow("Hand Magic", frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_input(key)

        self.shutdown()

    def shutdown(self):
        self.cap.release()
        self.hand_tracker.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def main():
    try:
        app = HandMagicApp()
        app.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
