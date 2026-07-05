# Hand Magic

A real-time desktop application that detects your hands via webcam and renders stunning visual effects directly on your hands, **OR** controls your computer using gestures.

Built with **Python 3.12+**, **OpenCV**, **MediaPipe**, **NumPy**, and **PyAutoGUI**.

## Features

- **Real-time Hand Tracking**: Detects 21 hand landmarks at ~30 FPS using MediaPipe Hands.
- **Dual-Hand Support**: Effects work simultaneously on both hands.
- **Multi-Language Sign Language Support**: Indonesian (Bahasa Isyarat), American Sign Language (ASL), and number gestures.
- **19 High-Quality Visual Effects**: (see below)
- **System Control Mode** (press `C`):
  - Move cursor with index finger
  - Pinch (thumb + index) to click
  - Scroll using open hand tilt
  - Play/Pause with OK gesture
  - Navigate pages with peace and other gestures
- **6 Layered Particle Systems** for rich visuals.
- **Extended Gesture Recognition** (19 gestures across 3 languages).
- **Interactive Controls**: E/Q/R/S/L/K/C/H:ESC.
- **Clean UI Overlay**: FPS, hands, active effect, language, control mode, and help text.

## System Control Gestures

| Gesture | Action |
|---------|--------|
| Point (index up) | Move cursor |
| Pinch (thumb + index) | Left click |
| OK sign | Play / Pause video |
| Peace sign (index + middle) | Page back |
| Middle + Ring up | Page forward |
| 3 fingers up (index+mid+ring) | Switch tab |
| Open hand tilt up | Scroll up |
| Open hand tilt down | Scroll down |

Other shortcuts (via keyboard):
- Thumbs up → Fullscreen
- Pointing → Copy
- Fist → Paste

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Press `C` to enter **Control Mode**. Press `C` again to return to visual effects mode.

Press `E` to cycle effects, `R` to reset particles, `S` to save a screenshot, `L` to cycle sign language, and `ESC` to exit.

## Project Structure

```
Handcam/
├── main.py              # Application entry point; OpenCV loop and keyboard controls
├── hand_tracker.py      # MediaPipe Hands wrapper for multi-hand 21-landmark detection
├── gesture.py           # Extended gesture recognizer (Indonesian, ASL, numbers)
├── effects.py           # EffectsManager — 19 visual effects and 6 particle systems
├── particle.py          # Particle and ParticleSystem for physics emitters
├── utils.py             # Helper functions: lerp, draw_glow_line, draw_soft_circle, etc.
├── requirements.txt     # Python dependencies
├── assets/              # Pre-generated texture images
│   ├── aura.png
│   ├── glow.png
│   └── spark.png
├── screenshots/         # Auto-created directory for saved frames
└── Readme.md            # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Press `E` to cycle effects, `R` to reset particles, `S` to save a screenshot, `L` to cycle sign language, and `ESC` to exit.

## Effect System Details

### Layered Rendering Order (back to front)

1. Rain particles (hologram)
2. Trail particles
3. Spark particles
4. Ambient particles
5. Orbital particles
6. Ember particles (fire, dragon)
7. Per-effect geometry (bolts, portal spokes, galaxy arms, etc.)

### Dual-Hand Support

Effects are rendered on every detected hand independently. Raise one or two hands to the camera to see effects apply to each hand simultaneously.

### Rendering Helpers

- `draw_glow_line` — Layered blurred lines for neon-like strokes.
- `draw_soft_circle` — Additive-blended soft circles using temporary overlays.
- `landmark_to_pixel` / `get_palm_center` — Normalized landmark → screen coordinates.

### Hand Topology

Effects use the standard 21-landmark topology:
- Fingertip indices: `[4, 8, 12, 16, 20]`
- Palm center: midpoint between wrist (`0`) and middle MCP (`9`)

## Sign Language Support

| Language | Toggle Key | Examples |
|----------|-----------|----------|
| Bahasa Isyarat (Indonesian) | Press `L` to cycle to | HALO, SALAM, APA KABAR?, TIDAK, BAGUS, OKE |
| ASL (American Sign Language) | Press `L` to cycle to | A, B, D, F, K, L, V, W, Y, ?, POINT |
| Numbers | Press `L` to cycle to | 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 |

## Gesture Recognition

| Gesture | Language | Display |
|---------|----------|---------|
| Open Hand | Indonesian | HALO |
| Peace Sign | Indonesian | SALAM |
| Thumbs Up | Indonesian | BAGUS! |
| Pointing | Indonesian | APA KABAR? |
| Fist | Indonesian | TIDAK |
| OK Sign | Indonesian | OKE / SETUJU |
| Sign A (fist+thumb side) | ASL | A |
| Sign B (all fingers up) | ASL | B |
| Sign L (L-shape) | ASL | L |
| Sign Y (shaka) | ASL | Y |
| Sign F (OK+3 up) | ASL | F |
| Sign D (index+thumb) | ASL | D |
| Sign K (split V) | ASL | K |
| Sign W (3 fingers) | ASL | W |
| Number 1 | Numbers | 1 |
| Number 2 | Numbers | 2 |
| Number 3 | Numbers | 3 |
| Number 4 | Numbers | 4 |
| Number 5 | Numbers | 5 |
| Number 6 | Numbers | 6 |
| Number 7 | Numbers | 7 |
| Number 8 | Numbers | 8 |
| Number 10 | Numbers | 10 |

## Requirements

- Python 3.12+
- opencv-python >= 4.9.0
- mediapipe == 0.10.13
- numpy >= 1.26.0
