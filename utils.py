"""
utils.py
Utility functions for drawing, math, and image processing.
"""

import cv2
import numpy as np


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    return tuple(max(0, min(255, int(lerp(c1[i], c2[i], t)))) for i in range(3))


def draw_glow_line(img, pt1, pt2, color, thickness=2, glow_radius=8):
    """Draw an anti-aliased glowing line by layering blurred lines."""
    overlay = img.copy()
    cv2.line(overlay, pt1, pt2, color, thickness + glow_radius)
    cv2.line(overlay, pt1, pt2, color, thickness + glow_radius // 2)
    cv2.line(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(img, 0.4, overlay, 0.6, 0, img)


def draw_soft_circle(img, center, radius, color, alpha=0.5):
    """Draw a soft glowing circle using a temporary overlay."""
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0, img)


def create_radial_gradient(size, center_color, edge_color):
    """Create a radial gradient image (square)."""
    h, w = size, size
    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    max_dist = np.sqrt(2) * (size // 2)
    norm = np.clip(dist / max_dist, 0, 1)
    
    gradient = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        gradient[:, :, i] = (center_color[i] * (1 - norm) + edge_color[i] * norm).astype(np.uint8)
    return gradient


def get_palm_center(landmarks):
    """Calculate palm center from wrist and middle finger MCP."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    cx = int((wrist.x + middle_mcp.x) / 2 * 1280)
    cy = int((wrist.y + middle_mcp.y) / 2 * 720)
    return (cx, cy)


def landmark_to_pixel(landmark, width, height):
    """Convert normalized landmark to pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)


def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)