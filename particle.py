"""
particle.py
Lightweight particle system for trails, sparks, and ambient effects.
"""

import random
import numpy as np
import cv2


class Particle:
    """Single particle with physics and lifecycle."""

    def __init__(self, x, y, vx, vy, life, color, size, gravity=0.0, fade_mode="linear"):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.life = float(life)
        self.max_life = float(life)
        self.color = color
        self.size = float(size)
        self.gravity = gravity
        self.fade_mode = fade_mode
        self.dead = False

    def update(self, dt):
        """Update particle physics. dt is time delta in seconds."""
        self.vy += self.gravity * dt * 60.0
        self.x += self.vx * dt * 60.0
        self.y += self.vy * dt * 60.0
        self.life -= dt
        if self.life <= 0:
            self.dead = True

    def get_alpha(self):
        """Return current alpha multiplier [0,1]."""
        if self.fade_mode == "linear":
            return max(0.0, self.life / self.max_life)
        elif self.fade_mode == "ease":
            t = self.life / self.max_life
            return t * t
        else:
            return max(0.0, self.life / self.max_life)

    def draw(self, img, glow_img=None):
        """Draw particle to image. Optional glow_img for additive blending."""
        alpha = self.get_alpha()
        if alpha <= 0:
            return
        px, py = int(self.x), int(self.y)
        r = max(1, int(self.size * alpha))
        color = tuple(int(c * alpha) for c in self.color)
        if glow_img is not None and r > 2:
            h, w = glow_img.shape[:2]
            scale = r * 2 / max(w, h)
            if scale > 0:
                sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = cv2.resize(glow_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
                x1 = px - sw // 2
                y1 = py - sh // 2
                x2 = x1 + sw
                y2 = y1 + sh
                ih, iw = img.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
                    return
                roi = img[y1:y2, x1:x2]
                glow_colored = resized.copy()
                for i in range(3):
                    glow_colored[:, :, i] = (glow_colored[:, :, i] * (self.color[i] / 255.0) * alpha).astype(np.uint8)
                cv2.add(roi, glow_colored, roi)
        else:
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)


class ParticleSystem:
    """Manager for a collection of particles."""

    def __init__(self, max_particles=500):
        self.particles = []
        self.max_particles = max_particles

    def emit(self, x, y, vx, vy, life, color, size, gravity=0.0, fade_mode="linear"):
        """Spawn a new particle if under budget."""
        if len(self.particles) >= self.max_particles:
            self.particles.pop(0)
        self.particles.append(Particle(x, y, vx, vy, life, color, size, gravity, fade_mode))

    def emit_burst(self, x, y, count, speed, life, color, size, gravity=0.0, fade_mode="linear"):
        """Emit a burst of particles in random directions."""
        for _ in range(count):
            angle = random.uniform(0, 2 * np.pi)
            spd = random.uniform(speed * 0.3, speed)
            vx = np.cos(angle) * spd
            vy = np.sin(angle) * spd
            self.emit(x, y, vx, vy, random.uniform(life * 0.5, life), color, size, gravity, fade_mode)

    def update(self, dt):
        """Update all particles and remove dead ones."""
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if not p.dead]

    def draw(self, img, glow_img=None):
        """Render all particles."""
        for p in self.particles:
            p.draw(img, glow_img)

    def clear(self):
        """Remove all particles."""
        self.particles.clear()