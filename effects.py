"""
effects.py
Multi-hand visual effects manager. Supports rendering effects on multiple hands simultaneously.
"""

import math
import random
import cv2
import numpy as np
from particle import ParticleSystem
from utils import (
    lerp, lerp_color, draw_glow_line, draw_soft_circle,
    get_palm_center, landmark_to_pixel
)


class EffectsManager:
    EFFECTS = [
        "glow", "trail", "aura", "spark", "outline",
        "fire", "golden", "laser", "energy",
        "lightning", "plasma", "vortex", "hologram",
        "portal", "galaxy", "ice", "neon_wave",
        "shield", "dragon", "morph_shape",
    ]

    def __init__(self, width, height, assets_dir):
        self.width = width
        self.height = height
        self.assets_dir = assets_dir
        self.time = 0.0
        self.active_effect = "lightning"
        self.manual_override = False

        self.trail_system  = ParticleSystem(max_particles=500)
        self.spark_system  = ParticleSystem(max_particles=600)
        self.ambient_system = ParticleSystem(max_particles=400)
        self.orbital_system = ParticleSystem(max_particles=300)
        self.rain_system   = ParticleSystem(max_particles=200)
        self.ember_system  = ParticleSystem(max_particles=300)

        self.glow_img  = self._generate_glow(64)
        self.spark_img = self._generate_spark(32)
        self.aura_img  = self._generate_glow(128)
        self.star_img  = self._generate_star(16)

        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        self.FINGERTIP_IDX = [4, 8, 12, 16, 20]

        self._lightning_cache  = []
        self._lightning_timer  = 0.0
        self._prev_pixels = None
        self._prev_palm   = None

    def _generate_glow(self, size):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        c = size / 2.0
        for y in range(size):
            for x in range(size):
                d = math.sqrt((x - c)**2 + (y - c)**2) / (size / 2.0)
                v = int(255 * (1 - d**1.5))
                v = max(0, min(255, v))
                img[y, x] = (v, v, v)
        return img

    def _generate_spark(self, size):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        c = size / 2.0
        for y in range(size):
            for x in range(size):
                d = math.sqrt((x - c)**2 + (y - c)**2) / (size / 3.0)
                v = max(0, min(255, int(255 * (1 - d)**2)))
                img[y, x] = (v, v, v)
        return img

    def _generate_star(self, size):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        c = size / 2.0
        for y in range(size):
            for x in range(size):
                dx, dy = abs(x - c), abs(y - c)
                d_diamond = (dx + dy) / (size / 2.0)
                d_cross   = min(dx, dy) / (size / 4.0) if min(dx, dy) > 0 else 0
                d = min(d_diamond, d_cross)
                v = max(0, min(255, int(255 * (1 - d)**2)))
                img[y, x] = (v, v, v)
        return img

    def set_effect(self, name):
        if name in self.EFFECTS:
            self.active_effect = name
            self.manual_override = True
            self.reset()

    def next_effect(self):
        idx = self.EFFECTS.index(self.active_effect)
        self.active_effect = self.EFFECTS[(idx + 1) % len(self.EFFECTS)]
        self.manual_override = True
        self.reset()

    def prev_effect(self):
        idx = self.EFFECTS.index(self.active_effect)
        self.active_effect = self.EFFECTS[(idx - 1) % len(self.EFFECTS)]
        self.manual_override = True
        self.reset()

    def reset(self):
        for s in (self.trail_system, self.spark_system, self.ambient_system,
                  self.orbital_system, self.rain_system, self.ember_system):
            s.clear()
        self._lightning_cache = []
        self._prev_pixels = None
        self._prev_palm   = None

    def update(self, dt, hands):
        self.time += dt

        if not self.manual_override:
            if len(hands) == 2:
                self.active_effect = "morph_shape"
            elif hands:
                primary = hands[0]
                gesture = primary.get("gesture", "unknown")
                self.active_effect = {
                    "open":      "portal",
                    "fist":      "dragon",
                    "peace":     "lightning",
                    "thumbs_up": "golden",
                    "pointing":  "laser",
                    "unknown":   "plasma",
                }.get(gesture, "plasma")
            else:
                self.active_effect = "plasma"

        for s in (self.trail_system, self.spark_system, self.ambient_system,
                  self.orbital_system, self.rain_system, self.ember_system):
            s.update(dt)

        for hand in hands:
            landmarks = hand.get("landmarks")
            gesture = hand.get("gesture", "unknown")
            self._spawn_particles(landmarks, gesture)

        if self.active_effect == "morph_shape" and len(hands) == 2:
            self._spawn_nexus_particles(hands)

        if hands:
            last = hands[-1]
            landmarks = last.get("landmarks")
            if landmarks and len(landmarks) >= 21:
                self._prev_pixels = [landmark_to_pixel(lm, self.width, self.height)
                                     for lm in landmarks]
                self._prev_palm = get_palm_center(landmarks)

    def _palm_px(self, landmarks):
        p = get_palm_center(landmarks)
        return (int(p[0] * self.width / 1280),
                int(p[1] * self.height / 720))

    def _spawn_particles(self, landmarks, gesture):
        if not landmarks or len(landmarks) < 21:
            return
        pixels    = [landmark_to_pixel(lm, self.width, self.height) for lm in landmarks]
        fingertips = [pixels[i] for i in self.FINGERTIP_IDX]
        palm_px    = self._palm_px(landmarks)
        t          = self.time
        fx         = self.active_effect

        if fx == "trail":
            for tip in fingertips:
                for _ in range(2):
                    self.trail_system.emit(
                        tip[0]+random.gauss(0,2), tip[1]+random.gauss(0,2),
                        vx=random.gauss(0,.8), vy=random.gauss(0,.8),
                        life=random.uniform(.3,.8),
                        color=lerp_color((100,200,255),(255,100,200),random.random()),
                        size=random.uniform(3,7), fade_mode="ease")

        elif fx == "laser":
            tip = pixels[8]
            for _ in range(4):
                a = random.gauss(0,.3); sp = random.uniform(2,5)
                self.trail_system.emit(
                    tip[0], tip[1],
                    vx=math.cos(a)*sp, vy=math.sin(a)*sp-1,
                    life=random.uniform(.15,.4),
                    color=(255, random.randint(50,150), random.randint(30,80)),
                    size=random.uniform(2,5), fade_mode="linear")

        elif fx == "spark":
            for tip in fingertips:
                if random.random() < .5:
                    a = random.uniform(0, math.tau); sp = random.uniform(2,5)
                    self.spark_system.emit(
                        tip[0], tip[1],
                        vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                        life=random.uniform(.4,1),
                        color=(random.randint(150,255), random.randint(200,255), 255),
                        size=random.uniform(2,5), gravity=2., fade_mode="linear")

        elif fx == "fire":
            for _ in range(5):
                self.spark_system.emit(
                    palm_px[0]+random.gauss(0,15), palm_px[1]+random.gauss(0,10),
                    vx=random.gauss(0,1.5), vy=random.uniform(-4,-1.5),
                    life=random.uniform(.3,.8),
                    color=(0, random.randint(50,150), random.randint(200,255)),
                    size=random.uniform(4,9), gravity=-2., fade_mode="ease")
            if random.random() < .3:
                tip = random.choice(fingertips)
                self.ember_system.emit(
                    tip[0], tip[1],
                    vx=random.gauss(0,1), vy=random.uniform(-2,-.5),
                    life=random.uniform(.5,1.2),
                    color=(255, random.randint(100,200), random.randint(0,50)),
                    size=random.uniform(1,3), gravity=-.5, fade_mode="linear")

        elif fx == "dragon":
            for _ in range(8):
                self.spark_system.emit(
                    palm_px[0]+random.gauss(0,20), palm_px[1]+random.gauss(5,8),
                    vx=random.gauss(0,3), vy=random.uniform(-5,-2),
                    life=random.uniform(.2,.6),
                    color=(random.randint(0,50), random.randint(100,200),
                           random.randint(200,255)),
                    size=random.uniform(5,12), gravity=-3., fade_mode="ease")
            for _ in range(2):
                tip = random.choice(fingertips)
                a = random.uniform(0, math.tau); sp = random.uniform(3,7)
                self.ember_system.emit(
                    tip[0], tip[1],
                    vx=math.cos(a)*sp, vy=math.sin(a)*sp-2,
                    life=random.uniform(.3,.8),
                    color=(255, random.randint(150,255), random.randint(0,100)),
                    size=random.uniform(1,4), gravity=-1., fade_mode="linear")

        elif fx == "energy":
            for tip in fingertips:
                if random.random() < .4:
                    a = random.uniform(0, math.tau); sp = random.uniform(.5,2)
                    self.ambient_system.emit(
                        tip[0], tip[1],
                        vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                        life=random.uniform(.3,.7),
                        color=lerp_color((255,150,50),(255,50,150),random.random()),
                        size=random.uniform(3,6), fade_mode="ease")

        elif fx == "golden":
            for tip in fingertips:
                if random.random() < .6:
                    a = random.uniform(0, math.tau); sp = random.uniform(.2,.8)
                    self.ambient_system.emit(
                        tip[0]+random.gauss(0,3), tip[1]+random.gauss(0,3),
                        vx=math.cos(a)*sp, vy=math.sin(a)*sp-.5,
                        life=random.uniform(.4,1),
                        color=(random.randint(30,80), random.randint(200,255),
                               random.randint(230,255)),
                        size=random.uniform(2,5), fade_mode="ease")

        elif fx == "lightning":
            for tip in fingertips:
                if random.random() < .6:
                    self.spark_system.emit(
                        tip[0]+random.gauss(0,5), tip[1]+random.gauss(0,5),
                        vx=random.gauss(0,3), vy=random.gauss(0,3),
                        life=random.uniform(.1,.3), color=(200,220,255),
                        size=random.uniform(2,5), fade_mode="linear")

        elif fx == "plasma":
            for tip in fingertips:
                if random.random() < .5:
                    a = t*3 + random.uniform(0, math.tau)
                    sp = random.uniform(.5,2)
                    hue = (math.sin(t*2 + tip[0]*.01)+1)/2
                    self.ambient_system.emit(
                        tip[0]+random.gauss(0,4), tip[1]+random.gauss(0,4),
                        vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                        life=random.uniform(.3,.8),
                        color=lerp_color((255,50,200),(50,200,255),hue),
                        size=random.uniform(3,7), fade_mode="ease")

        elif fx == "vortex":
            for tip in fingertips:
                if random.random() < .4:
                    a = t*4 + random.uniform(0,.5); r = random.uniform(5,20)
                    self.orbital_system.emit(
                        tip[0]+math.cos(a)*r, tip[1]+math.sin(a)*r,
                        vx=-math.sin(a)*3, vy=math.cos(a)*3,
                        life=random.uniform(.4,1),
                        color=lerp_color((100,255,200),(200,100,255),random.random()),
                        size=random.uniform(2,5), fade_mode="ease")

        elif fx == "hologram":
            if random.random() < .3:
                self.rain_system.emit(
                    palm_px[0]+random.randint(-50,50),
                    palm_px[1]+random.randint(-50,50),
                    vx=0, vy=random.uniform(1,3),
                    life=random.uniform(.3,.8), color=(0,255,150),
                    size=random.uniform(1,3), fade_mode="linear")

        elif fx == "portal":
            for i in range(3):
                a = t*2 + i*(math.tau/3)
                r = 40 + math.sin(t*3)*10
                self.orbital_system.emit(
                    palm_px[0]+math.cos(a)*r, palm_px[1]+math.sin(a)*r,
                    vx=-math.sin(a)*2, vy=math.cos(a)*2,
                    life=random.uniform(.3,.7),
                    color=lerp_color((150,50,255),(255,50,150),
                                     (math.sin(t+i)+1)/2),
                    size=random.uniform(3,6), fade_mode="ease")
            if random.random() < .4:
                self.spark_system.emit(
                    palm_px[0]+random.gauss(0,5), palm_px[1]+random.gauss(0,5),
                    vx=random.gauss(0,1), vy=random.gauss(0,1),
                    life=random.uniform(.1,.3), color=(220,180,255),
                    size=random.uniform(2,4), fade_mode="linear")

        elif fx == "galaxy":
            for _ in range(2):
                arm = random.randint(0,2)
                ba  = arm*(math.tau/3) + t*.5
                r   = random.uniform(10,50)
                sa  = ba + r*.05
                self.ambient_system.emit(
                    palm_px[0]+math.cos(sa)*r,
                    palm_px[1]+math.sin(sa)*r*.6,
                    vx=-math.sin(sa)*.5, vy=math.cos(sa)*.5,
                    life=random.uniform(.5,1.5),
                    color=lerp_color((200,150,255),(150,200,255),random.random()),
                    size=random.uniform(1,4), fade_mode="ease")

        elif fx == "ice":
            for tip in fingertips:
                if random.random() < .3:
                    a = random.uniform(0, math.tau); sp = random.uniform(.3,1.5)
                    self.ambient_system.emit(
                        tip[0]+random.gauss(0,3), tip[1]+random.gauss(0,3),
                        vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                        life=random.uniform(.5,1.2),
                        color=(random.randint(180,230), random.randint(230,255), 255),
                        size=random.uniform(2,5), fade_mode="ease")
            if random.random() < .2:
                self.rain_system.emit(
                    palm_px[0]+random.gauss(0,30), palm_px[1]+random.gauss(0,20),
                    vx=random.gauss(0,.5), vy=random.uniform(-.5,.5),
                    life=random.uniform(.5,1), color=(200,230,255),
                    size=random.uniform(4,8), fade_mode="ease")

        elif fx == "neon_wave":
            for tip in fingertips:
                if random.random() < .5:
                    self.trail_system.emit(
                        tip[0]+random.gauss(0,2), tip[1]+random.gauss(0,2),
                        vx=random.gauss(0,.5), vy=random.gauss(0,.5),
                        life=random.uniform(.3,.7),
                        color=lerp_color((255,0,150),(0,255,255),
                                         (math.sin(t*3+tip[0]*.02)+1)/2),
                        size=random.uniform(3,6), fade_mode="ease")

        elif fx == "shield":
            for i in range(2):
                a = t*1.5 + i*.8; r = 60
                self.orbital_system.emit(
                    palm_px[0]+math.cos(a)*r, palm_px[1]+math.sin(a)*r,
                    vx=-math.sin(a)*1.5, vy=math.cos(a)*1.5,
                    life=random.uniform(.5,1), color=(0,200,255),
                    size=random.uniform(3,5), fade_mode="ease")

    def render(self, img, hands):
        self.rain_system.draw(img, self.spark_img)
        self.trail_system.draw(img, self.glow_img)
        self.spark_system.draw(img, self.spark_img)
        self.ambient_system.draw(img, self.glow_img)
        self.orbital_system.draw(img, self.glow_img)
        self.ember_system.draw(img, self.star_img)

        if not hands:
            return img

        if self.active_effect == "morph_shape":
            self._render_morph_shape(img, hands)
            return img

        for hand in hands:
            landmarks = hand.get("landmarks")
            gesture = hand.get("gesture", "unknown")
            if not landmarks or len(landmarks) < 21:
                continue
            pixels = [landmark_to_pixel(lm, self.width, self.height)
                      for lm in landmarks]
            {
                "glow":      self._render_magic_glow,
                "trail":     self._render_energy_trail,
                "aura":      self._render_palm_aura,
                "spark":     self._render_sparks,
                "outline":   self._render_hand_outline,
                "fire":      self._render_fire,
                "golden":    self._render_golden,
                "laser":     self._render_laser,
                "energy":    self._render_energy,
                "lightning": self._render_lightning,
                "plasma":    self._render_plasma,
                "vortex":    self._render_vortex,
                "hologram":  self._render_hologram,
                "portal":    self._render_portal,
                "galaxy":    self._render_galaxy,
                "ice":       self._render_ice,
                "neon_wave": self._render_neon_wave,
                "shield":    self._render_shield,
                "dragon":    self._render_dragon,
            }.get(self.active_effect, lambda *a: None)(img, pixels, landmarks)

        return img

    def _render_magic_glow(self, img, pixels, lm=None):
        t = self.time
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            p = (math.sin(t*3+i*.7)+1)/2
            r = int(12+p*10); a = .3+p*.3
            c = lerp_color((255,200,50),(255,100,200),(math.sin(t+i)+1)/2)
            draw_soft_circle(img, tip, r, c, a)
            draw_soft_circle(img, tip, r//2, (255,255,255), a*.5)
            self._thin_ring(img, tip, r+8+int(p*5), c, .2+p*.2)

    def _render_energy_trail(self, img, pixels, lm=None):
        for tip in (pixels[j] for j in self.FINGERTIP_IDX):
            draw_soft_circle(img, tip, 10, (255,200,100), .5)
            draw_soft_circle(img, tip, 5,  (255,255,200), .3)

    def _render_palm_aura(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        s = (math.sin(t*2)+1)/2
        for layer in range(3):
            r = 50+layer*20+int(s*15)
            a = (.12-layer*.03)+s*.08
            c = lerp_color((255,200,100),(255,150,50),layer/3)
            draw_soft_circle(img, pp, r, c, max(.05,a))
        a2 = t*1.5
        draw_soft_circle(img,
                         (pp[0]+int(math.cos(a2)*30), pp[1]+int(math.sin(a2)*30)),
                         15, (255,255,200), .2)

    def _render_sparks(self, img, pixels, lm=None):
        for tip in (pixels[j] for j in self.FINGERTIP_IDX):
            draw_soft_circle(img, tip, 8, (200,220,255), .4)
            draw_soft_circle(img, tip, 4, (255,255,255), .3)

    def _render_hand_outline(self, img, pixels, lm=None):
        t = self.time
        for i,(a,b) in enumerate(self.HAND_CONNECTIONS):
            h = (math.sin(t*2+i*.3)+1)/2
            draw_glow_line(img, pixels[a], pixels[b],
                           lerp_color((100,255,100),(100,200,255),h),
                           2, 8)
        for px in pixels:
            cv2.circle(img, px, 4, (255,255,255), -1, cv2.LINE_AA)
            draw_soft_circle(img, px, 8, (150,255,150), .3)

    def _render_fire(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        p = (math.sin(t*4)+1)/2
        for layer in range(4):
            r = 30+layer*12+int(p*8)
            draw_soft_circle(img, pp, r,
                             [(0,50,255),(0,100,255),(50,150,255),(100,200,255)][layer],
                             max(.03, .15-layer*.03))

    def _render_golden(self, img, pixels, lm=None):
        t = self.time
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            p = (math.sin(t*2.5+i*.5)+1)/2
            r = int(14+p*8); a = .25+p*.25
            draw_soft_circle(img, tip, r, (50,215,255), a)
            draw_soft_circle(img, tip, r//2, (200,240,255), a*.6)
            self._thin_ring(img, tip, r+5, (50,215,255), .15+p*.15)

    def _render_laser(self, img, pixels, lm=None):
        tip = pixels[8]; wrist = pixels[0]; t = self.time
        p = (math.sin(t*5)+1)/2
        draw_glow_line(img, wrist, tip, (255,50,50), 3, 12)
        draw_glow_line(img, wrist, tip, (255,150,150), 1, 6)
        ov = img.copy()
        cv2.circle(ov, tip, int(10+p*6), (255,80,80), -1, cv2.LINE_AA)
        cv2.addWeighted(img, .5, ov, .5, 0, img)
        sz = 15+int(p*5)
        cv2.line(img, (tip[0]-sz,tip[1]), (tip[0]+sz,tip[1]), (255,100,100), 1, cv2.LINE_AA)
        cv2.line(img, (tip[0],tip[1]-sz), (tip[0],tip[1]+sz), (255,100,100), 1, cv2.LINE_AA)

    def _render_energy(self, img, pixels, lm=None):
        t = self.time
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            p = (math.sin(t*3+i*.8)+1)/2
            r = int(12+p*6); a = .3+p*.2
            c = lerp_color((255,150,50),(255,50,150),(math.sin(t*2+i)+1)/2)
            draw_soft_circle(img, tip, r, c, a)
            self._thin_ring(img, tip, r+6, c, .2)

    def _render_lightning(self, img, pixels, lm):
        t = self.time
        self._lightning_timer -= 1/30
        if self._lightning_timer <= 0:
            self._lightning_timer = random.uniform(.03,.1)
            self._lightning_cache = self._gen_bolts(pixels)

        for bolt in self._lightning_cache:
            self._draw_bolt(img, bolt, (180,200,255), 2, 6)
            self._draw_bolt(img, bolt, (255,255,255), 1, 3)

        for tip in (pixels[j] for j in self.FINGERTIP_IDX):
            draw_soft_circle(img, tip, 10, (150,180,255), .4)
            draw_soft_circle(img, tip, 5,  (220,230,255), .3)

        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            r = 18+int(math.sin(t*8+i)*5)
            self._thin_ring(img, tip, r, (100,150,255), .15)

    def _gen_bolts(self, pixels):
        bolts = []; fi = self.FINGERTIP_IDX
        for i in range(len(fi)-1):
            if random.random() < .6:
                bolts.append(self._bolt_path(pixels[fi[i]], pixels[fi[i+1]], 8, 15, .3))
        for _ in range(2):
            a,b = random.sample(fi, 2)
            if random.random() < .4:
                bolts.append(self._bolt_path(pixels[a], pixels[b], 10, 20, .2))
        for ti in fi:
            if random.random() < .3:
                bolts.append(self._bolt_path(pixels[0], pixels[ti], 6, 12, .2))
        return bolts

    def _bolt_path(self, s, e, segs=8, jit=15, bp=.3):
        pts = [s]
        for i in range(1, segs):
            t = i/segs
            pts.append((int(lerp(s[0],e[0],t)+random.gauss(0,jit)),
                        int(lerp(s[1],e[1],t)+random.gauss(0,jit))))
        pts.append(e)
        branches = []
        for i in range(1, len(pts)-1):
            if random.random() < bp:
                a = math.atan2(e[1]-s[1],e[0]-s[0]) + random.gauss(0,1)
                l = random.uniform(15,40)
                be = (int(pts[i][0]+math.cos(a)*l),
                      int(pts[i][1]+math.sin(a)*l))
                bp2 = [pts[i]]
                for j in range(1,4):
                    bt = j/4
                    bp2.append((int(lerp(pts[i][0],be[0],bt)+random.gauss(0,jit*.5)),
                                int(lerp(pts[i][1],be[1],bt)+random.gauss(0,jit*.5))))
                bp2.append(be); branches.append(bp2)
        return {"main": pts, "branches": branches}

    def _draw_bolt(self, img, bolt, color, thick, glow):
        pts = bolt["main"]
        if len(pts) >= 2:
            draw_glow_line(img, pts[0], pts[-1], color, thick, glow)
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], color, thick, cv2.LINE_AA)
        for br in bolt["branches"]:
            for i in range(len(br)-1):
                cv2.line(img, br[i], br[i+1], color, max(1,thick-1), cv2.LINE_AA)

    def _render_plasma(self, img, pixels, lm=None):
        t = self.time
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            for layer in range(3):
                ph = t*(2+layer*.5)+i*1.2+layer*.7
                ox = math.sin(ph)*8; oy = math.cos(ph*1.3)*8
                r = 15+layer*8+int(math.sin(ph*.7)*5)
                h = (math.sin(t*1.5+i*.8+layer*.5)+1)/2
                c = lerp_color((255,50,200),(50,200,255),h)
                draw_soft_circle(img, (int(tip[0]+ox),int(tip[1]+oy)),
                                 r, c, max(.03,.15-layer*.03))
            draw_soft_circle(img, tip, 5, (255,255,255), .4)
        fi = self.FINGERTIP_IDX
        for i in range(len(fi)-1):
            h = (math.sin(t*2+i)+1)/2
            self._wavy_line(img, pixels[fi[i]], pixels[fi[i+1]],
                            lerp_color((255,100,200),(100,200,255),h),
                            t*3+i, 5, 3)

    def _render_vortex(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        for arm in range(3):
            ba = arm*(math.tau/3)+t*2; prev = None
            for j in range(40):
                r = 5+j*1.5; a = ba+j*.15
                x = int(pp[0]+math.cos(a)*r); y = int(pp[1]+math.sin(a)*r)
                if prev:
                    h = (math.sin(t+j*.1+arm)+1)/2
                    draw_glow_line(img, prev, (x,y),
                                   lerp_color((100,255,200),(200,100,255),h), 2, 4)
                prev = (x,y)
        for layer in range(3):
            r = 8+layer*6
            draw_soft_circle(img, pp, r,
                             lerp_color((150,255,220),(220,150,255),layer/3),
                             max(.05,.3-layer*.08))
        self._thin_ring(img, pp, 55+int(math.sin(t*3)*5), (150,255,200), .2)

    def _render_hologram(self, img, pixels, lm):
        t = self.time
        ov = img.copy()
        for i,(a,b) in enumerate(self.HAND_CONNECTIONS):
            h = (math.sin(t*2+i*.2)+1)/2
            cv2.line(ov, pixels[a], pixels[b],
                     lerp_color((0,200,100),(0,255,200),h), 1, cv2.LINE_AA)
        for px in pixels:
            cv2.circle(ov, px, 3, (0,255,150), -1, cv2.LINE_AA)
        cv2.addWeighted(ov, .5, img, .5, 0, img)

        xs = [p[0] for p in pixels]; ys = [p[1] for p in pixels]
        mn_x,mx_x = min(xs)-20,max(xs)+20; mn_y,mx_y = min(ys)-20,max(ys)+20
        sy = mn_y+int(((math.sin(t*3)+1)/2)*(mx_y-mn_y))
        sov = img.copy()
        for dy in range(-8,9):
            y = sy+dy
            if mn_y <= y <= mx_y:
                cv2.line(sov, (mn_x,y),(mx_x,y), (0,255,150), 1, cv2.LINE_AA)
        cv2.line(sov, (mn_x,sy),(mx_x,sy), (0,255,150), 2, cv2.LINE_AA)
        cv2.addWeighted(sov, .3, img, .7, 0, img)

        gov = img.copy(); step = 25
        for gx in range(mn_x-(mn_x%step), mx_x+step, step):
            cv2.line(gov, (gx,mn_y),(gx,mx_y), (0,150,100), 1, cv2.LINE_AA)
        for gy in range(mn_y-(mn_y%step), mx_y+step, step):
            cv2.line(gov, (mn_x,gy),(mx_x,gy), (0,150,100), 1, cv2.LINE_AA)
        cv2.addWeighted(gov, .1, img, .9, 0, img)

        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            cv2.putText(img, f"{random.randint(0,255):02X}",
                        (tip[0]+10, tip[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, (0,255,150), 1, cv2.LINE_AA)

    def _render_portal(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        for i in range(5):
            r = 25+i*15
            ao = t*(2+i*.5)*(1 if i%2==0 else -1)
            h = (math.sin(t+i*.5)+1)/2
            c = lerp_color((150,50,255),(255,50,150),h)
            nd = 12+i*4
            for d in range(nd):
                a1 = ao+d*(math.tau/nd)
                a2 = a1+(math.tau/nd)*.6
                cv2.line(img,
                         (int(pp[0]+math.cos(a1)*r),int(pp[1]+math.sin(a1)*r)),
                         (int(pp[0]+math.cos(a2)*r),int(pp[1]+math.sin(a2)*r)),
                         c, 2, cv2.LINE_AA)
        for layer in range(4):
            r = 20-layer*4
            if r > 0:
                h = (math.sin(t*3+layer)+1)/2
                draw_soft_circle(img, pp, r,
                                 lerp_color((200,100,255),(255,100,200),h),
                                 .3-layer*.05)
        draw_soft_circle(img, pp, 5, (255,255,255), .5)

    def _render_galaxy(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        for arm in range(3):
            ba = arm*(math.tau/3)
            for j in range(60):
                r = 3+j*.8; sa = ba+j*.08+t*.3
                x = int(pp[0]+math.cos(sa)*r)
                y = int(pp[1]+math.sin(sa)*r*.6)
                h = (math.sin(j*.1+t+arm)+1)/2
                draw_soft_circle(img, (x,y), max(1,int(3-j*.03)),
                                 lerp_color((200,150,255),(150,200,255),h),
                                 max(.05,.8-j*.012))
        for layer in range(3):
            r = 10-layer*3
            if r > 0:
                draw_soft_circle(img, pp, r, (255,230,255), .4-layer*.1)
        rng = random.Random(int(t*10))
        for _ in range(15):
            a = rng.uniform(0, math.tau); r = rng.uniform(5,55)
            x = int(pp[0]+math.cos(a)*r); y = int(pp[1]+math.sin(a)*r*.6)
            tw = (math.sin(t*rng.uniform(5,15)+rng.uniform(0,6.28))+1)/2
            if tw > .7:
                cv2.circle(img, (x,y), 1, (255,255,255), -1, cv2.LINE_AA)

    def _render_ice(self, img, pixels, lm=None):
        t = self.time
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            p = (math.sin(t*2+i*.6)+1)/2
            draw_soft_circle(img, tip, int(12+p*6), (180,220,255), .3)
            draw_soft_circle(img, tip, int(6+p*3),  (220,240,255), .2)
            al = 15+int(p*8)
            for a in range(6):
                ang = a*(math.pi/3)+t*.3
                ex = int(tip[0]+math.cos(ang)*al)
                ey = int(tip[1]+math.sin(ang)*al)
                cv2.line(img, tip, (ex,ey), (180,220,255), 1, cv2.LINE_AA)
                mx,my = (tip[0]+ex)//2, (tip[1]+ey)//2
                for side in (-1,1):
                    ba = ang+side*.6; bl = al*.3
                    cv2.line(img, (mx,my),
                             (int(mx+math.cos(ba)*bl),int(my+math.sin(ba)*bl)),
                             (150,200,255), 1, cv2.LINE_AA)
        fi = self.FINGERTIP_IDX
        for i in range(len(fi)-1):
            self._frost_line(img, pixels[fi[i]], pixels[fi[i+1]], t, i)

    def _render_neon_wave(self, img, pixels, lm=None):
        t = self.time
        for i,(a,b) in enumerate(self.HAND_CONNECTIONS):
            w = (math.sin(t*4-i*.5)+1)/2
            c = lerp_color((255,0,150),(0,255,255),w)
            draw_glow_line(img, pixels[a], pixels[b], c,
                           2+int(w*2), 6+int(w*6))
        for i, px in enumerate(pixels):
            p = (math.sin(t*5-i*.3)+1)/2
            r = int(3+p*3)
            h = (math.sin(t*3+i*.2)+1)/2
            c = lerp_color((255,100,200),(100,200,255),h)
            cv2.circle(img, px, r, c, -1, cv2.LINE_AA)
            draw_soft_circle(img, px, r+4, c, .2+p*.2)

    def _render_shield(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        xs = [p[0] for p in pixels]; ys = [p[1] for p in pixels]
        cx,cy = sum(xs)//len(xs), sum(ys)//len(ys)
        sr = int(max(math.sqrt((p[0]-cx)**2+(p[1]-cy)**2) for p in pixels)+30)
        p = (math.sin(t*2)+1)/2
        self._thin_ring(img, (cx,cy), sr, (0,200,255), .2+p*.1)
        hs = 20
        for row in range(-5,6):
            for col in range(-5,6):
                hx = cx+col*hs*1.5
                hy = cy+row*hs*math.sqrt(3)+(col%2)*hs*math.sqrt(3)/2
                d = math.sqrt((hx-cx)**2+(hy-cy)**2)
                if d < sr:
                    a = max(.02, .15*(1-d/sr)+p*.05)
                    pts = [(int(hx+hs*.4*math.cos(math.pi/3*k+t*.2)),
                            int(hy+hs*.4*math.sin(math.pi/3*k+t*.2)))
                           for k in range(6)]
                    ov = img.copy()
                    cv2.polylines(ov, [np.array(pts,dtype=np.int32)],
                                  True, (0,int(150+p*105),255), 1, cv2.LINE_AA)
                    cv2.addWeighted(ov, a, img, 1-a, 0, img)
        draw_soft_circle(img, pp, 20, (0,200,255), .15+p*.1)

    def _render_dragon(self, img, pixels, lm):
        pp = self._palm_px(lm); t = self.time
        p = (math.sin(t*6)+1)/2
        colors = [(0,30,200),(0,60,220),(0,100,240),(30,150,255),(80,200,255)]
        for layer in range(5):
            r = 25+layer*15+int(p*10)
            draw_soft_circle(img, pp, r, colors[layer], max(.03,.2-layer*.035))
        draw_soft_circle(img, pp, 8, (200,230,255), .5+p*.2)
        for tip in (pixels[j] for j in self.FINGERTIP_IDX):
            self._fire_wisp(img, pp, tip, t)
        for i, tip in enumerate(pixels[j] for j in self.FINGERTIP_IDX):
            br = int(8+math.sin(t*8+i)*4)
            draw_soft_circle(img, tip, br, (50,150,255), .3)
            draw_soft_circle(img, tip, br//2, (200,230,255), .2)

    def _thin_ring(self, img, center, radius, color, alpha):
        if radius < 2: return
        ov = img.copy()
        cv2.circle(ov, center, radius,   color, 1, cv2.LINE_AA)
        cv2.circle(ov, center, radius+1, color, 1, cv2.LINE_AA)
        cv2.circle(ov, center, max(1,radius-1), color, 1, cv2.LINE_AA)
        a = min(1., max(0., alpha))
        cv2.addWeighted(ov, a, img, 1-a, 0, img)

    def _wavy_line(self, img, p1, p2, color, phase, amp=5, freq=3):
        dx,dy = p2[0]-p1[0], p2[1]-p1[1]
        l = math.sqrt(dx*dx+dy*dy)
        if l < 1: return
        nx,ny = -dy/l, dx/l
        n = max(2, int(l/3)); pts = []
        for i in range(n+1):
            t = i/n
            w = math.sin(t*freq*math.pi+phase)*amp
            pts.append((int(lerp(p1[0],p2[0],t)+nx*w),
                        int(lerp(p1[1],p2[1],t)+ny*w)))
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], color, 1, cv2.LINE_AA)

    def _frost_line(self, img, p1, p2, t, idx):
        dx,dy = p2[0]-p1[0], p2[1]-p1[1]
        l = math.sqrt(dx*dx+dy*dy)
        if l < 1: return
        nx,ny = -dy/l, dx/l
        cv2.line(img, p1, p2, (180,220,255), 1, cv2.LINE_AA)
        nb = max(1, int(l/15))
        for i in range(nb):
            tp = (i+.5)/nb
            bx,by = lerp(p1[0],p2[0],tp), lerp(p1[1],p2[1],tp)
            for side in (-1,1):
                bl = 5+math.sin(t*2+idx+i)*3
                cv2.line(img, (int(bx),int(by)),
                         (int(bx+nx*side*bl),int(by+ny*side*bl)),
                         (150,200,255), 1, cv2.LINE_AA)

    def _fire_wisp(self, img, s, e, t):
        dx,dy = e[0]-s[0], e[1]-s[1]
        l = math.sqrt(dx*dx+dy*dy)
        if l < 1: return
        nx,ny = -dy/l, dx/l
        n = max(3, int(l/4)); pts = []
        for i in range(n+1):
            f = i/n
            w = math.sin(f*8*math.pi+t*6)*(3+f*5)
            pts.append((int(lerp(s[0],e[0],f)+nx*w),
                        int(lerp(s[1],e[1],f)+ny*w)))
        for i in range(len(pts)-1):
            f = i/max(1,len(pts)-1)
            cv2.line(img, pts[i], pts[i+1],
                     lerp_color((200,230,255),(0,100,255),f), 2, cv2.LINE_AA)

    def _spawn_nexus_particles(self, hands):
        palms = []
        for hand in hands:
            lm = hand.get("landmarks")
            if not lm or len(lm) < 21:
                continue
            palms.append(self._palm_px(lm))
        if len(palms) < 2:
            return
        p1, p2 = palms[0], palms[1]
        for _ in range(5):
            fx = random.uniform(p1[0], p2[0])
            fy = random.uniform(p1[1], p2[1])
            self.orbital_system.emit(
                fx, fy,
                vx=random.gauss(0, 1.2), vy=random.gauss(0, 1.2),
                life=random.uniform(0.3, 0.9),
                color=lerp_color((120, 50, 255), (50, 220, 255), random.random()),
                size=random.uniform(1, 3), fade_mode="ease"
            )

    def _render_morph_shape(self, img, hands):
        t = self.time
        palms = []
        finger_lists = []
        for hand in hands:
            lm = hand.get("landmarks")
            if not lm or len(lm) < 21:
                continue
            palms.append(self._palm_px(lm))
            pixels = [landmark_to_pixel(l, self.width, self.height) for l in lm]
            finger_lists.append([pixels[i] for i in self.FINGERTIP_IDX])

        if not palms:
            return

        if len(palms) == 1:
            pp = palms[0]
            c = lerp_color((255, 100, 200), (100, 200, 255), (math.sin(t * 2) + 1) / 2)
            r = 22 + int(math.sin(t * 3) * 8)
            draw_soft_circle(img, pp, r + 10, c, 0.10)
            draw_soft_circle(img, pp, r + 4,  (200, 230, 255), 0.20)
            draw_soft_circle(img, pp, r,      (255, 255, 255), 0.45)
            return

        p1, p2 = palms
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 2:
            return

        nx, ny = -dy / dist, dx / dist
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        ang = math.atan2(dy, dx)

        hue1 = (math.sin(t * 1.5) + 1) / 2
        hue2 = (hue1 + 0.5) % 1
        ca1 = lerp_color((120, 50, 255), (255, 50, 150), hue1)
        ca2 = lerp_color((50, 220, 255), (0, 255, 200), hue2)
        cw  = (255, 255, 255)

        steps = 60
        bridge = []
        for i in range(steps + 1):
            f = i / steps
            wave = math.sin(f * math.pi * 4 + t * 5) * (0.12 * dist)
            bridge.append((
                int(p1[0] + f * dx + nx * wave),
                int(p1[1] + f * dy + ny * wave)
            ))
        for i in range(steps):
            h = (math.sin(t * 3 + i * 0.2) + 1) / 2
            c = lerp_color(ca1, ca2, h)
            draw_glow_line(img, bridge[i], bridge[i + 1], c, 2, 8)
            cv2.line(img, bridge[i], bridge[i + 1], c, 1, cv2.LINE_AA)

        base_r = max(18, min(70, dist * 0.3))
        for k in range(3):
            tilt = k * (math.pi / 3) + t * 0.8 * (1 if k % 2 == 0 else -1)
            pts = []
            n = 50
            for i in range(n + 1):
                a = i / n * math.tau
                x = math.cos(a) * base_r
                y = math.sin(a) * base_r * 0.35
                z = math.sin(a + tilt) * base_r * 0.45
                xr = x * math.cos(ang) - z * math.sin(ang)
                yr = y + math.sin(tilt + t) * 6
                pts.append((int(cx + xr), int(cy + yr)))
            rc = lerp_color((100, 50, 255), (255, 100, 150), k / 3)
            for i in range(n):
                draw_glow_line(img, pts[i], pts[i + 1], rc, 1, 5)

        size = base_r * 0.9
        n_pts = 18
        phi = math.pi * (3 - math.sqrt(5)); pts3 = []
        for i in range(n_pts):
            yv = 1 - (i / float(n_pts - 1)) * 2
            r = math.sqrt(1 - yv * yv)
            th = phi * i
            pts3.append((math.cos(th) * r, yv, math.sin(th) * r))

        ca = math.cos(t * 1.2); sa = math.sin(t * 1.2)
        cb = math.sin(t * 0.7); cs = math.cos(t * 0.7)
        TP = []
        for x, y, z in pts3:
            x1 = x * ca - z * sa
            z1 = x * sa + z * ca
            y1 = y * cb - z1 * cs
            z2 = y * cs + z1 * cb
            TP.append((x1, y1, z2))

        fov = 300; proj = []
        for x, y, z in TP:
            persp = fov / (fov + z * size * 0.3)
            proj.append((int(cx + x * size * persp), int(cy + y * size * persp * 0.75), z))

        max_d = 0.55
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                x0, y0, z0 = TP[i]; x1, y1, z1 = TP[j]
                d = math.hypot(x1 - x0, y1 - y0, z1 - z0)
                if d < max_d:
                    avg_z = (z0 + z1) * 0.5
                    if avg_z > -0.2:
                        h = (math.sin(t * 2 + d * 8) + 1) / 2
                        c = lerp_color(ca1, ca2, h)
                        a = max(1, int(1 + avg_z * 5))
                        draw_glow_line(img, proj[i][:2], proj[j][:2], c, a, int(3 + avg_z * 5))
                        cv2.line(img, proj[i][:2], proj[j][:2], c, 1, cv2.LINE_AA)

        tips_list = finger_lists if len(finger_lists) == 2 else (finger_lists + [[]])
        for tips in tips_list:
            for tip in tips:
                ang_t = math.atan2(tip[1] - cy, tip[0] - cx)
                off = 10 * math.sin(t * 6 + ang_t * 3)
                ex = int(cx + (tip[0] - cx) * 0.25 + math.cos(t * 4) * off)
                ey = int(cy + (tip[1] - cy) * 0.25 + math.sin(t * 4) * off)
                draw_glow_line(img, tip, (ex, ey), cw, 1, 6)
                draw_glow_line(img, tip, (ex, ey), ca2, 1, 4)

        cr = 7 + int(math.sin(t * 5) * 3)
        draw_soft_circle(img, (cx, cy), cr + 8, ca1, 0.12)
        draw_soft_circle(img, (cx, cy), cr + 3, ca2, 0.25)
        draw_soft_circle(img, (cx, cy), cr,     cw,  0.50)

        for pp in palms:
            pr = 9 + int(math.sin(t * 4 + pp[0] * 0.02) * 3)
            draw_soft_circle(img, pp, pr + 7, ca1, 0.14)
            draw_soft_circle(img, pp, pr,     cw,  0.40)
            cv2.circle(img, pp, 3, cw, -1, cv2.LINE_AA)
