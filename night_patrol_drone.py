"""
night_patrol_drone_v7.py  -  GUI for the night-patrol drone (all physics lives in patrol_core.py)

Keys:  SPACE pause | R reset | N night-look | G show ground truth | M satellite background
       F fast-forward (1x/2x/4x/8x) | T spawn an intruder | ESC quit
"""
import io
import math
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pygame
    import requests
except ImportError as _e:
    print("\nMISSING PACKAGE:", _e)
    print("Fix:  pip install numpy pygame requests   (or: python -m pip install -r requirements.txt)")
    input("\nPress Enter to close...")
    sys.exit(1)

import patrol_core as core

if sys.platform == "win32":                       # real pixels: no blurry text, window no longer grows with Windows display scaling
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# ------------------------------------------------------------------ settings
WIDTH, HEIGHT, SIDEBAR_W = 1320, 820, 380
MAP_W = WIDTH - SIDEBAR_W
FPS = 60
DT = 1.0 / FPS

# >>> SET THESE to the centre of the real ground you chose (hostel ground, substation, yard ...) <<<
ZONE_LAT, ZONE_LON, ZONE_ZOOM = 37.6213, -122.3790, 18      # placeholder - change me
USE_SATELLITE_AT_START = False

COMMAND_CENTER = "http://127.0.0.1:8000"
DRONE_ID = "DRONE-01"
# WHERE the intruder / not-intruder prediction runs:
#   "COMMAND_CENTER" : drone only sends observations, the command center predicts (default, as designed)
#   "ONBOARD"        : offline mode - prediction engine runs inside this program (no server needed)
PREDICTION_AT = "COMMAND_CENTER"

BG, PANEL = (5, 10, 8), (9, 17, 13)
BORDER, GREEN, GREEN_LIGHT, GREEN_DARK, MUTED = (25, 65, 40), (40, 255, 130), (170, 255, 205), (25, 100, 55), (75, 130, 95)
WHITE, YELLOW, ORANGE, RED, BLUE, GRAY = (235, 255, 240), (255, 220, 70), (255, 160, 40), (255, 55, 55), (80, 180, 255), (110, 130, 120)
STATUS_COLOR = {"UNVERIFIED": WHITE, "THREAT": RED, "SUSPICIOUS": YELLOW, "NOT_THREAT": GREEN, "NOT_A_PERSON": GRAY, "TENTATIVE": GRAY}
KIND_COLOR = {"INTRUDER": ORANGE, "STAFF": BLUE, "ANIMAL": (170, 150, 110)}

CACHE = Path.home() / ".night_patrol_satellite_cache"
CACHE.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ satellite background (zone-sized, metric scale)
class SatBG:
    def __init__(self, lat, lon, zoom, region_w_m, region_h_m, px_per_m):
        n = 2.0 ** zoom
        self.zoom = zoom
        mpp = 156543.03392 * math.cos(math.radians(lat)) / n          # metres per tile pixel
        gx = (lon + 180) / 360 * n * 256
        gy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256
        self.k = px_per_m * mpp                                          # screen px per tile px
        self.rw, self.rh = int(region_w_m * px_per_m), int(region_h_m * px_per_m)
        hw, hh = self.rw / 2 / self.k, self.rh / 2 / self.k
        self.x0, self.y0, self.x1, self.y1 = gx - hw, gy - hh, gx + hw, gy + hh
        self.tiles = [(tx, ty) for tx in range(int(self.x0 // 256), int(self.x1 // 256) + 1)
                      for ty in range(int(self.y0 // 256), int(self.y1 // 256) + 1)]
        self.raw, self.surf_tiles, self.lock = {}, {}, threading.Lock()
        self.surface, self.started, self.dirty = None, False, False

    def start(self):
        if self.started:
            return
        self.started = True
        for t in self.tiles:
            threading.Thread(target=self._fetch, args=t, daemon=True).start()

    def _fetch(self, tx, ty):
        path = CACHE / f"sat_{self.zoom}_{tx}_{ty}.jpg"
        try:
            if path.exists():
                data = path.read_bytes()
            else:
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{ty}/{tx}"
                r = requests.get(url, headers={"User-Agent": "NightPatrolGCS/7"}, timeout=6)
                if r.status_code != 200 or not r.content:
                    return
                path.write_bytes(r.content)
                data = r.content
            with self.lock:
                self.raw[(tx, ty)] = data
        except Exception:
            pass

    def pump(self):
        with self.lock:
            new = dict(self.raw)
            self.raw.clear()
        for key, data in new.items():
            try:
                self.surf_tiles[key] = pygame.image.load(io.BytesIO(data), "t.jpg").convert()
                self.dirty = True
            except Exception:
                pass
        if self.dirty:
            self.dirty = False
            tx0, ty0 = int(self.x0 // 256), int(self.y0 // 256)
            nx, ny = int(self.x1 // 256) - tx0 + 1, int(self.y1 // 256) - ty0 + 1
            mosaic = pygame.Surface((nx * 256, ny * 256))
            mosaic.fill((18, 27, 22))
            for (tx, ty), s in self.surf_tiles.items():
                mosaic.blit(s, ((tx - tx0) * 256, (ty - ty0) * 256))
            crop = pygame.Rect(int(self.x0 - tx0 * 256), int(self.y0 - ty0 * 256), int(self.x1 - self.x0), int(self.y1 - self.y0))
            crop.clamp_ip(mosaic.get_rect())
            self.surface = pygame.transform.smoothscale(mosaic.subsurface(crop), (self.rw, self.rh))

    def status(self):
        return len(self.surf_tiles), len(self.tiles)


# ------------------------------------------------------------------ synthetic thermal image from the SAME physics
def build_iron_lut():
    pts = [(0.0, (0, 0, 8)), (0.25, (50, 0, 110)), (0.5, (190, 30, 70)), (0.75, (250, 160, 20)), (1.0, (255, 255, 225))]
    xs = np.linspace(0, 1, 256)
    return np.stack([np.interp(xs, [p[0] for p in pts], [p[1][c] for p in pts]) for c in range(3)], 1).astype(np.uint8)


IRON = build_iron_lut()


def render_thermal(world, w=280, h=224):
    d, cam, zone = world.drone, world.cfg.cam, world.zone
    rng = np.random.default_rng(int(world.t * 8))
    img = 0.20 + 0.018 * rng.standard_normal((h, w))                 # sensor noise (NETD) + ground
    if d.alt < 3:
        return pygame.surfarray.make_surface(IRON[np.clip(img * 255, 0, 255).astype(np.uint8)].swapaxes(0, 1)), None
    ha, hl = cam.half_extent(d.alt)
    c, s = math.cos(math.radians(d.yaw)), math.sin(math.radians(d.yaw))
    ppx, ppy = w / (2 * ha), h / (2 * hl)

    def to_px(x, y):
        dx, dy = x - d.x, y - d.y
        return w / 2 + (-dx * s + dy * c) * ppx, h / 2 - (dx * c + dy * s) * ppy

    yy, xx = np.mgrid[0:h, 0:w]
    for tx, ty, r in zone.trees:                                       # canopy: slightly warmer texture
        u, v = to_px(tx, ty)
        img += 0.06 * (((xx - u) / (r * ppx)) ** 2 + ((yy - v) / (r * ppy)) ** 2 <= 1.0)
    for a in world.agents:
        u, v = to_px(a.x, a.y)
        if not (-10 < u < w + 10 and -10 < v < h + 10):
            continue
        sx, sy = max(0.8, 0.55 * a.d_c * ppx), max(0.8, 0.55 * a.d_c * ppy)   # blob size = physical size / GSD
        amp = (0.30 + 0.45 * a.d_temp / 4.0) * (0.15 if zone.in_canopy(a.x, a.y) else 1.0)
        img += amp * np.exp(-(((xx - u) / sx) ** 2 + ((yy - v) / sy) ** 2) / 2)
    surf = pygame.surfarray.make_surface(IRON[(np.clip(img, 0, 1) * 255).astype(np.uint8)].swapaxes(0, 1))
    return surf, to_px


# ------------------------------------------------------------------ drone -> command center link
class RemoteClassifier:
    """Sends OBSERVATIONS to the command center and receives its VERDICTS. All network I/O is in a thread.
    If the command center is unreachable no verdict arrives: tracks simply stay UNVERIFIED."""

    def __init__(self, url):
        self.url, self.online, self.running = url, False, True
        self.q = queue.Queue(maxsize=200)
        self.verdicts, self.lock, self.info = {}, threading.Lock(), {}
        self.n_sent = self.n_fail = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, obs):
        try:
            self.q.put_nowait(obs)
        except queue.Full:                           # link too slow: drop the oldest observation
            try:
                self.q.get_nowait()
                self.q.put_nowait(obs)
            except (queue.Empty, queue.Full):
                pass

    def poll(self, track_id):
        with self.lock:
            return self.verdicts.pop(track_id, None)

    def clear(self):
        with self.lock:
            self.verdicts.clear()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _loop(self):
        s = requests.Session()
        while self.running:
            if not self.info:
                try:
                    self.info = s.get(self.url + "/api/model", timeout=1.0).json().get("info", {})
                except Exception:
                    pass
            try:
                obs = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                r = s.post(self.url + "/api/observations", json=obs, timeout=1.5)
                r.raise_for_status()
                with self.lock:
                    self.verdicts[obs["track_id"]] = r.json()
                self.online, self.n_sent = True, self.n_sent + 1
            except Exception:
                self.online, self.n_fail = False, self.n_fail + 1


# ------------------------------------------------------------------ main application
class App:
    def __init__(self):
        pygame.init()
        try:                                    # SCALED: fixed 1320x820 layout is fitted to the screen, window is resizable
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("AI Night-Patrol Drone v7 - model-based simulation")
        self.clock = pygame.time.Clock()
        names = "consolas,dejavusansmono,couriernew"
        self.f_xs, self.f_sm, self.f_md, self.f_lg = [pygame.font.SysFont(names, s, bold=b) for s, b in ((10, False), (11, False), (12, True), (16, True))]
        self.cfg = core.Config()
        self.screen.fill(BG)
        pygame.font.SysFont("consolas,dejavusansmono,couriernew", 22, bold=True)
        self.screen.blit(pygame.font.SysFont("consolas,dejavusansmono,couriernew", 22, bold=True).render(
            "Loading threat model (first run trains it, ~10 s) ...", True, GREEN), (60, HEIGHT // 2))
        pygame.display.flip()
        pygame.event.pump()
        print("[START] window opened, loading model ...")
        if PREDICTION_AT == "ONBOARD":
            self.model = core.ThreatModel.load_or_train(self.cfg, verbose=True)
            self.remote = None
        else:                                       # the drone carries NO threat model - the command center predicts
            self.model, self.remote = None, RemoteClassifier(COMMAND_CENTER)
        z = self.cfg.zone
        self.scale = min((MAP_W - 80) / z.width, (598 - 140) / z.height)
        self.cx, self.cy = MAP_W // 2, 42 + 598 // 2 - 6
        self.sat = SatBG(ZONE_LAT, ZONE_LON, ZONE_ZOOM, z.width + 30, z.height + 30, self.scale)
        self.use_sat = USE_SATELLITE_AT_START
        if self.use_sat:
            self.sat.start()
        self.R_axis = np.arange(5, 361, 5)
        self.curve_det = [core.p_detect(self.cfg.cam, core.PERSON_DC, float(r)) for r in self.R_axis]
        self.curve_rec = [core.p_recognize(self.cfg.cam, core.PERSON_DC, float(r)) for r in self.R_axis]
        self.night, self.show_truth, self.paused = True, False, False
        self.speed_opts, self.speed_i = [1, 2, 4, 8], 0
        self.msg, self.msg_t = "SYSTEM READY", 4.0
        self.thermal, self.thermal_px, self.thermal_t = None, None, -1.0
        self.ov = pygame.Surface((MAP_W, HEIGHT), pygame.SRCALPHA)       # translucent map layer
        self.trails, self.dtrail = {}, []
        # command-center plumbing (all network I/O off the GUI thread)
        self.running, self.live = True, {}
        self.live_lock = threading.Lock()
        try:
            requests.post(COMMAND_CENTER + "/api/session/reset", timeout=1.5).raise_for_status()
            print("[COMMAND CENTER] session reset")
        except requests.RequestException as e:
            print(f"[COMMAND CENTER] offline ({e}); GUI still works")
        threading.Thread(target=self._world_sender, daemon=True).start()
        self.new_world()

    # ---------------- world / network
    def new_world(self):
        if self.remote:
            self.remote.clear()
        self.world = core.World(self.cfg, None, self.model, classifier=self.remote, drone_id=DRONE_ID)
        for k in ("STAFF", "ANIMAL", "INTRUDER"):
            self.world.add_agent(k)
        self.trails, self.dtrail = {}, []
        self.msg, self.msg_t = "NEW SESSION", 2.5

    def _world_sender(self):
        s = requests.Session()
        while self.running:
            with self.live_lock:
                p = dict(self.live)
            if p:
                try:
                    s.post(COMMAND_CENTER + "/api/world-state", json=p, timeout=1.0)
                except requests.RequestException:
                    pass
            time.sleep(0.5)

    def publish_world_state(self):
        w = self.world
        act = w.agents
        intr = [a for a in act if a.kind == "INTRUDER"]
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "total_targets_in_world": len(act),
                   "active_targets": len(act), "active_civilians": len(act) - len(intr), "active_intruders": len(intr),
                   "resolved_targets": 0,
                   "targets": [{"target_id": a.id, "category": core.TRUTH_LABEL[a.kind], "status": "ACTIVE", "visible": False,
                                "ai_label": "TRUTH", "confidence": 0.0} for a in act]}
        with self.live_lock:
            self.live = payload

    def handle_events_from_world(self):
        """Only banners here: the command center already received the observations and made the decision."""
        for ev in self.world.events:
            tr = ev["track"]
            if ev["type"] == "ALERT":
                self.msg, self.msg_t = f"CC VERDICT: THREAT {tr.id}  p={tr.p_threat:.2f}  incident {tr.incident_id}", 6.0
            elif ev["type"] == "MERGED":
                self.msg, self.msg_t = f"CC VERDICT: {tr.id} merged into open incident {tr.incident_id}", 3.0
            elif ev["type"] == "CLEARED":
                self.msg, self.msg_t = f"CC VERDICT: {tr.id} cleared - normal activity p={tr.p_threat:.2f}", 3.0
            elif ev["type"] == "CONFIRMED":
                self.msg, self.msg_t = f"{tr.id} hot-spot confirmed -> verifying, observations sent to command center", 2.0
        self.world.events.clear()

    # ---------------- drawing helpers
    def w2s(self, x, y):
        return int(self.cx + x * self.scale), int(self.cy + y * self.scale)

    def text(self, v, x, y, font=None, color=GREEN):
        self.screen.blit((font or self.f_sm).render(str(v), True, color), (x, y))

    def dashed_rect(self, rect, color, dash=8):
        x0, y0, x1, y1 = rect
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            for i in range(0, int(L), dash * 2):
                t0, t1 = i / L, min(L, i + dash) / L
                pygame.draw.line(self.screen, color, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                                 (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), 2)

    # ---------------- map helpers
    def pill(self, label, x, y, color):
        img = self.f_xs.render(label, True, color)
        r = pygame.Rect(x, y, img.get_width() + 10, img.get_height() + 4)
        bg = pygame.Surface(r.size, pygame.SRCALPHA)
        bg.fill((5, 10, 8, 205))
        self.screen.blit(bg, r.topleft)
        pygame.draw.rect(self.screen, color, r, 1)
        self.screen.blit(img, (r.x + 5, r.y + 2))

    def dashed_line(self, surf, color, a, b, width=2, dash=8, gap=6):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1:
            return
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        t = 0.0
        while t < L:
            t2 = min(L, t + dash)
            pygame.draw.line(surf, color, (a[0] + ux * t, a[1] + uy * t), (a[0] + ux * t2, a[1] + uy * t2), width)
            t += dash + gap

    def arrow(self, color, a, b, width=2, head=7):
        pygame.draw.line(self.screen, color, a, b, width)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for da in (2.6, -2.6):
            pygame.draw.line(self.screen, color, b, (b[0] + math.cos(ang + da) * head, b[1] + math.sin(ang + da) * head), width)

    def update_trails(self):
        w, d = self.world, self.world.drone
        if not self.dtrail or math.hypot(self.dtrail[-1][0] - d.x, self.dtrail[-1][1] - d.y) > 0.6:
            self.dtrail.append((d.x, d.y))
            del self.dtrail[:-140]
        for tr in w.tracker.tracks:
            if not tr.confirmed:
                continue
            q = self.trails.setdefault(tr.id, [])
            if not q or math.hypot(q[-1][0] - tr.pos[0], q[-1][1] - tr.pos[1]) > 0.5:
                q.append(tr.pos)
                del q[:-50]

    def draw_map(self):
        w, z, d, cfg = self.world, self.world.zone, self.world.drone, self.cfg
        view = pygame.Rect(0, 42, MAP_W, 598)
        pygame.draw.rect(self.screen, (5, 12, 9), view)
        if self.use_sat:
            self.sat.pump()
            if self.sat.surface:
                self.screen.blit(self.sat.surface, (self.cx - self.sat.rw // 2, self.cy - self.sat.rh // 2))
        old_clip = self.screen.get_clip()
        self.screen.set_clip(view)
        fl, ft = self.w2s(-z.width / 2, -z.height / 2)
        fr, fb = self.w2s(z.width / 2, z.height / 2)
        ov = self.ov
        ov.fill((0, 0, 0, 0))
        if not self.use_sat:
            pygame.draw.rect(self.screen, (9, 24, 16), (fl, ft, fr - fl, fb - ft))      # ground inside the fence
            for gx in range(-60, 61, 20):                                                 # metric grid every 20 m
                pygame.draw.line(self.screen, (17, 44, 30), self.w2s(gx, -z.height / 2), self.w2s(gx, z.height / 2), 1)
                self.text(str(gx), self.w2s(gx, 0)[0] - 8, fb + 4, self.f_xs, MUTED)
            for gy in range(-40, 41, 20):
                pygame.draw.line(self.screen, (17, 44, 30), self.w2s(-z.width / 2, gy), self.w2s(z.width / 2, gy), 1)
                self.text(str(gy), fl - 26, self.w2s(0, gy)[1] - 6, self.f_xs, MUTED)
        # tree canopies (thermal cannot see through them)
        for tx, ty, r in z.trees:
            p, R = self.w2s(tx, ty), int(r * self.scale)
            pygame.draw.circle(ov, (28, 105, 58, 80), p, R)
            pygame.draw.circle(ov, (60, 170, 100, 150), p, R, 1)
        # authorised walkway: soft band + dashed centre line
        wk = [self.w2s(*q) for q in z.walkway]
        pygame.draw.lines(ov, (215, 195, 120, 38), False, wk, int(2.6 * self.scale))
        for i in range(len(wk) - 1):
            self.dashed_line(ov, (225, 205, 130, 230), wk[i], wk[i + 1], 2, 9, 7)
        # restricted zone around the asset
        r0 = z.restricted_rect()
        rx0, ry0, rx1, ry1 = *self.w2s(r0[0], r0[1]), *self.w2s(r0[2], r0[3])
        pygame.draw.rect(ov, (255, 60, 60, 30), (rx0, ry0, rx1 - rx0, ry1 - ry0))
        for a_, b_ in (((rx0, ry0), (rx1, ry0)), ((rx1, ry0), (rx1, ry1)), ((rx1, ry1), (rx0, ry1)), ((rx0, ry1), (rx0, ry0))):
            self.dashed_line(ov, (235, 80, 80, 255), a_, b_, 2, 9, 6)
        self.screen.blit(ov, (0, 0))
        self.text("RESTRICTED ZONE", rx0 + 6, ry0 + 5, self.f_xs, (235, 90, 90))
        # protected asset
        bx, by, bw, bh = z.building
        x0, y0 = self.w2s(bx - bw / 2, by - bh / 2)
        rect = pygame.Rect(x0, y0, int(bw * self.scale), int(bh * self.scale))
        pygame.draw.rect(self.screen, (66, 84, 92), rect)
        pygame.draw.rect(self.screen, (165, 190, 200), rect, 2)
        lab = self.f_xs.render("PROTECTED ASSET", True, WHITE)
        self.screen.blit(lab, (rect.centerx - lab.get_width() // 2, rect.centery - lab.get_height() // 2))
        self.text("authorised walkway", wk[1][0] + 8, wk[1][1] - 16, self.f_xs, (225, 205, 130))
        # fence with posts, gate, home pad
        pygame.draw.rect(self.screen, (232, 232, 120), (fl, ft, fr - fl, fb - ft), 3)
        for gx in range(-60, 61, 10):
            for gy in (-40, 40):
                q = self.w2s(gx, gy)
                pygame.draw.rect(self.screen, (255, 255, 160), (q[0] - 2, q[1] - 2, 5, 5))
        for gy in range(-30, 31, 10):
            for gx in (-60, 60):
                q = self.w2s(gx, gy)
                pygame.draw.rect(self.screen, (255, 255, 160), (q[0] - 2, q[1] - 2, 5, 5))
        gx_, gy_ = self.w2s(*z.gate)
        pygame.draw.circle(self.screen, YELLOW, (gx_, gy_), 9, 2)
        self.pill("GATE", gx_ + 13, gy_ - 9, YELLOW)
        hx, hy = self.w2s(*z.home)
        pygame.draw.rect(self.screen, GREEN, (hx - 11, hy - 11, 22, 22), 2)
        pygame.draw.circle(self.screen, GREEN, (hx, hy), 7, 1)
        self.text("H", hx - 3, hy - 6, self.f_xs, GREEN)
        # patrol loop with direction arrows and the active waypoint
        pts = [self.w2s(*p) for p in w.ring]
        for i in range(len(pts)):
            a_, b_ = pts[i], pts[(i + 1) % len(pts)]
            pygame.draw.line(self.screen, (28, 105, 62), a_, b_, 1)
        for i, p in enumerate(pts):
            on = (i == w.wp_i and w.mode == "PATROL")
            pygame.draw.circle(self.screen, YELLOW if on else GREEN_DARK, p, 5 if on else 2)
        if w.mode == "PATROL":
            self.dashed_line(self.screen, (255, 220, 70), self.w2s(d.x, d.y), pts[w.wp_i], 1, 4, 5)
        # camera footprint
        ha, hl = cfg.cam.half_extent(d.alt)
        c, s = math.cos(math.radians(d.yaw)), math.sin(math.radians(d.yaw))
        corners = [self.w2s(d.x + u * c - v * s, d.y + u * s + v * c) for u, v in ((hl, -ha), (hl, ha), (-hl, ha), (-hl, -ha))]
        ov.fill((0, 0, 0, 0))
        tracking = w.mode == "TRACKING"
        pygame.draw.polygon(ov, (255, 70, 70, 26) if tracking else (40, 255, 140, 18), corners)
        pygame.draw.polygon(ov, (255, 120, 120, 150) if tracking else (130, 255, 190, 110), corners, 1)
        self.screen.blit(ov, (0, 0))
        dp = self.w2s(d.x, d.y)
        # ground truth (evaluation overlay)
        if self.show_truth:
            for a in w.agents:
                p, col = self.w2s(a.x, a.y), KIND_COLOR[a.kind]
                if a.kind == "INTRUDER":
                    pygame.draw.polygon(self.screen, col, [(p[0], p[1] - 7), (p[0] - 6, p[1] + 5), (p[0] + 6, p[1] + 5)], 2)
                elif a.kind == "STAFF":
                    pygame.draw.circle(self.screen, col, p, 6, 2)
                else:
                    pygame.draw.rect(self.screen, col, (p[0] - 5, p[1] - 5, 10, 10), 2)
                self.text(a.id, p[0] + 9, p[1] - 15, self.f_xs, col)
        # tracks = what the command center believes: one circle + one short label each
        tnow = pygame.time.get_ticks() / 1000.0
        labels = []
        for tr in w.tracker.tracks:
            if not tr.confirmed:
                continue
            col, p = STATUS_COLOR[tr.status], self.w2s(*tr.pos)
            if tr.status == "THREAT":                                                      # one soft pulsing ring
                k = (math.sin(tnow * 5) + 1) / 2
                pygame.draw.circle(self.screen, (255, 90, 90), p, 14 + int(6 * k), 1)
            pygame.draw.circle(self.screen, col, p, 8, 2)
            lab = f"{tr.id[-3:]} {tr.status}" + (f" {tr.p_threat:.2f}" if tr.p_threat is not None else "")
            labels.append((lab, col, p))
        # drone body
        for off in (45, 135, 225, 315):
            a = math.radians(d.yaw + off)
            qq = (int(dp[0] + math.cos(a) * 10), int(dp[1] + math.sin(a) * 10))
            pygame.draw.line(self.screen, GRAY, dp, qq, 2)
            pygame.draw.circle(self.screen, YELLOW, qq, 3)
        pygame.draw.circle(self.screen, WHITE, dp, 5)
        pygame.draw.line(self.screen, RED if tracking else GREEN, dp,
                         (dp[0] + math.cos(math.radians(d.yaw)) * 22, dp[1] + math.sin(math.radians(d.yaw)) * 22), 3)
        labels.insert(0, (f"DRONE {d.alt:.0f} m  {w.mode}", RED if tracking else GREEN, dp))
        placed = []                                                                        # labels never overlap each other
        for text, colr, (ax, ay) in labels:
            img = self.f_xs.render(text, True, colr)
            r = pygame.Rect(ax + 14, ay - 8, img.get_width() + 8, img.get_height() + 4)
            for _ in range(10):
                if not any(r.colliderect(q) for q in placed):
                    break
                r.y += r.height + 2
            placed.append(r)
            bg = pygame.Surface(r.size, pygame.SRCALPHA)
            bg.fill((5, 10, 8, 190))
            self.screen.blit(bg, r.topleft)
            self.screen.blit(img, (r.x + 4, r.y + 2))
        # compass, scale bar, legend
        cxn, cyn = MAP_W - 36, 120
        pygame.draw.circle(self.screen, (25, 65, 40), (cxn, cyn), 16, 1)
        self.arrow(GREEN_LIGHT, (cxn, cyn + 9), (cxn, cyn - 11), 2, 5)
        self.text("N", cxn - 3, cyn - 31, self.f_xs, GREEN_LIGHT)
        sb = int(20 * self.scale)
        pygame.draw.line(self.screen, WHITE, (22, 618), (22 + sb, 618), 3)
        pygame.draw.line(self.screen, WHITE, (22, 613), (22, 623), 2)
        pygame.draw.line(self.screen, WHITE, (22 + sb, 613), (22 + sb, 623), 2)
        self.text("20 m", 28 + sb, 612, self.f_xs, WHITE)
        lx = 190
        for name, colr in (("THREAT", RED), ("SUSPICIOUS", YELLOW), ("NOT_THREAT", GREEN), ("UNVERIFIED", WHITE)):
            pygame.draw.circle(self.screen, colr, (lx, 618), 5, 2)
            img = self.f_xs.render(name, True, colr)
            self.screen.blit(img, (lx + 10, 612))
            lx += img.get_width() + 32
        self.text("shaded box = camera view   (G truth | N night | M map | F speed | F11 fullscreen)", lx + 6, 612, self.f_xs, MUTED)
        self.screen.set_clip(old_clip)
        if self.night:
            dark = pygame.Surface((MAP_W, 598))
            dark.fill((228, 234, 228))
            self.screen.blit(dark, (0, 42), special_flags=pygame.BLEND_RGB_MULT)

    def draw_chart(self, x, y, w, h):
        pygame.draw.rect(self.screen, (7, 14, 10), (x, y, w, h))
        pygame.draw.rect(self.screen, BORDER, (x, y, w, h), 1)
        self.text("P(task) vs SLANT RANGE - person, 640x512 thermal", x + 6, y + 3, self.f_xs, GREEN_LIGHT)
        L, R, T, B = x + 30, x + w - 8, y + 18, y + h - 16
        for frac in (0, 0.5, 1.0):
            yy = B - frac * (B - T)
            pygame.draw.line(self.screen, (20, 45, 30), (L, yy), (R, yy), 1)
            self.text(f"{frac:.1f}", x + 4, yy - 5, self.f_xs, MUTED)
        rmax = float(self.R_axis[-1])
        px = lambda r: L + (R - L) * r / rmax
        py = lambda p: B - p * (B - T)
        for curve, col in ((self.curve_det, GREEN), (self.curve_rec, YELLOW)):
            pygame.draw.lines(self.screen, col, False, [(px(r), py(p)) for r, p in zip(self.R_axis, curve)], 2)
        cfg, cam = self.cfg, self.cfg.cam
        marks = [(core.edge_range(cam, cfg.h_search), "SEARCH", BLUE), (core.edge_range(cam, cfg.h_verify), "VERIFY", GREEN_LIGHT),
                 (340.0, "OLD SIM", RED)]
        if self.world.drone.alt > 3:
            marks.append((core.edge_range(cam, self.world.drone.alt), "NOW", WHITE))
        for r, lab, col in marks:
            pygame.draw.line(self.screen, col, (px(r), T), (px(r), B), 1)
            self.text(lab, min(px(r) + 3, R - 44), T + {"SEARCH": 0, "VERIFY": 22, "NOW": 11, "OLD SIM": 0}[lab], self.f_xs, col)
        for r in (0, 100, 200, 300):
            self.text(str(r), px(r) - 6, B + 2, self.f_xs, MUTED)
        self.text("green = detect", R - 120, T + 36, self.f_xs, GREEN)
        self.text("yellow = recognise", R - 120, T + 48, self.f_xs, YELLOW)

    def draw_sidebar(self):
        x = MAP_W
        w, d, cfg, cam = self.world, self.world.drone, self.cfg, self.cfg.cam
        pygame.draw.rect(self.screen, PANEL, (x, 0, SIDEBAR_W, HEIGHT))
        pygame.draw.line(self.screen, BORDER, (x, 0), (x, HEIGHT), 2)
        self.text("DRONE COMMAND & CONTROL", x + 14, 8, self.f_lg, GREEN_LIGHT)
        self.text("MODEL-BASED NIGHT PATROL - SMALL FENCED COMPOUND", x + 14, 28, self.f_xs, MUTED)
        mode_col = RED if w.mode == "TRACKING" else YELLOW if w.mode in ("RTL", "SWAP") else GREEN
        e_min = d.soc * cfg.battery_wh * 60 / max(d.power_w, 1)
        rows = [("MODE", w.mode, mode_col), ("ALTITUDE", f"{d.alt:.0f} m  (search {cfg.h_search:.0f} / verify {cfg.h_verify:.0f})", GREEN),
                ("GROUND SPEED", f"{d.speed:.1f} m/s", GREEN), ("BATTERY", f"{d.soc * 100:.0f} %  ({e_min:.0f} min left, {d.power_w:.0f} W)", GREEN if d.soc > 0.3 else RED),
                ("SWEEPS / SWAPS", f"{w.sweeps} / {w.swaps}", GREEN), ("LOOP", f"{w.n_lanes} lanes @ {w.lane_sp:.0f} m, {core.ring_length(w.ring) / cfg.v_cruise:.0f} s", GREEN),
                ("TRACKS (conf.)", str(sum(1 for t in w.tracker.tracks if t.confirmed)), GREEN),
                ("SIM TIME", f"{int(w.t // 60):02d}:{int(w.t % 60):02d}  x{self.speed_opts[self.speed_i]}", GREEN),
                ("PREDICTION AT", "COMMAND CENTER" if self.remote else "ONBOARD (offline mode)", GREEN),
                ("CC LINK", ("ONLINE  sent %d" % self.remote.n_sent if self.remote.online else "OFFLINE - no verdicts") if self.remote else "n/a",
                 GREEN if (self.remote and self.remote.online) or not self.remote else RED)]
        y = 46
        for lab, val, col in rows:
            self.text(lab, x + 14, y, self.f_xs, MUTED)
            self.text(val, x + 128, y, self.f_sm, col)
            y += 15
        # thermal view
        ty = 200
        pygame.draw.rect(self.screen, (0, 0, 0), (x + 12, ty, 356, 246))
        pygame.draw.rect(self.screen, RED, (x + 12, ty, 356, 246), 1)
        self.text("SIMULATED THERMAL FRAME (same physics as detector)", x + 18, ty + 2, self.f_xs, RED)
        if self.thermal is None or w.t - self.thermal_t >= 0.125 or self.paused:
            self.thermal, self.thermal_px = render_thermal(w)
            self.thermal_t = w.t
        ix = x + 12 + (356 - 280) // 2
        self.screen.blit(self.thermal, (ix, ty + 16))
        if self.thermal_px:
            for tr in w.tracker.tracks:
                if tr.confirmed:
                    u, v = self.thermal_px(*tr.pos)
                    if 0 < u < 280 and 0 < v < 224:
                        col = STATUS_COLOR[tr.status]
                        pygame.draw.rect(self.screen, col, (ix + u - 9, ty + 16 + v - 9, 18, 18), 1)
                        self.text(tr.id[-3:], ix + u - 9, ty + 16 + v + 10, self.f_xs, col)
        if d.alt > 3:
            ha, hl = cam.half_extent(d.alt)
            self.text(f"{2 * ha:.0f} x {2 * hl:.0f} m   GSD {cam.gsd(d.alt) * 100:.1f} cm/px", ix + 3, ty + 18, self.f_xs, WHITE)
        # sensor-physics readout
        sy = 452
        pygame.draw.rect(self.screen, (7, 14, 10), (x + 12, sy, 356, 84))
        pygame.draw.rect(self.screen, BORDER, (x + 12, sy, 356, 84), 1)
        self.text("SENSOR PHYSICS (Johnson criteria)", x + 18, sy + 3, self.f_md, GREEN_LIGHT)
        Re = core.edge_range(cam, max(d.alt, 5))
        cyc = core.cycles_on_target(cam, core.PERSON_DC, Re, 4.0, cfg.atm_beta_km)
        npix = core.PERSON_DC / (Re * cam.ifov)
        self.text(f"IFOV {cam.ifov * 1e3:.3f} mrad  FOV {math.degrees(cam.hfov):.1f}x{math.degrees(cam.vfov):.1f} deg", x + 18, sy + 20, self.f_xs, MUTED)
        self.text(f"Person at image corner (R={Re:.0f} m): {npix:.1f} px = {cyc:.1f} cycles", x + 18, sy + 34, self.f_xs, WHITE)
        self.text(f"P(detect)={core.ttpf(cyc, 1.0):.2f}   P(recognise)={core.ttpf(cyc, 4.0):.2f}", x + 18, sy + 48,
                  self.f_xs, GREEN if core.ttpf(cyc, 4.0) > 0.8 else YELLOW)
        self.text("High = detect hot-spot only; descend to verify." if d.alt > cfg.h_verify + 6 else "Low = shape resolved: classification valid.",
                  x + 18, sy + 62, self.f_xs, YELLOW)
        self.draw_chart(x + 12, 542, 356, 118)
        # evaluation panel
        ey = 666
        m = w.metrics()
        pygame.draw.rect(self.screen, (7, 14, 10), (x + 12, ey, 356, 148))
        pygame.draw.rect(self.screen, BORDER, (x + 12, ey, 356, 148), 1)
        self.text("LIVE EVALUATION (ground truth used only here)", x + 18, ey + 3, self.f_md, GREEN_LIGHT)
        fmt = lambda v, f: "n/a" if v != v else f % v
        info = (self.remote.info if self.remote else self.model.info) or {}
        lines = [f"Hostile persons done/alerted : {m['intruders']} / {round(m['alert_rate'] * m['intruders']) if m['intruders'] else 0}",
                 f"Alert rate                 : {fmt(m['alert_rate'] * 100 if m['intruders'] else float('nan'), '%.0f %%')}",
                 f"Alert before reaching asset: {fmt(m['alert_before_asset'] * 100 if m['intruders'] else float('nan'), '%.0f %%')}",
                 f"Mean time to alert         : {fmt(m['mean_alert_delay_s'], '%.0f s')}",
                 f"False alerts (not hostile)   : {w.false_alerts}",
                 f"Track decisions TP/FP/FN/TN: {m['cm']['TP']}/{m['cm']['FP']}/{m['cm']['FN']}/{m['cm']['TN']}",
                 f"Model at CC (held-out): acc {info.get('accuracy', 0) * 100:.0f}%  recall {info.get('recall', 0) * 100:.0f}%  AUC {info.get('auc', 0):.2f}",
                 f"Merged repeats {w.merged_alerts} | animals filtered {w.gate['animal_filtered']}/{w.gate['animal_filtered'] + w.gate['animal_passed']}"]
        for i, l in enumerate(lines):
            self.text(l, x + 18, ey + 20 + i * 15, self.f_xs, WHITE if i < 6 else MUTED)

    def draw_top_bar(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, MAP_W, 42))
        pygame.draw.line(self.screen, BORDER, (0, 42), (MAP_W, 42), 2)
        z = self.cfg.zone
        self.text("AI NIGHT-PATROL GCS", 12, 11, self.f_lg, GREEN_LIGHT)
        self.text(f"ZONE {z.width:.0f} x {z.height:.0f} m | thermal + Kalman + threat model | PID flight control | wind {self.cfg.wind_sigma:.1f}", 220, 15, self.f_xs, (130, 185, 150))
        if self.use_sat:
            n, tot = self.sat.status()
            self.text(f"SAT {n}/{tot}", MAP_W - 70, 15, self.f_xs, GREEN if n == tot else YELLOW)
        if self.msg_t > 0:
            col = RED if "ALERT" in self.msg else GREEN
            rect = pygame.Rect(12, 50, MAP_W - 24, 28)
            pygame.draw.rect(self.screen, (10, 20, 14), rect)
            pygame.draw.rect(self.screen, col, rect, 2)
            self.text(self.msg, rect.x + 10, rect.y + 7, self.f_md, col)

    def draw_table(self):
        y0 = 642
        pygame.draw.rect(self.screen, (7, 14, 10), (8, y0, MAP_W - 16, 172))
        pygame.draw.rect(self.screen, BORDER, (8, y0, MAP_W - 16, 172), 1)
        cols = [15, 82, 172, 242, 312, 390, 470, 535, 610, 720]
        heads = ["TRACK", "STATUS", "P(THREAT)", "SPEED", "LOITER", "PROXIMITY", "ROUTE DEV", "AGE", "SIM TRUTH*", "DRONE ACTION"]
        for c, h in zip(cols, heads):
            self.text(h, c, y0 + 6, self.f_xs, MUTED)
        w = self.world
        trs = sorted([t for t in w.tracker.tracks if t.confirmed], key=lambda t: t.id)[-8:]
        for i, tr in enumerate(trs):
            y = y0 + 24 + i * 16
            f = tr.features(w.zone) if tr.hist else [0, 0, 0, 0]          # [speed, loiter, proximity, route_dev]
            col = STATUS_COLOR[tr.status]
            act = ("IGNORE (not a person)" if tr.status == "NOT_A_PERSON" else "SHADOW" if tr.alerted else
                   "VERIFY" if tr.status == "UNVERIFIED" else "RESUME PATROL")
            shown = tr.status != "NOT_A_PERSON"
            vals = [tr.id, tr.status, f"{tr.p_threat:.2f}" if tr.p_threat is not None else "--",
                    f"{f[0]:.1f} m/s" if shown else "--", f"{f[1]:.2f}" if shown else "--",
                    f"{f[2]:.0f} m" if shown else "--", f"{f[3]:.1f} m" if shown else "--",
                    f"{w.t - tr.t0:.0f} s", w.truth_label(tr), act]
            for c, v in zip(cols, vals):
                self.text(v, c, y, self.f_xs, col)
        self.text("* SIM TRUTH is only for scoring. The drone sends speed, loitering, proximity and route deviation (no temperature); the COMMAND CENTER decides.",
                  15, y0 + 154, self.f_xs, (70, 95, 78))

    # ---------------- loop
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                elif e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_n:
                    self.night = not self.night
                elif e.key == pygame.K_g:
                    self.show_truth = not self.show_truth
                elif e.key == pygame.K_f:
                    self.speed_i = (self.speed_i + 1) % len(self.speed_opts)
                elif e.key == pygame.K_m:
                    self.use_sat = not self.use_sat
                    if self.use_sat:
                        self.sat.start()
                elif e.key == pygame.K_r:
                    try:
                        requests.post(COMMAND_CENTER + "/api/session/reset", timeout=1.0)
                    except requests.RequestException:
                        pass
                    self.new_world()
                elif e.key == pygame.K_t:
                    a = self.world.add_agent("INTRUDER")
                    self.msg, self.msg_t = f"MANUAL HOSTILE PERSON {a.id} ENTERED FENCE", 3.0
        return True

    def run(self):
        while self.handle_events():
            if not self.paused:
                for _ in range(self.speed_opts[self.speed_i]):
                    self.world.step(DT)
                self.msg_t = max(0.0, self.msg_t - DT * self.speed_opts[self.speed_i])
                self.handle_events_from_world()
                self.publish_world_state()
                self.update_trails()
            self.screen.fill(BG)
            self.draw_map()
            self.draw_sidebar()
            self.draw_top_bar()
            self.draw_table()
            if self.paused:
                self.text("PAUSED", MAP_W // 2 - 30, 84, self.f_lg, YELLOW)
            pygame.display.flip()
            self.clock.tick(FPS)
        self.running = False
        if self.remote:
            self.remote.running = False
        try:
            requests.post(COMMAND_CENTER + "/api/session/stop", timeout=1.5)
        except requests.RequestException:
            pass
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    try:
        print("[START] launching night patrol simulation ...")
        App().run()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nERROR above - copy it and send it. Press Enter to close...")