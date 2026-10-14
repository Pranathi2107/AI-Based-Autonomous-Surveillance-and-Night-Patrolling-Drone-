"""
patrol_core.py  -  MATHEMATICAL MODEL + SIMULATION CORE  (no pygame in here)
=============================================================================
AI night-patrol drone for a SMALL FENCED COMPOUND (default 120 m x 80 m).

Everything is in SI units (metres, seconds). The pygame GUI only draws what
this file computes, and `python patrol_core.py --eval` runs the very same
logic headless for Monte-Carlo validation.

MODEL BLOCKS (each one is a section below)
  1. Thermal camera        IFOV, FOV, ground footprint, GSD
  2. Johnson criteria      cycles on target -> P(detect), P(recognise)  (TTPF)
  3. Search theory         coverage rate, lane spacing, revisit time
  4. Drone                 point-mass kinematics + energy model + PID flight controller + wind
  5. Target behaviour      intruder / authorised staff / animal agents (Poisson arrivals)
  6. Tracking              constant-velocity Kalman filter + M-of-N confirmation
  7. Threat classifier     logistic regression on TRACK features (no ground truth)
  8. Mission logic         search high -> descend to verify -> alert -> resume
  9. Metrics + CLI         Monte-Carlo evaluation

CONTROLLERS (changed from plain P):
  * velocity loop  (Drone.step)       PID, derivative on measurement, anti-windup, accel saturation
  * follow loop    (World._mission)   PID on position error + target-velocity feed-forward
  * wind gusts (Ornstein-Uhlenbeck acceleration disturbance) are simulated so the integral term matters.

HONEST LIMITS: camera numbers are a representative 640x512 / 12 um / 13.5 mm
uncooled thermal payload - replace them with your real camera. The classifier
is trained on simulated tracks because no real intrusion data exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from threat_engine import ThreatEngine          # the prediction brain (lives at the command center)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets"


# ---------------------------------------------------------------- helpers
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def poisson(rng, lam):
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-clamp(x, -30.0, 30.0)))


# ============================================================ 0. PID CONTROLLER
class PID2D:
    """2-axis PID controller.

        u = kp * e  +  I  +  kd * (d ref/dt - d meas/dt)

    * derivative on MEASUREMENT (the reference derivative is supplied separately, default 0)
      -> no derivative kick when the setpoint jumps (e.g. at waypoint changes)
    * conditional-integration anti-windup: the integrator only accumulates while the output is NOT saturated
    * integral clamp (i_lim, per axis) and output magnitude saturation (out_lim)
    """

    def __init__(self, kp, ki, kd, i_lim, out_lim):
        self.kp, self.ki, self.kd, self.i_lim, self.out_lim = kp, ki, kd, i_lim, out_lim
        self.reset()

    def reset(self):
        self.ix = self.iy = 0.0
        self.prev = None

    def step(self, ref, meas, dt, meas_vel=None, ref_vel=(0.0, 0.0)):
        ex, ey = ref[0] - meas[0], ref[1] - meas[1]
        if meas_vel is None:                                   # estimate measurement rate by finite difference
            if self.prev is None or dt <= 0:
                meas_vel = ref_vel                             # first call: no derivative action
            else:
                meas_vel = ((meas[0] - self.prev[0]) / dt, (meas[1] - self.prev[1]) / dt)
        self.prev = (meas[0], meas[1])
        ux = self.kp * ex + self.ix + self.kd * (ref_vel[0] - meas_vel[0])
        uy = self.kp * ey + self.iy + self.kd * (ref_vel[1] - meas_vel[1])
        m = math.hypot(ux, uy)
        if m > self.out_lim:                                   # saturated: clamp output, freeze integrator
            ux, uy = ux * self.out_lim / m, uy * self.out_lim / m
        elif self.ki > 0.0:
            self.ix = clamp(self.ix + self.ki * ex * dt, -self.i_lim, self.i_lim)
            self.iy = clamp(self.iy + self.ki * ey * dt, -self.i_lim, self.i_lim)
        return ux, uy


# ============================================================ 1. THERMAL CAMERA
@dataclass
class ThermalCamera:
    px_h: int = 640            # pixels across-track
    px_v: int = 512            # pixels along-track
    pitch_m: float = 12e-6     # detector pixel pitch
    focal_m: float = 13.5e-3   # lens focal length

    @property
    def ifov(self):            # rad per pixel  (instantaneous field of view)
        return self.pitch_m / self.focal_m

    @property
    def hfov(self):
        return 2 * math.atan(self.px_h * self.pitch_m / (2 * self.focal_m))

    @property
    def vfov(self):
        return 2 * math.atan(self.px_v * self.pitch_m / (2 * self.focal_m))

    def half_extent(self, alt):
        """Nadir-looking camera: (half across-track, half along-track) ground footprint [m]."""
        return alt * math.tan(self.hfov / 2), alt * math.tan(self.vfov / 2)

    def gsd(self, slant_range):
        return slant_range * self.ifov     # metres of ground per pixel


def edge_range(cam, alt):
    """Slant range to the far corner of the footprint (worst case inside the image)."""
    return alt * math.sqrt(1 + math.tan(cam.hfov / 2) ** 2 + math.tan(cam.vfov / 2) ** 2)


# ============================================================ 2. JOHNSON CRITERIA
# N50 = number of resolvable cycles across the target needed for 50 % task success
N50 = {"detect": 1.0, "recognize": 4.0, "identify": 6.4}


def ttpf(n_cycles, n50):
    """Target Transfer Probability Function (Johnson / NVESD)."""
    if n_cycles <= 0:
        return 0.0
    x = n_cycles / n50
    e = 2.7 + 0.7 * x
    return x ** e / (1.0 + x ** e)


def cycles_on_target(cam, d_c, slant_range, d_temp=4.0, beta_km=0.5):
    """1 cycle = 2 pixels. Scaled by atmospheric transmittance and relative thermal contrast."""
    n_pix = d_c / (slant_range * cam.ifov)
    tau = math.exp(-beta_km * slant_range / 1000.0)
    return 0.5 * n_pix * tau * min(1.0, d_temp / 4.0)


def p_detect(cam, d_c, R, d_temp=4.0, beta_km=0.5):
    return ttpf(cycles_on_target(cam, d_c, R, d_temp, beta_km), N50["detect"])


def p_recognize(cam, d_c, R, d_temp=4.0, beta_km=0.5):
    return ttpf(cycles_on_target(cam, d_c, R, d_temp, beta_km), N50["recognize"])


def range_at_n50(cam, d_c, task):
    """Slant range where P(task)=50 %  (ideal air):  R = d_c / (2 * N50 * IFOV)."""
    return d_c / (2.0 * N50[task] * cam.ifov)


PERSON_DC = 0.42   # sqrt(top-down area 0.5 m x 0.35 m)  [m]


def design_altitude(cam, task, p_req, d_c=PERSON_DC, d_temp=4.0, beta_km=0.5, h_max=120):
    """Highest altitude where the worst image corner still meets P(task) >= p_req."""
    best = 5
    for h in range(5, h_max + 1):
        R = edge_range(cam, h)
        p = (p_detect if task == "detect" else p_recognize)(cam, d_c, R, d_temp, beta_km)
        if p >= p_req:
            best = h
    return best


# ============================================================ 4b. ENERGY MODEL
def hover_power_w(mass_kg=2.2, rotor_area_m2=0.2027, fm=0.65, eta=0.80, rho=1.225):
    """Momentum theory:  P = T^1.5 / sqrt(2 rho A) / (FM * eta)."""
    T = mass_kg * 9.81
    return T ** 1.5 / math.sqrt(2 * rho * rotor_area_m2) / (fm * eta)


# ============================================================ 5a. ZONE
@dataclass
class Zone:
    width: float = 120.0
    height: float = 80.0
    gate: tuple = (-60.0, 25.0)
    walkway: list = field(default_factory=lambda: [(-60, 25), (-25, 25), (-25, 4), (20, 4), (20, -2)])
    building: tuple = (20.0, -8.0, 18.0, 10.0)          # cx, cy, w, h  (the protected asset)
    restricted_margin: float = 8.0
    trees: list = field(default_factory=lambda: [(-35, -20, 7), (-10, -25, 6), (40, 22, 8), (45, -25, 6), (0, 20, 5)])
    home: tuple = (-55.0, -35.0)

    def inside(self, x, y, margin=0.0):
        return abs(x) <= self.width / 2 + margin and abs(y) <= self.height / 2 + margin

    def restricted_rect(self):
        cx, cy, w, h = self.building
        m = self.restricted_margin
        return cx - w / 2 - m, cy - h / 2 - m, cx + w / 2 + m, cy + h / 2 + m

    def in_restricted(self, x, y):
        x0, y0, x1, y1 = self.restricted_rect()
        return x0 <= x <= x1 and y0 <= y <= y1

    def in_canopy(self, x, y):
        return any(math.hypot(x - tx, y - ty) <= r for tx, ty, r in self.trees)

    def dist_to_asset(self, x, y):
        """Distance [m] from a point to the protected building (0 if inside)."""
        cx, cy, w, h = self.building
        return math.hypot(max(abs(x - cx) - w / 2, 0.0), max(abs(y - cy) - h / 2, 0.0))

    def dist_to_walkway(self, x, y):
        w = self.walkway
        return min(seg_dist(x, y, *w[i], *w[i + 1]) for i in range(len(w) - 1))

    def perimeter_point(self, rng):
        W, H = self.width, self.height
        u = rng.uniform(0, 2 * (W + H))
        if u < W:
            return -W / 2 + u, -H / 2, (0, 1)
        u -= W
        if u < H:
            return W / 2, -H / 2 + u, (-1, 0)
        u -= H
        if u < W:
            return W / 2 - u, H / 2, (0, -1)
        u -= W
        return -W / 2, H / 2 - u, (1, 0)

    def nearest_fence_point(self, x, y):
        W, H = self.width, self.height
        d = {"l": x + W / 2, "r": W / 2 - x, "t": y + H / 2, "b": H / 2 - y}
        k = min(d, key=d.get)
        return {"l": (-W / 2, y), "r": (W / 2, y), "t": (x, -H / 2), "b": (x, H / 2)}[k]


# ============================================================ CONFIG
@dataclass
class Config:
    zone: Zone = field(default_factory=Zone)
    cam: ThermalCamera = field(default_factory=ThermalCamera)
    # altitudes: None -> derived from the Johnson model (see __post_init__)
    h_search: float | None = None
    h_verify: float | None = None
    overlap: float = 0.90
    # flight
    v_cruise: float = 6.0
    v_follow_max: float = 7.0
    v_corner: float = 2.5
    a_max: float = 3.0
    climb_rate: float = 4.0
    yaw_rate_deg: float = 120.0
    wp_radius: float = 3.0
    # ---- flight controllers (PID) ----
    # velocity loop: accel_cmd = kp*(v_cmd - v) + I + kd*(0 - dv/dt)      [m/s^2]
    kp_vel: float = 2.0
    ki_vel: float = 0.8
    kd_vel: float = 0.05
    i_lim_vel: float = 1.5           # max integral contribution [m/s^2]
    # follow loop: v_cmd = v_target + kp*e + I + kd*(v_target - v_drone)   [m/s]
    kp_follow: float = 1.2
    ki_follow: float = 0.05
    kd_follow: float = 0.1
    i_lim_follow: float = 2.0        # max integral contribution [m/s]
    # wind: Ornstein-Uhlenbeck acceleration disturbance on the drone
    wind_sigma: float = 0.3          # [m/s^2], 0 = no wind
    wind_tau: float = 5.0            # gust correlation time [s]
    # energy
    battery_wh: float = 74.0
    k_speed_w: float = 0.04          # extra watts per (m/s)^3
    soc_rtl: float = 0.25
    swap_seconds: float = 8.0
    # sensing
    frame_rate: float = 8.0
    sigma_pos: float = 1.0           # target position measurement noise [m] (GPS + pixel)
    fa_density: float = 2e-5         # false detections per frame per m^2 of footprint
    canopy_factor: float = 0.15      # thermal cannot see through foliage
    atm_beta_km: float = 0.5         # clear-air LWIR extinction [1/km]
    # tracker
    gate_m: float = 4.0
    confirm_m: int = 3
    confirm_n: int = 5
    lost_after_s: float = 5.0
    dormant_ttl: float = 45.0        # how long a lost confirmed track is remembered [s]
    reid_margin: float = 0.5         # speed margin added to the track's own speed for the re-acquisition gate [m/s]
    reid_max_m: float = 20.0         # never re-acquire farther than this (zone is small: avoids merging different people)
    reid_ambiguity: float = 1.5      # refuse to revive if two candidates score within this factor
    settle_s: float = 6.0            # Kalman settling time before (re)classifying a revived track
    q_acc: float = 0.3
    t_class: float = 10.0
    min_rec: int = 8
    person_aspect_max: float = 2.15   # shape gate: blob elongation above this = animal-like, not a person
    # decision
    alert_thr: float = 0.5
    suspicious_thr: float = 0.3
    shadow_s: float = 15.0
    obs_period: float = 1.0          # drone sends one observation per confirmed track every N seconds
    engine_w_new: float = 0.5        # command-center smoothing weight per observation
    incident_radius: float = 40.0    # alerts closer than this ...
    incident_window: float = 150.0   # ... and sooner than this are merged into one incident [s]
    max_track_s: float = 35.0
    lost_follow_s: float = 4.0
    # world arrival rates [1/s]  (Poisson)
    rates: dict = field(default_factory=lambda: {"INTRUDER": 1 / 100, "STAFF": 1 / 120, "ANIMAL": 1 / 70})

    def __post_init__(self):
        if self.h_search is None:
            self.h_search = float(design_altitude(self.cam, "detect", 0.99))
        if self.h_verify is None:
            self.h_verify = float(max(15, design_altitude(self.cam, "recognize", 0.95)))
        self.p_hover_w = hover_power_w()

    def vel_pid(self):
        """Velocity-loop controller for the drone."""
        return PID2D(self.kp_vel, self.ki_vel, self.kd_vel, self.i_lim_vel, self.a_max)

    def follow_pid(self):
        """Position-follow controller used in TRACKING mode."""
        return PID2D(self.kp_follow, self.ki_follow, self.kd_follow, self.i_lim_follow, self.v_follow_max)


# ============================================================ 3. SEARCH THEORY / LANES
def plan_ring(cfg, alt=None):
    """Boustrophedon (lawn-mower) lanes closed into a loop. Lane spacing <= overlap * swath."""
    z, cam = cfg.zone, cfg.cam
    alt = alt or cfg.h_search
    ha, hl = cam.half_extent(alt)
    n = max(2, math.ceil(z.height / (cfg.overlap * 2 * ha)))
    n += n % 2                                   # even -> loop closes on the same side
    sp = z.height / n
    inset = min(0.8 * hl, z.width / 4)
    x0, x1 = -z.width / 2 + inset, z.width / 2 - inset
    pts = []
    for i in range(n):
        y = -z.height / 2 + (i + 0.5) * sp
        pts += [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]
    return pts, n, sp


def ring_length(pts):
    return sum(math.hypot(pts[i][0] - pts[(i + 1) % len(pts)][0], pts[i][1] - pts[(i + 1) % len(pts)][1])
               for i in range(len(pts)))


def koopman_random_search(cover_rate, area, t):
    """Classic random-search detection probability  P = 1 - exp(-Q t / A)."""
    return 1.0 - math.exp(-cover_rate * t / area)


# ============================================================ 4. DRONE
class Drone:
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.x, self.y = cfg.zone.home
        self.vx = self.vy = 0.0
        self.yaw = 0.0
        self.alt = 0.0
        self.soc = 1.0
        self.power_w = cfg.p_hover_w
        self.pid = cfg.vel_pid()                    # velocity-loop controller
        self.wx = self.wy = 0.0                     # wind gust acceleration (Ornstein-Uhlenbeck) [m/s^2]

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    def step(self, dt, v_cmd, alt_cmd, hold_yaw=False):
        c = self.cfg
        airborne = self.alt > 0.5
        # --- wind gust disturbance (only when flying)
        if airborne and c.wind_sigma > 0:
            k = math.sqrt(2 * dt / c.wind_tau) * c.wind_sigma
            self.wx += -self.wx / c.wind_tau * dt + k * self.rng.gauss(0, 1)
            self.wy += -self.wy / c.wind_tau * dt + k * self.rng.gauss(0, 1)
        else:
            self.wx = self.wy = 0.0
        if not airborne:
            self.pid.reset()                        # no integrator build-up on the ground
        # --- velocity PID -> commanded acceleration (already saturated at a_max)
        ax, ay = self.pid.step(v_cmd, (self.vx, self.vy), dt)
        self.vx += (ax + self.wx) * dt
        self.vy += (ay + self.wy) * dt
        sp = self.speed
        if sp > c.v_follow_max:
            self.vx, self.vy = self.vx * c.v_follow_max / sp, self.vy * c.v_follow_max / sp
        self.x += self.vx * dt
        self.y += self.vy * dt
        if not hold_yaw and sp > 0.8:
            diff = (math.degrees(math.atan2(self.vy, self.vx)) - self.yaw + 180) % 360 - 180
            m = c.yaw_rate_deg * dt
            self.yaw += clamp(diff, -m, m)
        dz = clamp(alt_cmd - self.alt, -c.climb_rate * dt, c.climb_rate * dt)
        self.alt = max(0.0, self.alt + dz)
        self.power_w = c.p_hover_w + c.k_speed_w * self.speed ** 3
        self.soc = max(0.0, self.soc - self.power_w * dt / 3600.0 / c.battery_wh)


# ============================================================ 5b. TARGET AGENTS
class Agent:
    """INTRUDER / STAFF (authorised) / ANIMAL.  Ground truth - the drone never reads `kind`."""

    def __init__(self, kind, ident, zone, rng, t0):
        self.kind, self.id, self.zone, self.rng, self.t0 = kind, ident, zone, rng, t0
        self.alive = True
        self.vx = self.vy = 0.0
        self.wx = self.wy = 0.0                     # Ornstein-Uhlenbeck meander
        z = zone
        if kind == "ANIMAL":
            self.x, self.y, n = z.perimeter_point(rng)
            self.heading = math.atan2(n[1], n[0]) + rng.uniform(-0.6, 0.6)
            self.speed = rng.uniform(0.3, 2.0)
            self.turn_timer = rng.uniform(2, 5)
            self.life = rng.uniform(40, 120)
            self.age = 0.0
            self.aspect = clamp(rng.gauss(2.7, 0.3), 1.8, 3.8)
            self.d_c = clamp(rng.gauss(0.42, 0.05), 0.30, 0.60)
            self.d_temp = 3.0
            self.phase = "ROAM"
            return
        self.aspect = clamp(rng.gauss(1.6, 0.15), 1.2, 2.1)
        self.d_c = clamp(rng.gauss(0.42, 0.04), 0.32, 0.55)
        self.d_temp = 4.0
        self.phase = "GO"
        self.stealth = False
        if kind == "STAFF":
            self.x, self.y = z.gate
            shortcut = rng.random() < 0.10                      # 10 % cut the corner
            self.route = [z.walkway[-1]] if shortcut else list(z.walkway[1:])
            self.exit_route = [z.gate] if shortcut else list(reversed(z.walkway[:-1]))
            self.speed, self.dwell, self.dwell_sigma = rng.uniform(1.2, 1.5), rng.uniform(20, 60), 0.25
        else:
            self.stealth = rng.random() < 0.20                  # 20 % tailgate through the gate
            if self.stealth:
                self.x, self.y = z.gate
                self.route = list(z.walkway[1:])
                self.exit_route = list(reversed(z.walkway[:-1]))
                self.speed, self.dwell = rng.uniform(0.6, 0.9), rng.uniform(40, 90)
            else:
                self.x, self.y, _ = z.perimeter_point(rng)
                bx, by, bw, bh = z.building
                goal = (bx + rng.uniform(-bw / 2 - 6, bw / 2 + 6), by + rng.uniform(-bh / 2 - 6, bh / 2 + 6))
                self.route = [goal]
                self.exit_route = [z.nearest_fence_point(*goal)]
                self.speed, self.dwell = rng.uniform(0.9, 1.6), rng.uniform(30, 90)
            self.dwell_sigma = 0.4
        self.dwell_left = self.dwell
        self.center = None
        self.reached_restricted_t = None

    def _ou(self, dt, sigma, tau):
        k = math.sqrt(2 * dt / tau) * sigma
        self.wx += -self.wx / tau * dt + k * self.rng.gauss(0, 1)
        self.wy += -self.wy / tau * dt + k * self.rng.gauss(0, 1)

    def step(self, dt):
        rng = self.rng
        if self.kind == "ANIMAL":
            self.life -= dt
            self.age += dt
            self.turn_timer -= dt
            if self.turn_timer <= 0:
                self.heading += rng.gauss(0, 1.0)
                self.speed = rng.uniform(0.3, 2.0)
                self.turn_timer = rng.uniform(2, 5)
            self.vx, self.vy = self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.life <= 0 or (self.age > 10 and not self.zone.inside(self.x, self.y, 2.0)):
                self.alive = False
            return
        if self.phase == "DWELL":
            self.dwell_left -= dt
            self._ou(dt, self.dwell_sigma, 3.0)
            self.vx = self.wx - 0.2 * (self.x - self.center[0])
            self.vy = self.wy - 0.2 * (self.y - self.center[1])
            if self.dwell_left <= 0:
                self.phase, self.route = "LEAVE", list(self.exit_route)
        else:
            if not self.route:
                self.alive = False
                return
            wx, wy = self.route[0]
            d = math.hypot(wx - self.x, wy - self.y)
            if d < 1.0:
                self.route.pop(0)
                if not self.route:
                    if self.phase == "GO":
                        self.phase, self.center = "DWELL", (self.x, self.y)
                    else:
                        self.alive = False
                return
            self._ou(dt, 0.15, 2.0)
            self.vx = self.speed * (wx - self.x) / d + self.wx
            self.vy = self.speed * (wy - self.y) / d + self.wy
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.kind == "INTRUDER" and self.reached_restricted_t is None and self.zone.in_restricted(self.x, self.y):
            self.reached_restricted_t = True


# ============================================================ SENSOR
class Sensor:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng

    def assess(self, drone, a):
        """Johnson readout for any agent (used by the GUI overlay)."""
        cfg, cam = self.cfg, self.cfg.cam
        dx, dy = a.x - drone.x, a.y - drone.y
        R = math.sqrt(max(drone.alt, 0.5) ** 2 + dx * dx + dy * dy)
        cyc = cycles_on_target(cam, a.d_c, R, a.d_temp, cfg.atm_beta_km)
        c, s = math.cos(math.radians(drone.yaw)), math.sin(math.radians(drone.yaw))
        ha, hl = cam.half_extent(drone.alt)
        inside = abs(-dx * s + dy * c) <= ha and abs(dx * c + dy * s) <= hl
        return {"range": R, "n_pix": a.d_c / (R * cam.ifov), "cycles": cyc, "inside": inside,
                "p_det": ttpf(cyc, N50["detect"]), "p_rec": ttpf(cyc, N50["recognize"])}

    def measure(self, a, R, occluded):
        cfg, cam, rng = self.cfg, self.cfg.cam, self.rng
        n_pix = a.d_c / (R * cam.ifov)
        cyc = cycles_on_target(cam, a.d_c, R, a.d_temp, cfg.atm_beta_km)
        occ = cfg.canopy_factor if occluded else 1.0
        pd = ttpf(cyc, N50["detect"]) * occ
        if rng.random() >= pd:
            return None
        prec = ttpf(cyc, N50["recognize"]) * occ
        rec = rng.random() < prec
        aspect = None
        if rec:   # shape is resolved -> noisy aspect-ratio measurement, worse when few pixels
            minor_px = n_pix / math.sqrt(a.aspect)
            aspect = max(1.0, rng.gauss(a.aspect, 3.0 / max(minor_px, 1.0)))
        sg = cfg.sigma_pos
        return {"x": a.x + rng.gauss(0, sg), "y": a.y + rng.gauss(0, sg), "agent_id": a.id,
                "aspect": aspect, "rec": rec, "p_det": pd, "p_rec": prec, "range": R, "cycles": cyc}

    def frame(self, drone, agents, zone):
        cfg, cam, rng = self.cfg, self.cfg.cam, self.rng
        if drone.alt < 3.0:
            return []
        ha, hl = cam.half_extent(drone.alt)
        c, s = math.cos(math.radians(drone.yaw)), math.sin(math.radians(drone.yaw))
        out = []
        for a in agents:
            dx, dy = a.x - drone.x, a.y - drone.y
            if abs(-dx * s + dy * c) > ha or abs(dx * c + dy * s) > hl:
                continue
            R = math.sqrt(drone.alt ** 2 + dx * dx + dy * dy)
            det = self.measure(a, R, zone.in_canopy(a.x, a.y))
            if det:
                out.append(det)
        for _ in range(poisson(rng, cfg.fa_density * 4 * ha * hl)):      # clutter / false alarms
            u, v = rng.uniform(-hl, hl), rng.uniform(-ha, ha)
            out.append({"x": drone.x + u * c - v * s, "y": drone.y + u * s + v * c, "agent_id": None,
                        "aspect": None, "rec": False, "p_det": 0.0, "p_rec": 0.0,
                        "range": drone.alt, "cycles": 0.0})
        return out


# ============================================================ 6. KALMAN TRACKER
class KF1D:
    """Constant-velocity Kalman filter for one axis.  state = [position, velocity]."""

    def __init__(self, z, sigma, v_sigma=3.0):
        self.p, self.v = z, 0.0
        self.P00, self.P01, self.P11 = sigma ** 2, 0.0, v_sigma ** 2

    def predict(self, dt, q):
        self.p += self.v * dt
        P00 = self.P00 + dt * (2 * self.P01 + dt * self.P11) + q * dt ** 3 / 3
        P01 = self.P01 + dt * self.P11 + q * dt ** 2 / 2
        P11 = self.P11 + q * dt
        self.P00, self.P01, self.P11 = P00, P01, P11

    def update(self, z, r):
        S = self.P00 + r
        k0, k1 = self.P00 / S, self.P01 / S
        y = z - self.p
        self.p += k0 * y
        self.v += k1 * y
        self.P11 = self.P11 - k1 * self.P01
        self.P00 = (1 - k0) * self.P00
        self.P01 = (1 - k0) * self.P01


class Track:
    def __init__(self, tid, det, t, cfg):
        self.cfg, self.id, self.t0, self.last_t, self.last_hit_t = cfg, tid, t, t, t
        self.kx, self.ky = KF1D(det["x"], cfg.sigma_pos), KF1D(det["y"], cfg.sigma_pos)
        self.flags = deque([True], maxlen=cfg.confirm_n)
        self.hist = deque(maxlen=int(8.5 * cfg.frame_rate))
        self.aspects, self.rec_count = [], 0
        self.votes = {}
        self.confirmed = self.announced = self.alerted = self.cleared = False
        self.status, self.p_threat = "TENTATIVE", None
        self.ignore_until = -1.0
        self.alert_t = None
        self.last_report = -1e9
        self.last_obs = -1e9
        self.incident_id = None
        self.gate_done = False
        self.fresh_since, self.revived, self.lost_t = t, 0, None
        self.last_det = det
        self._ingest(det, t, None)

    @property
    def pos(self):
        return self.kx.p, self.ky.p

    @property
    def vel(self):
        return self.kx.v, self.ky.v

    @property
    def speed(self):
        return math.hypot(*self.vel)

    def predict(self, t):
        dt = t - self.last_t
        if dt > 0:
            self.kx.predict(dt, self.cfg.q_acc)
            self.ky.predict(dt, self.cfg.q_acc)
            self.last_t = t

    def _ingest(self, det, t, zone):
        if det["agent_id"] is not None:
            self.votes[det["agent_id"]] = self.votes.get(det["agent_id"], 0) + 1
        if det["aspect"] is not None:
            self.aspects.append(det["aspect"])
            self.rec_count += 1
        self.last_det, self.last_hit_t = det, t
        x, y = self.pos
        zone = zone or self.cfg.zone
        self.hist.append((t, x, y, zone.dist_to_walkway(x, y), zone.dist_to_asset(x, y)))

    def update(self, det, t, zone):
        r = self.cfg.sigma_pos ** 2
        self.kx.update(det["x"], r)
        self.ky.update(det["y"], r)
        self.flags.append(True)
        self._ingest(det, t, zone)

    def miss(self):
        self.flags.append(False)

    @property
    def last_pos(self):
        return self.hist[-1][1], self.hist[-1][2]

    def revive(self, det, t, zone):
        """Re-acquire a remembered track: keep identity/status, restart the filter at the new detection."""
        c = self.cfg
        self.kx, self.ky = KF1D(det["x"], c.sigma_pos), KF1D(det["y"], c.sigma_pos)
        self.last_t, self.fresh_since, self.lost_t = t, t, None
        self.flags = deque([True] * c.confirm_m, maxlen=c.confirm_n)
        self.hist.clear()
        self.revived += 1
        confident_normal = (self.status == "NOT_THREAT" and self.p_threat is not None and self.p_threat < 0.15) \
            or self.status == "NOT_A_PERSON"
        if not self.alerted and not confident_normal:      # uncertain result -> give the drone a second look
            self.status, self.p_threat, self.cleared, self.ignore_until = "UNVERIFIED", None, False, -1.0
        self._ingest(det, t, zone)

    def truth_id(self):
        return max(self.votes, key=self.votes.get) if self.votes else None

    def mean_aspect(self):
        last = self.aspects[-30:]
        return sum(last) / len(last) if last else None

    def features(self, zone, window_s=8.0):
        """BEHAVIOUR ONLY - no temperature anywhere.
        speed      [m/s]  Kalman-filtered ground speed
        loiter     [0..1] 1 - (net displacement / path length) over the last 8 s: 0 = walking purposefully, 1 = hanging around
        proximity  [m]    mean distance to the protected asset (smaller = riskier)
        route_dev  [m]    mean distance from the authorised walkway (larger = riskier)"""
        h = list(self.hist)
        t_end = h[-1][0]
        seg = [e for e in h if t_end - e[0] <= window_s] or h
        path = sum(math.hypot(seg[i + 1][1] - seg[i][1], seg[i + 1][2] - seg[i][2]) for i in range(len(seg) - 1))
        net = math.hypot(seg[-1][1] - seg[0][1], seg[-1][2] - seg[0][2])
        eff = clamp(net / path, 0.0, 1.0) if path > 1e-6 else 0.0
        route_dev = sum(e[3] for e in seg) / len(seg)
        proximity = sum(e[4] for e in seg) / len(seg)
        return [self.speed, 1.0 - eff, proximity, route_dev]


class Tracker:
    def __init__(self, cfg, zone):
        self.cfg, self.zone = cfg, zone
        self.tracks, self.closed, self.dormant, self._n = [], [], [], 0

    def update(self, t, dets):
        cfg = self.cfg
        for tr in self.tracks:
            tr.predict(t)
        pairs = []
        for i, tr in enumerate(self.tracks):
            px, py = tr.pos
            for j, d in enumerate(dets):
                dist = math.hypot(d["x"] - px, d["y"] - py)
                if dist <= cfg.gate_m:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            self.tracks[i].update(dets[j], t, self.zone)
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.miss()
        for j, d in enumerate(dets):
            if j not in used_d:
                tr, was_dormant = self._revive(d, t)
                if tr is None:
                    self._n += 1
                    self.tracks.append(Track(f"TRK-{self._n:03d}", d, t, cfg))
                elif was_dormant:
                    self.tracks.append(tr)
        keep = []
        for tr in self.tracks:
            if not tr.confirmed and sum(tr.flags) >= cfg.confirm_m:
                tr.confirmed, tr.status = True, "UNVERIFIED"
            dead = (not tr.confirmed and len(tr.flags) == cfg.confirm_n and sum(tr.flags) < cfg.confirm_m) or \
                   (t - tr.last_hit_t > cfg.lost_after_s)
            if dead and tr.confirmed:
                tr.lost_t = t
                self.dormant.append(tr)          # remember it: the same person will come back into view
            elif dead:
                self.closed.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep
        self.dormant = [d for d in self.dormant if t - d.lost_t <= cfg.dormant_ttl]

    def _revive(self, det, t):
        """Kinematic + appearance re-identification of a lost / coasting confirmed track.
        Returns (track, was_dormant) or (None, False)."""
        cfg, cands = self.cfg, []
        coasting = [tr for tr in self.tracks if tr.confirmed and t - tr.last_hit_t > 0.5]
        for tr, dormant in [(x, True) for x in self.dormant] + [(x, False) for x in coasting]:
            gap = t - tr.last_hit_t
            gate = min(cfg.reid_max_m, cfg.gate_m + (tr.speed + cfg.reid_margin) * gap)
            d = math.hypot(det["x"] - tr.last_pos[0], det["y"] - tr.last_pos[1])
            if d > gate:
                continue
            if det["aspect"] is not None and len(tr.aspects) >= 5:           # shape must agree (person vs animal)
                if abs(det["aspect"] - sum(tr.aspects[-30:]) / len(tr.aspects[-30:])) > 0.9:
                    continue
            cands.append((d / gate, id(tr), tr, dormant))
        if not cands:
            return None, False
        cands.sort(key=lambda c: (c[0], c[1]))
        if len(cands) > 1 and cands[1][0] < cfg.reid_ambiguity * max(cands[0][0], 0.05):
            return None, False                                                # ambiguous -> start a new track (safe)
        _, _, tr, dormant = cands[0]
        if dormant:
            self.dormant.remove(tr)
        tr.revive(det, t, self.zone)
        return tr, dormant


# ============================================================ 7. THREAT CLASSIFIER
FEATURES = ["speed", "loiter", "proximity", "route_dev"]


class ThreatModel:
    """Logistic regression  P(threat | x) = sigmoid(w . z + b),  z = (x - mu)/sd."""

    def __init__(self, w, b, mu, sd, info=None):
        self.w, self.b, self.mu, self.sd, self.info = np.array(w), float(b), np.array(mu), np.array(sd), info or {}

    def predict(self, f):
        z = (np.array(f) - self.mu) / self.sd
        return sigmoid(float(self.w @ z + self.b))

    # ---- persistence
    @staticmethod
    def path():
        return DATA_DIR / "threat_model.json"

    def to_dict(self):
        return {"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "features": FEATURES, "info": self.info}

    def save(self):
        DATA_DIR.mkdir(exist_ok=True)
        json.dump(self.to_dict(), open(self.path(), "w"), indent=1)

    @classmethod
    def load_or_train(cls, cfg, force=False, verbose=True):
        p = cls.path()
        if p.exists() and not force:
            d = json.load(open(p))
            return cls(d["w"], d["b"], d["mu"], d["sd"], d.get("info"))
        if verbose:
            print("[MODEL] Training threat classifier on simulated tracks ...")
        return train_model(cfg, verbose=verbose)


def make_sample(cfg, kind, seed):
    """Run one agent through the SAME sensor + Kalman chain the drone uses; return its track features."""
    rng = random.Random(seed)
    zone = cfg.zone
    a = None
    for _ in range(6):
        a = Agent(kind, "x", zone, rng, 0.0)
        burn, t = rng.uniform(0, 60), 0.0
        while t < burn and a.alive:
            a.step(0.25)
            t += 0.25
        if a.alive:
            break
    else:
        return None
    sensor, trk = Sensor(cfg, rng), None
    dtf, tt = 1.0 / cfg.frame_rate, 0.0
    while tt < cfg.t_class + 4 and a.alive:
        a.step(dtf)
        tt += dtf
        alt = max(cfg.h_verify, cfg.h_search - cfg.climb_rate * tt)       # descend-to-verify profile
        R = math.sqrt(alt ** 2 + rng.uniform(0, 3) ** 2)
        det = sensor.measure(a, R, zone.in_canopy(a.x, a.y))
        if det:
            if trk is None:
                trk = Track("S", det, tt, cfg)
            else:
                trk.predict(tt)
                trk.update(det, tt, zone)
        elif trk is not None:
            trk.predict(tt)
            trk.miss()
    if trk is None or trk.rec_count < cfg.min_rec or tt - trk.t0 < cfg.t_class - 1:
        return None
    if trk.mean_aspect() >= cfg.person_aspect_max:          # rejected by the person-shape gate -> never reaches the classifier
        return None
    return trk.features(zone)


def generate_dataset(cfg, n_total=1500, seed=1):
    lam = cfg.rates
    tot = sum(lam.values())
    rows, k = [], 0
    for kind, rate in lam.items():                  # class priors = arrival-rate proportions
        for _ in range(round(n_total * rate / tot)):
            k += 1
            f = make_sample(cfg, kind, seed * 100003 + k)
            if f:
                rows.append(f + [1.0 if kind == "INTRUDER" else 0.0, kind])
    random.Random(seed).shuffle(rows)
    return rows


def auc_score(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_model(cfg, verbose=True):
    rows = generate_dataset(cfg)
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "threat_dataset.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(FEATURES + ["label", "kind"])
        wr.writerows(rows)
    X = np.array([r[:4] for r in rows], float)
    y = np.array([r[4] for r in rows], float)
    n_tr = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w, b = np.zeros(4), 0.0
    for _ in range(2000):                                       # batch gradient descent, cross-entropy + L2
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.3 * (Z.T @ (p - ytr) / len(Z) + 1e-3 * w)
        b -= 0.3 * float((p - ytr).mean())
    zte = (Xte - mu) / sd
    pte = 1 / (1 + np.exp(-(zte @ w + b)))
    pred = pte >= cfg.alert_thr
    tp, fp = int((pred & (yte == 1)).sum()), int((pred & (yte == 0)).sum())
    fn, tn = int((~pred & (yte == 1)).sum()), int((~pred & (yte == 0)).sum())
    info = {"n_train": int(n_tr), "n_test": int(len(Xte)), "accuracy": (tp + tn) / max(1, len(yte)),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "auc": auc_score(yte, pte), "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "weights": dict(zip(FEATURES, [round(float(v), 3) for v in w]))}
    m = ThreatModel(w, b, mu, sd, info)
    m.save()
    if verbose:
        print(f"[MODEL] held-out test: acc={info['accuracy']:.3f} precision={info['precision']:.3f} "
              f"recall={info['recall']:.3f} AUC={info['auc']:.3f}  {info['confusion']}")
    return m


# ============================================================ 8. WORLD + MISSION
def approach_speed(dist, v_cruise, a_max, v_end, r_acc):
    return min(v_cruise, math.sqrt(v_end ** 2 + 2 * 0.7 * a_max * max(dist - r_acc, 0.0)))


PREFIX = {"INTRUDER": "HOS", "STAFF": "BEN", "ANIMAL": "ANI"}
# SIMULATOR ground truth, expressed as BEHAVIOUR (what the classifier is judged against) - not as identity
TRUTH_LABEL = {"INTRUDER": "HOSTILE", "STAFF": "BENIGN", "ANIMAL": "NON-PERSON"}


class LocalClassifier:
    """Prediction running ON the drone. Used only for offline Monte-Carlo or as an explicit fallback.
    The real deployment uses a remote classifier: observations go to the command center, verdicts come back."""

    def __init__(self, model, cfg):
        self.engine = ThreatEngine(model.to_dict(), alert_thr=cfg.alert_thr, suspicious_thr=cfg.suspicious_thr,
                                   w_new=cfg.engine_w_new, incident_radius=cfg.incident_radius,
                                   incident_window=cfg.incident_window, clear_age=cfg.t_class + 2)
        self.pending = {}

    def submit(self, obs):
        self.pending[obs["track_id"]] = self.engine.assess(obs)

    def poll(self, track_id):
        return self.pending.pop(track_id, None)


class World:
    def __init__(self, cfg=None, seed=None, model=None, verbose=False, classifier=None, drone_id="DRONE-01"):
        self.cfg = cfg or Config()
        self.rng = random.Random(seed)
        self.zone = self.cfg.zone
        self.drone_id = drone_id
        if classifier is None:                       # offline / Monte-Carlo: prediction engine runs in-process
            model = model or ThreatModel.load_or_train(self.cfg, verbose=verbose)
            classifier = LocalClassifier(model, self.cfg)
        self.model, self.classifier = model, classifier
        self.sensor = Sensor(self.cfg, self.rng)
        self.tracker = Tracker(self.cfg, self.zone)
        self.drone = Drone(self.cfg, self.rng)
        self.follow_pid = self.cfg.follow_pid()      # PID used while following a track
        self.ring, self.n_lanes, self.lane_sp = plan_ring(self.cfg)
        self.t = 0.0
        self.cam_acc, self.cam_dt = 0.0, 1.0 / self.cfg.frame_rate
        self.agents, self.log, self.counter = [], {}, {k: 0 for k in PREFIX}
        self.mode, self.wp_i, self.follow, self.track_t0, self.swap_t = "PATROL", 0, None, 0.0, 0.0
        self.v_cmd, self.alt_cmd = (0.0, 0.0), 0.0
        self.sweeps = self.swaps = 0
        self.events, self.last_dets = [], []
        self.false_alerts = 0
        self.merged_alerts = 0
        self.gate = {"person_ok": 0, "person_rejected": 0, "animal_filtered": 0, "animal_passed": 0}
        self.cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        self.patrol_time = 0.0
        self.follow_err_sum, self.follow_err_n = 0.0, 0     # follow-quality metric (mean distance drone <-> aim point)

    # ---------------- agents
    def add_agent(self, kind):
        self.counter[kind] += 1
        ident = f"{PREFIX[kind]}-{self.counter[kind]:02d}"
        a = Agent(kind, ident, self.zone, self.rng, self.t)
        self.agents.append(a)
        self.log[ident] = {"kind": kind, "t0": self.t, "t_first_det": None, "t_alert": None,
                           "t_restricted": None, "t_exit": None}
        return a

    def _spawn(self, dt):
        for kind, lam in self.cfg.rates.items():
            if self.rng.random() < lam * dt:
                self.add_agent(kind)

    # ---------------- step
    def step(self, dt):
        self.t += dt
        self._spawn(dt)
        for a in self.agents:
            a.step(dt)
            rec = self.log[a.id]
            if a.kind == "INTRUDER" and a.reached_restricted_t is True and rec["t_restricted"] is None:
                rec["t_restricted"] = self.t
            if not a.alive:
                rec["t_exit"] = self.t
        self.agents = [a for a in self.agents if a.alive]
        self._mission(dt)
        self.drone.step(dt, self.v_cmd, self.alt_cmd, hold_yaw=(self.mode == "TRACKING"))
        if self.mode == "PATROL":
            self.patrol_time += dt
        self.cam_acc += dt
        while self.cam_acc >= self.cam_dt:
            self.cam_acc -= self.cam_dt
            self._camera_frame()

    def _camera_frame(self):
        dets = self.sensor.frame(self.drone, self.agents, self.zone)
        self.last_dets = dets
        for d in dets:
            if d["agent_id"] and self.log[d["agent_id"]]["t_first_det"] is None:
                self.log[d["agent_id"]]["t_first_det"] = self.t
        self.tracker.update(self.t, dets)
        self._decide()

    def truth_of(self, tr):
        tid = tr.truth_id()
        return self.log[tid]["kind"] if tid else "CLUTTER"

    def truth_label(self, tr):
        return TRUTH_LABEL.get(self.truth_of(tr), "CLUTTER")

    def _event(self, typ, tr):
        self.events.append({"type": typ, "track": tr, "t": self.t, "truth": self.truth_of(tr),
                            "drone": (self.drone.x, self.drone.y, self.drone.alt, self.drone.speed, self.drone.yaw),
                            "mode": self.mode})

    def _make_obs(self, tr, feats):
        """What the drone tells the command center about one tracked person. No decision, no label."""
        d, (px, py), (vx, vy) = self.drone, tr.pos, tr.vel
        return {
            "drone_id": self.drone_id, "sim_time": round(self.t, 2), "track_id": tr.id,
            "features": None if feats is None else {k: round(float(v), 4) for k, v in zip(FEATURES, feats)},
            "perception": {"person_like": True, "shape_frames": tr.rec_count},   # thermal = detection + position only
            "track_age_s": round(self.t - tr.t0, 1), "revived": tr.revived,
            "position": {"x": round(px, 2), "y": round(py, 2)}, "velocity": {"vx": round(vx, 2), "vy": round(vy, 2)},
            "detection_probability": round(float(tr.last_det["p_det"]), 4), "range_m": round(float(tr.last_det["range"]), 1),
            "drone": {"x": round(d.x, 2), "y": round(d.y, 2), "altitude": round(d.alt, 1), "speed": round(d.speed, 2),
                      "heading": round(d.yaw % 360, 1), "mode": self.mode},
            "sim_ground_truth": self.truth_label(tr),       # EVALUATION ONLY (HOSTILE/BENIGN) - never used for the prediction
        }

    def _apply_verdict(self, tr, v):
        """The command center decided; the drone only reacts (keep following / shadow / resume patrol)."""
        if v.get("p_threat") is None:
            return
        tr.p_threat, tr.status, tr.incident_id = v["p_threat"], v["status"], v.get("incident_id")
        truth, a = self.truth_of(tr), v.get("alert", "NONE")
        if a in ("NEW", "MERGED") and not tr.alerted:
            tr.alerted, tr.alert_t = True, self.t
            if truth == "INTRUDER":
                rec = self.log[tr.truth_id()]
                if rec["t_alert"] is None:
                    rec["t_alert"] = self.t
            if a == "NEW":
                self.cm["TP" if truth == "INTRUDER" else "FP"] += 1
                if truth != "INTRUDER":
                    self.false_alerts += 1
                self._event("ALERT", tr)
            else:
                self.merged_alerts += 1
                self._event("MERGED", tr)
        elif a == "CLEARED" and not tr.cleared and not tr.alerted:
            tr.cleared = True
            self.cm["FN" if truth == "INTRUDER" else "TN"] += 1
            self._event("CLEARED", tr)

    def _decide(self):
        cfg = self.cfg
        for tr in self.tracker.tracks:
            if not tr.confirmed:
                continue
            if not tr.announced:
                tr.announced = True
                self._event("CONFIRMED", tr)
            ready = (self.t - tr.t0 >= cfg.t_class and tr.rec_count >= cfg.min_rec
                     and self.t - tr.fresh_since >= cfg.settle_s)
            if ready and not tr.gate_done and tr.status == "UNVERIFIED":          # PERSON GATE (shape, not temperature)
                tr.gate_done = True
                person = tr.mean_aspect() < cfg.person_aspect_max
                kind = self.truth_of(tr)
                self.gate["person_ok" if kind != "ANIMAL" and person else
                          "person_rejected" if kind != "ANIMAL" else
                          "animal_passed" if person else "animal_filtered"] += 1
                if not person:
                    tr.status = "NOT_A_PERSON"
            if tr.status == "NOT_A_PERSON":
                continue                                                          # nothing is sent about animals / clutter
            if self.t - tr.last_obs >= cfg.obs_period:
                tr.last_obs = self.t
                self.classifier.submit(self._make_obs(tr, tr.features(self.zone) if ready else None))
            v = self.classifier.poll(tr.id)
            if v is not None:
                self._apply_verdict(tr, v)

    # ---------------- mission state machine
    def _go(self, tx, ty, v_end=0.0, v=None):
        d, c = self.drone, self.cfg
        dist = math.hypot(tx - d.x, ty - d.y)
        sp = approach_speed(dist, v or c.v_cruise, c.a_max, v_end, c.wp_radius)
        self.v_cmd = (sp * (tx - d.x) / dist, sp * (ty - d.y) / dist) if dist > 1e-6 else (0.0, 0.0)
        return dist

    def _release(self, tr, hold=30.0):
        if tr is not None:
            tr.ignore_until = self.t + hold
        self.follow, self.mode = None, "PATROL"

    def _mission(self, dt):
        cfg, d = self.cfg, self.drone
        if self.mode in ("PATROL", "TRACKING") and d.soc <= cfg.soc_rtl:
            self.mode, self.follow = "RTL", None
        if self.mode == "PATROL":
            self.alt_cmd = cfg.h_search
            wx, wy = self.ring[self.wp_i]
            if self._go(wx, wy, cfg.v_corner) < cfg.wp_radius:
                self.wp_i = (self.wp_i + 1) % len(self.ring)
                self.sweeps += self.wp_i == 0
            cand = [tr for tr in self.tracker.tracks
                    if tr.confirmed and tr.status == "UNVERIFIED" and self.t >= tr.ignore_until]
            if cand and d.alt > 5:
                self.follow = min(cand, key=lambda tr: math.hypot(tr.pos[0] - d.x, tr.pos[1] - d.y))
                self.mode, self.track_t0 = "TRACKING", self.t
                self.follow_pid.reset()                                   # fresh integrator for each new follow
        elif self.mode == "TRACKING":
            tr = self.follow
            lost = tr is None or tr not in self.tracker.tracks or self.t - tr.last_hit_t > cfg.lost_follow_s
            if lost or self.t - self.track_t0 > cfg.max_track_s:
                self._release(tr)
                return
            if tr.status == "NOT_A_PERSON":
                self._release(tr, 120.0)
                return
            if (tr.status in ("NOT_THREAT", "SUSPICIOUS") and self.t - tr.t0 >= cfg.t_class + 2) or \
               (tr.alerted and self.t - tr.alert_t >= cfg.shadow_s):
                self._release(tr, 60.0)
                return
            self.alt_cmd = cfg.h_verify
            # ---- PID follow: aim at the predicted target position, feed-forward the target velocity
            vx, vy = tr.vel
            aim = (tr.pos[0] + 0.8 * vx, tr.pos[1] + 0.8 * vy)
            corr = self.follow_pid.step(aim, (d.x, d.y), dt, meas_vel=(d.vx, d.vy), ref_vel=(vx, vy))
            cx, cy = vx + corr[0], vy + corr[1]
            m = math.hypot(cx, cy)
            if m > cfg.v_follow_max:
                cx, cy = cx * cfg.v_follow_max / m, cy * cfg.v_follow_max / m
            self.v_cmd = (cx, cy)
            self.follow_err_sum += math.hypot(aim[0] - d.x, aim[1] - d.y)
            self.follow_err_n += 1
        elif self.mode == "RTL":
            hx, hy = self.zone.home
            dist = self._go(hx, hy, 0.0)
            self.alt_cmd = cfg.h_search if dist > 6 else 0.0
            if dist < 2.0 and d.alt < 1.0:
                self.mode, self.swap_t, self.v_cmd = "SWAP", cfg.swap_seconds, (0.0, 0.0)
                d.pid.reset()
        elif self.mode == "SWAP":
            self.v_cmd, self.alt_cmd = (0.0, 0.0), 0.0
            d.pid.reset()
            self.swap_t -= dt
            if self.swap_t <= 0:
                d.soc, self.mode, self.swaps = 1.0, "PATROL", self.swaps + 1

    # ---------------- metrics
    def metrics(self):
        L = list(self.log.values())
        intr = [r for r in L if r["kind"] == "INTRUDER" and (r["t_exit"] is not None or r["t_alert"] is not None)]
        al = [r for r in intr if r["t_alert"] is not None]
        det = [r for r in intr if r["t_first_det"] is not None]
        before = [r for r in al if r["t_restricted"] is None or r["t_alert"] <= r["t_restricted"]]
        hours = max(self.t, 1.0) / 3600.0
        return {
            "intruders": len(intr),
            "alert_rate": len(al) / len(intr) if intr else float("nan"),
            "seen_rate": len(det) / len(intr) if intr else float("nan"),
            "alert_before_asset": len(before) / len(intr) if intr else float("nan"),
            "mean_alert_delay_s": float(np.mean([r["t_alert"] - r["t0"] for r in al])) if al else float("nan"),
            "false_alerts_per_h": self.false_alerts / hours,
            "sweeps": self.sweeps, "battery_swaps": self.swaps,
            "patrol_fraction": self.patrol_time / max(self.t, 1.0),
            "follow_err_m": self.follow_err_sum / self.follow_err_n if self.follow_err_n else float("nan"),
            "cm": dict(self.cm), "gate": dict(self.gate),
        }


# ============================================================ 9. REPORTS / CLI
def sensor_report(cfg):
    cam, z = cfg.cam, cfg.zone
    print("=" * 74)
    print("THERMAL CAMERA  %dx%d px, %.0f um pitch, %.1f mm lens" % (cam.px_h, cam.px_v, cam.pitch_m * 1e6, cam.focal_m * 1e3))
    print("  IFOV = pitch/focal = %.3f mrad   HFOV = %.1f deg   VFOV = %.1f deg"
          % (cam.ifov * 1e3, math.degrees(cam.hfov), math.degrees(cam.vfov)))
    print("\nJOHNSON RANGES for a person (critical dimension %.2f m), P = 50 %%:" % PERSON_DC)
    for task in ("detect", "recognize", "identify"):
        print("  %-10s N50=%.1f cycles  ->  R50 = %6.0f m" % (task, N50[task], range_at_n50(cam, PERSON_DC, task)))
    print("\nWHY 340 m 'CLASSIFICATION' WAS NOT POSSIBLE (the old simulation):")
    R = 340.0
    print("  at 340 m: person = %.1f pixels wide, P(detect)=%.2f, P(recognise)=%.4f"
          % (PERSON_DC / (R * cam.ifov), p_detect(cam, PERSON_DC, R), p_recognize(cam, PERSON_DC, R)))
    print("\nALTITUDE TRADE-OFF (nadir camera, worst image corner, clear air):")
    print("  alt  swath  footprint     GSD    P(det)  P(rec)  lanes  loop[s]  Koopman P(detect in 60 s)")
    for h in (20, 30, 40, 60, 80, 100, 120):
        ha, hl = cam.half_extent(h)
        Re = edge_range(cam, h)
        pts, n, sp = plan_ring(cfg, h)
        T = ring_length(pts) / cfg.v_cruise
        Q = 2 * ha * cfg.v_cruise
        print("  %3d  %5.1f  %5.1fx%-5.1f  %5.1fcm   %.3f   %.3f   %3d   %6.0f   %.2f"
              % (h, 2 * ha, 2 * ha, 2 * hl, cam.gsd(Re) * 100, p_detect(cam, PERSON_DC, Re),
                 p_recognize(cam, PERSON_DC, Re), n, T, koopman_random_search(Q, z.width * z.height, 60)))
    print("\nDESIGN: search altitude h_search = %.0f m (P_detect >= 0.99 at image corner)" % cfg.h_search)
    print("        verify altitude h_verify = %.0f m (P_recognise >= 0.95 at image corner)" % cfg.h_verify)
    pts, n, sp = plan_ring(cfg)
    print("        %d lanes, spacing %.1f m, loop %.0f m = %.0f s at %.1f m/s"
          % (n, sp, ring_length(pts), ring_length(pts) / cfg.v_cruise, cfg.v_cruise))
    e_wh = cfg.battery_wh * (1 - cfg.soc_rtl)
    p = cfg.p_hover_w + cfg.k_speed_w * cfg.v_cruise ** 3
    print("ENERGY: hover %.0f W, cruise %.0f W -> %.1f min usable per battery (to %d %% reserve)"
          % (cfg.p_hover_w, p, e_wh / p * 60, cfg.soc_rtl * 100))
    print("CONTROL: PID  velocity loop kp=%.2f ki=%.2f kd=%.2f | follow loop kp=%.2f ki=%.2f kd=%.2f | wind sigma=%.2f m/s^2"
          % (cfg.kp_vel, cfg.ki_vel, cfg.kd_vel, cfg.kp_follow, cfg.ki_follow, cfg.kd_follow, cfg.wind_sigma))
    print("=" * 74)


def run_episode(cfg, seed, minutes, model, dt=0.05):
    w = World(cfg, seed, model)
    for _ in range(int(minutes * 60 / dt)):
        w.step(dt)
    return w.metrics()


def evaluate(n_runs, minutes, alts):
    base = Config()
    model = ThreatModel.load_or_train(base)
    rows = []
    print("\nMONTE-CARLO (PID): %d runs x %.0f min per setting (Poisson arrivals, seeds 0..%d)"
          % (n_runs, minutes, n_runs - 1))
    print(" h_search  intruders  alerted  seen   before-asset  mean delay[s]  false alerts/h  patrol%  follow err[m]")
    for h in alts:
        cfg = Config(h_search=float(h))
        ms = [run_episode(cfg, s, minutes, model) for s in range(n_runs)]
        agg = {k: float(np.nanmean([m[k] for m in ms])) for k in
               ("alert_rate", "seen_rate", "alert_before_asset", "mean_alert_delay_s", "false_alerts_per_h",
                "patrol_fraction", "follow_err_m")}
        agg["intruders"] = sum(m["intruders"] for m in ms)
        rows.append({"h_search": h, **agg})
        print("   %4.0f m   %6d     %.2f    %.2f      %.2f         %6.1f          %5.1f        %.2f      %5.2f"
              % (h, agg["intruders"], agg["alert_rate"], agg["seen_rate"], agg["alert_before_asset"],
                 agg["mean_alert_delay_s"], agg["false_alerts_per_h"], agg["patrol_fraction"], agg["follow_err_m"]))
    Path("results").mkdir(exist_ok=True)
    out = "results/monte_carlo.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Night-patrol drone: model report / training / Monte-Carlo")
    ap.add_argument("--train", action="store_true", help="retrain classifier and print held-out metrics")
    ap.add_argument("--eval", type=int, metavar="N", help="Monte-Carlo with N runs per altitude")
    ap.add_argument("--minutes", type=float, default=20.0)
    a = ap.parse_args()
    cfg = Config()
    sensor_report(cfg)
    if a.train:
        train_model(cfg)
    else:
        m = ThreatModel.load_or_train(cfg)
        print("classifier:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.info.items() if k != "weights"})
        print("weights   :", m.info.get("weights"))
    if a.eval:
        evaluate(a.eval, a.minutes, [30, 60, cfg.h_search, 110])