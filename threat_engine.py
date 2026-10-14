"""
threat_engine.py - THE PREDICTION BRAIN (runs at the COMMAND CENTER, standard library only)

The drone never decides "threat or not". It only sends OBSERVATIONS about a tracked PERSON. The features are
BEHAVIOUR ONLY - temperature is deliberately not used (everybody has the same body heat, and a thermal camera
tens of metres away cannot give a reliable temperature reading anyway):
  speed      how fast the person moves                 loiter     hanging around vs walking purposefully (0..1)
  proximity  distance to the protected asset [m]       route_dev  distance from the authorised walkway [m]
Observation example:

    {"track_id": "TRK-012", "sim_time": 83.4,
     "features": {"speed": 1.21, "loiter": 0.12, "proximity": 9.4, "route_dev": 18.4},
     "position": {"x": 12.3, "y": -4.1}, "track_age_s": 14.0, ...}

This engine turns an observation into a VERDICT:

    logistic regression      p_raw = sigmoid( w . ((x - mu)/sd) + b )
    temporal smoothing       p     = (1 - w_new) * p_old + w_new * p_raw       (per track)
    decision                 p >= 0.5 THREAT | p >= 0.3 SUSPICIOUS | else NOT_THREAT
    incident merging         a new THREAT within 40 m / 150 s of an open incident updates that
                             incident instead of raising a second alert for the same person
"""
import json
import math
import time
from pathlib import Path

FEATURES = ["speed", "loiter", "proximity", "route_dev"]


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class ThreatEngine:
    def __init__(self, params, alert_thr=0.5, suspicious_thr=0.3, w_new=0.5,
                 incident_radius=40.0, incident_window=150.0, clear_age=12.0):
        self.w = [float(v) for v in params["w"]]
        self.b = float(params["b"])
        self.mu = [float(v) for v in params["mu"]]
        self.sd = [float(v) for v in params["sd"]]
        self.features = params.get("features", FEATURES)
        self.info = params.get("info", {})
        self.alert_thr, self.suspicious_thr, self.w_new = alert_thr, suspicious_thr, w_new
        self.incident_radius, self.incident_window, self.clear_age = incident_radius, incident_window, clear_age
        self.reset()

    @classmethod
    def from_file(cls, path, **kw):
        return cls(json.load(open(path)), **kw)

    def reset(self):
        self.tracks, self.incidents, self.n_incidents = {}, [], 0

    # ------------------------------------------------------------ model
    def contributions(self, feats):
        """Per-feature push on the log-odds of THREAT (+ = towards threat, - = away). Makes every verdict explainable."""
        return {k: round(w * ((feats[k] - m) / s), 2) for w, k, m, s in zip(self.w, self.features, self.mu, self.sd)}

    def score(self, feats):
        return _sigmoid(sum(self.contributions(feats).values()) + self.b)

    # ------------------------------------------------------------ main entry point
    def assess(self, obs):
        tid = obs["track_id"]
        now = obs.get("sim_time", time.time())
        st = self.tracks.setdefault(tid, {"p": None, "alerted": False, "cleared": False, "incident": None})
        feats = obs.get("features")
        if not feats:                                            # hot-spot seen, shape/behaviour not yet resolved
            return self._verdict(tid, None, None, "UNVERIFIED", "NONE", st, None)
        p_raw = self.score(feats)
        st["p"] = p_raw if st["p"] is None else (1 - self.w_new) * st["p"] + self.w_new * p_raw
        p = st["p"]
        status = "THREAT" if p >= self.alert_thr else "SUSPICIOUS" if p >= self.suspicious_thr else "NOT_THREAT"
        alert = "NONE"
        pos = obs.get("position", {"x": 0.0, "y": 0.0})
        if status == "THREAT":
            inc = self._find_incident(now, pos)
            if not st["alerted"]:
                st["alerted"] = True
                if inc is not None:
                    alert = "MERGED"
                else:
                    self.n_incidents += 1
                    inc = {"id": f"INC-{self.n_incidents:03d}", "t": now, "x": pos["x"], "y": pos["y"], "first_t": now}
                    self.incidents.append(inc)
                    alert = "NEW"
                st["incident"] = inc["id"]
            else:
                alert = "UPDATE"
            if inc is not None:
                inc.update(t=now, x=pos["x"], y=pos["y"])
        elif not st["alerted"] and not st["cleared"] and obs.get("track_age_s", 0.0) >= self.clear_age:
            st["cleared"] = True
            alert = "CLEARED"
        return self._verdict(tid, p, p_raw, status, alert, st, self.contributions(feats))

    def _find_incident(self, now, pos):
        for inc in self.incidents:
            if now - inc["t"] <= self.incident_window and \
               math.hypot(pos["x"] - inc["x"], pos["y"] - inc["y"]) <= self.incident_radius:
                return inc
        return None

    @staticmethod
    def _verdict(tid, p, p_raw, status, alert, st, why):
        level = "LOW" if p is None else "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.5 else "LOW"
        return {"track_id": tid, "status": status, "alert": alert, "incident_id": st["incident"],
                "p_threat": None if p is None else round(p, 4), "p_raw": None if p_raw is None else round(p_raw, 4),
                "alert_level": level, "why": why, "model": "logistic-regression-v1"}