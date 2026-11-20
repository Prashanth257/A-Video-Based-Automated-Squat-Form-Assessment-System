"""
CODE 2 (squat): Annotate a recorded video with
  - the CTR-GCN/Transformer/LSTM Good/Bad verdict (re-evaluated every second), PLUS
  - joint angles (hip / knee / torso lean), rep counting, squat-specific error checks,
    colour-coded skeleton, live knee-angle graph, per-frame + per-rep CSVs, ROM / symmetry /
    smoothness plots and a text report.

UPDATED for ENHANCED Code 1:
  * HybridNet now MATCHES the enhanced Code 1 architecture exactly
    (8 attention heads, 256 FFN, GELU, 2-layer frame_proj, 2-layer LSTM, 2-layer head)
  * Loads smoothing settings (SMOOTH_ALPHA, VIS_THRESH) from checkpoint
  * Uses IDENTICAL window normalization as Code 1 (per-window, not per-video)
  * Handles data leakage fix: frame gap split or video-level split
  * Tolerates old checkpoints (with warning)
  * Better error handling and progress reporting

  * USES atan2 WITH CROSS PRODUCT for joint angle calculation
    instead of arccos (mathematically equivalent, alternative approach)

  * ENHANCED: Degenerate case handling, SEQ_LEN consistency check,
    improved progress output, comprehensive error handling
"""

import os, warnings, traceback
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION  <-  EDIT THESE
# =====================================================================
INPUT_DIR       = r"E:\25PHD1029\videosREHAB26_Zenodo\Ex6"                 # folder containing your videos
OUTPUT_ROOT     = r"E:\25PHD1029\videosREHAB26_Zenodo\Ex6\New folder"          # one sub-folder per video is created here
VIDEO_EXTS      = (".mp4", ".avi", ".mov", ".mkv")
CHECKPOINT_PATH = r"D:\OUTPUT\hybrid_model_checkpoint.pth"   # ABSOLUTE PATH from Code 1

SEQ_LEN        = 45          # MUST match Code 1
TARGET_FPS     = 30          # MUST match Code 1
EVAL_EVERY_SEC = 1.0         # re-run the classifier every N seconds
ANALYSIS_ALPHA = 0.6         # smoothing for the angle/rep analytics (every frame)

# --- squat rules (same values as Code 1's sanity report) ---------------
KNEE_DEPTH_MAX    = 90      # deepest knee angle in a rep must be <= this
TORSO_LEAN_MAX    = 40       # torso vs vertical (deg); above = leaning too far forward
KNEE_SPAN_MIN     = 0.35      # knee distance / hip distance; below = knees caving
VALGUS_KNEE_ANGLE = 140      # judge knee caving / asymmetry only when knee is bent below this
ASYM_MAX          = 10       # left/right knee angle difference (deg)

# --- rep counting -------------------------------------------------------
REP_DOWN_DEFAULT = 120       # knee angle below this = "Down"
REP_UP_DEFAULT   = 160       # knee angle above this = "Up"
MIN_REP_INTERVAL = 0.6       # seconds between counted reps
ADAPT_WINDOW_S   = 10.0      # thresholds adapt to the last N seconds of motion
ADAPT_MIN_AMP    = 40        # ...but only if the knee moved at least this many degrees
ANGLE_AVG_N      = 5         # moving average (frames) applied to hip/knee angles
SIDE_LOCK_FRAMES = 30        # frames used to decide left / right / both visibility
GRAPH_LEN        = 150       # frames shown in the live graph

MEDIAPIPE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16

# =====================================================================
# 2. MODEL  (must be identical to ENHANCED Code 1)
# =====================================================================
class CTR_GC(nn.Module):
    def __init__(self, in_c, out_c, num_joints):
        super().__init__()
        self.refine = nn.Conv2d(in_c, out_c, 1)
        self.pa = nn.Parameter(torch.randn(out_c, num_joints, num_joints) * 0.01)
        self.alpha = nn.Parameter(torch.ones(1))
        self.register_buffer("adj", torch.eye(num_joints).unsqueeze(0))
    def forward(self, x):
        x = self.refine(x)
        A = self.alpha * self.adj + self.pa
        return torch.einsum('bctv,cvw->bctw', x, A)

class CTRGCN_Block(nn.Module):
    def __init__(self, in_c, out_c, num_joints, dropout=0.5):
        super().__init__()
        self.gcn = CTR_GC(in_c, out_c, num_joints)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (1, 1), (4, 0)),
            nn.BatchNorm2d(out_c), nn.Dropout(dropout))
        self.res = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        return F.relu(self.tcn(self.gcn(x)) + self.res(x))

class HybridNet(nn.Module):
    def __init__(self, in_channels=12, num_joints=17, num_classes=2,
                 seq_len=45, d_model=64):
        super().__init__()
        self.gcn1 = CTRGCN_Block(in_channels, 32, num_joints, dropout=0.3)
        self.gcn2 = CTRGCN_Block(32, d_model, num_joints, dropout=0.3)

        self.joint_emb = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))
        self.time_emb  = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        nn.init.trunc_normal_(self.joint_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb,  std=0.02)

        self.tf_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256,
            dropout=0.2, batch_first=True, norm_first=True, activation='gelu')
        self.tf_norm = nn.LayerNorm(d_model)

        self.frame_proj = nn.Sequential(
            nn.Linear(num_joints * d_model, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))

        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=False)

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.gcn2(self.gcn1(x))
        B, d, T, V = x.shape
        x = x.permute(0, 2, 3, 1) + self.joint_emb + self.time_emb[:, :T]
        x = self.tf_layer(x.reshape(B * T, V, d)).reshape(B, T, V, d)
        x = self.tf_norm(x)
        x = self.frame_proj(x.reshape(B, T, V * d))
        out, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], out.mean(dim=1)], dim=1))

# =====================================================================
# 3. SHARED HELPERS  (identical to Code 1)
# =====================================================================
class LandmarkSmoother:
    """EMA smoothing; a landmark whose visibility is low keeps its previous position."""
    def __init__(self, alpha=0.6, vis_thresh=0.5):
        self.alpha, self.vis_thresh = alpha, vis_thresh
        self.prev = None
    def update(self, pts, vis=None):
        pts = np.asarray(pts, dtype=np.float32)
        if self.prev is None:
            self.prev = pts.copy()
            return self.prev.copy()
        new = self.alpha * pts + (1.0 - self.alpha) * self.prev
        if vis is not None:
            low = np.asarray(vis) < self.vis_thresh
            new[low] = self.prev[low]
        self.prev = new
        return new.copy()

def angle_3pts_atan2(a, b, c):
    """
    Angle at b (degrees) formed by a-b-c using atan2 with cross product.

    Formula: theta = atan2(||u x v||, u . v)
    where u = a - b, v = c - b

    Works in 2D or 3D, returns angle in range [0, 180] degrees.
    Handles degenerate cases (collinear points).
    """
    ba = np.asarray(a, float) - b
    bc = np.asarray(c, float) - b

    cross = np.cross(ba, bc)
    cross_mag = np.linalg.norm(cross)
    dot = np.dot(ba, bc)

    angle_rad = np.arctan2(cross_mag, dot)
    angle_deg = float(np.degrees(angle_rad))
    angle_deg = np.clip(angle_deg, 0.0, 180.0)

    # Degenerate case: collinear points with negative dot -> ~180 deg
    if cross_mag < 1e-6 and dot < 0:
        angle_deg = 180.0

    return angle_deg

def squat_metrics(p):
    """
    p: (17,3) world coordinates -> squat metrics.
    Uses atan2 with cross product for hip/knee angles, arccos for torso lean.
    """
    mid_sh  = (p[L_SHOULDER] + p[R_SHOULDER]) / 2
    mid_hip = (p[L_HIP] + p[R_HIP]) / 2
    torso = mid_sh - mid_hip

    lean = float(np.degrees(np.arccos(np.clip(abs(torso[1]) / (np.linalg.norm(torso) + 1e-8), 0, 1))))

    hip_w  = np.linalg.norm(p[L_HIP] - p[R_HIP])
    knee_w = np.linalg.norm(p[L_KNEE] - p[R_KNEE])

    return {
        "hip_l":  angle_3pts_atan2(p[L_SHOULDER], p[L_HIP], p[L_KNEE]),
        "hip_r":  angle_3pts_atan2(p[R_SHOULDER], p[R_HIP], p[R_KNEE]),
        "knee_l": angle_3pts_atan2(p[L_HIP], p[L_KNEE], p[L_ANKLE]),
        "knee_r": angle_3pts_atan2(p[R_HIP], p[R_KNEE], p[R_ANKLE]),
        "lean": lean,
        "span_ratio": float(knee_w / (hip_w + 1e-8)),
    }

def normalize_sequence(coords):
    """Hip-centre + body-scale + per-joint standardisation, applied PER WINDOW.
    MUST match Code 1 exactly for reliable predictions."""
    mid_hip = (coords[:, L_HIP] + coords[:, R_HIP]) / 2
    coords = coords - mid_hip[:, None, :]
    shoulder_mid = (coords[:, L_SHOULDER] + coords[:, R_SHOULDER]) / 2
    body_scale = np.linalg.norm(shoulder_mid, axis=-1).mean()
    if body_scale > 1e-6:
        coords = coords / body_scale
    mean = coords.mean(axis=0, keepdims=True)
    std  = coords.std(axis=0, keepdims=True) + 1e-8
    return (coords - mean) / std

def window_features(raw_window):
    """(SEQ_LEN,17,3) raw smoothed coords -> (12, SEQ_LEN, 17) model input.
    MUST match Code 1 exactly."""
    w = normalize_sequence(raw_window)
    vel  = np.diff(w, axis=0, append=w[-1:])
    acc  = np.diff(vel, axis=0, append=vel[-1:])
    jerk = np.diff(acc, axis=0, append=acc[-1:])
    return np.concatenate([w, vel, acc, jerk], axis=-1).transpose(2, 0, 1)

# =====================================================================
# 4. REP COUNTER + SQUAT ANALYZER
# =====================================================================
def rep_smoothness(times, angles):
    """Sum of changes in angular speed per second (lower = smoother)."""
    t, a = np.asarray(times, float), np.asarray(angles, float)
    if len(t) < 3:
        return 0.0
    dts = np.diff(t)
    dts[dts <= 0] = 1e-6
    vel = np.abs(np.diff(a) / dts)
    return float(np.sum(np.abs(np.diff(vel))) / max(t[-1] - t[0], 1e-6))

class RepCounter:
    """Up/Down hysteresis on the smoothed knee angle with thresholds that adapt to recent motion."""
    def __init__(self):
        self.state = None          # "Up" / "Down"
        self.reps = 0
        self.last_rep_t = -1e9
        self.rep_open = False      # True after an Up->Down transition
        self.hist = deque()
        self.down_th, self.up_th = REP_DOWN_DEFAULT, REP_UP_DEFAULT

    def _thresholds(self):
        if len(self.hist) >= 10:
            vals = np.array([a for _, a in self.hist])
            lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
            if hi - lo >= ADAPT_MIN_AMP:
                return lo + 0.35 * (hi - lo), lo + 0.80 * (hi - lo)
        return REP_DOWN_DEFAULT, REP_UP_DEFAULT

    def update(self, angle, t):
        """Returns None, 'rep_start', 'rep_counted' or 'rep_ignored'."""
        self.hist.append((t, angle))
        while self.hist and t - self.hist[0][0] > ADAPT_WINDOW_S:
            self.hist.popleft()
        self.down_th, self.up_th = self._thresholds()

        if self.state is None:
            if angle >= self.up_th:
                self.state = "Up"
            elif angle <= self.down_th:
                self.state = "Down"
            return None
        if self.state == "Up" and angle <= self.down_th:
            self.state, self.rep_open = "Down", True
            return "rep_start"
        if self.state == "Down" and angle >= self.up_th:
            self.state = "Up"
            if not self.rep_open:
                return None
            self.rep_open = False
            if t - self.last_rep_t >= MIN_REP_INTERVAL:
                self.reps += 1
                self.last_rep_t = t
                return "rep_counted"
            return "rep_ignored"
        return None

class SquatAnalyzer:
    def __init__(self):
        self.counter = RepCounter()
        self.rows, self.rep_rows = [], []
        self.knee_buf = deque(maxlen=ANGLE_AVG_N)
        self.hip_buf  = deque(maxlen=ANGLE_AVG_N)
        self.prev = None                       # (t, knee_s, hip_s)
        self.vis_sum, self.vis_n = np.zeros(2), 0
        self.side, self.side_locked = "both", False
        self.cur = None                        # rep currently in progress
        self.angle_hist = deque(maxlen=GRAPH_LEN)
        self.time_hist  = deque(maxlen=GRAPH_LEN)
        self.rep_marks = []                    # (time, min_knee, rep_no)
        self.msgs, self.rep_msg, self.rep_msg_until = [], "", -1.0
        self.status = {"back": None, "legs": None}
        self.metrics, self.knee_s, self.hip_s = None, np.nan, np.nan

    def _update_side(self, vis):
        """Decide once whether one leg is clearly better visible (side-on filming) or both are."""
        if self.side_locked:
            return
        self.vis_sum += np.array([vis[[L_HIP, L_KNEE, L_ANKLE]].mean(),
                                  vis[[R_HIP, R_KNEE, R_ANKLE]].mean()])
        self.vis_n += 1
        if self.vis_n >= SIDE_LOCK_FRAMES:
            l, r = self.vis_sum / self.vis_n
            self.side = "left" if l - r > 0.2 else ("right" if r - l > 0.2 else "both")
            self.side_locked = True

    def update(self, pts, vis, t, frame_no, pbad=None):
        m = squat_metrics(pts)
        self._update_side(vis)
        if self.side == "left":
            hip, knee = m["hip_l"], m["knee_l"]
        elif self.side == "right":
            hip, knee = m["hip_r"], m["knee_r"]
        else:
            hip, knee = (m["hip_l"] + m["hip_r"]) / 2, (m["knee_l"] + m["knee_r"]) / 2
        self.knee_buf.append(knee); self.hip_buf.append(hip)
        knee_s, hip_s = float(np.mean(self.knee_buf)), float(np.mean(self.hip_buf))
        self.metrics, self.knee_s, self.hip_s = m, knee_s, hip_s

        # angular velocity
        kv = hv = np.nan
        if self.prev is not None:
            dt = t - self.prev[0]
            if dt > 0:
                kv = abs(knee_s - self.prev[1]) / dt
                hv = abs(hip_s - self.prev[2]) / dt
        self.prev = (t, knee_s, hip_s)

        # live graph data
        self.angle_hist.append(knee_s); self.time_hist.append(t)

        # rep counting + per-rep tracking
        event = self.counter.update(knee_s, t)
        if event == "rep_start":
            self.cur = {"t0": t, "min_knee": knee_s, "t_bottom": t, "max_lean": m["lean"],
                        "min_ratio": np.inf, "pbad": [], "times": [], "angles": []}
        if self.cur is not None:
            c = self.cur
            if knee_s < c["min_knee"]:
                c["min_knee"], c["t_bottom"] = knee_s, t
            c["max_lean"] = max(c["max_lean"], m["lean"])
            if knee_s < VALGUS_KNEE_ANGLE and self.side == "both":
                c["min_ratio"] = min(c["min_ratio"], m["span_ratio"])
            if pbad is not None:
                c["pbad"].append(pbad)
            c["times"].append(t); c["angles"].append(knee_s)
        if event == "rep_counted":
            self._finish_rep(t)
        elif event == "rep_ignored":
            self.cur = None

        # live form checks
        bent = knee_s < VALGUS_KNEE_ANGLE
        lean_bad = m["lean"] > TORSO_LEAN_MAX
        caving = bool(self.side == "both" and bent and m["span_ratio"] < KNEE_SPAN_MIN)
        asym = bool(self.side == "both" and bent and abs(m["knee_l"] - m["knee_r"]) > ASYM_MAX)
        self.msgs = []
        if lean_bad: self.msgs.append("Leaning too far forward - chest up.")
        if caving:   self.msgs.append("Knees caving in - push knees out.")
        if asym:     self.msgs.append("Uneven legs - balance your weight.")
        self.status = {"back": not lean_bad, "legs": not (caving or asym)}

        self.rows.append({
            "Frame": frame_no, "Timestamp (s)": round(t, 3),
            "Reps": self.counter.reps, "State": self.counter.state or "-", "Side": self.side,
            "Hip Angle": round(hip_s, 2), "Knee Angle": round(knee_s, 2),
            "Left Hip Angle": round(m["hip_l"], 2), "Right Hip Angle": round(m["hip_r"], 2),
            "Left Knee Angle": round(m["knee_l"], 2), "Right Knee Angle": round(m["knee_r"], 2),
            "Torso Lean Angle": round(m["lean"], 2), "Knee Span Ratio": round(m["span_ratio"], 3),
            "Hip Velocity (deg/s)": round(hv, 2) if not np.isnan(hv) else np.nan,
            "Knee Velocity (deg/s)": round(kv, 2) if not np.isnan(kv) else np.nan,
            "Model_P_Bad": pbad if pbad is not None else np.nan,
            "Err_Torso_Lean": int(lean_bad), "Err_Knee_Caving": int(caving), "Err_Asymmetry": int(asym),
        })

    def _finish_rep(self, t):
        c, self.cur = self.cur, None
        n = self.counter.reps
        depth_ok  = c["min_knee"] <= KNEE_DEPTH_MAX
        lean_ok   = c["max_lean"] <= TORSO_LEAN_MAX
        valgus_ok = not (np.isfinite(c["min_ratio"]) and c["min_ratio"] < KNEE_SPAN_MIN)
        pbad = float(np.mean(c["pbad"])) if c["pbad"] else np.nan
        issues = []
        if not depth_ok:  issues.append("too shallow")
        if not lean_ok:   issues.append("too much forward lean")
        if not valgus_ok: issues.append("knees caved")
        self.rep_rows.append({
            "Rep": n, "Start (s)": round(c["t0"], 2), "End (s)": round(t, 2),
            "Duration (s)": round(t - c["t0"], 2),
            "Min Knee Angle": round(c["min_knee"], 1), "Max Torso Lean": round(c["max_lean"], 1),
            "Min Knee Span Ratio": round(c["min_ratio"], 3) if np.isfinite(c["min_ratio"]) else np.nan,
            "Depth OK": depth_ok, "Lean OK": lean_ok, "Knees OK": valgus_ok,
            "Smoothness": round(rep_smoothness(c["times"], c["angles"]), 2),
            "Model P(bad)": round(pbad, 3) if not np.isnan(pbad) else np.nan,
            "Issues": ", ".join(issues) if issues else "none",
        })
        self.rep_marks.append((c["t_bottom"], c["min_knee"], n))
        self.rep_msg = f"Rep {n}: " + (", ".join(issues) if issues else "good")
        self.rep_msg_until = t + 2.5

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def rep_dataframe(self):
        return pd.DataFrame(self.rep_rows)

# =====================================================================
# 5. DRAWING
# =====================================================================
def draw_stick_overlay(frame, px, status):
    """Skeleton from pixel landmarks; torso green/orange by lean, legs by knee form."""
    h, w = frame.shape[:2]
    sc = max(0.6, h / 720.0); th = max(2, int(3 * sc))
    P = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in px]
    GREEN, ORANGE, GRAY, BLUE = (0, 220, 0), (0, 165, 255), (200, 200, 200), (255, 200, 200)
    col = lambda ok: BLUE if ok is None else (GREEN if ok else ORANGE)
    back_c, legs_c = col(status.get("back")), col(status.get("legs"))
    def limb(a, b, c): cv2.line(frame, P[a], P[b], c, th, cv2.LINE_AA)
    limb(11, 12, GRAY); limb(23, 24, GRAY)
    for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]: limb(a, b, GRAY)
    limb(11, 23, back_c); limb(12, 24, back_c)
    for a, b in [(23, 25), (25, 27), (24, 26), (26, 28)]: limb(a, b, legs_c)
    for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(frame, P[i], max(3, int(4 * sc)), (0, 0, 255), -1)

def draw_live_graph(frame, an):
    """Knee-angle trace with the 'good depth' zone and a marker at the bottom of each rep."""
    if len(an.angle_hist) < 2:
        return
    H, W = frame.shape[:2]
    sc = max(0.6, H / 720.0)
    gw, gh, mg = int(400 * sc), int(150 * sc), int(16 * sc)
    x0, y0 = W - gw - mg, H - gh - mg - int(10 * sc)
    ang = np.array(an.angle_hist); tim = np.array(an.time_hist)
    vmin, vmax = ang.min() - 5, ang.max() + 5
    if vmax - vmin < 1e-3: vmin, vmax = vmin - 1, vmax + 1
    tmin, tmax = tim[0], tim[-1]
    trange = tmax - tmin if tmax - tmin > 0 else 1.0
    ypos = lambda v: int(y0 + gh - mg - (v - vmin) / (vmax - vmin) * (gh - 2 * mg))
    xpos = lambda t: int(x0 + mg + (t - tmin) / trange * (gw - 2 * mg))
    fs = 0.4 * sc

    cv2.rectangle(frame, (x0, y0), (x0 + gw, y0 + gh), (25, 25, 25), -1)
    lo, hi = max(vmin, 0), min(vmax, KNEE_DEPTH_MAX)
    if hi > lo:                                            # good-depth zone
        cv2.rectangle(frame, (x0 + 1, ypos(hi)), (x0 + gw - 1, ypos(lo)), (0, 100, 0), -1)
        cv2.putText(frame, "Depth zone", (x0 + 6, max(y0 + int(30 * sc), ypos(hi) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + gw, y0 + gh), (150, 150, 150), 2)
    cv2.putText(frame, "Knee angle", (x0 + 6, y0 + int(18 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, (255, 255, 255), 1)
    for i in range(5):                                     # grid + labels
        v = vmin + (vmax - vmin) * i / 4
        yg = ypos(v)
        cv2.line(frame, (x0 + mg, yg), (x0 + gw - mg, yg), (60, 60, 60), 1)
        cv2.putText(frame, f"{v:.0f}", (x0 + gw - int(40 * sc), yg - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1)
    pts = [(xpos(t), ypos(a)) for t, a in zip(tim, ang)]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), max(1, int(2 * sc)))
    for rt, ra, rn in an.rep_marks:                        # rep markers
        if tmin <= rt <= tmax:
            cx, cy = xpos(rt), ypos(ra)
            cv2.circle(frame, (cx, cy), max(3, int(5 * sc)), (0, 0, 255), -1)
            cv2.putText(frame, str(rn), (cx + int(8 * sc), cy - int(8 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (0, 0, 255), 2)

def draw_hud(frame, an, verdict, pose_ok, t):
    H, W = frame.shape[:2]
    sc = max(0.6, H / 720.0)
    lines = []                                             # (text, colour, font scale)
    if verdict["cls"] is None:
        lines.append(("Warming up...", (200, 200, 200), 1.0))
    else:
        good = verdict["cls"] == 0
        lines.append((("GOOD FORM" if good else "BAD FORM") + f"  {verdict['conf']:.0%}",
                      (0, 200, 0) if good else (0, 0, 220), 1.1))
    lines.append((f"Reps: {an.counter.reps}   State: {an.counter.state or '-'}", (255, 255, 255), 0.7))
    if pose_ok and an.metrics is not None:
        lines.append((f"Knee {an.knee_s:5.1f} deg  Hip {an.hip_s:5.1f} deg  Lean {an.metrics['lean']:4.1f} deg",
                      (255, 255, 0), 0.6))
    elif not pose_ok:
        lines.append(("No pose detected", (0, 0, 255), 0.7))
    for msg in an.msgs[:3]:
        lines.append((msg, (0, 165, 255), 0.55))
    if an.rep_msg and t < an.rep_msg_until:
        lines.append((an.rep_msg, (255, 200, 0), 0.65))

    heights = [int(32 * fs * sc) + int(8 * sc) for _, _, fs in lines]
    x1, y1 = int(10 * sc), int(10 * sc)
    x2 = min(W - 10, x1 + int(560 * sc))
    y2 = min(H - 10, y1 + sum(heights) + int(10 * sc))
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)
    y = y1 + int(6 * sc)
    for (text, color, fs), hgt in zip(lines, heights):
        y += hgt
        cv2.putText(frame, text, (x1 + int(10 * sc), y - int(8 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * sc, color, max(1, int(2 * sc)), cv2.LINE_AA)

# =====================================================================
# 6. REPORTING
# =====================================================================
def rom_stats(df):
    rows = []
    for c in [c for c in df.columns if c.endswith("Angle")]:
        rows.append({"Angle": c, "Min": df[c].min(), "Mean": df[c].mean(), "Max": df[c].max(),
                     "Std": df[c].std(), "ROM": df[c].max() - df[c].min()})
    return pd.DataFrame(rows)

def plot_rom_bar(rom_df, path):
    if rom_df.empty: return
    plt.figure(figsize=(8, 4))
    plt.bar(rom_df["Angle"], rom_df["ROM"])
    plt.ylabel("ROM (deg)"); plt.title("Range of Motion (Max - Min) per Angle")
    plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_smoothness(rep_df, path):
    if rep_df.empty: return
    plt.figure(figsize=(8, 4))
    plt.plot(rep_df["Rep"], rep_df["Smoothness"], marker="o")
    plt.xlabel("Rep"); plt.ylabel("Smoothness (lower is smoother)"); plt.title("Rep Smoothness")
    plt.grid(True); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_hip_knee(df, path):
    if df.empty: return
    plt.figure(figsize=(10, 4))
    for col, lab in [("Hip Angle", "Hip"), ("Knee Angle", "Knee"), ("Torso Lean Angle", "Torso lean")]:
        if col in df.columns:
            plt.plot(df["Timestamp (s)"], df[col], label=lab)
    plt.axhline(KNEE_DEPTH_MAX, color="g", ls=":", lw=1, label=f"Depth target ({KNEE_DEPTH_MAX} deg)")
    plt.axhline(TORSO_LEAN_MAX, color="r", ls=":", lw=1, label=f"Lean limit ({TORSO_LEAN_MAX} deg)")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)"); plt.title("Hip, Knee and Torso Lean over Time")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_symmetry(df, path):
    if df.empty: return
    plt.figure(figsize=(10, 4))
    t = df["Timestamp (s)"]
    plt.plot(t, df["Left Knee Angle"], label="Left knee")
    plt.plot(t, df["Right Knee Angle"], label="Right knee")
    plt.plot(t, (df["Left Knee Angle"] - df["Right Knee Angle"]).abs(), "--", label="Abs diff")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)"); plt.title("Left vs Right Knee Symmetry")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(path); plt.close()

def write_report(path, an, df, rep_df, verdict_log, fps, ckpt_info):
    n_rep = len(rep_df)
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("SQUAT ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Video: {os.path.basename(INPUT_VIDEO)}\n")
        f.write(f"Frames analysed: {len(df)} ({len(df) / fps:.1f} s at {fps:.1f} fps)\n")
        f.write(f"Camera side used for angles: {an.side}\n")
        f.write(f"Reps counted: {an.counter.reps}\n")
        f.write(f"\nModel Info:\n")
        f.write(f"  - Accuracy: {ckpt_info.get('accuracy', 'N/A')}\n")
        f.write(f"  - F1 Score: {ckpt_info.get('f1_score', 'N/A')}\n")
        f.write(f"  - Best Epoch: {ckpt_info.get('best_epoch', 'N/A')}\n")
        f.write(f"  - Split Method: {ckpt_info.get('split_method', 'Unknown')}\n")
        f.write(f"  - Smoothing: alpha={ckpt_info.get('smooth_alpha', 'N/A')}, "
                f"vis_thresh={ckpt_info.get('vis_thresh', 'N/A')}\n\n")

        f.write("Angle Calculation Method:\n")
        f.write("  - Joint angles (hip_l, hip_r, knee_l, knee_r): atan2 with cross product\n")
        f.write("    Formula: theta = atan2(||u x v||, u . v)\n")
        f.write("    Range: [0, 180] deg\n")
        f.write("    Note: Mathematically equivalent to arccos; alternative approach chosen for consistency with Code 1.\n")
        f.write("  - Torso lean: arccos (for numerical stability)\n\n")

        if verdict_log:
            bad = np.mean([c == 1 for _, c, _ in verdict_log])
            f.write(f"Model verdict: {bad:.0%} of {len(verdict_log)} windows flagged BAD FORM\n\n")
        if n_rep:
            f.write("Per-rep results:\n")
            for _, r in rep_df.iterrows():
                f.write(f"- Rep {int(r['Rep'])}: min knee {r['Min Knee Angle']:.1f} deg, "
                        f"max lean {r['Max Torso Lean']:.1f} deg, issues: {r['Issues']}\n")
            f.write("\n")
        shallow = int((~rep_df["Depth OK"]).sum()) if n_rep else 0
        lean = int((~rep_df["Lean OK"]).sum()) if n_rep else 0
        knees = int((~rep_df["Knees OK"]).sum()) if n_rep else 0
        f.write("Issue summary (reps affected / frames flagged):\n")
        f.write(f"- Insufficient depth: {shallow}/{n_rep} reps\n")
        f.write(f"- Excessive forward lean: {lean}/{n_rep} reps, {int(df['Err_Torso_Lean'].sum())} frames\n")
        f.write(f"- Knees caving: {knees}/{n_rep} reps, {int(df['Err_Knee_Caving'].sum())} frames\n")
        f.write(f"- Left/right asymmetry: {int(df['Err_Asymmetry'].sum())} frames\n\n")
        f.write("Suggested drills:\n")
        if shallow: f.write("- Depth: paused box squats, ankle and hip mobility, tempo work.\n")
        if lean:    f.write("- Forward lean: goblet / front squats, upper-back strength, cue 'chest up'.\n")
        if knees:   f.write("- Knee caving: banded squats, monster walks, cue 'knees out over toes'.\n")
        if int(df["Err_Asymmetry"].sum()):
            f.write("- Asymmetry: single-leg work, check stance width and weight distribution.\n")
        if not (shallow or lean or knees):
            f.write("- No rule-based issues found; keep it up.\n")
        f.write("\nNote: knee-caving and asymmetry are only judged when both legs are clearly visible "
                "(front/back/oblique view).\n")
        f.write("\n" + "="*70 + "\n")

# =====================================================================
# 7. PROCESS VIDEO  (MAIN LOOP)
# =====================================================================
print("\n" + "="*80)
print("CODE 2: SQUAT ANALYSIS WITH ENHANCED CODE 1 MODEL")
print("Angle Calculation: atan2 with cross product (alternative to arccos)")
print("="*80 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Load checkpoint ---
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"\nERROR: Checkpoint not found at:\n   {CHECKPOINT_PATH}\n")
    print("SOLUTIONS:")
    print("   1. Run Code 1 FIRST to train the model and save checkpoint")
    print("   2. Verify the path exists: D:\\OUTPUT\\hybrid_model_checkpoint.pth")
    print("   3. Update CHECKPOINT_PATH in this code if your path is different\n")
    exit(1)

try:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"[OK] Checkpoint loaded successfully")
except Exception as e:
    print(f"\nERROR: Failed to load checkpoint:")
    print(f"   {e}\n")
    print("SOLUTIONS:")
    print("   - Checkpoint file may be corrupted")
    print("   - Re-run Code 1 to train and save a new checkpoint\n")
    exit(1)

# Verify SEQ_LEN consistency
ckpt_seq_len = ckpt.get("seq_len", SEQ_LEN)
if ckpt_seq_len != SEQ_LEN:
    print(f"\nWARNING: SEQ_LEN mismatch!")
    print(f"   Checkpoint was trained with SEQ_LEN={ckpt_seq_len}")
    print(f"   Code 2 is using SEQ_LEN={SEQ_LEN}")
    print(f"   Using checkpoint's value...\n")
    SEQ_LEN = ckpt_seq_len

# Load smoothing settings from checkpoint
MODEL_ALPHA = ckpt.get("smooth_alpha", 0.6)
MODEL_VIS_THRESH = ckpt.get("vis_thresh", 0.5)
ckpt_info = {
    "accuracy": ckpt.get("accuracy", "N/A"),
    "f1_score": ckpt.get("f1_score", "N/A"),
    "best_epoch": ckpt.get("best_epoch", "N/A"),
    "split_method": ckpt.get("split_method", "Unknown"),
    "smooth_alpha": MODEL_ALPHA,
    "vis_thresh": MODEL_VIS_THRESH
}

# Warn if old checkpoint
if "smooth_alpha" not in ckpt:
    print("\nWARNING: This checkpoint was trained with OLD Code 1")
    print("   (missing smoothing settings). Predictions may be less reliable.")
    print("   Please retrain with the new Code 1.\n")
else:
    print(f"\nCheckpoint info loaded:")
    print(f"   Smoothing: alpha={MODEL_ALPHA}, vis_thresh={MODEL_VIS_THRESH}")
    print(f"   Seq_len: {ckpt_seq_len}")
    print(f"   Split method: {ckpt_info['split_method']}")
    print(f"   Model accuracy: {ckpt_info['accuracy']}")
    print(f"   Angle calculation: atan2 with cross product (hip/knee)")
    print(f"                      arccos (torso lean)\n")

# Load model state
try:
    model = HybridNet(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("[OK] Model loaded and ready.\n")
except Exception as e:
    print(f"\nERROR: Failed to load model state:")
    print(f"   {e}\n")
    print("SOLUTIONS:")
    print("   - Make sure HybridNet here is identical to HybridNet in Code 1")
    print("   - Ensure SEQ_LEN = 45 matches Code 1 training")
    print("   - Re-run Code 1 to create a compatible checkpoint\n")
    exit(1)

# --- Find all videos in the folder ---
videos = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VIDEO_EXTS))
print(f"Found {len(videos)} video(s) in {INPUT_DIR}")
summary_rows = []     # one row per video (videos named "transposed" are left out)

for vid_name in videos:
    INPUT_VIDEO  = os.path.join(INPUT_DIR, vid_name)
    OUTPUT_DIR   = os.path.join(OUTPUT_ROOT, os.path.splitext(vid_name)[0])
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "#"*80 + f"\nVIDEO: {vid_name}\n" + "#"*80)

    # --- Open video ---
    print(f"Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"\nERROR: Cannot open video:\n   {INPUT_VIDEO}\n")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
        print(f"  FPS not detected, using default: {fps}")
    else:
        print(f"  FPS: {fps:.1f}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Resolution: {W}x{H}, Total frames: {total_frames}\n")

    # --- Setup video writer ---
    print(f"Setting up output video: {OUTPUT_VIDEO}")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out_vid.isOpened():
        print(f"\nERROR: Cannot write to {OUTPUT_VIDEO}")
        print("SOLUTION: Ensure OUTPUT_DIR exists and is writable\n")
        continue
    print("[OK] Video writer ready.\n")

    # --- Setup pose estimation ---
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  min_detection_confidence=0.5)

    # CRITICAL: Use SAME smoother settings as Code 1 training
    model_smoother    = LandmarkSmoother(MODEL_ALPHA, MODEL_VIS_THRESH)
    analysis_smoother = LandmarkSmoother(ANALYSIS_ALPHA, MODEL_VIS_THRESH)
    pixel_smoother    = LandmarkSmoother(ANALYSIS_ALPHA, MODEL_VIS_THRESH)

    window_buf  = deque(maxlen=SEQ_LEN)
    step        = fps / TARGET_FPS if fps > TARGET_FPS * 1.1 else 1.0
    next_keep   = 0.0
    eval_interval = max(1, int(round(fps * EVAL_EVERY_SEC)))
    last_eval   = -10 ** 9

    analyzer    = SquatAnalyzer()
    verdict     = {"cls": None, "conf": 0.0, "pbad": None}
    verdict_log = []
    frame_idx   = 0

    print("="*80)
    print("PROCESSING VIDEO")
    print("="*80 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps
            keep = frame_idx + 1e-9 >= next_keep
            if keep:
                next_keep += step

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_ok = bool(res.pose_world_landmarks and res.pose_landmarks)

            if pose_ok:
                wl = res.pose_world_landmarks.landmark
                raw = np.array([[wl[i].x, wl[i].y, wl[i].z] for i in MEDIAPIPE_TO_COCO17], dtype=np.float32)
                vis = np.array([wl[i].visibility for i in MEDIAPIPE_TO_COCO17], dtype=np.float32)

                # --- classifier (uses MODEL_ALPHA smoothing) --
                if keep:
                    window_buf.append(model_smoother.update(raw, vis))
                if len(window_buf) == SEQ_LEN and frame_idx - last_eval >= eval_interval:
                    try:
                        x = torch.tensor(window_features(np.array(window_buf)),
                                         dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
                        cls = int(np.argmax(probs))
                        verdict = {"cls": cls, "conf": float(probs[cls]), "pbad": float(probs[1])}
                        verdict_log.append((t, cls, float(probs[cls])))
                        last_eval = frame_idx
                    except Exception as e:
                        print(f"[!] Error during inference at frame {frame_idx}: {e}")

                # --- squat analytics (angles, reps, error checks) ----
                analyzer.update(analysis_smoother.update(raw, vis), vis, t, frame_idx, verdict["pbad"])

                # --- skeleton (uses ANALYSIS_ALPHA smoothing) ----
                lm = res.pose_landmarks.landmark
                px = np.array([[l.x * W, l.y * H] for l in lm], dtype=np.float32)
                pv = np.array([l.visibility for l in lm], dtype=np.float32)
                draw_stick_overlay(frame, pixel_smoother.update(px, pv), analyzer.status)

            draw_hud(frame, analyzer, verdict, pose_ok, t)
            draw_live_graph(frame, analyzer)

            # Progress bar
            progress = frame_idx / max(total_frames, 1)
            cv2.rectangle(frame, (0, H - 6), (int(W * progress), H), (0, 255, 0), -1)

            out_vid.write(frame)
            frame_idx += 1

            # Show time elapsed and ETA
            if frame_idx % max(1, total_frames // 10) == 0:
                elapsed_s = frame_idx / fps
                total_s = total_frames / fps
                eta_s = (total_frames - frame_idx) / fps if frame_idx > 0 else 0
                progress_pct = frame_idx / max(total_frames, 1)
                print(f"  [{progress_pct:>5.1%}] {frame_idx:>6}/{total_frames} frames | "
                      f"Time: {elapsed_s:>6.1f}s / {total_s:>6.1f}s | "
                      f"ETA: {eta_s:>6.1f}s | Reps: {analyzer.counter.reps}")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR during video processing:\n   {e}")
        traceback.print_exc()
    finally:
        cap.release()
        out_vid.release()
        pose.close()
        print(f"\n[OK] Resources cleaned up.")

    print(f"\n[OK] Annotated video saved -> {OUTPUT_VIDEO}\n")

    # =================================================================
    # 8. SAVE ANALYTICS & REPORTS
    # =================================================================
    print("="*80)
    print("SAVING ANALYSIS RESULTS")
    print("="*80 + "\n")

    df = analyzer.dataframe()
    rep_df = analyzer.rep_dataframe()

    if df.empty:
        print("[!] No pose was detected in the video - nothing to analyse.")
    else:
        # Save CSVs
        df.to_csv(os.path.join(OUTPUT_DIR, "squat_frames.csv"), index=False)
        rep_df.to_csv(os.path.join(OUTPUT_DIR, "squat_reps.csv"), index=False)
        print(f"[OK] Saved frame-by-frame analysis: squat_frames.csv ({len(df)} frames)")
        print(f"[OK] Saved per-rep summary: squat_reps.csv ({len(rep_df)} reps)")

        # ROM stats
        rom_df = rom_stats(df)
        rom_df.to_csv(os.path.join(OUTPUT_DIR, "squat_rom_stats.csv"), index=False)

        # Plots
        plot_rom_bar(rom_df, os.path.join(OUTPUT_DIR, "rom_plot.png"))
        plot_hip_knee(df, os.path.join(OUTPUT_DIR, "hip_knee_time.png"))
        plot_symmetry(df, os.path.join(OUTPUT_DIR, "symmetry.png"))
        plot_smoothness(rep_df, os.path.join(OUTPUT_DIR, "rep_smoothness.png"))
        print(f"[OK] Saved 4 analysis plots")

        # Report
        write_report(os.path.join(OUTPUT_DIR, "squat_report.txt"), analyzer, df, rep_df,
                     verdict_log, fps, ckpt_info)
        print(f"[OK] Saved detailed report: squat_report.txt")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Reps counted: {analyzer.counter.reps}")
        print(f"Camera side: {analyzer.side}")
        if not rep_df.empty:
            print(f"\nPer-rep breakdown:")
            print(rep_df[["Rep", "Min Knee Angle", "Max Torso Lean", "Model P(bad)", "Issues"]].to_string(index=False))
        print(f"\n{'='*80}")
        print(f"All results saved to: {OUTPUT_DIR}/")
        print(f"{'='*80}\n")

        # --- collect this video's numbers for the average CSV (skip "transposed" videos) ---
        if "transposed" not in vid_name.lower():
            summary_rows.append({
                "Video": vid_name,
                "Reps": analyzer.counter.reps,
                "Duration (s)": round(len(df) / fps, 1),
                "Mean Knee Angle": df["Knee Angle"].mean(),
                "Min Knee Angle": df["Knee Angle"].min(),
                "Knee ROM": df["Knee Angle"].max() - df["Knee Angle"].min(),
                "Mean Hip Angle": df["Hip Angle"].mean(),
                "Mean Torso Lean": df["Torso Lean Angle"].mean(),
                "Avg Min Knee Angle per Rep": rep_df["Min Knee Angle"].mean() if not rep_df.empty else np.nan,
                "Avg Max Torso Lean per Rep": rep_df["Max Torso Lean"].mean() if not rep_df.empty else np.nan,
                "Avg Rep Duration (s)": rep_df["Duration (s)"].mean() if not rep_df.empty else np.nan,
                "Avg Rep Smoothness": rep_df["Smoothness"].mean() if not rep_df.empty else np.nan,
                "Reps Too Shallow": int((~rep_df["Depth OK"]).sum()) if not rep_df.empty else 0,
                "Reps Too Much Lean": int((~rep_df["Lean OK"]).sum()) if not rep_df.empty else 0,
                "Reps Knees Caved": int((~rep_df["Knees OK"]).sum()) if not rep_df.empty else 0,
                "Avg Model P(bad)": df["Model_P_Bad"].mean(),
                "Model % Windows BAD": (100 * np.mean([c == 1 for _, c, _ in verdict_log])
                                        if verdict_log else np.nan),
            })

# =====================================================================
# 9. AVERAGE CSV ACROSS ALL VIDEOS (except "transposed")
# =====================================================================
if summary_rows:
    avg_df = pd.DataFrame(summary_rows)
    avg_row = avg_df.mean(numeric_only=True).round(3)
    avg_row["Video"] = f"AVERAGE ({len(avg_df)} videos)"
    avg_df = pd.concat([avg_df, pd.DataFrame([avg_row])], ignore_index=True)
    avg_path = os.path.join(OUTPUT_ROOT, "average_all_videos.csv")
    avg_df.to_csv(avg_path, index=False)
    print(f"[OK] Saved average over {len(summary_rows)} videos -> {avg_path}")
else:
    print("[!] No videos available for the average CSV.")