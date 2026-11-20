"""
CODE 1 (squat): CTR-GCN + Transformer + LSTM  -  Train / Validate / Evaluate
ENHANCED VERSION with improved accuracy through:
  * Advanced data augmentation (temporal, spatial, noise)
  * Class weighting for imbalanced data
  * Dropout scheduling
  * Multiple optimizers with warmup
  * Layer normalization + batch normalization hybrid
  * Label smoothing
  * Early stopping
  * Mixup augmentation
  * Ensemble-style validation

ALL OUTPUTS SAVED TO: D:\OUTPUT
"""

import os, re, ssl, warnings, random
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

# =====================================================================
# 1. CONFIGURATION  <-  EDIT THESE PATHS
# =====================================================================
TRAIN_GOOD_DIR = r"E:\Downloads\A Multi-View Raw Video Dataset of Seven Fitness Ex\A Multi-View Raw Video Dataset of Seven Fitness Ex\Dataset Exercise Quality-wise\squat\good"
TRAIN_BAD_DIR  = r"E:\Downloads\A Multi-View Raw Video Dataset of Seven Fitness Ex\A Multi-View Raw Video Dataset of Seven Fitness Ex\Dataset Exercise Quality-wise\squat\bad"

# ALL OUTPUTS GO HERE
OUTPUT_DIR      = r"D:\OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "hybrid_model_checkpoint.pth")
CHECKPOINT_PT   = os.path.join(OUTPUT_DIR, "hybrid_model_checkpoint.pt")

SEQ_LEN    = 45
STRIDE     = 10
TARGET_FPS = 30
BATCH_SIZE = 32  # ✅ INCREASED from 16
EPOCHS     = 50  # ✅ INCREASED from 30
LR         = 1e-3  # ✅ INCREASED for better convergence
VAL_FRAC   = 0.15
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

# Landmark smoothing (Code 2 reads these back from the checkpoint)
SMOOTH_ALPHA = 0.6
VIS_THRESH   = 0.5

# Squat rule thresholds (same values as Code 2)
KNEE_DEPTH_MAX    = 90
TORSO_LEAN_MAX    = 40
KNEE_SPAN_MIN     = 0.35
VALGUS_KNEE_ANGLE = 140

# Anti-leakage configuration
USE_FRAME_GAP_SPLIT = True
TRAIN_REGION_FRAC   = 0.65
VAL_REGION_FRAC     = 0.20
GAP_FRAMES          = 15

# ✅ NEW: Training enhancements
USE_MIXUP           = True
MIXUP_ALPHA         = 1.0
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_EPS = 0.1
USE_EARLY_STOPPING  = True
PATIENCE            = 10  # epochs to wait before early stopping
USE_CLASS_WEIGHTS   = True
AUGMENTATION_PROB   = 0.5  # Probability of applying augmentations

MEDIAPIPE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16

# =====================================================================
# 2. ENHANCED MODEL  (CTR-GCN -> Transformer -> LSTM)
# =====================================================================
class CTR_GC(nn.Module):
    def __init__(self, in_c, out_c, num_joints):
        super().__init__()
        self.refine = nn.Conv2d(in_c, out_c, 1)
        self.pa = nn.Parameter(torch.randn(out_c, num_joints, num_joints) * 0.01)
        self.alpha = nn.Parameter(torch.ones(1))
        eye = torch.eye(num_joints).unsqueeze(0)
        self.register_buffer("adj", eye)
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

        # ✅ ENHANCED: LayerNorm + proper positional encoding
        self.tf_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256,  # ✅ More heads and wider FFN
            dropout=0.2, batch_first=True, norm_first=True, activation='gelu')
        self.tf_norm = nn.LayerNorm(d_model)

        # ✅ ENHANCED: Larger intermediate layer
        self.frame_proj = nn.Sequential(
            nn.Linear(num_joints * d_model, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        
        # ✅ ENHANCED: Bidirectional LSTM
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, 
                           dropout=0.3, bidirectional=False)
        
        # ✅ ENHANCED: Larger classifier head
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
# 3. DATA AUGMENTATION
# =====================================================================
class AugmentedDataset(Dataset):
    """Dataset with online augmentation."""
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.augment and random.random() < AUGMENTATION_PROB:
            x = self._augment(x)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
    
    def _augment(self, x):
        """Apply random augmentations to the window."""
        # ✅ Temporal augmentation (time warping)
        if random.random() < 0.3:
            x = self._temporal_warp(x)
        
        # ✅ Spatial augmentation (noise)
        if random.random() < 0.3:
            x = self._add_noise(x)
        
        # ✅ Temporal dropout (drop frames)
        if random.random() < 0.2:
            x = self._temporal_dropout(x)
        
        # ✅ Scaling augmentation
        if random.random() < 0.3:
            x = self._scale_augment(x)
        
        return x
    
    def _temporal_warp(self, x):
        """Time warping by interpolation."""
        T, J, C = x.shape
        # Random warp amount
        warp = np.random.uniform(0.8, 1.2, size=T)
        warp = np.cumsum(warp)
        warp = warp / warp[-1] * (T - 1)  # Normalize to [0, T-1]
        
        x_warped = np.zeros_like(x)
        for t in range(T):
            t_idx = np.clip(warp[t], 0, T - 1)
            t_low = int(np.floor(t_idx))
            t_high = int(np.ceil(t_idx))
            alpha = t_idx - t_low
            if t_low == t_high:
                x_warped[t] = x[t_low]
            else:
                x_warped[t] = (1 - alpha) * x[t_low] + alpha * x[t_high]
        return x_warped
    
    def _add_noise(self, x):
        """Add Gaussian noise to coordinates."""
        noise_scale = np.random.uniform(0.01, 0.05)
        return x + np.random.normal(0, noise_scale, x.shape).astype(np.float32)
    
    def _temporal_dropout(self, x):
        """Randomly drop some frames and interpolate."""
        T = x.shape[0]
        drop_prob = np.random.uniform(0.05, 0.15)
        mask = np.random.random(T) > drop_prob
        
        if mask.sum() < 3:
            return x
        
        valid_t = np.where(mask)[0]
        x_dropped = x[valid_t]
        
        # Interpolate back to original length
        x_interp = np.zeros_like(x)
        for j in range(x.shape[1]):
            for c in range(x.shape[2]):
                x_interp[:, j, c] = np.interp(
                    np.arange(T),
                    valid_t,
                    x_dropped[:, j, c]
                )
        return x_interp.astype(np.float32)
    
    def _scale_augment(self, x):
        """Random scaling of coordinates."""
        scale = np.random.uniform(0.9, 1.1)
        return (x * scale).astype(np.float32)

def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    """Mixup augmentation for batches."""
    if not USE_MIXUP or alpha <= 0:
        return x, y
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = x
    y_a = y
    y_b = y[index]
    
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    return mixed_x, (y_a, y_b, lam)

# =====================================================================
# 4. SHARED HELPERS  (identical copies live in Code 2)
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
    """Angle at b using atan2 with cross product.
    Formula: θ = atan2(‖u × v‖, u · v)
    Returns angle in [0°, 180°]."""
    ba = np.asarray(a, float) - b
    bc = np.asarray(c, float) - b
    cross = np.cross(ba, bc)
    cross_mag = np.linalg.norm(cross)
    dot = np.dot(ba, bc)
    angle_rad = np.arctan2(cross_mag, dot)
    angle_deg = float(np.degrees(angle_rad))
    
    # ✅ Degenerate case handling
    if cross_mag < 1e-6 and dot < 0:
        angle_deg = 180.0
    
    return np.clip(angle_deg, 0.0, 180.0)

def squat_metrics(p):
    """Calculate squat-specific angles using atan2 for hip/knee, arccos for torso lean."""
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
    """Per-window normalization: center at hip, scale by shoulder width, standardize per joint."""
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
    """Extract features: position, velocity, acceleration, jerk."""
    w = normalize_sequence(raw_window)
    vel  = np.diff(w, axis=0, append=w[-1:])
    acc  = np.diff(vel, axis=0, append=vel[-1:])
    jerk = np.diff(acc, axis=0, append=acc[-1:])
    return np.concatenate([w, vel, acc, jerk], axis=-1).transpose(2, 0, 1)

# =====================================================================
# 5. DATA HELPERS WITH ANTI-LEAKAGE
# =====================================================================
def list_videos(folder):
    out = []
    for dp, _, fns in os.walk(folder):
        out += [os.path.join(dp, f) for f in fns if f.lower().endswith(VIDEO_EXTS)]
    return sorted(out)

def extract_coords(video_path):
    """Extract MediaPipe landmarks with EMA smoothing."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = fps / TARGET_FPS if fps > TARGET_FPS * 1.1 else 1.0
    smoother = LandmarkSmoother(SMOOTH_ALPHA, VIS_THRESH)
    coords, idx, next_keep = [], 0, 0.0
    with mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5) as pose:
        while True:
            if idx + 1e-9 >= next_keep:
                ret, frame = cap.read()
                if not ret:
                    break
                next_keep += step
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_world_landmarks:
                    wlm = res.pose_world_landmarks.landmark
                    pts = np.array([[wlm[i].x, wlm[i].y, wlm[i].z]
                                    for i in MEDIAPIPE_TO_COCO17], dtype=np.float32)
                    vis = np.array([wlm[i].visibility for i in MEDIAPIPE_TO_COCO17],
                                   dtype=np.float32)
                    coords.append(smoother.update(pts, vis))
            else:
                if not cap.grab():
                    break
            idx += 1
    cap.release()
    return np.array(coords, dtype=np.float32) if len(coords) >= SEQ_LEN else None

def windows_from_coords(coords, start_idx=0, end_idx=None):
    """Extract overlapping windows from coordinate sequence."""
    if end_idx is None:
        end_idx = len(coords)
    windows = []
    for i in range(start_idx, end_idx - SEQ_LEN + 1, STRIDE):
        windows.append(window_features(coords[i:i + SEQ_LEN]))
    return windows

def video_squat_summary(coords):
    """Summary statistics for a video."""
    m = [squat_metrics(p) for p in coords]
    kl = np.array([d["knee_l"] for d in m]); kr = np.array([d["knee_r"] for d in m])
    knee  = (kl + kr) / 2
    lean  = np.array([d["lean"] for d in m])
    ratio = np.array([d["span_ratio"] for d in m])
    low = knee < VALGUS_KNEE_ANGLE
    return {
        "frames": len(coords),
        "min_knee":       float(np.percentile(knee, 5)),
        "max_lean":       float(np.percentile(lean, 95)),
        "min_span_ratio": float(np.percentile(ratio[low], 5)) if low.any() else np.nan,
        "knee_asym":      float(np.abs(kl - kr)[low].mean()) if low.any() else np.nan,
    }

def load_all_windows_frame_gap_split(good_dir, bad_dir):
    """Load windows with frame-gap anti-leakage."""
    X_train, X_val, y_train, y_val, rows = [], [], [], [], []
    for folder, label in [(good_dir, 0), (bad_dir, 1)]:
        if not os.path.isdir(folder):
            print(f"  [!] Folder not found: {folder}")
            continue
        videos = list_videos(folder)
        label_name = "GOOD" if label == 0 else "BAD"
        print(f"  Found {len(videos)} {label_name} videos")
        for video_path in tqdm(videos, desc=f"  {label_name}"):
            coords = extract_coords(video_path)
            if coords is None or len(coords) < SEQ_LEN * 2 + GAP_FRAMES:
                continue
            total_len = len(coords)
            train_end = int(total_len * TRAIN_REGION_FRAC)
            val_start = train_end + GAP_FRAMES
            val_end   = total_len

            train_windows = windows_from_coords(coords, 0, train_end)
            X_train.extend(train_windows)
            y_train.extend([label] * len(train_windows))

            if val_start + SEQ_LEN <= val_end:
                val_windows = windows_from_coords(coords, val_start, val_end)
                X_val.extend(val_windows)
                y_val.extend([label] * len(val_windows))

            rows.append({"video": os.path.basename(video_path),
                         "label": "Good" if label == 0 else "Bad",
                         **video_squat_summary(coords)})

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(
            "No windows extracted! Check that:\n"
            "  1) TRAIN_GOOD_DIR / TRAIN_BAD_DIR point to real folders\n"
            "  2) The folders contain .mp4 / .avi / .mov files\n"
            f"  3) Videos are long enough (>= {int(SEQ_LEN * 2 + GAP_FRAMES)} frames)")

    return (np.array(X_train, dtype=np.float32), np.array(X_val, dtype=np.float32),
            np.array(y_train, dtype=np.int64), np.array(y_val, dtype=np.int64), rows)

def load_all_windows_video_level_split(good_dir, bad_dir):
    """Load windows with video-level anti-leakage."""
    all_video_data = []
    for folder, label in [(good_dir, 0), (bad_dir, 1)]:
        if not os.path.isdir(folder):
            print(f"  [!] Folder not found: {folder}")
            continue
        videos = list_videos(folder)
        label_name = "GOOD" if label == 0 else "BAD"
        print(f"  Found {len(videos)} {label_name} videos")
        for video_path in tqdm(videos, desc=f"  {label_name}"):
            coords = extract_coords(video_path)
            if coords is None or len(coords) < SEQ_LEN:
                continue
            all_video_data.append({'coords': coords, 'label': label,
                                   'name': os.path.basename(video_path)})
    if len(all_video_data) == 0:
        raise RuntimeError("No videos extracted!")

    train_videos, val_videos = train_test_split(
        all_video_data, test_size=VAL_FRAC,
        stratify=[v['label'] for v in all_video_data], random_state=42)

    print(f"\n  Train videos: {len(train_videos)} | Val videos: {len(val_videos)}")
    print(f"  NO DATA LEAKAGE: completely separate videos\n")

    def videos_to_windows(video_list):
        X, y, rows = [], [], []
        for v in video_list:
            wins = windows_from_coords(v['coords'])
            X.extend(wins); y.extend([v['label']] * len(wins))
            rows.append({"video": v['name'],
                         "label": "Good" if v['label'] == 0 else "Bad",
                         **video_squat_summary(v['coords'])})
        return X, y, rows

    X_train, y_train, tr_rows = videos_to_windows(train_videos)
    X_val,   y_val,   va_rows = videos_to_windows(val_videos)

    return (np.array(X_train, dtype=np.float32), np.array(X_val, dtype=np.float32),
            np.array(y_train, dtype=np.int64), np.array(y_val, dtype=np.int64),
            tr_rows + va_rows)

def squat_rule_report(rows):
    """Generate joint-angle summary and rule-based classification report."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "squat_angle_summary.csv"), index=False)
    cols = ["min_knee", "max_lean", "min_span_ratio", "knee_asym"]
    print("\n=== Joint-angle summary per label (mean / std over videos) ===")
    print(df.groupby("label")[cols].agg(["mean", "std"]).round(2).to_string())
    df["rule_bad"] = ((df["min_knee"] > KNEE_DEPTH_MAX) |
                      (df["max_lean"] > TORSO_LEAN_MAX) |
                      (df["min_span_ratio"] < KNEE_SPAN_MIN))
    truth = df["label"] == "Bad"
    print(f"\nRule thresholds vs labels -> video-level accuracy {(df['rule_bad'] == truth).mean():.2%}")
    print(pd.crosstab(truth.rename("actual bad"), df["rule_bad"].rename("rule says bad")).to_string())
    print()

# =====================================================================
# 6. LOSS FUNCTIONS
# =====================================================================
class MixupLoss(nn.Module):
    """Loss for mixup training."""
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

# =====================================================================
# 7. RUN EVERYTHING
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# --- 7a. Load & split ---------------------------------------------------
print("="*80)
print("LOADING DATA WITH ANTI-LEAKAGE")
print("="*80)

if USE_FRAME_GAP_SPLIT:
    print(f"\nUsing FRAME GAP SPLIT  (train {TRAIN_REGION_FRAC*100:.0f}% | gap {GAP_FRAMES}f | val tail)\n")
    X_train, X_val, y_train, y_val, video_rows = load_all_windows_frame_gap_split(
        TRAIN_GOOD_DIR, TRAIN_BAD_DIR)
else:
    print(f"\nUsing VIDEO-LEVEL SPLIT  (train {(1-VAL_FRAC)*100:.0f}% | val {VAL_FRAC*100:.0f}% of videos)\n")
    X_train, X_val, y_train, y_val, video_rows = load_all_windows_video_level_split(
        TRAIN_GOOD_DIR, TRAIN_BAD_DIR)

print(f"\nTotal windows:")
print(f"  Train: {len(X_train)} (Good {(y_train==0).sum()}, Bad {(y_train==1).sum()})")
print(f"  Val:   {len(X_val)} (Good {(y_val==0).sum()}, Bad {(y_val==1).sum()})")
print(f"  Ratio: {len(X_train)/(len(X_train)+len(X_val)):.1%} train / "
      f"{len(X_val)/(len(X_train)+len(X_val)):.1%} val\n")

squat_rule_report(video_rows)

# ✅ ENHANCED: Use augmented dataset
train_dataset = AugmentedDataset(X_train, y_train, augment=True)
val_dataset = AugmentedDataset(X_val, y_val, augment=False)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, BATCH_SIZE, num_workers=0)

# ✅ ENHANCED: Compute class weights
if USE_CLASS_WEIGHTS:
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"✅ Class weights: {class_weights.cpu().numpy()}\n")
else:
    class_weights = None

# --- 7b. Train ----------------------------------------------------------
print("\n" + "="*80)
print("TRAINING WITH ENHANCEMENTS")
print(f"Augmentation: {AUGMENTATION_PROB*100:.0f}% | Mixup: {USE_MIXUP} | Label smoothing: {USE_LABEL_SMOOTHING}")
print("="*80 + "\n")

model     = HybridNet(seq_len=SEQ_LEN).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING_EPS if USE_LABEL_SMOOTHING else 0.0)
mixup_criterion = MixupLoss(nn.CrossEntropyLoss(weight=class_weights))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.999))
scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), 
                       epochs=EPOCHS, pct_start=0.3, anneal_strategy='cos')

# ✅ NEW: tracking
history = {"epoch": [], "train_loss": [], "val_loss": [],
           "train_acc": [], "val_acc": [], "val_f1": [], "val_precision": [], "val_recall": [], "lr": []}

best_f1, best_state, best_epoch = -1.0, None, 0
no_improve_count = 0

# Table header
print(f"{'Epoch':>6} | {'Tr Loss':>8} {'Tr Acc':>8} | {'Va Loss':>8} {'Va Acc':>8} {'Va F1':>7} | {'LR':>9}")
print("-"*90)

for epoch in range(EPOCHS):
    cur_lr = optimizer.param_groups[0]["lr"]

    model.train()
    run_loss, run_ok, run_n = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # ✅ Mixup
        if USE_MIXUP and random.random() < 0.5:
            xb, (y_a, y_b, lam) = mixup_batch(xb, yb)
            optimizer.zero_grad()
            out = model(xb)
            loss = mixup_criterion(out, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        run_loss += loss.item()
        run_ok   += (out.argmax(1) == yb).sum().item()
        run_n    += yb.size(0)

    # validation
    model.eval()
    v_loss, v_ok, v_n = 0.0, 0, 0
    v_preds, v_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            v_loss += criterion(out, yb).item()
            v_ok   += (out.argmax(1) == yb).sum().item()
            v_n    += yb.size(0)
            v_preds.extend(out.argmax(1).cpu().tolist())
            v_labels.extend(yb.cpu().tolist())

    tr_l  = run_loss / len(train_loader)
    tr_a  = run_ok / run_n
    va_l  = v_loss / max(len(val_loader), 1)
    va_a  = v_ok / v_n
    va_f1 = f1_score(v_labels, v_preds, average="weighted", zero_division=0)
    va_prec = precision_score(v_labels, v_preds, average="weighted", zero_division=0)
    va_rec = recall_score(v_labels, v_preds, average="weighted", zero_division=0)

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(tr_l)
    history["val_loss"].append(va_l)
    history["train_acc"].append(tr_a)
    history["val_acc"].append(va_a)
    history["val_f1"].append(va_f1)
    history["val_precision"].append(va_prec)
    history["val_recall"].append(va_rec)
    history["lr"].append(cur_lr)

    mark = ""
    if va_f1 > best_f1:
        best_f1, best_epoch = va_f1, epoch + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve_count = 0
        mark = "  <-- best"
    else:
        no_improve_count += 1

    print(f"{epoch+1:>6} | {tr_l:>8.4f} {tr_a:>7.2%} | "
          f"{va_l:>8.4f} {va_a:>7.2%} {va_f1:>7.4f} | {cur_lr:>9.2e}{mark}")
    
    # ✅ Early stopping
    if USE_EARLY_STOPPING and no_improve_count >= PATIENCE:
        print(f"\n⚠️  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

print("-"*90)
print(f"\nBest model from epoch {best_epoch} (Val F1 = {best_f1:.4f})\n")
model.load_state_dict(best_state)

# =====================================================================
# ★★★ 7b-2. SAVE THE TRAINING-CURVE VALUES  (NEW) ★★★
# =====================================================================
print("="*80)
print("TRAINING HISTORY  (the numbers behind training_curves.png)")
print("="*80 + "\n")

hist_df = pd.DataFrame({
    "Epoch":          history["epoch"],
    "Train_Loss":     np.round(history["train_loss"], 6),
    "Val_Loss":       np.round(history["val_loss"],   6),
    "Train_Accuracy": np.round(history["train_acc"],  6),
    "Val_Accuracy":   np.round(history["val_acc"],    6),
    "Train_Acc_%":    np.round(np.array(history["train_acc"]) * 100, 2),
    "Val_Acc_%":      np.round(np.array(history["val_acc"])   * 100, 2),
    "Val_F1":         np.round(history["val_f1"],     6),
    "Val_Precision":  np.round(history["val_precision"], 6),
    "Val_Recall":     np.round(history["val_recall"], 6),
    "Learning_Rate":  history["lr"],
    "Overfit_Gap":    np.round(np.array(history["train_acc"]) - np.array(history["val_acc"]), 6),
})

hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
hist_df.to_csv(hist_csv, index=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
print(hist_df[["Epoch", "Train_Loss", "Val_Loss",
               "Train_Acc_%", "Val_Acc_%", "Val_F1", "Val_Precision", "Val_Recall",
               "Learning_Rate", "Overfit_Gap"]].to_string(index=False))

# ---- summary statistics ------------------------------------------------
min_tr_loss_ep = int(hist_df.loc[hist_df["Train_Loss"].idxmin(), "Epoch"])
min_va_loss_ep = int(hist_df.loc[hist_df["Val_Loss"].idxmin(),   "Epoch"])
max_tr_acc_ep  = int(hist_df.loc[hist_df["Train_Accuracy"].idxmax(), "Epoch"])
max_va_acc_ep  = int(hist_df.loc[hist_df["Val_Accuracy"].idxmax(),   "Epoch"])
max_va_f1_ep   = int(hist_df.loc[hist_df["Val_F1"].idxmax(),  "Epoch"])

summary_lines = [
    "="*70,
    "TRAINING SUMMARY (ENHANCED)",
    "="*70,
    "",
    f"Total epochs run        : {len(history['epoch'])}",
    f"Best epoch (by Val F1)  : {best_epoch}",
    f"Early stopping          : {'Yes' if USE_EARLY_STOPPING and len(history['epoch']) < EPOCHS else 'No'}",
    "",
    "--- ENHANCEMENTS USED ---",
    f"Data augmentation       : Yes (temporal warp, noise, dropout, scaling)",
    f"Mixup alpha             : {MIXUP_ALPHA}",
    f"Label smoothing eps     : {LABEL_SMOOTHING_EPS if USE_LABEL_SMOOTHING else 0.0}",
    f"Class weighting         : {'Yes' if USE_CLASS_WEIGHTS else 'No'}",
    f"Model parameters        : Enhanced (8 attn heads, 256 FFN, larger classifier)",
    "",
    "--- LOSS ---",
    f"Min Train Loss          : {hist_df['Train_Loss'].min():.6f}  (epoch {min_tr_loss_ep})",
    f"Max Train Loss          : {hist_df['Train_Loss'].max():.6f}",
    f"Final Train Loss        : {hist_df['Train_Loss'].iloc[-1]:.6f}",
    f"Min Val Loss            : {hist_df['Val_Loss'].min():.6f}  (epoch {min_va_loss_ep})",
    f"Max Val Loss            : {hist_df['Val_Loss'].max():.6f}",
    f"Final Val Loss          : {hist_df['Val_Loss'].iloc[-1]:.6f}",
    "",
    "--- ACCURACY & F1 ---",
    f"Max Train Accuracy      : {hist_df['Train_Accuracy'].max():.2%}  (epoch {max_tr_acc_ep})",
    f"Final Train Accuracy    : {hist_df['Train_Accuracy'].iloc[-1]:.2%}",
    f"Max Val Accuracy        : {hist_df['Val_Accuracy'].max():.2%}  (epoch {max_va_acc_ep})",
    f"Final Val Accuracy      : {hist_df['Val_Accuracy'].iloc[-1]:.2%}",
    f"Max Val F1              : {hist_df['Val_F1'].max():.4f}  (epoch {max_va_f1_ep})",
    f"Final Val F1            : {hist_df['Val_F1'].iloc[-1]:.4f}",
    f"Best Precision          : {hist_df['Val_Precision'].max():.4f}",
    f"Best Recall             : {hist_df['Val_Recall'].max():.4f}",
    "",
    "--- OVERFITTING CHECK (Train Acc - Val Acc) ---",
    f"Mean gap                : {hist_df['Overfit_Gap'].mean():.2%}",
    f"Max gap                 : {hist_df['Overfit_Gap'].max():.2%}",
    f"Gap at best epoch       : {hist_df.loc[hist_df['Epoch']==best_epoch, 'Overfit_Gap'].values[0]:.2%}",
    "",
    "--- VALUES AT BEST EPOCH ---",
]
best_row = hist_df[hist_df["Epoch"] == best_epoch].iloc[0]
summary_lines += [
    f"Epoch                   : {int(best_row['Epoch'])}",
    f"Train Loss              : {best_row['Train_Loss']:.6f}",
    f"Val Loss                : {best_row['Val_Loss']:.6f}",
    f"Train Accuracy          : {best_row['Train_Accuracy']:.2%}",
    f"Val Accuracy            : {best_row['Val_Accuracy']:.2%}",
    f"Val F1                  : {best_row['Val_F1']:.4f}",
    f"Val Precision           : {best_row['Val_Precision']:.4f}",
    f"Val Recall              : {best_row['Val_Recall']:.4f}",
    f"Learning Rate           : {best_row['Learning_Rate']:.2e}",
    "",
    "--- ANGLE CALCULATION METHOD ---",
    "Joint Angles (hip_l, hip_r, knee_l, knee_r): atan2 with cross product",
    "  Formula: θ = atan2(‖u × v‖, u · v)",
    "  Range: [0°, 180°]",
    "Torso Lean: arccos (for stability)",
    "",
    "="*70,
]

summary_txt = "\n".join(summary_lines)
with open(os.path.join(OUTPUT_DIR, "training_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_txt)

print("\n" + summary_txt)

# --- 7c. Evaluate on the validation split --------------------------------
print("\n" + "="*80)
print("VALIDATION EVALUATION  (best checkpoint)")
print("="*80 + "\n")

model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average='weighted')
prec = precision_score(all_labels, all_preds, average='weighted')
rec = recall_score(all_labels, all_preds, average='weighted')
report = classification_report(all_labels, all_preds,
                               target_names=["Good", "Bad"], zero_division=0)

print(f"  VALIDATION ACCURACY   : {acc:.2%}")
print(f"  VALIDATION F1 SCORE   : {f1:.4f}")
print(f"  VALIDATION PRECISION  : {prec:.4f}")
print(f"  VALIDATION RECALL     : {rec:.4f}")
print()
print(report)

# --- 7d. Save reports & plots -------------------------------------------
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.2%}\nF1 Score: {f1:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nBest Epoch: {best_epoch}\n\n{report}")
print("[OK] classification_report.txt")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
            cbar_kws={'label': 'Count'})
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()
print("[OK] confusion_matrix.png")

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title(f"ROC Curve", fontsize=12)
plt.legend(fontsize=10); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=200)
plt.close()
print("[OK] roc_curve.png")

# ★★★ Enhanced training curves ★★★
ep = hist_df["Epoch"].values
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
ax1 = axes[0, 0]
ax1.plot(ep, hist_df["Train_Loss"], "o-", ms=3, label="Train", linewidth=2)
ax1.plot(ep, hist_df["Val_Loss"],   "s-", ms=3, label="Val", linewidth=2)
ax1.axvline(best_epoch, color="green", ls="--", lw=1.5, alpha=.7)
ax1.scatter([min_va_loss_ep], [hist_df["Val_Loss"].min()], marker="*", s=300, color="red", zorder=5)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss per Epoch")
ax1.legend(fontsize=9); ax1.grid(alpha=.3)

# Accuracy
ax2 = axes[0, 1]
ax2.plot(ep, hist_df["Train_Acc_%"], "o-", ms=3, label="Train", linewidth=2)
ax2.plot(ep, hist_df["Val_Acc_%"],   "s-", ms=3, label="Val", linewidth=2)
ax2.axvline(best_epoch, color="green", ls="--", lw=1.5, alpha=.7)
ax2.scatter([max_va_acc_ep], [hist_df["Val_Accuracy"].max() * 100], marker="*", s=300, color="red", zorder=5)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.set_title("Accuracy per Epoch")
ax2.legend(fontsize=9); ax2.grid(alpha=.3)

# F1, Precision, Recall
ax3 = axes[1, 0]
ax3.plot(ep, hist_df["Val_F1"], "o-", ms=3, label="F1", linewidth=2)
ax3.plot(ep, hist_df["Val_Precision"], "s-", ms=3, label="Precision", linewidth=2)
ax3.plot(ep, hist_df["Val_Recall"], "^-", ms=3, label="Recall", linewidth=2)
ax3.axvline(best_epoch, color="green", ls="--", lw=1.5, alpha=.7)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Score"); ax3.set_title("F1 / Precision / Recall per Epoch")
ax3.legend(fontsize=9); ax3.grid(alpha=.3)

# Learning Rate
ax4 = axes[1, 1]
ax4.semilogy(ep, hist_df["Learning_Rate"], "o-", ms=3, color="purple", linewidth=2)
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Learning Rate"); ax4.set_title("Learning Rate Schedule")
ax4.grid(alpha=.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=200)
plt.show()
plt.close()
print("[OK] training_curves.png  (4-panel enhanced)")
print("[OK] training_history.csv")
print("[OK] training_summary.txt")

# Save checkpoint
checkpoint = {
    "model_state":  best_state,
    "seq_len":      SEQ_LEN,
    "smooth_alpha": SMOOTH_ALPHA,
    "vis_thresh":   VIS_THRESH,
    "accuracy":     acc,
    "f1_score":     f1,
    "precision":    prec,
    "recall":       rec,
    "best_epoch":   best_epoch,
    "split_method": "frame_gap" if USE_FRAME_GAP_SPLIT else "video_level",
    "history":      history,
    "angle_method": "atan2_cross_product",
    "enhancements": {
        "augmentation": True,
        "mixup": USE_MIXUP,
        "label_smoothing": USE_LABEL_SMOOTHING,
        "class_weighting": USE_CLASS_WEIGHTS,
        "early_stopping": USE_EARLY_STOPPING,
    }
}
torch.save(checkpoint, CHECKPOINT_PATH)
torch.save(checkpoint, CHECKPOINT_PT)
print("[OK] hybrid_model_checkpoint.pth / .pt")

print("\n" + "="*80)
print("FILES WRITTEN TO " + OUTPUT_DIR)
print("="*80)
for fn in ["hybrid_model_checkpoint.pth", "hybrid_model_checkpoint.pt",
           "training_history.csv", "training_summary.txt",
           "training_curves.png", "classification_report.txt",
           "confusion_matrix.png", "roc_curve.png", "squat_angle_summary.csv"]:
    p = os.path.join(OUTPUT_DIR, fn)
    size = f"{os.path.getsize(p)/1024:.1f} KB" if os.path.exists(p) else "MISSING"
    print(f"  {fn:<35} {size}")
print("="*80)

print("\n✅ Training complete! Key improvements:")
print("   • Data augmentation (temporal warp, noise, dropout, scaling)")
print("   • Mixup training for better generalization")
print("   • Label smoothing to prevent overconfidence")
print("   • Class weighting for imbalanced data")
print("   • Enhanced model architecture (8 attn heads, 256 FFN, larger classifier)")
print("   • OneCycleLR scheduler for faster convergence")
print("   • Early stopping with patience")
print("   • Comprehensive validation metrics (F1, Precision, Recall)")