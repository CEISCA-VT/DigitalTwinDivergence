#!/usr/bin/env python3
"""
TerraSentia IMU/Frame/Timing Diagnostic + Frozen Zero-Shot Predictive Rollout Study
==================================================================================

This is a PREPROCESSING / EVALUATION diagnostic, not a new model experiment.

The neural architecture and all 30 i2Nav checkpoints remain frozen.

Official TerraSentia calibration facts encoded below come from:
  jrcuaranv/terrasentia-dataset, sensor_parameters.txt
  - ZED IMU frame: x forward, y left, z up
  - T_imu_left: left-camera -> ZED-IMU
  - Tl_robotIMU: robot-IMU -> left-camera
  - wheel left-right track width: 0.26 m

Therefore the metadata-supported robot-IMU -> body-like (ZED-IMU) rotation is:
  R_body_robotimu = R_zedimu_left @ R_left_robotimu

The official repository's extract_data_from_rosbag.py timestamps extracted data
using the rosbag message time `t`. We therefore retain bag time as the primary
pipeline, but explicitly audit header-time cadence and run a header-relative
sensitivity pipeline because some bag topics may exhibit batched receipt times.

No axis/sign/lag candidate is selected by TerraSentia ATE. Candidate sweeps are
diagnostic only.

Outputs answer:
1) Is the robot IMU axis/frame interpretation correct?
2) Is timestamp behavior a likely source of the huge global ATE?
3) Does the metadata-supported transform change the fixed baseline?
4) Does the frozen i2Nav model improve anchored 1/5/10/30/60 s prediction?
5) Is TerraSentia global ATE trustworthy enough to headline?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


SEQUENCES = [
    "ts_2022_06_09_13h16m39s_one_row",
    "ts_2022_06_15_11h48m34s_four_rows",
    "ts_2022_09_01_11h20m00s_two_random",
    "ts_2022_09_01_12h32m56s_double_loop_corridor",
    "ts_2022_09_06_12h37m11s_four_rows",
]

FEATURE_NAMES = [
    "odo_speed_mps",
    "imu_yaw_rate_radps",
    "odo_accel_mps2",
    "imu_yaw_accel_radps2",
    "abs_yaw_rate_radps",
    "abs_odo_accel_mps2",
]

HORIZONS_S = (1, 5, 10, 30, 60)

MOTOR_COLS = [
    "front_left.linear_speed",
    "front_right.linear_speed",
    "back_left.linear_speed",
    "back_right.linear_speed",
]
LEFT_COLS = ["front_left.linear_speed", "back_left.linear_speed"]
RIGHT_COLS = ["front_right.linear_speed", "back_right.linear_speed"]

TRACK_WIDTH_M = 0.26

# Official sensor_parameters.txt:
# "Transformation from left camera to zed-imu"
R_ZEDIMU_LEFT = np.array([
    [ 0.008566743072747,  0.002937311261940,  0.999958988714200],
    [-0.999961868128753,  0.001961224964540,  0.008560264215503],
    [-0.001935776313654, -0.999993527115831,  0.002953683950084],
], dtype=float)

# "Transformation: Robot IMU to left camera frame (estimated with Kalibr)"
R_LEFT_ROBOTIMU = np.array([
    [-0.118051884740142, -0.991715886014659, -0.050629575699013],
    [ 0.044898991662278,  0.045603162098676, -0.997950115063029],
    [ 0.991991851305751, -0.120083108857007,  0.039143504061662],
], dtype=float)

R_BODY_ROBOTIMU = R_ZEDIMU_LEFT @ R_LEFT_ROBOTIMU


def progress(msg: str):
    """Timestamped progress output that appears immediately in PowerShell."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def wrap_angle(x):
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return np.arctan2(siny_cosp, cosy_cosp)


def finite(x):
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 20:
        return float("nan")
    a, b = a[g], b[g]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    g = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[g]-b[g])**2))) if g.any() else float("nan")


def robust_hz(t):
    t = np.asarray(t, float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 1e-8)]
    return float(1.0 / np.median(d)) if len(d) else float("nan")


def write_rows(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def smooth(x, hz=10.0):
    x = np.asarray(x, float)
    if savgol_filter is None or len(x) < 15:
        # short centered moving average
        w = max(3, int(round(hz*0.5)) | 1)
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
    w = max(5, int(round(hz*0.7)) | 1)
    if w >= len(x):
        w = (len(x)-1) if (len(x)-1)%2 else (len(x)-2)
    if w < 5:
        return x
    return savgol_filter(x, window_length=w, polyorder=2, mode="interp")


def timestamp_arrays(df: pd.DataFrame):
    if "bag_timestamp_ns" in df:
        bag = pd.to_numeric(df["bag_timestamp_ns"], errors="coerce").to_numpy(float)*1e-9
    elif "timestamp_s" in df:
        bag = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(float)
    else:
        raise RuntimeError("CSV lacks bag_timestamp_ns/timestamp_s")
    header = None
    if "timestamp_ns" in df:
        header = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(float)*1e-9
    return bag, header


def clean_time_and_df(df: pd.DataFrame, mode: str):
    bag, header = timestamp_arrays(df)
    if mode == "bag":
        t = bag.copy()
    elif mode == "header_relative":
        if header is not None and np.all(np.isfinite(header)) and np.ptp(header) > 1:
            # preserve header cadence, anchor each stream to first bag receipt
            t = bag[0] + (header - header[0])
        else:
            t = bag.copy()
    else:
        raise ValueError(mode)

    g = np.isfinite(t)
    df = df.loc[g].copy()
    t = t[g]
    order = np.argsort(t)
    t = t[order]
    df = df.iloc[order].reset_index(drop=True)
    keep = np.r_[True, np.diff(t) > 1e-8]
    return df.loc[keep].reset_index(drop=True), t[keep]


def stream_audit(df: pd.DataFrame):
    bag, header = timestamp_arrays(df)
    out = {
        "bag_hz": robust_hz(bag),
        "bag_duration_s": float(np.nanmax(bag)-np.nanmin(bag)),
    }
    if header is not None and np.isfinite(header).sum() > 10:
        out.update({
            "header_hz": robust_hz(header),
            "header_duration_s": float(np.nanmax(header)-np.nanmin(header)),
            "header_minus_bag_first_s": float(header[np.isfinite(header)][0]-bag[np.isfinite(bag)][0]),
        })
        h = header - header[0]
        b = bag - bag[0]
        g = np.isfinite(h)&np.isfinite(b)
        if g.sum()>10 and np.dot(h[g],h[g])>1e-12:
            slope=float(np.dot(h[g],b[g])/np.dot(h[g],h[g]))
            residual=b[g]-slope*h[g]
            out["bag_vs_header_relative_slope"]=slope
            out["bag_vs_header_residual_p95_ms"]=float(np.percentile(np.abs(residual),95)*1000)
    else:
        out["header_hz"]=float("nan")
    return out


class OriginalDualGRU(nn.Module):
    def __init__(self, dv, dw, hidden=64, layers=2, dropout=.1, amin=1., amax=10.):
        super().__init__()
        self.dv=float(dv); self.dw=float(dw); self.amin=float(amin); self.amax=float(amax)
        self.gru=nn.GRU(6, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers>1 else 0.)
        self.trunk=nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU())
        self.dynamics_head=nn.Linear(hidden,2)
        self.q_head=nn.Linear(hidden,2)
    def forward(self,x):
        y,_=self.gru(x)
        h=self.trunk(y[:,-1])
        d=self.dynamics_head(h)
        dyn=torch.stack([self.dv*torch.tanh(d[:,0]), self.dw*torch.tanh(d[:,1])],dim=-1)
        q=self.q_head(h)
        a=self.amin+(self.amax-self.amin)*torch.sigmoid(q)
        return dyn,a


@dataclass
class CK:
    label: str
    model: OriginalDualGRU
    mean: np.ndarray
    std: np.ndarray
    window: int


def load_ck(path: Path, device):
    c=torch.load(path,map_location="cpu",weights_only=False)
    dv=float(c.get("dv_limit",c.get("delta_v_limit")))
    dw=float(c.get("domega_limit",c.get("delta_omega_limit")))
    m=OriginalDualGRU(dv,dw,
                      hidden=int(c.get("hidden_size",64)),
                      layers=int(c.get("num_layers",2)),
                      amin=float(c.get("alpha_min",1.)),
                      amax=float(c.get("alpha_max",10.)))
    m.load_state_dict(c["model_state_dict"],strict=True); m.to(device); m.eval()
    rep=int(c.get("replicate",0)); test=str(c.get("test_sequence",path.parent.name))
    return CK(f"seed{rep}_{test}",m,np.asarray(c["feature_mean"],np.float32),
              np.asarray(c["feature_std"],np.float32),int(c.get("window",20)))


@dataclass
class Seq:
    name: str
    grid: np.ndarray
    dt: float
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    ref_w_pose: np.ndarray
    ref_w_twist: np.ndarray
    ref_v: np.ndarray
    v: np.ndarray
    vl: np.ndarray
    vr: np.ndarray
    omega_raw_z: np.ndarray
    omega_body_z: np.ndarray
    omega_wheel: np.ndarray
    imu_xyz: np.ndarray
    features_raw: np.ndarray
    features_body: np.ndarray
    audit: dict[str,Any]


def interp_col(df,t,col,grid):
    a=pd.to_numeric(df[col],errors="coerce").to_numpy(float)
    g=np.isfinite(a)&np.isfinite(t)
    if g.sum()<2:
        return np.full(len(grid),np.nan)
    return np.interp(grid,t[g],a[g])


def prepare_sequence(seqdir:Path,name:str,hz:float,mode:str)->Seq:
    mp,ip,rp=seqdir/"motors.csv",seqdir/"imu.csv",seqdir/"reference_ekf.csv"
    m0=pd.read_csv(mp,low_memory=False); i0=pd.read_csv(ip,low_memory=False); r0=pd.read_csv(rp,low_memory=False)
    for c in MOTOR_COLS:
        if c not in m0: raise RuntimeError(f"{name}: motors missing {c}")
    if "angular_velocity.z" not in i0: raise RuntimeError(f"{name}: imu missing angular_velocity.z")
    for c in ["pose.pose.position.x","pose.pose.position.y",
              "pose.pose.orientation.x","pose.pose.orientation.y",
              "pose.pose.orientation.z","pose.pose.orientation.w"]:
        if c not in r0: raise RuntimeError(f"{name}: reference missing {c}")

    audit={"sequence":name,"mode":mode,
           "motors":stream_audit(m0),"imu":stream_audit(i0),"reference":stream_audit(r0)}

    m,tm=clean_time_and_df(m0,mode); i,ti=clean_time_and_df(i0,mode); r,tr=clean_time_and_df(r0,mode)
    t0=max(tm[0],ti[0],tr[0]); t1=min(tm[-1],ti[-1],tr[-1])
    if t1-t0<20: raise RuntimeError(f"{name}/{mode}: only {t1-t0:.1f}s overlap")
    dt=1/hz; grid=np.arange(t0,t1+0.25*dt,dt)

    wheels=np.column_stack([interp_col(m,tm,c,grid) for c in MOTOR_COLS])
    v=np.nanmean(wheels,axis=1)
    vl=np.nanmean(np.column_stack([interp_col(m,tm,c,grid) for c in LEFT_COLS]),axis=1)
    vr=np.nanmean(np.column_stack([interp_col(m,tm,c,grid) for c in RIGHT_COLS]),axis=1)
    omega_wheel=(vr-vl)/TRACK_WIDTH_M

    axes=[]
    for ax in "xyz":
        col=f"angular_velocity.{ax}"
        if col in i:
            axes.append(interp_col(i,ti,col,grid))
        else:
            axes.append(np.full(len(grid),np.nan))
    imu_xyz=np.column_stack(axes)
    omega_raw=imu_xyz[:,2]
    if np.all(np.isfinite(imu_xyz)):
        body=(R_BODY_ROBOTIMU @ imu_xyz.T).T
        omega_body=body[:,2]
    else:
        omega_body=omega_raw.copy()

    x=interp_col(r,tr,"pose.pose.position.x",grid)
    y=interp_col(r,tr,"pose.pose.position.y",grid)
    qx=interp_col(r,tr,"pose.pose.orientation.x",grid)
    qy=interp_col(r,tr,"pose.pose.orientation.y",grid)
    qz=interp_col(r,tr,"pose.pose.orientation.z",grid)
    qw=interp_col(r,tr,"pose.pose.orientation.w",grid)
    yaw=wrap_angle(quat_to_yaw(qx,qy,qz,qw))
    yaw_u=np.unwrap(yaw)
    ref_w_pose=smooth(np.gradient(yaw_u,dt),hz)

    if "twist.twist.angular.z" in r:
        ref_w_twist=interp_col(r,tr,"twist.twist.angular.z",grid)
    else:
        ref_w_twist=ref_w_pose.copy()
    if "twist.twist.linear.x" in r:
        ref_v=interp_col(r,tr,"twist.twist.linear.x",grid)
    else:
        vx=np.gradient(x,dt); vy=np.gradient(y,dt)
        ref_v=vx*np.cos(yaw)+vy*np.sin(yaw)

    def feats(om):
        a=np.gradient(v,dt); wa=np.gradient(om,dt)
        return np.column_stack([v,om,a,wa,np.abs(om),np.abs(a)]).astype(np.float32)

    audit.update({
        "duration_s":float(grid[-1]-grid[0]),
        "samples":len(grid),
        "ref_twist_vs_pose_yawrate_corr":corr(ref_w_twist,ref_w_pose),
        "ref_twist_vs_pose_yawrate_rmse":rmse(ref_w_twist,ref_w_pose),
        "raw_z_vs_pose_yawrate_corr":corr(omega_raw,ref_w_pose),
        "body_z_vs_pose_yawrate_corr":corr(omega_body,ref_w_pose),
        "wheel_yaw_vs_pose_yawrate_corr":corr(omega_wheel,ref_w_pose),
        "raw_z_vs_ref_twist_corr":corr(omega_raw,ref_w_twist),
        "body_z_vs_ref_twist_corr":corr(omega_body,ref_w_twist),
        "wheel_yaw_vs_ref_twist_corr":corr(omega_wheel,ref_w_twist),
    })

    return Seq(name,grid,dt,x,y,yaw,ref_w_pose,ref_w_twist,ref_v,v,vl,vr,
               omega_raw,omega_body,omega_wheel,imu_xyz,feats(omega_raw),feats(omega_body),audit)


def lag_diag(signal,target,hz,maxlag=2.0):
    s=np.asarray(signal,float); t=np.asarray(target,float)
    best=None
    maxn=int(round(maxlag*hz))
    zero_c=corr(s,t); zero_r=rmse(s,t)
    for k in range(-maxn,maxn+1):
        if k<0: a,b=s[-k:],t[:len(t)+k]
        elif k>0: a,b=s[:-k],t[k:]
        else: a,b=s,t
        c=corr(a,b); r=rmse(a,b)
        if np.isfinite(c):
            score=abs(c)
            if best is None or score>best[0]:
                best=(score,k,c,r)
    return {
        "zero_lag_corr":zero_c,"zero_lag_rmse":zero_r,
        "best_abs_corr":best[0] if best else np.nan,
        "best_lag_s":best[1]/hz if best else np.nan,
        "best_signed_corr":best[2] if best else np.nan,
        "best_lag_rmse":best[3] if best else np.nan,
    }


def predict(ck:CK, features:np.ndarray, device, batch=8192):
    z=(features-ck.mean)/np.maximum(ck.std,1e-6)
    n=len(z); dyn=np.zeros((n,2),float)
    if n<ck.window: return dyn
    sw=np.lib.stride_tricks.sliding_window_view(z.astype(np.float32),window_shape=ck.window,axis=0).transpose(0,2,1)
    out=[]
    with torch.no_grad():
        for st in range(0,len(sw),batch):
            b=torch.from_numpy(np.array(sw[st:st+batch],np.float32,copy=True)).to(device)
            d,_=ck.model(b); out.append(d.cpu().numpy())
    dyn[ck.window-1:]=np.concatenate(out,axis=0)
    return dyn


def integrate(x0,y0,yaw0,v,w,dt):
    n=len(v); p=np.zeros((n,3),float); p[0]=[x0,y0,yaw0]
    for k in range(1,n):
        dth=w[k-1]*dt; mid=p[k-1,2]+0.5*dth; ds=v[k-1]*dt
        p[k,0]=p[k-1,0]+ds*math.cos(mid)
        p[k,1]=p[k-1,1]+ds*math.sin(mid)
        p[k,2]=wrap_angle(p[k-1,2]+dth)
    return p


def global_metrics(seq:Seq,v,w):
    p=integrate(seq.x[0],seq.y[0],seq.yaw[0],v,w,seq.dt)
    pe=np.hypot(p[:,0]-seq.x,p[:,1]-seq.y)
    he=np.rad2deg(np.abs(wrap_angle(p[:,2]-seq.yaw)))
    return {
        "ate_rmse_m":float(np.sqrt(np.mean(pe**2))),
        "position_mae_m":float(np.mean(pe)),
        "final_position_error_m":float(pe[-1]),
        "heading_mae_deg":float(np.mean(he)),
        "heading_rmse_deg":float(np.sqrt(np.mean(he**2))),
        "final_heading_error_deg":float(he[-1]),
    },p


def anchored_rollouts(seq:Seq,v,w,horizon_s:int,stride_s:float=1.0):
    h=int(round(horizon_s/seq.dt))
    stride=max(1,int(round(stride_s/seq.dt)))
    terr=[]; herr=[]
    for k in range(0,len(v)-h,stride):
        # h+1 states need h controls
        pp=integrate(seq.x[k],seq.y[k],seq.yaw[k],v[k:k+h+1],w[k:k+h+1],seq.dt)
        dx=pp[-1,0]-seq.x[k+h]; dy=pp[-1,1]-seq.y[k+h]
        terr.append(math.hypot(dx,dy))
        herr.append(abs(float(wrap_angle(pp[-1,2]-seq.yaw[k+h])))*180/math.pi)
    a=np.asarray(terr); b=np.asarray(herr)
    return {
        "n_anchors":len(a),
        "terminal_trans_rmse_m":float(np.sqrt(np.mean(a*a))) if len(a) else np.nan,
        "terminal_trans_mae_m":float(np.mean(a)) if len(a) else np.nan,
        "terminal_heading_rmse_deg":float(np.sqrt(np.mean(b*b))) if len(b) else np.nan,
        "terminal_heading_mae_deg":float(np.mean(b)) if len(b) else np.nan,
    }


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--terrasentia-root",type=Path,required=True)
    p.add_argument("--checkpoint-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("terrasentia_diagnostic_results"))
    p.add_argument("--device",default="cuda")
    p.add_argument("--hz",type=float,default=10.)
    p.add_argument("--batch-size",type=int,default=8192)
    p.add_argument("--max-checkpoints",type=int,default=None)
    p.add_argument("--no-plots",action="store_true")
    return p.parse_args()


def main():
    run_start = time.time()
    args=parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    progress("="*88)
    progress("TerraSentia frame/timing + frozen predictive diagnostic STARTED")
    progress("="*88)
    progress(f"TerraSentia root : {args.terrasentia_root.resolve()}")
    progress(f"Checkpoint root  : {args.checkpoint_root.resolve()}")
    progress(f"Output directory : {out}")
    progress(f"Requested device : {args.device}")
    progress(f"Actual device    : {device}")
    if device.type == "cuda":
        progress(f"GPU              : {torch.cuda.get_device_name(0)}")
    progress("Scanning for frozen i2Nav checkpoints...")

    ckpaths=sorted(args.checkpoint_root.rglob("gru_dual_checkpoint.pt"))
    if args.max_checkpoints is not None: ckpaths=ckpaths[:args.max_checkpoints]
    if args.max_checkpoints is None and len(ckpaths)!=30:
        raise RuntimeError(f"Expected 30 checkpoints, found {len(ckpaths)}")
    progress(f"Found {len(ckpaths)} checkpoint files.")
    cks=[]
    for ci,p in enumerate(ckpaths, start=1):
        cks.append(load_ck(p,device))
        if ci == 1 or ci % 5 == 0 or ci == len(ckpaths):
            progress(f"  loaded checkpoint {ci:02d}/{len(ckpaths)}")
    progress("Checkpoint loading complete.")

    meta={
        "official_track_width_m":TRACK_WIDTH_M,
        "R_zedimu_left":R_ZEDIMU_LEFT.tolist(),
        "R_left_robotimu":R_LEFT_ROBOTIMU.tolist(),
        "R_body_robotimu":R_BODY_ROBOTIMU.tolist(),
        "R_body_robotimu_det":float(np.linalg.det(R_BODY_ROBOTIMU)),
        "R_body_robotimu_orthonormal_error":float(np.linalg.norm(R_BODY_ROBOTIMU.T@R_BODY_ROBOTIMU-np.eye(3))),
        "body_convention":"ZED IMU convention x forward, y left, z up per official sensor_parameters.txt",
        "transform_derivation":"R_body_robotimu = R_zedimu_left @ R_left_robotimu",
        "selection_policy":"official calibration only; candidate axis/sign/lag sweep is diagnostic and never used to choose headline preprocessing",
    }
    write_json(out/"official_metadata_transform.json",meta)

    all_audits=[]; lagrows=[]; refrows=[]; wheelrows=[]; pipelines=[]; anchored=[]; ckrows=[]
    primary_pred={}

    modes=["bag","header_relative"]
    prepared={}
    progress("Stage 1/5: preparing TerraSentia streams and running frame/timing audits...")
    prep_total=len(modes)*len(SEQUENCES)
    prep_idx=0
    for mode in modes:
        progress(f"  Time mode: {mode}")
        for name in SEQUENCES:
            prep_idx += 1
            seq_start=time.time()
            progress(f"    [{prep_idx:02d}/{prep_total}] preparing {name}")
            s=prepare_sequence(args.terrasentia_root/name,name,args.hz,mode)
            progress(
                f"        ready: {len(s.grid)} samples, "
                f"{s.audit['duration_s']:.1f}s, "
                f"raw-z corr={s.audit['raw_z_vs_pose_yawrate_corr']:.3f}, "
                f"official-body-z corr={s.audit['body_z_vs_pose_yawrate_corr']:.3f} "
                f"({time.time()-seq_start:.1f}s)"
            )
            prepared[(mode,name)]=s
            a={"sequence":name,"mode":mode,"duration_s":s.audit["duration_s"],"samples":s.audit["samples"]}
            for stream in ("motors","imu","reference"):
                for k,v in s.audit[stream].items(): a[f"{stream}_{k}"]=v
            for k in ["ref_twist_vs_pose_yawrate_corr","ref_twist_vs_pose_yawrate_rmse",
                      "raw_z_vs_pose_yawrate_corr","body_z_vs_pose_yawrate_corr","wheel_yaw_vs_pose_yawrate_corr",
                      "raw_z_vs_ref_twist_corr","body_z_vs_ref_twist_corr","wheel_yaw_vs_ref_twist_corr"]:
                a[k]=s.audit[k]
            all_audits.append(a)

            refrows.append({
                "sequence":name,"mode":mode,
                "ref_twist_vs_pose_corr":corr(s.ref_w_twist,s.ref_w_pose),
                "ref_twist_vs_pose_rmse_radps":rmse(s.ref_w_twist,s.ref_w_pose),
            })
            wheelrows.append({
                "sequence":name,"mode":mode,
                "wheel_yaw_vs_pose_corr":corr(s.omega_wheel,s.ref_w_pose),
                "wheel_yaw_vs_pose_rmse":rmse(s.omega_wheel,s.ref_w_pose),
                "wheel_yaw_vs_ref_twist_corr":corr(s.omega_wheel,s.ref_w_twist),
                "mean_wheel_speed_vs_ref_v_corr":corr(s.v,s.ref_v),
                "mean_wheel_speed_vs_ref_v_rmse":rmse(s.v,s.ref_v),
            })

            candidates={
                "raw_imu_z":s.omega_raw_z,
                "official_body_z":s.omega_body_z,
                "wheel_diff_yaw":s.omega_wheel,
            }
            if np.all(np.isfinite(s.imu_xyz)):
                for j,ax in enumerate("xyz"):
                    candidates[f"imu_{ax}_plus"]=s.imu_xyz[:,j]
                    candidates[f"imu_{ax}_minus"]=-s.imu_xyz[:,j]
            for targetname,target in [("pose_yaw_derivative",s.ref_w_pose),("reference_twist_z",s.ref_w_twist)]:
                for cname,signal in candidates.items():
                    row={"sequence":name,"mode":mode,"target":targetname,"candidate":cname}
                    row.update(lag_diag(signal,target,args.hz,2.0))
                    lagrows.append(row)

    progress("Stage 1/5 complete. Writing diagnostic audit tables...")
    write_rows(out/"time_frame_audit.csv",all_audits)
    write_rows(out/"imu_axis_sign_lag_diagnostic.csv",lagrows)
    write_rows(out/"reference_yaw_consistency.csv",refrows)
    write_rows(out/"wheel_kinematics_diagnostic.csv",wheelrows)

    # Three predeclared pipelines:
    # A legacy=current preprocessing; B metadata transform, same official bag time;
    # C metadata transform + header-relative cadence sensitivity.
    specs=[
        ("legacy_raw_z_bag","bag","raw"),
        ("official_body_z_bag","bag","body"),
        ("official_body_z_header_relative","header_relative","body"),
    ]

    progress("Stage 2/5: evaluating fixed baselines + all frozen checkpoints...")
    pipe_total=len(specs)*len(SEQUENCES)
    pipe_idx=0
    for pi,(label,mode,omkind) in enumerate(specs, start=1):
        progress(f"  Pipeline {pi}/{len(specs)}: {label}")
        for si,name in enumerate(SEQUENCES, start=1):
            pipe_idx += 1
            seq_start=time.time()
            progress(f"    [{pipe_idx:02d}/{pipe_total}] sequence {si}/{len(SEQUENCES)}: {name}")
            s=prepared[(mode,name)]
            om=s.omega_raw_z if omkind=="raw" else s.omega_body_z
            feats=s.features_raw if omkind=="raw" else s.features_body

            # fixed baseline
            gm,p=global_metrics(s,s.v,om)
            row={"pipeline":label,"sequence":name,"method":"fixed_physics",**gm,
                 "yaw_rate_vs_pose_corr":corr(om,s.ref_w_pose),
                 "yaw_rate_vs_ref_twist_corr":corr(om,s.ref_w_twist)}
            pipelines.append(row); primary_pred[(label,name,"fixed")]=p
            progress(
                f"        fixed baseline: ATE={gm['ate_rmse_m']:.3f} m, "
                f"heading MAE={gm['heading_mae_deg']:.2f} deg"
            )
            for H in HORIZONS_S:
                anchored.append({"pipeline":label,"sequence":name,"method":"fixed_physics","horizon_s":H,
                                 **anchored_rollouts(s,s.v,om,H)})

            # wheel-differential baseline is metadata-supported but not the neural-input pipeline.
            gm_w,pw=global_metrics(s,s.v,s.omega_wheel)
            pipelines.append({"pipeline":label,"sequence":name,"method":"wheel_diff_physics",**gm_w,
                              "yaw_rate_vs_pose_corr":corr(s.omega_wheel,s.ref_w_pose),
                              "yaw_rate_vs_ref_twist_corr":corr(s.omega_wheel,s.ref_w_twist)})
            for H in HORIZONS_S:
                anchored.append({"pipeline":label,"sequence":name,"method":"wheel_diff_physics","horizon_s":H,
                                 **anchored_rollouts(s,s.v,s.omega_wheel,H)})

            dynbank=[]
            for ci,ck in enumerate(cks, start=1):
                d=predict(ck,feats,device,args.batch_size); dynbank.append(d)
                vv=s.v+d[:,0]; ww=om+d[:,1]
                gm_ck,_=global_metrics(s,vv,ww)
                rr={"pipeline":label,"sequence":name,"checkpoint":ck.label,**gm_ck}
                for H in (5,10,30):
                    ar=anchored_rollouts(s,vv,ww,H)
                    rr[f"anchored{H}_trans_rmse_m"]=ar["terminal_trans_rmse_m"]
                    rr[f"anchored{H}_heading_rmse_deg"]=ar["terminal_heading_rmse_deg"]
                ckrows.append(rr)
                if ci == 1 or ci % 5 == 0 or ci == len(cks):
                    progress(
                        f"        checkpoint {ci:02d}/{len(cks)}: "
                        f"ATE={gm_ck['ate_rmse_m']:.3f} m, "
                        f"A5={rr['anchored5_trans_rmse_m']:.3f} m, "
                        f"A10={rr['anchored10_trans_rmse_m']:.3f} m"
                    )

            db=np.stack(dynbank,axis=0)
            for ename,dyn in [
                ("mean_checkpoint_ensemble",np.mean(db,axis=0)),
                ("median_checkpoint_ensemble",np.median(db,axis=0)),
            ]:
                vv=s.v+dyn[:,0]; ww=om+dyn[:,1]
                gm_e,pe=global_metrics(s,vv,ww)
                pipelines.append({"pipeline":label,"sequence":name,"method":ename,**gm_e,
                                  "yaw_rate_vs_pose_corr":corr(ww,s.ref_w_pose),
                                  "yaw_rate_vs_ref_twist_corr":corr(ww,s.ref_w_twist)})
                primary_pred[(label,name,ename)]=pe
                for H in HORIZONS_S:
                    anchored.append({"pipeline":label,"sequence":name,"method":ename,"horizon_s":H,
                                     **anchored_rollouts(s,vv,ww,H)})
                if ename == "mean_checkpoint_ensemble":
                    progress(
                        f"        mean ensemble: ATE={gm_e['ate_rmse_m']:.3f} m, "
                        f"heading MAE={gm_e['heading_mae_deg']:.2f} deg"
                    )
            progress(f"        sequence complete in {time.time()-seq_start:.1f}s")

    progress("Stage 2/5 complete. Writing raw pipeline/checkpoint results...")
    write_rows(out/"pipeline_global_metrics.csv",pipelines)
    write_rows(out/"anchored_predictive_rollouts.csv",anchored)
    write_rows(out/"checkpoint_predictive_distribution_raw.csv",ckrows)

    progress("Stage 3/5: aggregating global and anchored predictive metrics...")
    # Aggregate summaries.
    pdf=pd.DataFrame(pipelines); adf=pd.DataFrame(anchored); cdf=pd.DataFrame(ckrows)
    macro=[]
    for (pipe,method),g in pdf.groupby(["pipeline","method"]):
        macro.append({
            "pipeline":pipe,"method":method,"n_sequences":g["sequence"].nunique(),
            "macro_ate_rmse_m":g["ate_rmse_m"].mean(),
            "macro_heading_mae_deg":g["heading_mae_deg"].mean(),
            "macro_yaw_rate_vs_pose_corr":g["yaw_rate_vs_pose_corr"].mean(),
        })
    pd.DataFrame(macro).to_csv(out/"pipeline_macro_global.csv",index=False)

    amacro=[]
    for (pipe,method,H),g in adf.groupby(["pipeline","method","horizon_s"]):
        amacro.append({
            "pipeline":pipe,"method":method,"horizon_s":H,"n_sequences":g["sequence"].nunique(),
            "macro_terminal_trans_rmse_m":g["terminal_trans_rmse_m"].mean(),
            "macro_terminal_heading_rmse_deg":g["terminal_heading_rmse_deg"].mean(),
        })
    am=pd.DataFrame(amacro)
    # Improvement vs fixed for same pipeline/horizon.
    imp=[]
    for pipe in am["pipeline"].unique():
        for H in HORIZONS_S:
            b=am[(am.pipeline==pipe)&(am.method=="fixed_physics")&(am.horizon_s==H)]
            e=am[(am.pipeline==pipe)&(am.method=="mean_checkpoint_ensemble")&(am.horizon_s==H)]
            if len(b) and len(e):
                bv=float(b.iloc[0].macro_terminal_trans_rmse_m); ev=float(e.iloc[0].macro_terminal_trans_rmse_m)
                bh=float(b.iloc[0].macro_terminal_heading_rmse_deg); eh=float(e.iloc[0].macro_terminal_heading_rmse_deg)
                imp.append({
                    "pipeline":pipe,"horizon_s":H,
                    "fixed_trans_rmse_m":bv,"ensemble_trans_rmse_m":ev,
                    "trans_improvement_pct":100*(bv-ev)/bv if bv else np.nan,
                    "fixed_heading_rmse_deg":bh,"ensemble_heading_rmse_deg":eh,
                    "heading_improvement_pct":100*(bh-eh)/bh if bh else np.nan,
                })
    am.to_csv(out/"anchored_predictive_macro.csv",index=False)
    write_rows(out/"anchored_predictive_improvement.csv",imp)

    # Checkpoint distribution as nested training uncertainty.
    drows=[]
    for (pipe,seq),g in cdf.groupby(["pipeline","sequence"]):
        for metric in ["ate_rmse_m","anchored5_trans_rmse_m","anchored10_trans_rmse_m","anchored30_trans_rmse_m"]:
            a=finite(g[metric])
            drows.append({"pipeline":pipe,"sequence":seq,"metric":metric,"n_checkpoints":len(a),
                          "mean":np.mean(a),"std":np.std(a,ddof=1) if len(a)>1 else np.nan,
                          "p05":np.percentile(a,5),"p95":np.percentile(a,95)})
    write_rows(out/"checkpoint_predictive_distribution_summary.csv",drows)
    progress("Stage 3/5 complete.")

    progress("Stage 4/5: computing final diagnostic decision audit...")
    # Decision audit: do not decide using ATE.
    aud=pd.DataFrame(all_audits)
    bag=aud[aud["mode"]=="bag"]
    official_corr=bag["body_z_vs_pose_yawrate_corr"].to_numpy(float)
    raw_corr=bag["raw_z_vs_pose_yawrate_corr"].to_numpy(float)
    ref_corr=bag["ref_twist_vs_pose_yawrate_corr"].to_numpy(float)
    ref_bag_hz=bag["reference_bag_hz"].to_numpy(float)
    ref_header_hz=bag["reference_header_hz"].to_numpy(float)

    frame_resolved=bool(np.sum(official_corr>=0.70)>=4)
    reference_consistent=bool(np.sum(ref_corr>=0.75)>=4)
    batching_suspected=bool(np.sum(ref_bag_hz>500)>=3 and np.sum((ref_header_hz>20)&(ref_header_hz<500))>=3)

    # anchored transfer on metadata+bag pipeline
    idf=pd.DataFrame(imp)
    sub=idf[idf.pipeline=="official_body_z_bag"]
    pred_ok=bool(
        len(sub)>=3
        and np.nanmean(sub[sub.horizon_s.isin([5,10])].trans_improvement_pct)>0
    )

    if frame_resolved and reference_consistent:
        ate_status="reasonable_to_report_with_reference_caveat"
    else:
        ate_status="do_not_headline_yet"

    final={
        "schema":"terrasentia_frame_timing_predictive_diagnostic_v1",
        "official_transform_is_metadata_supported":True,
        "frame_resolved_by_official_transform":frame_resolved,
        "reference_twist_pose_consistent":reference_consistent,
        "reference_bag_timestamp_batching_suspected":batching_suspected,
        "global_ate_status":ate_status,
        "anchored_predictive_transfer_positive_5_10s":pred_ok,
        "mean_raw_z_pose_corr_bag":float(np.nanmean(raw_corr)),
        "mean_official_body_z_pose_corr_bag":float(np.nanmean(official_corr)),
        "mean_reference_twist_pose_corr_bag":float(np.nanmean(ref_corr)),
        "interpretation":[
            "Official sensor transform is applied independently of TerraSentia performance.",
            "Axis/sign/lag sweeps are diagnostic only and must not be used to cherry-pick a headline transform.",
            "If frame/timing remains unresolved, preserve zero-shot predictive-rollout results but do not headline global ATE.",
            "If anchored 5/10s gains persist under the official transform, that directly supports predictive digital-twin fidelity.",
        ],
    }
    write_json(out/"final_diagnostic_verdict.json",final)
    progress("Stage 4/5 complete.")

    progress("Stage 5/5: generating plots..." if not args.no_plots else "Stage 5/5: plots disabled.")
    if not args.no_plots:
        import matplotlib.pyplot as plt
        pd.set_option("display.max_columns",100)
        plotdir=out/"plots"; plotdir.mkdir(exist_ok=True)

        # signal overlays, first 60s
        for name in SEQUENCES:
            s=prepared[("bag",name)]
            n=min(len(s.grid),int(60/ s.dt))
            t=s.grid[:n]-s.grid[0]
            fig=plt.figure(figsize=(10,4))
            plt.plot(t,s.ref_w_pose[:n],label="reference pose derivative")
            plt.plot(t,s.ref_w_twist[:n],label="reference twist z")
            plt.plot(t,s.omega_raw_z[:n],label="raw IMU z")
            plt.plot(t,s.omega_body_z[:n],label="official transformed body z")
            plt.plot(t,s.omega_wheel[:n],label="wheel-diff yaw")
            plt.xlabel("time [s]"); plt.ylabel("yaw rate [rad/s]"); plt.title(name)
            plt.legend(ncol=2); plt.tight_layout()
            fig.savefig(plotdir/f"{name}_yawrate_overlay.png",dpi=180); plt.close(fig)

        # anchored macro
        iim=pd.DataFrame(imp)
        for pipe in iim.pipeline.unique():
            q=iim[iim.pipeline==pipe]
            fig=plt.figure(figsize=(7,4))
            x=np.arange(len(q)); width=.36
            plt.bar(x-width/2,q.fixed_trans_rmse_m,width,label="fixed")
            plt.bar(x+width/2,q.ensemble_trans_rmse_m,width,label="ensemble")
            plt.xticks(x,[str(int(h))+" s" for h in q.horizon_s])
            plt.ylabel("anchored terminal translation RMSE [m]")
            plt.title(pipe); plt.legend(); plt.tight_layout()
            fig.savefig(plotdir/f"{pipe}_anchored_translation.png",dpi=180); plt.close(fig)

        # pipeline macro ATE
        mm=pd.DataFrame(macro)
        q=mm[mm.method.isin(["fixed_physics","mean_checkpoint_ensemble"])]
        piv=q.pivot(index="pipeline",columns="method",values="macro_ate_rmse_m")
        fig=plt.figure(figsize=(8,4))
        x=np.arange(len(piv)); width=.36
        plt.bar(x-width/2,piv["fixed_physics"],width,label="fixed")
        plt.bar(x+width/2,piv["mean_checkpoint_ensemble"],width,label="ensemble")
        plt.xticks(x,piv.index,rotation=20,ha="right"); plt.ylabel("macro ATE RMSE [m]")
        plt.legend(); plt.tight_layout()
        fig.savefig(plotdir/"pipeline_macro_ate.png",dpi=180); plt.close(fig)

    elapsed=time.time()-run_start
    progress("Stage 5/5 complete.")
    print("="*100, flush=True)
    print("DIAGNOSTIC COMPLETE", flush=True)
    print("="*100, flush=True)
    print(json.dumps(final,indent=2), flush=True)
    print("Outputs:",out, flush=True)
    progress(f"Total elapsed time: {elapsed/60.0:.1f} minutes")
    return 0


if __name__=="__main__":
    raise SystemExit(main())