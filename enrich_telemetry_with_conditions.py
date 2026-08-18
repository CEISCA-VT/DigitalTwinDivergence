#!/usr/bin/env python3
"""
Enrich the 20 formal benign rover telemetry logs with operating-condition features.

Features to derive:
- v_k: estimated forward speed (m/s)
- omega_k: turn rate (rad/s)  
- delta_v_k: track-speed asymmetry magnitude (m/s)
- V_k: IMU vibration measure (RMS acceleration)
- surface: smooth vs rough
- speed: low vs medium
- turn_type: straight / gentle_turn / sharp_turn
- vibration_level: low / high
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

def get_benign_logs():
    """Return list of formal benign CSV files."""
    raw_logs_path = Path("raw_logs/telemetry")
    logs = sorted(raw_logs_path.glob("speed-*_attack-none_trial-*.csv"))
    
    # Filter to formal logs only (named with surface/speed in proper format)
    formal_logs = [
        log for log in logs
        if ('kitchen_floor' in log.name or 'concrete' in log.name) and
           'attack-none' in log.name
    ]
    
    return formal_logs[:20]  # Take only 20 formal logs

def load_telemetry(log_file):
    """Load a telemetry log."""
    return pd.read_csv(log_file)

def compute_kinematics(df):
    """
    Compute kinematic features from encoder and IMU data.
    
    Returns:
        DataFrame with new columns
    """
    df = df.copy()
    
    # Constants (from motion model)
    DRIVE_DIAMETER_M = 0.0523
    ENCODER_COUNTS_PER_REV = 1092
    TRACK_WIDTH_M = 0.141  # effective track width for turns
    
    METERS_PER_COUNT = (np.pi * DRIVE_DIAMETER_M) / ENCODER_COUNTS_PER_REV
    
    # Compute delta encoder ticks (differences)
    df['enc_left_delta'] = df['enc_left'].diff().fillna(0)
    df['enc_right_delta'] = df['enc_right'].diff().fillna(0)
    
    # Compute meters traveled per track
    df['dist_left_m'] = df['enc_left_delta'] * METERS_PER_COUNT
    df['dist_right_m'] = df['enc_right_delta'] * METERS_PER_COUNT
    
    # Mean distance (forward motion)
    df['dist_straight_m'] = (df['dist_left_m'] + df['dist_right_m']) / 2.0
    
    # Asymmetry (differential distance -> turn)
    df['dist_diff_m'] = df['dist_right_m'] - df['dist_left_m']
    
    # Sample period (assuming 10 Hz)
    dt = 0.1  # seconds
    
    # Forward speed (m/s)
    df['v_forward_mps'] = df['dist_straight_m'] / dt
    
    # Turn rate (rad/s)
    # omega = (dist_right - dist_left) / track_width / dt
    df['omega_rad_s'] = df['dist_diff_m'] / TRACK_WIDTH_M / dt
    
    # Track-speed asymmetry magnitude
    df['delta_v_asymmetry'] = np.abs(df['dist_right_m'] - df['dist_left_m']) / dt
    
    return df

def compute_vibration(df, window_size=5):
    """
    Compute IMU vibration measure as RMS of vertical acceleration anomaly.
    
    V_k = RMS(a_z - mean(a_z)) over short window
    """
    df = df.copy()
    
    # High-pass filter to remove bias
    az_filtered = df['az'].rolling(window=window_size, center=True).mean()
    az_anomaly = (df['az'] - az_filtered).fillna(0)
    
    # RMS over window
    df['vibration_rms'] = az_anomaly.rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    ).fillna(0)
    
    return df

def label_operating_conditions(df, log_name):
    """
    Label operating conditions based on filename and derived features.
    """
    df = df.copy()
    
    # Parse surface from filename
    if 'smooth_kitchen_floor' in log_name:
        df['surface'] = 'SMOOTH'
    elif 'rough_permeable_concrete' in log_name:
        df['surface'] = 'ROUGH'
    else:
        df['surface'] = 'UNKNOWN'
    
    # Parse speed from filename
    if 'speed-low' in log_name:
        df['speed_setting'] = 'LOW'
    elif 'speed-medium' in log_name:
        df['speed_setting'] = 'MEDIUM'
    else:
        df['speed_setting'] = 'UNKNOWN'
    
    # Classify turn type based on turn rate
    df['turn_type'] = df['omega_rad_s'].apply(lambda x: classify_turn(x))
    
    # Classify vibration level based on percentile
    vibration_threshold = df['vibration_rms'].quantile(0.5)  # median
    df['vibration_level'] = df['vibration_rms'].apply(
        lambda x: 'HIGH' if x > vibration_threshold else 'LOW'
    )
    
    return df

def classify_turn(omega):
    """Classify turn rate into turn type."""
    abs_omega = abs(omega)
    if abs_omega < 0.1:  # < 0.1 rad/s
        return 'STRAIGHT'
    elif abs_omega < 0.5:  # 0.1-0.5 rad/s
        return 'GENTLE_TURN'
    else:  # > 0.5 rad/s
        return 'SHARP_TURN'

def enrich_telemetry():
    """Main function to enrich all telemetry with operating conditions."""
    
    logs = get_benign_logs()
    print(f"Processing {len(logs)} formal benign logs...")
    print("=" * 100)
    
    all_enriched_logs = []
    
    for i, log_file in enumerate(logs, 1):
        log_name = log_file.name
        print(f"\n[{i}/{len(logs)}] {log_name}")
        
        # Load
        df = load_telemetry(log_file)
        
        # Enrich
        df = compute_kinematics(df)
        df = compute_vibration(df, window_size=5)
        df = label_operating_conditions(df, log_name)
        
        # Select and reorder columns
        result_cols = [
            # Original timing
            't_wall_unix_s', 'http_latency_ms',
            # Motor commands
            'L', 'R', 'millis',
            # Encoder data
            'enc_left', 'enc_right',
            # New kinematics
            'v_forward_mps', 'omega_rad_s', 'delta_v_asymmetry',
            # IMU
            'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'temp',
            # Vibration
            'vibration_rms',
            # GPS
            'gps_valid', 'lat', 'lon', 'alt_m', 'speed_mps',
            # Operating conditions
            'surface', 'speed_setting', 'turn_type', 'vibration_level'
        ]
        
        # Keep only available columns
        available_cols = [c for c in result_cols if c in df.columns]
        df_result = df[available_cols].copy()
        
        # Add metadata
        df_result['log_file'] = log_name
        df_result['surface'] = df['surface'].iloc[0]
        df_result['speed_setting'] = df['speed_setting'].iloc[0]
        
        all_enriched_logs.append(df_result)
        
        # Print statistics for this log
        print(f"  Samples: {len(df)}")
        print(f"  Surface: {df['surface'].iloc[0]}, Speed: {df['speed_setting'].iloc[0]}")
        print(f"  Speed range: {df['v_forward_mps'].min():.3f} to {df['v_forward_mps'].max():.3f} m/s")
        print(f"  Turn rate range: {df['omega_rad_s'].min():.3f} to {df['omega_rad_s'].max():.3f} rad/s")
        print(f"  Vibration range: {df['vibration_rms'].min():.3f} to {df['vibration_rms'].max():.3f}")
        print(f"  Turn types: {df['turn_type'].value_counts().to_dict()}")
    
    # Combine all logs
    print("\n" + "=" * 100)
    print("Combining all logs into single enriched dataset...")
    
    enriched_df = pd.concat(all_enriched_logs, ignore_index=True)
    
    # Save
    output_file = Path("enriched_telemetry_20runs.csv")
    enriched_df.to_csv(output_file, index=False)
    
    print(f"✓ Saved enriched telemetry to {output_file}")
    print(f"  Total rows: {len(enriched_df)}")
    print(f"  Total runs: {enriched_df['log_file'].nunique()}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    print(f"\nSurface breakdown:")
    print(enriched_df['surface'].value_counts())
    
    print(f"\nSpeed breakdown:")
    print(enriched_df['speed_setting'].value_counts())
    
    print(f"\nTurn type breakdown:")
    print(enriched_df['turn_type'].value_counts())
    
    print(f"\nVibration level breakdown:")
    print(enriched_df['vibration_level'].value_counts())
    
    print(f"\nForward speed statistics (m/s):")
    print(enriched_df['v_forward_mps'].describe())
    
    print(f"\nTurn rate statistics (rad/s):")
    print(enriched_df['omega_rad_s'].describe())
    
    print(f"\nVibration statistics:")
    print(enriched_df['vibration_rms'].describe())
    
    return enriched_df

if __name__ == "__main__":
    enrich_telemetry()
