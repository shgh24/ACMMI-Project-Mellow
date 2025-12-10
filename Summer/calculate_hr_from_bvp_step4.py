#!/usr/bin/env python3
"""
Calculate IBI (Inter-Beat Interval) and HR (Heart Rate) from BVP data.

Uses signal processing to:
1. Filter BVP signal (0.5-5 Hz bandpass)
2. Detect peaks
3. Calculate IBI from peak intervals
4. Calculate instantaneous and smoothed HR

Requirements: pip install pandas numpy scipy
"""

import os
import glob
import sys
import re

try:
    import pandas as pd
    import numpy as np
    from scipy import signal
except ImportError as e:
    print("Error: Missing required packages. Please install: pip install pandas numpy scipy")
    print(f"Details: {e}")
    sys.exit(1)

# Configuration
RAW_DATA_DIR = "/Users/summerghorbani/Documents/MIT_projects/Molly/data/cleaned_raw_data"
OUTPUT_DIR = "/Users/summerghorbani/Documents/MIT_projects/Molly/data/cleaned_raw_data"

def calculate_hr_from_bvp(bvp_filepath, output_filepath):
    """
    Process BVP data to calculate IBI and HR.
    
    Args:
        bvp_filepath: Path to BVP CSV file
        output_filepath: Path to save HR CSV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load BVP data
        df_bvp = pd.read_csv(bvp_filepath)
        
        if df_bvp.empty:
            print(f"    Warning: BVP file is empty")
            return False
        
        # Check required columns
        if 'timestamp_unix' not in df_bvp.columns or 'bvp_value' not in df_bvp.columns:
            print(f"    Warning: Missing required columns (timestamp_unix, bvp_value)")
            return False
        
        # Sort by timestamp
        df_bvp = df_bvp.sort_values('timestamp_unix').reset_index(drop=True)
        
        # Extract time and BVP values
        t = df_bvp['timestamp_unix'].values  # Unix timestamp in seconds
        bvp = df_bvp['bvp_value'].values
        
        if len(t) < 100:
            print(f"    Warning: Not enough BVP samples ({len(t)})")
            return False
        
        # Calculate sampling frequency
        time_diffs = np.diff(t)
        median_diff = np.median(time_diffs)
        fs = 1.0 / median_diff  # Sampling frequency in Hz
        
        print(f"    Sampling frequency: {fs:.2f} Hz")
        print(f"    BVP samples: {len(bvp):,}")
        
        # Bandpass filter (0.5–5 Hz typical for BVP)
        try:
            b, a = signal.butter(3, [0.5/(fs/2), 5/(fs/2)], btype='band')
            bvp_filt = signal.filtfilt(b, a, bvp)
        except Exception as e:
            print(f"    Warning: Could not filter BVP signal: {e}")
            bvp_filt = bvp
        
        # Peak detection
        # At least 0.4s between beats (~150 BPM max)
        min_distance = int(fs * 0.4)
        peaks, _ = signal.find_peaks(bvp_filt, distance=min_distance)
        
        print(f"    Detected peaks: {len(peaks)}")
        
        if len(peaks) < 2:
            print(f"    Warning: Not enough peaks detected")
            return False
        
        # Compute IBI (Inter-Beat Interval in seconds)
        ibi = np.diff(t[peaks])
        
        # Instantaneous HR (BPM)
        hr_inst = 60.0 / ibi
        
        # Filter outliers (HR between 40-200 BPM)
        valid_mask = (hr_inst >= 40) & (hr_inst <= 200)
        hr_inst_filtered = hr_inst[valid_mask]
        ibi_filtered = ibi[valid_mask]
        
        # Create HR time series aligned with midpoint between beats
        hr_time = t[peaks][1:]  # Time of each IBI (start of interval)
        hr_time_filtered = hr_time[valid_mask]
        
        print(f"    Valid HR samples: {len(hr_inst_filtered):,} (filtered from {len(hr_inst)})")
        
        if len(hr_inst_filtered) < 2:
            print(f"    Warning: Not enough valid HR samples after filtering")
            return False
        
        # Smooth HR using a rolling mean over 30s window (with 1s step)
        window_size = 30  # seconds
        step_size = 1  # seconds
        
        hr_smooth_time = np.arange(hr_time_filtered[0], hr_time_filtered[-1], step_size)
        hr_smooth = []
        
        for t0 in hr_smooth_time:
            # Find HR values within window [t0, t0 + window_size]
            in_window = (hr_time_filtered >= t0) & (hr_time_filtered < t0 + window_size)
            if np.any(in_window):
                hr_smooth.append(np.mean(hr_inst_filtered[in_window]))
            else:
                hr_smooth.append(np.nan)
        
        hr_smooth = np.array(hr_smooth)
        
        print(f"    Smoothed HR samples: {len(hr_smooth):,}")
        
        # Create output DataFrames
        # 1. Instantaneous HR (one value per IBI)
        df_hr_inst = pd.DataFrame({
            'timestamp_unix': hr_time_filtered,
            'ibi_seconds': ibi_filtered,
            'hr_bpm': hr_inst_filtered
        })
        df_hr_inst['timestamp_iso'] = pd.to_datetime(df_hr_inst['timestamp_unix'], unit='s', utc=True)
        df_hr_inst['timestamp_iso_ny'] = df_hr_inst['timestamp_iso'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
        # 2. Smoothed HR (1Hz sampling)
        df_hr_smooth = pd.DataFrame({
            'timestamp_unix': hr_smooth_time,
            'hr_bpm_smoothed': hr_smooth
        })
        df_hr_smooth['timestamp_iso'] = pd.to_datetime(df_hr_smooth['timestamp_unix'], unit='s', utc=True)
        df_hr_smooth['timestamp_iso_ny'] = df_hr_smooth['timestamp_iso'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
        # Remove NaN values from smoothed data
        df_hr_smooth = df_hr_smooth.dropna(subset=['hr_bpm_smoothed'])
        
        # Save both versions
        # Instantaneous HR
        inst_output = output_filepath.replace('.csv', '_instantaneous.csv')
        df_hr_inst.to_csv(inst_output, index=False)
        print(f"    ✓ Saved instantaneous HR: {os.path.basename(inst_output)} ({len(df_hr_inst)} samples)")
        
        # Smoothed HR
        smooth_output = output_filepath
        df_hr_smooth.to_csv(smooth_output, index=False)
        print(f"    ✓ Saved smoothed HR: {os.path.basename(smooth_output)} ({len(df_hr_smooth)} samples)")
        
        return True
        
    except Exception as e:
        print(f"    Error processing BVP: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_bvp_files(raw_data_dir):
    """Find all BVP files in the raw data directory."""
    pattern = os.path.join(raw_data_dir, "*_raw_bvp.csv")
    bvp_files = glob.glob(pattern)
    return sorted(bvp_files)

def extract_participant_info(filename):
    """Extract participant base ID and hand from filename."""
    basename = os.path.basename(filename)
    # Pattern: MIT001L_raw_bvp.csv
    match = re.match(r'^([A-Z]+\d+)([LR])_raw_bvp\.csv', basename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def main():
    """Main function."""
    print("=" * 80)
    print("Calculate HR from BVP Data")
    print("=" * 80)
    print()
    
    # Find all BVP files
    print(f"Scanning {RAW_DATA_DIR} for BVP files...")
    bvp_files = find_bvp_files(RAW_DATA_DIR)
    
    if not bvp_files:
        print("No BVP files found. Please run extract_raw_avro_data.py first.")
        return
    
    print(f"Found {len(bvp_files)} BVP files")
    print()
    
    # Process each BVP file
    successful = 0
    failed = 0
    
    for bvp_file in bvp_files:
        base_id, hand = extract_participant_info(bvp_file)
        if not base_id or not hand:
            print(f"Warning: Could not extract info from {os.path.basename(bvp_file)}")
            failed += 1
            continue
        
        print(f"\n{base_id}{hand}:")
        print(f"  Processing: {os.path.basename(bvp_file)}")
        
        # Output filename
        output_file = os.path.join(OUTPUT_DIR, f"{base_id}{hand}_raw_hr.csv")
        
        # Calculate HR
        success = calculate_hr_from_bvp(bvp_file, output_file)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Done! Processed {successful} files successfully, {failed} failed")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

