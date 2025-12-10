#!/usr/bin/env python3
"""
Script to extract raw data from Avro files in participant_data folders
and match with timing from all_participants_tasks_updated.xlsx.

Based on Empatica's instructions for reading Embrace Plus Avro files.

Requirements: pip install pandas openpyxl avro-python3
"""

import os
import glob
import sys
import re
from datetime import datetime

try:
    import pandas as pd
    from avro.datafile import DataFileReader
    from avro.io import DatumReader
    import json
except ImportError:
    print("Error: Missing required packages. Please install: pip install pandas openpyxl avro-python3")
    sys.exit(1)

def extract_participant_base_id(participant_id):
    """
    Extract base ID from participant ID.
    Examples: 
      MIT003L-3YK9T1L1P2 -> MIT003
      MIT001 -> MIT001
    """
    if not participant_id or pd.isna(participant_id):
        return None
    
    participant_str = str(participant_id).strip()
    # Extract MITXXX pattern (may or may not have L/R)
    match = re.match(r'([A-Z]+\d+)', participant_str)
    if match:
        return match.group(1)
    return None

def load_excel_data(excel_file):
    """Load Excel file with experiment timing"""
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded Excel file: {excel_file}")
        print(f"  Shape: {df.shape}")
        
        # Find columns
        col_map = {}
        for col in df.columns:
            col_str = str(col).strip().lower()
            if 'participant' in col_str and 'id' in col_str:
                col_map['participant'] = col
            elif 'start time' in col_str and 'local' in col_str:
                col_map['start_time'] = col
            elif 'end time' in col_str and 'local' in col_str:
                col_map['end_time'] = col
            elif 'date' in col_str:
                col_map['date'] = col
        
        return df, col_map
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, None

def find_raw_data_folder(base_dir, date_str, participant_full_id):
    """
    Find the raw_data folder for a specific participant and date.
    Path: data/participant_data/{date}/{participant_full_id}/raw_data/
    """
    # Try exact match first
    exact_path = os.path.join(base_dir, 'data', 'participant_data', date_str, participant_full_id, 'raw_data','v6')
    if os.path.exists(exact_path):
        return exact_path
    
    # Try to find by pattern (in case of slight variations)
    date_path = os.path.join(base_dir, 'data', 'participant_data', date_str)
    if os.path.exists(date_path):
        # Look for folders matching the participant pattern
        base_id = extract_participant_base_id(participant_full_id)
        if base_id:
            # Try both L and R hands
            for hand in ['L', 'R']:
                pattern = f"{base_id}{hand}*"
                matching_dirs = glob.glob(os.path.join(date_path, pattern))
                for dir_path in matching_dirs:
                    raw_data_path = os.path.join(dir_path, 'raw_data','v6')
                    if os.path.exists(raw_data_path):
                        return raw_data_path
    
    return None

def read_avro_file(avro_file_path):
    """
    Read an Avro file and extract sensor data.
    Returns: dict with sensor data arrays
    """
    try:
        reader = DataFileReader(open(avro_file_path, "rb"), DatumReader())
        data = next(reader)
        reader.close()
        
        avro_version = (data['schemaVersion']['major'], 
                       data['schemaVersion']['minor'], 
                       data['schemaVersion']['patch'])
        
        result = {
            'avro_version': avro_version,
            'raw_data': {}
        }
        
        # Extract EDA
        if 'eda' in data['rawData']:
            eda = data['rawData']['eda']
            timestamps = [round(eda['timestampStart'] + i * (1e6 / eda['samplingFrequency']))
                         for i in range(len(eda['values']))]
            result['raw_data']['eda'] = {
                'timestamps': timestamps,
                'values': eda['values']
            }
        
        # Extract Temperature
        if 'temperature' in data['rawData']:
            tmp = data['rawData']['temperature']
            timestamps = [round(tmp['timestampStart'] + i * (1e6 / tmp['samplingFrequency']))
                         for i in range(len(tmp['values']))]
            result['raw_data']['temperature'] = {
                'timestamps': timestamps,
                'values': tmp['values']
            }
        
        # Extract Accelerometer
        if 'accelerometer' in data['rawData']:
            acc = data['rawData']['accelerometer']
            timestamps = [round(acc['timestampStart'] + i * (1e6 / acc['samplingFrequency']))
                         for i in range(len(acc['x']))]
            
            # Convert ADC counts to g
            if avro_version < (6, 5, 0):
                delta_physical = acc['imuParams']['physicalMax'] - acc['imuParams']['physicalMin']
                delta_digital = acc['imuParams']['digitalMax'] - acc['imuParams']['digitalMin']
                x_g = [val * delta_physical / delta_digital for val in acc['x']]
                y_g = [val * delta_physical / delta_digital for val in acc['y']]
                z_g = [val * delta_physical / delta_digital for val in acc['z']]
            else:
                conversion_factor = acc['imuParams']['conversionFactor']
                x_g = [val * conversion_factor for val in acc['x']]
                y_g = [val * conversion_factor for val in acc['y']]
                z_g = [val * conversion_factor for val in acc['z']]
            
            result['raw_data']['accelerometer'] = {
                'timestamps': timestamps,
                'x': x_g,
                'y': y_g,
                'z': z_g
            }
        
        # Extract BVP (for heart rate)
        if 'bvp' in data['rawData']:
            bvp = data['rawData']['bvp']
            timestamps = [round(bvp['timestampStart'] + i * (1e6 / bvp['samplingFrequency']))
                         for i in range(len(bvp['values']))]
            result['raw_data']['bvp'] = {
                'timestamps': timestamps,
                'values': bvp['values']
            }
        
        # Extract Systolic Peaks (for heart rate)
        if 'systolicPeaks' in data['rawData']:
            sps = data['rawData']['systolicPeaks']
            result['raw_data']['systolic_peaks'] = {
                'timestamps': sps['peaksTimeNanos']
            }
        
        return result
        
    except Exception as e:
        print(f"  Error reading Avro file {avro_file_path}: {e}")
        return None

def extract_all_avro_data_from_folder(raw_data_folder):
    """
    Extract data from all Avro files in a folder and combine.
    """
    avro_files = glob.glob(os.path.join(raw_data_folder, '*.avro'))
    
    if not avro_files:
        print(f"  No Avro files found in {raw_data_folder}")
        return None
    
    print(f"  Found {len(avro_files)} Avro files")
    
    # Combine data from all files
    combined_data = {
        'eda': {'timestamps': [], 'values': []},
        'temperature': {'timestamps': [], 'values': []},
        'accelerometer': {'timestamps': [], 'x': [], 'y': [], 'z': []},
        'bvp': {'timestamps': [], 'values': []},
        'systolic_peaks': {'timestamps': []}
    }
    
    for avro_file in sorted(avro_files):
        print(f"  Reading: {os.path.basename(avro_file)}")
        data = read_avro_file(avro_file)
        
        if data and 'raw_data' in data:
            raw_data = data['raw_data']
            
            # Combine EDA
            if 'eda' in raw_data:
                combined_data['eda']['timestamps'].extend(raw_data['eda']['timestamps'])
                combined_data['eda']['values'].extend(raw_data['eda']['values'])
            
            # Combine Temperature
            if 'temperature' in raw_data:
                combined_data['temperature']['timestamps'].extend(raw_data['temperature']['timestamps'])
                combined_data['temperature']['values'].extend(raw_data['temperature']['values'])
            
            # Combine Accelerometer
            if 'accelerometer' in raw_data:
                combined_data['accelerometer']['timestamps'].extend(raw_data['accelerometer']['timestamps'])
                combined_data['accelerometer']['x'].extend(raw_data['accelerometer']['x'])
                combined_data['accelerometer']['y'].extend(raw_data['accelerometer']['y'])
                combined_data['accelerometer']['z'].extend(raw_data['accelerometer']['z'])
            
            # Combine BVP
            if 'bvp' in raw_data:
                combined_data['bvp']['timestamps'].extend(raw_data['bvp']['timestamps'])
                combined_data['bvp']['values'].extend(raw_data['bvp']['values'])
            
            # Combine Systolic Peaks
            if 'systolic_peaks' in raw_data:
                combined_data['systolic_peaks']['timestamps'].extend(raw_data['systolic_peaks']['timestamps'])
    
    return combined_data

def filter_data_by_time(data, start_time_unix, end_time_unix):
    """
    Filter sensor data by experiment start and end times (in microseconds).
    """
    filtered = {}
    
    for sensor, sensor_data in data.items():
        if not sensor_data['timestamps']:
            filtered[sensor] = sensor_data
            continue
        
        timestamps = sensor_data['timestamps']
        
        # Filter indices
        indices = [i for i, ts in enumerate(timestamps) 
                  if start_time_unix <= ts <= end_time_unix]
        
        if not indices:
            filtered[sensor] = {k: [] for k in sensor_data.keys()}
            continue
        
        # Filter data
        filtered_sensor = {'timestamps': [timestamps[i] for i in indices]}
        
        for key, values in sensor_data.items():
            if key != 'timestamps' and values:
                filtered_sensor[key] = [values[i] for i in indices]
        
        filtered[sensor] = filtered_sensor
    
    return filtered

def save_sensor_data_to_csv(data, participant_base_id, hand, output_dir):
    """
    Save extracted sensor data to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save EDA
    if data['eda']['timestamps']:
        df_eda = pd.DataFrame({
            'timestamp_unix_micro': data['eda']['timestamps'],
            'eda_value': data['eda']['values']
        })
        # Convert to seconds and datetime
        df_eda['timestamp_unix'] = df_eda['timestamp_unix_micro'] / 1e6
        df_eda['timestamp_iso'] = pd.to_datetime(df_eda['timestamp_unix'], unit='s', utc=True)
        
        filename = f"{participant_base_id}{hand}_raw_eda.csv"
        df_eda.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename} ({len(df_eda)} rows)")
    
    # Save Temperature
    if data['temperature']['timestamps']:
        df_temp = pd.DataFrame({
            'timestamp_unix_micro': data['temperature']['timestamps'],
            'temperature_celsius': data['temperature']['values']
        })
        df_temp['timestamp_unix'] = df_temp['timestamp_unix_micro'] / 1e6
        df_temp['timestamp_iso'] = pd.to_datetime(df_temp['timestamp_unix'], unit='s', utc=True)
        
        filename = f"{participant_base_id}{hand}_raw_temperature.csv"
        df_temp.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename} ({len(df_temp)} rows)")
    
    # Save Accelerometer
    if data['accelerometer']['timestamps']:
        df_acc = pd.DataFrame({
            'timestamp_unix_micro': data['accelerometer']['timestamps'],
            'acc_x_g': data['accelerometer']['x'],
            'acc_y_g': data['accelerometer']['y'],
            'acc_z_g': data['accelerometer']['z']
        })
        df_acc['timestamp_unix'] = df_acc['timestamp_unix_micro'] / 1e6
        df_acc['timestamp_iso'] = pd.to_datetime(df_acc['timestamp_unix'], unit='s', utc=True)
        
        filename = f"{participant_base_id}{hand}_raw_accelerometer.csv"
        df_acc.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename} ({len(df_acc)} rows)")
    
    # Save BVP
    if data['bvp']['timestamps']:
        df_bvp = pd.DataFrame({
            'timestamp_unix_micro': data['bvp']['timestamps'],
            'bvp_value': data['bvp']['values']
        })
        df_bvp['timestamp_unix'] = df_bvp['timestamp_unix_micro'] / 1e6
        df_bvp['timestamp_iso'] = pd.to_datetime(df_bvp['timestamp_unix'], unit='s', utc=True)
        
        filename = f"{participant_base_id}{hand}_raw_bvp.csv"
        df_bvp.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename} ({len(df_bvp)} rows)")
    
    # Save Systolic Peaks
    if data['systolic_peaks']['timestamps']:
        df_peaks = pd.DataFrame({
            'timestamp_unix_nano': data['systolic_peaks']['timestamps']
        })
        df_peaks['timestamp_unix'] = df_peaks['timestamp_unix_nano'] / 1e9
        df_peaks['timestamp_iso'] = pd.to_datetime(df_peaks['timestamp_unix'], unit='s', utc=True)
        
        filename = f"{participant_base_id}{hand}_raw_systolic_peaks.csv"
        df_peaks.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename} ({len(df_peaks)} rows)")

def main():
    """Main function"""
    print("=" * 80)
    print("Extract Raw Avro Data for Participants")
    print("=" * 80)
    print()
    
    # Paths
    # Go up from scripts/scripts_preprocessing/ to analyses/ to Molly/
    analyses_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.dirname(analyses_dir)  # Go up one more level to get to Molly/
    
    # Use the edited Excel file from Results/Tasks folder
    results_tasks_dir = os.path.join(base_dir, 'Results', 'Tasks')
    excel_file = os.path.join(results_tasks_dir, 'all_participants_tasks_edited_20251120.xlsx')
    
    # If the specific file doesn't exist, try to find the latest edited file
    if not os.path.exists(excel_file):
        # Look for any edited file in Results/Tasks
        edited_files = glob.glob(os.path.join(results_tasks_dir, 'all_participants_tasks_edited_*.xlsx'))
        if edited_files:
            # Sort by modification time and use the latest
            excel_file = max(edited_files, key=os.path.getmtime)
            print(f"Using latest edited file: {os.path.basename(excel_file)}")
        else:
            # Fallback to original file location
            excel_file = os.path.join(base_dir, 'data', 'all_participants_tasks_updated.xlsx')
            print(f"Warning: No edited file found, using: {excel_file}")
    
    # Output directory is in data folder (under Molly/)
    output_dir = os.path.join(base_dir, 'data', 'cleaned_raw_data')
    
    # Load Excel file
    print("Step 1: Loading experiment timing from Excel...")
    df, col_map = load_excel_data(excel_file)
    if df is None:
        return
    
    print(f"  Found columns: {col_map}")
    
    # Process each participant
    print("\nStep 2: Grouping by participant to find full experiment duration...")
    
    # Group by participant to find overall start and end times
    participant_experiments = {}
    
    for idx, row in df.iterrows():
        participant_id = str(row.get(col_map.get('participant'), '')).strip()
        
        # Extract base ID (without hand designation)
        base_id = extract_participant_base_id(participant_id)
        if not base_id:
            continue
        
        # Get experiment times
        start_time = row.get(col_map.get('start_time'))
        end_time = row.get(col_map.get('end_time'))
        
        if pd.isna(start_time) or pd.isna(end_time):
            continue
        
        # Convert to timezone-aware datetime
        start_dt = pd.Timestamp(start_time)
        end_dt = pd.Timestamp(end_time)
        
        # Localize to NY timezone if naive
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize('America/New_York')
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize('America/New_York')
        
        # Track min start and max end for each participant
        if base_id not in participant_experiments:
            participant_experiments[base_id] = {
                'start': start_dt,
                'end': end_dt,
                'all_times': [(start_dt, end_dt)]
            }
        else:
            participant_experiments[base_id]['start'] = min(participant_experiments[base_id]['start'], start_dt)
            participant_experiments[base_id]['end'] = max(participant_experiments[base_id]['end'], end_dt)
            participant_experiments[base_id]['all_times'].append((start_dt, end_dt))
    
    print(f"  Found {len(participant_experiments)} unique participants")
    
    # Process each participant with their full experiment duration
    print("\nStep 3: Processing participants...")
    processed_count = 0
        
    
    for base_id, exp_data in participant_experiments.items():
        start_dt = exp_data['start']
        end_dt = exp_data['end']
        
        # Convert to UTC and then to Unix timestamp in microseconds
        start_unix_micro = int(start_dt.tz_convert('UTC').timestamp() * 1e6)
        end_unix_micro = int(end_dt.tz_convert('UTC').timestamp() * 1e6)
        
        # Extract date from start_time (use NY time for folder matching)
        date_str = start_dt.strftime('%Y-%m-%d')
        
        print(f"\n{base_id}:")
        print(f"  Full experiment: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')} (NY time)")
        print(f"  Duration: {(end_dt - start_dt).total_seconds() / 60:.1f} minutes")
        print(f"  Date folder: {date_str}")
        
        # Find raw_data folder
        date_path = os.path.join(base_dir, 'data', 'participant_data', date_str)
        if not os.path.exists(date_path):
            print(f"  ❌ Date folder not found: {date_path}")
            continue
        
        # Look for participant folders for BOTH left and right hands
        for hand in ['L', 'R']:
            pattern = f"{base_id}{hand}*"
            matching_dirs = glob.glob(os.path.join(date_path, pattern))
            
            if not matching_dirs:
                print(f"  ⚠️  No {hand} hand folder found")
                continue
            
            for participant_dir in matching_dirs:
                participant_full_id = os.path.basename(participant_dir)
                raw_data_folder = os.path.join(participant_dir, 'raw_data', 'v6')
                
                if not os.path.exists(raw_data_folder):
                    print(f"  ⚠️  No raw_data/v6 folder in: {participant_dir}")
                    continue
                
                print(f"  ✓ Found {hand} hand: {participant_full_id}")
                
                # Extract data from ALL Avro files in folder
                print(f"    Reading and combining all Avro files...")
                data = extract_all_avro_data_from_folder(raw_data_folder)
                if not data:
                    print(f"    ❌ No data extracted")
                    continue
                
                # Show data range before filtering
                if data['eda']['timestamps']:
                    first_ts = min(data['eda']['timestamps']) / 1e6
                    last_ts = max(data['eda']['timestamps']) / 1e6
                    first_dt = pd.Timestamp(first_ts, unit='s', tz='UTC').tz_convert('America/New_York')
                    last_dt = pd.Timestamp(last_ts, unit='s', tz='UTC').tz_convert('America/New_York')
                    print(f"    Raw data range: {first_dt.strftime('%Y-%m-%d %H:%M:%S')} to {last_dt.strftime('%Y-%m-%d %H:%M:%S')} (NY time)")
                
                # Filter by FULL experiment time
                print(f"    Filtering to experiment period...")
                filtered_data = filter_data_by_time(data, start_unix_micro, end_unix_micro)
                
                # Show filtered data stats
                total_filtered = sum(len(filtered_data[sensor]['timestamps']) for sensor in filtered_data.keys())
                print(f"    ✓ Filtered data points: {total_filtered:,}")
                
                # Save to CSV
                print(f"    Saving to CSV files...")
                save_sensor_data_to_csv(filtered_data, base_id, hand, output_dir)
                processed_count += 1
    
    print("\n" + "=" * 80)
    print(f"Done! Processed {processed_count} participants")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

