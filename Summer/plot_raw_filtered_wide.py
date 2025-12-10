#!/usr/bin/env python3
"""
Plot raw participant data with filtered EDA in WIDE format for better visibility.
- Wider figure size for better signal detail
- Applies Butterworth low-pass filter (0.6 Hz) to EDA
- Creates subplot layout: EDA, Accelerometer, HR, Temperature
- Adds task background colors and annotations
- Saves high-quality PNG files

Based on plot_cleaned_data.py structure
"""

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    from scipy.signal import butter, filtfilt
except ImportError as e:
    print(f"Error: Missing required package. Please install: pip install pandas matplotlib openpyxl numpy scipy")
    print(f"Details: {e}")
    exit(1)

import os
import glob
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_DIR = "/Users/summerghorbani/Documents/MIT_projects/Molly/data/cleaned_raw_data"
# EXCEL_FILE = "/Users/summerghorbani/Documents/MIT_projects/Molly/data/all_participants_tasks_updated.xlsx"
# OUTPUT_DIR = "/Users/summerghorbani/Documents/MIT_projects/Molly/analyses/plots_raw_wide"
EXCEL_FILE ="/Users/summerghorbani/Documents/MIT_projects/Molly/Results/Tasks/all_participants_tasks_edited_20251120.xlsx"

OUTPUT_DIR = "/Users/summerghorbani/Documents/MIT_projects/Molly/Results/plots_raw_wide"

# Filter parameters
EDA_SAMPLING_RATE = 4       # Hz
EDA_CUTOFF_FREQ = 2.0       # Hz cutoff for low-pass filter
FILTER_ORDER = 2            # Butterworth filter order

def butter_lowpass(cutoff, fs, order=2):
    """Design low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Ensure normalized cutoff is < 1.0 (required by scipy)
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.95  # Cap at 95% of Nyquist
        print(f"  Warning: Cutoff frequency {cutoff} Hz exceeds Nyquist ({nyquist} Hz). Using {normal_cutoff * nyquist:.2f} Hz instead.")
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """Apply low-pass Butterworth filter."""
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

def convert_to_naive_ny_time(dt):
    """Convert timezone-aware NY time to naive datetime."""
    if pd.isna(dt):
        return dt
    if hasattr(dt, 'tz') and dt.tz is not None:
        if str(dt.tz) != 'America/New_York':
            dt_ny = dt.tz_convert('America/New_York')
        else:
            dt_ny = dt
        return dt_ny.tz_localize(None)
    return dt

def load_excel_for_tasks(excel_file):
    """Load Excel file to get task information."""
    try:
        df = pd.read_excel(excel_file)
        
        # Find relevant columns
        participant_col = None
        task_name_col = None
        task_type_col = None
        start_time_col = None
        end_time_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'participant' in col_lower and ('id' in col_lower or 'identifier' in col_lower):
                participant_col = col
            if 'task' in col_lower and 'name' in col_lower:
                task_name_col = col
            if 'task' in col_lower and 'type' in col_lower:
                task_type_col = col
            if 'start' in col_lower and 'time' in col_lower:
                start_time_col = col
            if 'end' in col_lower and 'time' in col_lower:
                end_time_col = col
        
        if not participant_col and len(df.columns) > 0:
            participant_col = df.columns[0]
        
        return df, participant_col, task_name_col, task_type_col, start_time_col, end_time_col
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, None, None, None, None, None

def get_task_phases_from_excel(excel_df, participant_col, task_name_col, task_type_col,
                               start_time_col, end_time_col, participant_base_id):
    """Get task phases from Excel file with NY time."""
    phase_colors = {
        "Pre-Session": "#A09B90",
        "prestudy relaxation": "#83CEF7",
        "Descriptive_Stress": "#E74040",
        "Stroop_Stress": "#BF3434",
        "Math_Stress": "#922A2A",
        "content performance task ": "#68D852",
        "Post-study Relaxation": "#A54DF7",
    }
    
    phases = []
    if excel_df is None or excel_df.empty:
        return phases
    
    participant_tasks = excel_df[excel_df[participant_col].str.contains(participant_base_id, case=False, na=False)]
    
    for idx, task_row in participant_tasks.iterrows():
        task_name = str(task_row.get(task_name_col, '')).strip() if task_name_col else ''
        task_start = pd.to_datetime(task_row.get(start_time_col, None), errors='coerce') if start_time_col else None
        task_end = pd.to_datetime(task_row.get(end_time_col, None), errors='coerce') if end_time_col else None
        
        # Convert to NY time (naive)
        if pd.notna(task_start):
            if task_start.tz is None:
                task_start_ny = task_start.tz_localize('America/New_York')
            else:
                task_start_ny = task_start.tz_convert('America/New_York')
            task_start_ny = convert_to_naive_ny_time(task_start_ny)
        else:
            task_start_ny = None
        
        if pd.notna(task_end):
            if task_end.tz is None:
                task_end_ny = task_end.tz_localize('America/New_York')
            else:
                task_end_ny = task_end.tz_convert('America/New_York')
            task_end_ny = convert_to_naive_ny_time(task_end_ny)
        else:
            task_end_ny = None
        
        # Find matching color
        task_name_lower = task_name.lower().strip()
        color = "#CCCCCC"
        
        if 'pre' in task_name_lower and ('session' in task_name_lower or 'study' not in task_name_lower):
            color = phase_colors.get("Pre-Session", "#A09B90")
        elif 'pre' in task_name_lower and ('relax' in task_name_lower or 'study' in task_name_lower):
            color = phase_colors.get("prestudy relaxation", "#83CEF7")
        elif 'descriptive' in task_name_lower:
            color = phase_colors.get("Descriptive_Stress", "#E74040")
        elif 'stroop' in task_name_lower:
            color = phase_colors.get("Stroop_Stress", "#BF3434")
        elif 'math' in task_name_lower:
            color = phase_colors.get("Math_Stress", "#922A2A")
        elif 'content' in task_name_lower or ('performance' in task_name_lower and 'task' in task_name_lower):
            color = phase_colors.get("content performance task ", "#68D852")
        elif 'post' in task_name_lower and ('relax' in task_name_lower or 'study' in task_name_lower):
            color = phase_colors.get("Post-study Relaxation", "#A54DF7")
        
        # Extract Task Type
        task_type = None
        if task_type_col and ('content' in task_name_lower or ('performance' in task_name_lower and 'task' in task_name_lower)):
            task_type = str(task_row.get(task_type_col, '')).strip()
            if task_type == 'nan' or task_type == '':
                task_type = None
            else:
                task_type_lower = task_type.lower()
                if 'video' in task_type_lower and ('anxiety' in task_type_lower or 'anxious' in task_type_lower):
                    task_type = 'mindfulness'
        
        if pd.notna(task_start_ny) and pd.notna(task_end_ny):
            phases.append({
                'name': task_name,
                'start': task_start_ny,
                'end': task_end_ny,
                'color': color,
                'task_type': task_type
            })
    
    return phases

def load_raw_data(raw_data_dir, participant_base_id):
    """Load raw data files for a participant."""
    data = {'L': {}, 'R': {}}
    
    for hand in ['L', 'R']:
        # Try calculated HR first, fall back to BVP
        hr_file = f"{participant_base_id}{hand}_raw_hr.csv"
        bvp_file = f"{participant_base_id}{hand}_raw_bvp.csv"
        
        sensor_files = {
            'eda': f"{participant_base_id}{hand}_raw_eda.csv",
            'accelerometer': f"{participant_base_id}{hand}_raw_accelerometer.csv",
            'temperature': f"{participant_base_id}{hand}_raw_temperature.csv",
            'hr': hr_file if os.path.exists(os.path.join(raw_data_dir, hr_file)) else bvp_file
        }
        
        for sensor, filename in sensor_files.items():
            filepath = os.path.join(raw_data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # Convert timestamp to NY time
                    if 'timestamp_iso' in df.columns:
                        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce')
                        if df['timestamp_iso'].dt.tz is not None:
                            df['timestamp_iso_ny'] = df['timestamp_iso'].dt.tz_convert('America/New_York').dt.tz_localize(None)
                        else:
                            df['timestamp_iso_ny'] = df['timestamp_iso'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
                    elif 'timestamp_iso_ny' in df.columns:
                        df['timestamp_iso_ny'] = pd.to_datetime(df['timestamp_iso_ny'], errors='coerce')
                    
                    data[hand][sensor] = df
                except Exception as e:
                    print(f"  Warning: Could not load {filename}: {e}")
                    data[hand][sensor] = pd.DataFrame()
            else:
                data[hand][sensor] = pd.DataFrame()
    
    return data

def calculate_accelerometer_magnitude(df):
    """Calculate accelerometer magnitude from x, y, z."""
    if all(col in df.columns for col in ['acc_x_g', 'acc_y_g', 'acc_z_g']):
        df['acc_magnitude'] = np.sqrt(df['acc_x_g']**2 + df['acc_y_g']**2 + df['acc_z_g']**2)
    return df

def apply_eda_filter(df, fs=4, cutoff=0.6, order=2):
    """Apply Butterworth low-pass filter to EDA signal."""
    if 'eda_value' in df.columns and len(df) > 10:
        eda_filtered = butter_lowpass_filter(df['eda_value'].values, cutoff, fs, order)
        df['eda_filtered'] = eda_filtered
    return df

def estimate_sampling_frequency(timestamps):
    """Estimate sampling frequency from timestamps."""
    time_diffs = np.diff(timestamps)
    median_diff = np.median(time_diffs[time_diffs > 0])
    fs = 1.0 / median_diff if median_diff > 0 else 4.0
    return fs

def plot_participant_data(participant_base_id, data, task_phases, output_dir):
    """Plot participant raw data with filtered EDA in WIDE format."""
    # Check if we have any data
    has_data = any(not data[hand][sensor].empty 
                   for hand in ['L', 'R'] 
                   for sensor in data[hand].keys())
    
    if not has_data:
        print(f"  Warning: No data found for participant {participant_base_id}")
        return
    
    # Pre-process data
    for hand in ['L', 'R']:
        # Note: Accelerometer magnitude calculation removed - we plot X, Y, Z separately
        
        # Apply EDA filter
        if 'eda' in data[hand] and not data[hand]['eda'].empty:
            fs = estimate_sampling_frequency(data[hand]['eda']['timestamp_unix'].values)
            data[hand]['eda'] = apply_eda_filter(data[hand]['eda'], fs=fs, 
                                                 cutoff=EDA_CUTOFF_FREQ, order=FILTER_ORDER)
    
    # Create MUCH WIDER and TALLER figure for Google Slides
    fig, axes = plt.subplots(4, 2, figsize=(48, 27))  # 48×27 inches - 16:9 ratio for slides
    fig.suptitle(f'Participant {participant_base_id} - Raw Data (Filtered EDA)', 
                 fontsize=26, fontweight='bold', y=0.996)
    
    # Plot configuration
    plot_config = {
        'eda': {
            'column': 'eda_filtered',
            'ylabel': 'EDA (µS) - Filtered',
            'color': 'blue',
            'row': 0
        },
        'accelerometer': {
            'columns': ['acc_x_g', 'acc_y_g', 'acc_z_g'],  # Plot X, Y, Z separately
            'ylabel': 'Accelerometer (g)',
            'colors': ['red', 'green', 'blue'],  # X=red, Y=green, Z=blue
            'labels': ['X', 'Y', 'Z'],
            'row': 1
        },
        'hr': {
            'column': 'hr_bpm_smoothed',
            'column_alt': 'hr_bpm',
            'column_fallback': 'bvp_value',
            'ylabel': 'Heart Rate (BPM)',
            'ylabel_fallback': 'BVP',
            'color': 'red',
            'row': 2
        },
        'temperature': {
            'column': 'temperature_celsius',
            'ylabel': 'Temperature (°C)',
            'color': 'orange',
            'row': 3
        }
    }
    
    # Collect phase labels for legend
    phase_labels = []
    for phase in task_phases:
        label = phase['name']
        if 'content' in label.lower() or ('performance' in label.lower() and 'task' in label.lower()):
            label = "Molly App Intervention"
            if phase.get('task_type') and phase['task_type'] != 'nan' and phase['task_type'] != '':
                label = f"Molly App Intervention ({phase['task_type']})"
        if label not in [p['name'] for p in phase_labels]:
            phase_labels.append({'name': label, 'color': phase['color']})
    
    # Collect all times and data for axis limits
    all_x_data_ny = []
    all_y_values = {metric: [] for metric in plot_config.keys()}
    
    for metric, config in plot_config.items():
        # Special handling for accelerometer (X, Y, Z)
        if metric == 'accelerometer' and 'columns' in config:
            col_names = config['columns']
        else:
            col_names = [config.get('column')]
        
        for hand in ['L', 'R']:
            if metric not in data[hand] or data[hand][metric].empty:
                continue
            
            df = data[hand][metric].copy()
            
            # For accelerometer, collect all X, Y, Z values
            if metric == 'accelerometer' and 'columns' in config:
                for col_name in col_names:
                    if col_name in df.columns and 'timestamp_iso_ny' in df.columns:
                        valid_mask = pd.notna(df[col_name]) & pd.notna(df['timestamp_iso_ny'])
                        if valid_mask.any():
                            df_valid = df[valid_mask]
                            all_x_data_ny.extend(df_valid['timestamp_iso_ny'].tolist())
                            all_y_values[metric].extend(df_valid[col_name].tolist())
            else:
                # For other metrics, use standard column selection
                col_name = config.get('column')
                plot_col = None
                if col_name in df.columns:
                    plot_col = col_name
                elif config.get('column_alt') and config['column_alt'] in df.columns:
                    plot_col = config['column_alt']
                elif config.get('column_fallback') and config['column_fallback'] in df.columns:
                    plot_col = config['column_fallback']
                
                if plot_col and 'timestamp_iso_ny' in df.columns:
                    valid_mask = pd.notna(df[plot_col]) & pd.notna(df['timestamp_iso_ny'])
                    if valid_mask.any():
                        df_valid = df[valid_mask]
                        all_x_data_ny.extend(df_valid['timestamp_iso_ny'].tolist())
                        all_y_values[metric].extend(df_valid[plot_col].tolist())
    
    # Include task phase times in X-axis limits
    all_task_times_ny = []
    for phase in task_phases:
        if pd.notna(phase['start']):
            all_task_times_ny.append(phase['start'])
        if pd.notna(phase['end']):
            all_task_times_ny.append(phase['end'])
    
    all_x_times_ny = all_x_data_ny + all_task_times_ny
    
    # Determine X and Y limits
    if all_x_times_ny:
        x_min_ny = min(all_x_times_ny)
        x_max_ny = max(all_x_times_ny)
    else:
        x_min_ny = x_max_ny = None
    
    y_limits = {}
    y_ticks = {}
    for metric in plot_config.keys():
        if all_y_values[metric]:
            y_min = min(all_y_values[metric])
            y_max = max(all_y_values[metric])
            y_range = y_max - y_min
            if y_range > 0:
                y_limits[metric] = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
                y_ticks[metric] = np.linspace(y_limits[metric][0], y_limits[metric][1], 8)  # More ticks
            else:
                y_limits[metric] = (y_min - 1, y_max + 1)
                y_ticks[metric] = np.linspace(y_min - 1, y_max + 1, 8)
        else:
            y_limits[metric] = None
            y_ticks[metric] = None
    
    # Plot data
    for metric, config in plot_config.items():
        row = config['row']
        
        # Left hand
        ax_left = axes[row, 0]
        if metric in data['L'] and not data['L'][metric].empty:
            df_l = data['L'][metric].copy()
            
            # Special handling for accelerometer (plot X, Y, Z separately)
            if metric == 'accelerometer' and 'columns' in config:
                for col_name, color, label in zip(config['columns'], config['colors'], config['labels']):
                    if col_name in df_l.columns and 'timestamp_iso_ny' in df_l.columns:
                        valid_mask = pd.notna(df_l[col_name]) & pd.notna(df_l['timestamp_iso_ny'])
                        if valid_mask.any():
                            df_l_valid = df_l[valid_mask]
                            ax_left.plot(df_l_valid['timestamp_iso_ny'], df_l_valid[col_name],
                                       color=color, linewidth=2.5, zorder=2, alpha=0.9, label=label)
            else:
                # Standard plotting for other metrics
                col_name = config.get('column')
                plot_col = None
                if col_name in df_l.columns:
                    plot_col = col_name
                elif config.get('column_alt') and config['column_alt'] in df_l.columns:
                    plot_col = config['column_alt']
                elif config.get('column_fallback') and config['column_fallback'] in df_l.columns:
                    plot_col = config['column_fallback']
                
                if plot_col and 'timestamp_iso_ny' in df_l.columns:
                    valid_mask = pd.notna(df_l[plot_col]) & pd.notna(df_l['timestamp_iso_ny'])
                    if valid_mask.any():
                        df_l_valid = df_l[valid_mask]
                        ax_left.plot(df_l_valid['timestamp_iso_ny'], df_l_valid[plot_col],
                                   color=config['color'], linewidth=2.5, zorder=2, alpha=0.9)  # Thicker lines
        
        ax_left.set_title(f'Left Hand - {metric.upper()}', fontsize=15, fontweight='bold')
        
        # Right hand
        ax_right = axes[row, 1]
        if metric in data['R'] and not data['R'][metric].empty:
            df_r = data['R'][metric].copy()
            
            # Special handling for accelerometer (plot X, Y, Z separately)
            if metric == 'accelerometer' and 'columns' in config:
                for col_name, color, label in zip(config['columns'], config['colors'], config['labels']):
                    if col_name in df_r.columns and 'timestamp_iso_ny' in df_r.columns:
                        valid_mask = pd.notna(df_r[col_name]) & pd.notna(df_r['timestamp_iso_ny'])
                        if valid_mask.any():
                            df_r_valid = df_r[valid_mask]
                            ax_right.plot(df_r_valid['timestamp_iso_ny'], df_r_valid[col_name],
                                        color=color, linewidth=2.5, zorder=2, alpha=0.9, label=label)
            else:
                # Standard plotting for other metrics
                col_name = config.get('column')
                plot_col = None
                if col_name in df_r.columns:
                    plot_col = col_name
                elif config.get('column_alt') and config['column_alt'] in df_r.columns:
                    plot_col = config['column_alt']
                elif config.get('column_fallback') and config['column_fallback'] in df_r.columns:
                    plot_col = config['column_fallback']
                
                if plot_col and 'timestamp_iso_ny' in df_r.columns:
                    valid_mask = pd.notna(df_r[plot_col]) & pd.notna(df_r['timestamp_iso_ny'])
                    if valid_mask.any():
                        df_r_valid = df_r[valid_mask]
                        ax_right.plot(df_r_valid['timestamp_iso_ny'], df_r_valid[plot_col],
                                    color=config['color'], linewidth=2.5, zorder=2, alpha=0.9)  # Thicker lines
        
        ax_right.set_title(f'Right Hand - {metric.upper()}', fontsize=15, fontweight='bold')
        
        # Add legend for accelerometer
        if metric == 'accelerometer':
            ax_left.legend(loc='upper right', fontsize=11, framealpha=0.9)
            ax_right.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Set Y-axis limits and ticks - ALWAYS use same limits for both hands for comparison
        if y_limits[metric]:
            ax_left.set_ylim(y_limits[metric])
            ax_right.set_ylim(y_limits[metric])  # Same limits for both hands
            if y_ticks[metric] is not None:
                ax_left.set_yticks(y_ticks[metric])
                ax_right.set_yticks(y_ticks[metric])  # Same ticks for both hands
                ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        else:
            # If no limits calculated, get current limits from both axes and use the wider range
            left_ylim = ax_left.get_ylim()
            right_ylim = ax_right.get_ylim()
            combined_ylim = (min(left_ylim[0], right_ylim[0]), max(left_ylim[1], right_ylim[1]))
            ax_left.set_ylim(combined_ylim)
            ax_right.set_ylim(combined_ylim)  # Ensure same limits
        
        # Set X-axis limits
        if x_min_ny and x_max_ny:
            ax_left.set_xlim(x_min_ny, x_max_ny)
            ax_right.set_xlim(x_min_ny, x_max_ny)
        
        # Set labels and grid
        ax_left.set_ylabel(config['ylabel'], fontsize=14)
        ax_right.set_ylabel(config['ylabel'], fontsize=14)
        ax_left.grid(True, alpha=0.3, zorder=1, linewidth=1.0)
        ax_right.grid(True, alpha=0.3, zorder=1, linewidth=1.0)
        
        # Add task phase backgrounds
        for phase in task_phases:
            for ax in [ax_left, ax_right]:
                ax.axvspan(phase['start'], phase['end'], alpha=0.2, color=phase['color'], zorder=0)
                
                # Add task type labels
                if phase.get('task_type') and phase['task_type'] != 'nan' and phase['task_type'] != '':
                    y_lim = ax.get_ylim()
                    center_time = phase['start'] + (phase['end'] - phase['start']) / 2
                    y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.5
                    ax.text(center_time, y_pos, phase['task_type'],
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='black', zorder=3,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5))
        
        # Hide X-axis ticks except on bottom row
        if row != 3:
            ax_left.set_xticks([])
            ax_right.set_xticks([])
            ax_left.set_xticklabels([])
            ax_right.set_xticklabels([])
        else:
            ax_left.set_xlabel('Time', fontsize=12)
            ax_right.set_xlabel('Time', fontsize=12)
            
            # Format x-axis with MANY MORE ticks for wider plot
            if x_min_ny and x_max_ny:
                # Create many tick points for maximum detail visibility
                num_ticks = 15  # Much more ticks for 48-inch wide plot
                x_ticks_ny = pd.date_range(start=x_min_ny, end=x_max_ny, periods=num_ticks)
                x_ticks_ny = [convert_to_naive_ny_time(t) if hasattr(t, 'tz') and t.tz is not None else t for t in x_ticks_ny]
                ax_left.set_xticks(x_ticks_ny)
                ax_right.set_xticks(x_ticks_ny)
            
            # Format x-ticks
            def format_time_tick(x, pos):
                try:
                    if isinstance(x, (int, float)):
                        dt = mdates.num2date(x)
                    else:
                        dt = x
                    if pd.notna(dt):
                        if isinstance(dt, pd.Timestamp):
                            return dt.strftime('%d %b\n%H:%M')
                        return pd.to_datetime(dt).strftime('%d %b\n%H:%M')
                    return ''
                except:
                    return ''
            
            ax_left.xaxis.set_major_formatter(FuncFormatter(format_time_tick))
            ax_right.xaxis.set_major_formatter(FuncFormatter(format_time_tick))
            
            # Larger tick labels
            for ax in [ax_left, ax_right]:
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
    
    # Add legend with larger font
    if phase_labels:
        handles = []
        labels = []
        for phase_label in phase_labels:
            handles.append(Patch(facecolor=phase_label['color'], alpha=0.2, edgecolor=phase_label['color'], linewidth=2))
            labels.append(phase_label['name'])
        
        fig.legend(handles, labels, loc='lower center', ncol=min(len(phase_labels), 4),
                  bbox_to_anchor=(0.5, -0.03), fontsize=15, frameon=True,
                  framealpha=1.0, edgecolor='black', fancybox=True)
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save figure with optimized quality (200 DPI for lighter files)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{participant_base_id}_raw_filtered_wide.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"  Saved wide plot: {output_file}")

def find_participants_in_raw_data(raw_data_dir):
    """Find all unique participant IDs from raw data files."""
    participants = set()
    
    for filename in glob.glob(os.path.join(raw_data_dir, "*_raw_eda.csv")):
        basename = os.path.basename(filename)
        match = re.match(r'^([A-Z]+\d+)([LR])_raw_', basename)
        if match:
            base_id = match.group(1)
            participants.add(base_id)
    
    return sorted(participants)

def main():
    """Main function."""
    print("=" * 80)
    print("Plot Raw Data with Filtered EDA (WIDE FORMAT)")
    print("=" * 80)
    print()
    
    # Load Excel file
    print(f"Loading task information from {EXCEL_FILE}...")
    excel_df, participant_col, task_name_col, task_type_col, start_time_col, end_time_col = load_excel_for_tasks(EXCEL_FILE)
    
    if excel_df is None:
        print("Error: Could not load Excel file.")
        return
    
    print()
    
    # Find participants
    print(f"Scanning {RAW_DATA_DIR} for raw data files...")
    participants = find_participants_in_raw_data(RAW_DATA_DIR)
    
    if not participants:
        print("No raw data files found. Please run extract_raw_avro_data.py first.")
        return
    
    print(f"Found {len(participants)} participants: {', '.join(participants)}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each participant
    for participant_base_id in participants:
        print(f"\nProcessing {participant_base_id}...")
        
        # Load raw data
        data = load_raw_data(RAW_DATA_DIR, participant_base_id)
        
        # Get task phases
        task_phases = get_task_phases_from_excel(excel_df, participant_col, task_name_col, task_type_col,
                                                 start_time_col, end_time_col, participant_base_id)
        
        # Create plot
        plot_participant_data(participant_base_id, data, task_phases, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("Wide plotting complete!")
    print(f"PNG files saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

