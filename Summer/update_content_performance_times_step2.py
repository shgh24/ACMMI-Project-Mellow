#!/usr/bin/env python3
"""
Script to extract content performance task times from mtrack Excel files
and update the all_participants_tasks.xlsx file.

Tasks to extract: diary, mandala, video_anxiety

Requirements: pip install pandas openpyxl
"""

import os
import glob
import sys
import re

try:
    import pandas as pd
except ImportError:
    print("Error: Missing required packages. Please install: pip install pandas openpyxl")
    sys.exit(1)

def extract_participant_id_from_filename(filename):
    """Extract participant ID from filename like MIT001_SCENARIO_202511041236.xlsx"""
    basename = os.path.basename(filename)
    match = re.match(r'([A-Z]+\d+)', basename)
    if match:
        return match.group(1)
    return None

def find_task_row_in_mtrack_file(filepath):
    """
    Read mtrack Excel file and find row with diary, mandala, or video_anxiety.
    Returns: dict with task_type, start_time, end_time, or None if not found
    """
    try:
        df = pd.read_excel(filepath)
        
        # Search all columns for the task types
        task_keywords = ['diary', 'mandala', 'video_anxiety']
        task_type = None
        task_row_idx = None
        
        # Search through all rows and columns
        for idx, row in df.iterrows():
            row_str = ' '.join([str(val).lower() if pd.notna(val) else '' for val in row.values])
            for keyword in task_keywords:
                if keyword in row_str:
                    task_type = keyword
                    task_row_idx = idx
                    break
            if task_row_idx is not None:
                break
        
        if task_row_idx is None:
            return None
        
        task_row = df.iloc[task_row_idx]
        
        # Find start time and end time columns
        # Common column names for start/end times
        start_cols = ['start time', 'start_time', 'start', 'begin', 'begin time', 'begin_time', 
                     'start time (local)', 'start_time_local', 'start local']
        end_cols = ['end time', 'end_time', 'end', 'finish', 'finish time', 'finish_time',
                   'end time (local)', 'end_time_local', 'end local']
        
        start_time = None
        end_time = None
        start_col_name = None
        end_col_name = None
        
        # Search for start time column
        for col in df.columns:
            col_str = str(col).lower().strip()
            if any(start_key in col_str for start_key in start_cols):
                val = task_row[col]
                if pd.notna(val):
                    start_time = val
                    start_col_name = col
                    break
        
        # Search for end time column
        for col in df.columns:
            col_str = str(col).lower().strip()
            if any(end_key in col_str for end_key in end_cols):
                val = task_row[col]
                if pd.notna(val):
                    end_time = val
                    end_col_name = col
                    break
        
        if start_time is None or end_time is None:
            print(f"    Warning: Could not find start/end times. Start: {start_time}, End: {end_time}")
            print(f"    Available columns: {df.columns.tolist()}")
            print(f"    Task row values: {task_row.to_dict()}")
            return None
        
        return {
            'task_type': task_type,
            'start_time': start_time,
            'end_time': end_time,
            'start_col': start_col_name,
            'end_col': end_col_name
        }
        
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_excel_file(excel_path):
    """Load Excel file and identify column names"""
    try:
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel file: {excel_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def find_excel_columns(df):
    """Identify column names in Excel file"""
    col_map = {}
    
    # Common column name variations
    participant_cols = ['Participant ID', 'Participant_ID', 'participant_id', 'Participant', 'ID']
    task_name_cols = ['Task Name', 'Task_Name', 'task_name', 'Task', 'TaskName']
    task_type_cols = ['Task Type', 'Task_Type', 'task_type', 'TaskType']
    start_time_cols = ['Start Time (Local)', 'Start_Time_Local', 'start_time_local', 'Start Time', 'StartTime', 'Start Time (Local)']
    end_time_cols = ['End Time (Local)', 'End_Time_Local', 'end_time_local', 'End Time', 'EndTime', 'End Time (Local)']
    
    for col in df.columns:
        col_str = str(col).strip()
        # Participant ID
        if not col_map.get('participant') and any(p in col_str for p in participant_cols):
            col_map['participant'] = col
        # Task Name
        if not col_map.get('task_name') and any(t in col_str for t in task_name_cols):
            col_map['task_name'] = col
        # Task Type
        if not col_map.get('task_type') and any(t in col_str for t in task_type_cols):
            col_map['task_type'] = col
        # Start Time
        if not col_map.get('start_time') and any(s in col_str for s in start_time_cols):
            col_map['start_time'] = col
        # End Time
        if not col_map.get('end_time') and any(e in col_str for e in end_time_cols):
            col_map['end_time'] = col
    
    return col_map

def extract_participant_base_id(participant_id):
    """Extract base ID from participant ID (e.g., MIT001 from MIT001L-3YK9T1L1LK)"""
    if not participant_id or pd.isna(participant_id):
        return None
    participant_str = str(participant_id).strip()
    # Extract MITXXX pattern
    match = re.match(r'([A-Z]+\d+)', participant_str)
    if match:
        return match.group(1)
    return participant_str

def main():
    """Main function"""
    print("=" * 80)
    print("Update Content Performance Task Times from Mtrack Files")
    print("=" * 80)
    print()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mtrack_dir = os.path.join(base_dir, 'data', 'mtrack')
    excel_file = os.path.join(base_dir, 'data', 'all_participants_tasks.xlsx')
    output_file = os.path.join(base_dir, 'data', 'all_participants_tasks_updated.xlsx')
    
    # Step 1: Load Excel file
    print("Step 1: Loading Excel file...")
    df = load_excel_file(excel_file)
    if df is None:
        return
    
    # Step 2: Identify columns
    print("\nStep 2: Identifying columns...")
    col_map = find_excel_columns(df)
    print(f"  Found columns: {col_map}")
    
    if not col_map.get('participant') or not col_map.get('task_name'):
        print("  Error: Could not find required columns (Participant ID, Task Name)")
        return
    
    # Step 3: Extract task times from mtrack files
    print("\nStep 3: Extracting task times from mtrack files...")
    mtrack_files = glob.glob(os.path.join(mtrack_dir, 'MIT*_SCENARIO*.xlsx'))
    print(f"  Found {len(mtrack_files)} mtrack files")
    
    task_times = {}  # {participant_base_id: {task_type, start_time, end_time}}
    
    for mtrack_file in mtrack_files:
        participant_id = extract_participant_id_from_filename(mtrack_file)
        if not participant_id:
            print(f"  Warning: Could not extract participant ID from {mtrack_file}")
            continue
        
        result = find_task_row_in_mtrack_file(mtrack_file)
        if result:
            task_times[participant_id] = result
            print(f"  {participant_id}: {result['task_type']} - {result['start_time']} to {result['end_time']}")
        else:
            print(f"  {participant_id}: No task found")
    
    print(f"\n  Extracted times for {len(task_times)} participants")
    
    # Step 4: Update Excel file
    print("\nStep 4: Updating Excel file...")
    df_updated = df.copy()
    
    updated_count = 0
    
    for idx, row in df_updated.iterrows():
        participant_id = str(row.get(col_map['participant'], '')).strip()
        task_name = str(row.get(col_map['task_name'], '')).strip()
        
        # Check if this is a content performance task row
        if 'content' in task_name.lower() or ('performance' in task_name.lower() and 'task' in task_name.lower()):
            # Extract base ID
            base_id = extract_participant_base_id(participant_id)
            
            if base_id and base_id in task_times:
                times = task_times[base_id]
                
                # Update start time
                if col_map.get('start_time'):
                    start_time = times['start_time']
                    if start_time is not None:
                        # Convert to datetime if needed, preserving the format
                        try:
                            # If it's already a datetime, use it directly
                            if isinstance(start_time, pd.Timestamp):
                                start_dt = start_time
                            elif isinstance(start_time, str):
                                # Parse string to datetime
                                start_dt = pd.to_datetime(start_time)
                            else:
                                # Try to convert to datetime
                                start_dt = pd.to_datetime(start_time)
                            
                            # Remove timezone info if present (Excel doesn't support timezone-aware datetimes)
                            if hasattr(start_dt, 'tz') and start_dt.tz is not None:
                                start_dt = start_dt.tz_localize(None)
                            
                            df_updated.at[idx, col_map['start_time']] = start_dt
                            print(f"  Updated {base_id} start time: {start_dt}")
                        except Exception as e:
                            print(f"  Warning: Could not parse start time for {base_id}: {e}")
                            print(f"    Value: {start_time}, Type: {type(start_time)}")
                
                # Update end time
                if col_map.get('end_time'):
                    end_time = times['end_time']
                    if end_time is not None:
                        try:
                            # If it's already a datetime, use it directly
                            if isinstance(end_time, pd.Timestamp):
                                end_dt = end_time
                            elif isinstance(end_time, str):
                                # Parse string to datetime
                                end_dt = pd.to_datetime(end_time)
                            else:
                                # Try to convert to datetime
                                end_dt = pd.to_datetime(end_time)
                            
                            # Remove timezone info if present (Excel doesn't support timezone-aware datetimes)
                            if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                                end_dt = end_dt.tz_localize(None)
                            
                            df_updated.at[idx, col_map['end_time']] = end_dt
                            print(f"  Updated {base_id} end time: {end_dt}")
                            updated_count += 1
                        except Exception as e:
                            print(f"  Warning: Could not parse end time for {base_id}: {e}")
                            print(f"    Value: {end_time}, Type: {type(end_time)}")
                
                # Update task type if available
                if col_map.get('task_type') and times.get('task_type'):
                    df_updated.at[idx, col_map['task_type']] = times['task_type']
                    print(f"  Updated {base_id} task type: {times['task_type']}")
    
    print(f"\n  Updated {updated_count} rows")
    
    # Step 5: Remove timezone info from all datetime columns before saving
    print(f"\nStep 5: Preparing data for Excel export...")
    # Check all datetime columns and make them timezone-naive
    for col in df_updated.columns:
        # Check if column contains datetime data
        if pd.api.types.is_datetime64_any_dtype(df_updated[col]):
            # Convert timezone-aware datetimes to naive
            if df_updated[col].dtype.tz is not None:
                df_updated[col] = df_updated[col].dt.tz_localize(None)
            else:
                # Check individual values for timezone info
                for idx in df_updated.index:
                    val = df_updated.at[idx, col]
                    if pd.notna(val):
                        if isinstance(val, pd.Timestamp) and val.tz is not None:
                            df_updated.at[idx, col] = val.tz_localize(None)
    
    # Step 6: Save updated Excel file
    print(f"\nStep 6: Saving updated Excel file...")
    try:
        df_updated.to_excel(output_file, index=False)
        print(f"  Saved to: {output_file}")
    except Exception as e:
        print(f"  Error saving file: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()

