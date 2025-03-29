import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def load_tensorboard_logs(path):
    """
    Load tensorboard logs from the given path.
    
    This function first tries to use EventAccumulator, then falls back to reading
    YAML log files if tensorboard logs aren't available.
    
    Args:
        path: Path to the log directory
        
    Returns:
        Dictionary of metrics
    """
    data = defaultdict(list)
    
    # First try reading tensorboard logs directly
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()  # Load all data written so far
        
        # Check if there are any scalar values
        if event_acc.Tags().get("scalars"):
            for tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                for event in events:
                    data[tag].append(event.value)
            
            # If we found data, return it
            if any(len(values) > 0 for values in data.values()):
                return data
    except Exception as e:
        print(f"Warning: Error loading tensorboard logs: {e}")
    
    # Fall back to YAML log files if no tensorboard data
    yaml_log_path = os.path.join(path, "training_log.yaml")
    if os.path.exists(yaml_log_path):
        try:
            import yaml
            with open(yaml_log_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            # Convert to our format
            if yaml_data:
                return yaml_data
        except Exception as e:
            print(f"Warning: Error loading YAML logs: {e}")
    
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function