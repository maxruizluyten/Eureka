import subprocess
import os
import json
import logging
import time
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with absolute path
from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1, max_wait_time=300):
    """
    Wait for training to start and complete.
    
    Args:
        rl_filepath: Path to the log file
        log_status: Whether to log status
        iter_num: Iteration number for logging
        response_id: Response ID for logging
        max_wait_time: Maximum time to wait in seconds before returning anyway
    """
    start_time = time.time()
    last_modified = os.path.getmtime(rl_filepath) if os.path.exists(rl_filepath) else start_time
    
    # Ensure that the RL training has started before moving on
    while True:
        # Check for timeout
        current_time = time.time()
        if current_time - start_time > max_wait_time:
            if log_status:
                logging.warning(f"Iteration {iter_num}: Code Run {response_id} timed out waiting for training to start after {max_wait_time}s")
            break
                
        # Check file content
        rl_log = file_to_string(rl_filepath)
        
        # First check if training has completed - this takes precedence
        if "Training complete" in rl_log or "All runs completed" in rl_log:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} training already completed!")
            return
        
        # Now check if training has started
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
        
        # Check if file has been updated - but only if we haven't found any other indicators
        if os.path.exists(rl_filepath):
            current_modified = os.path.getmtime(rl_filepath)
            if current_modified > last_modified:
                last_modified = current_modified
            # Only assume stuck if file hasn't been modified for a very long time and there's content in it
            elif current_time - last_modified > 120 and len(rl_log) > 100:  # 2 minutes without updates and has content
                if log_status:
                    logging.warning(f"Iteration {iter_num}: Code Run {response_id} log file hasn't been updated for 120s, assuming stuck")
                break
                
        # Wait a bit before checking again
        time.sleep(1)
        
    # Also check for training completion
    rl_log = file_to_string(rl_filepath)
    if "Training complete" not in rl_log and "All runs completed" not in rl_log:
        if log_status:
            logging.info(f"Iteration {iter_num}: Code Run {response_id} waiting for training to complete...")
        
        # Wait for training completion with timeout
        completion_start_time = time.time()
        last_content_length = len(rl_log)
        while True:
            # Check for timeout
            current_time = time.time()
            if current_time - completion_start_time > max_wait_time:
                if log_status:
                    logging.warning(f"Iteration {iter_num}: Code Run {response_id} timed out waiting for training completion after {max_wait_time}s")
                break
                
            # Check file content
            rl_log = file_to_string(rl_filepath)
            if "Training complete" in rl_log or "All runs completed" in rl_log:
                if log_status:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} training completed!")
                return  # Return immediately when complete
            
            # Check if file content has increased (better than checking modification time)
            if len(rl_log) > last_content_length:
                last_content_length = len(rl_log)
                last_modified = time.time()  # Reset the timer when content increases
            elif current_time - last_modified > 120:  # If no new content for 2 minutes
                if "fps step:" in rl_log:  # Only if training had actually started
                    logging.warning(f"Iteration {iter_num}: Code Run {response_id} training appears to be complete but missing completion message")
                    break
                
            # Wait a bit before checking again
            time.sleep(1)

if __name__ == "__main__":
    print(get_freest_gpu())