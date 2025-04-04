import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import time 
import yaml
import sys
import psutil

# Get the absolute path to the cfg directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(SCRIPT_DIR, "cfg")

# Add strategist to path for importing the OpenAI client
STRATEGIST_PATH = "/home/mr971/strategist"
if STRATEGIST_PATH not in sys.path:
    sys.path.append(STRATEGIST_PATH)

# Import AzureOpenAIClient instead of using openai directly
from strategist.openai_client import AzureOpenAIClient, MODEL_TO_NAME

# Import local modules using relative paths
import os
# Add the utils directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")
sys.path.append(utils_dir)

# Import utils modules directly
from misc import set_freest_gpu, block_until_training, filter_traceback
from file_utils import find_files_with_substring, load_tensorboard_logs
from create_task import create_task
from extract_task_code import file_to_string, get_function_signature

# Set EUREKA_ROOT_DIR to the Eureka directory, not the current working directory
EUREKA_ROOT_DIR = "/home/mr971/strategist/Eureka/eureka"

def log_openai_messages(messages, output_dir, iter_num=0, response_id=0):
    """Log the messages sent to OpenAI to a file in the output directory"""
    # Create path if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Log messages to a JSON file
    messages_file = os.path.join(output_dir, f"openai_messages_iter{iter_num}_response{response_id}.json")
    with open(messages_file, 'w') as f:
        json.dump(messages, f, indent=2)
    
    # Also create a more readable text version
    text_file = os.path.join(output_dir, f"openai_messages_iter{iter_num}_response{response_id}.txt")
    with open(text_file, 'w') as f:
        for msg in messages:
            f.write(f"Role: {msg['role']}\n")
            f.write(f"Content:\n{msg['content']}\n")
            f.write("-" * 80 + "\n")
    
    logging.info(f"Logged OpenAI messages to {messages_file} and {text_file}")

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    # Initialize AzureOpenAIClient instead of setting openai.api_key
    llm_client = AzureOpenAIClient(model=cfg.model)
    logging.info(f"Using Azure OpenAI API with model: {cfg.model} (mapped to: {llm_client.model})")

    task = cfg.env.task
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)

    env_name = cfg.env.env_name.lower()
    # Crafter-specific environment handling
    env_parent = 'crafter'
    # Load config file to get the game prompt file
    if hasattr(cfg.env, 'config_path') and os.path.exists(cfg.env.config_path):
        with open(cfg.env.config_path, 'r') as f:
            config_content = yaml.safe_load(f)
            # Get game_prompt_file from the config
            if 'game_prompt_file' in config_content:
                prompt_path = f"/home/mr971/strategist/prompts/{config_content['game_prompt_file']}"
                logging.info(f"Using prompt file from config: {prompt_path}")
                
                # Load the prompt file to get the task description
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r') as pf:
                        prompt_content = pf.read()
                        # Extract task description from the YAML file
                        if 'goal_context' in prompt_content:
                            task_description = prompt_content.split('goal_context: |')[1].strip().split('\n\n')[0].strip()
                            logging.info(f"Using task description from prompt file: {task_description}")
                        if 'game_context' in prompt_content:
                            environment_details = prompt_content.split('game_context: |')[1].strip().split('\n\n')[0].strip()
                            logging.info(f"Using environment details from prompt file: {environment_details}")
    
    task_file = f'/home/mr971/strategist/Eureka/eureka/envs/crafter/crafter.py'
    task_obs_file = f'/home/mr971/strategist/Eureka/eureka/envs/crafter/crafter_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    
    # Set output file path for crafter
    output_file = f"{workspace_dir}/crafter_task_{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'prompts')
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description, environment_details=environment_details)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            
            # Generate multiple samples in a loop since Azure client doesn't support 'n' parameter
            for response_id in range(min(chunk_size, cfg.sample - total_samples)):
                for attempt in range(1000):
                    try:
                        # Log the messages being sent to OpenAI
                        log_openai_messages(messages, f"{workspace_dir}/openai_logs", iter, response_id)
                        
                        # Use AzureOpenAIClient instead of direct openai call
                        response_cur = llm_client.get_completion(
                            messages=messages,
                            max_tokens=3000,
                            #temperature=cfg.temperature
                        )
                        
                        # Extract choices from the response - format adaptation
                        choices = []
                        for choice in response_cur.choices:
                            choices.append({"message": {"content": choice.message.content}})
                        
                        # Add just one response at a time
                        responses.append(choices[0])
                        total_samples += 1
                        
                        # Extract token usage information
                        if hasattr(response_cur, 'usage'):
                            if 'prompt_tokens' not in locals():
                                prompt_tokens = response_cur.usage.prompt_tokens
                            completion_tokens = response_cur.usage.completion_tokens
                            total_token += response_cur.usage.total_tokens
                            total_completion_token += completion_tokens
                        
                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                            print("Current Chunk Size", chunk_size)
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(1)
                
                # Break the outer loop if we couldn't generate a response after max attempts
                if attempt == 999:
                    break
            
            if len(responses) < 1:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(min(cfg.sample, len(responses))):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    
            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError
            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                file.writelines(code_string + '\n')

            with open(f"{workspace_dir}/env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"{workspace_dir}/env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"{workspace_dir}/env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                # Run the crafter environment with the generated reward function
                print(f"Running crafter environment with the generated reward function")
                process = subprocess.Popen(['python', '-u', f'{EUREKA_ROOT_DIR}/utils/run_crafter.py',
                                          f'--reward_file={workspace_dir}/env_iter{iter}_response{response_id}.py',
                                          f'--config_path={cfg.env.config_path if hasattr(cfg.env, "config_path") else ""}',
                                          f'--iterations={cfg.max_iterations}',
                                          f'--seed=42',
                                          f'--use_wandb'
                                        ],  
                                          stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
        
        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            try:
                # Add timeout to prevent hanging indefinitely
                logging.info(f"Iteration {iter}: Waiting for subprocess {response_id} to complete (timeout: 30s)")
                rl_run.communicate()  # 30 second timeout
                logging.info(f"Iteration {iter}: Subprocess {response_id} completed or timed out")
            except subprocess.TimeoutExpired:
                logging.info(f"Iteration {iter}: Code Run {response_id} subprocess timed out, forcibly terminating")
                # Try to terminate the process and its children
                try:
                    parent = psutil.Process(rl_run.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                            logging.info(f"Terminated child process {child.pid}")
                        except:
                            pass
                    rl_run.terminate()
                    logging.info(f"Terminated parent process {rl_run.pid}")
                    # Give processes time to terminate
                    time.sleep(2)
                    # If still running, try to kill
                    if rl_run.poll() is None:
                        logging.info(f"Process {rl_run.pid} still running, killing it")
                        rl_run.kill()
                except Exception as e:
                    logging.error(f"Error terminating subprocess: {e}")
                
            rl_filepath = f"{workspace_dir}/env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"{workspace_dir}/env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                found_tensorboard_dir = False
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        found_tensorboard_dir = True
                        break 
                
                if found_tensorboard_dir:
                    tensorboard_logdir = line.split(':')[-1].strip() 
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs.get('gt_reward', [])).shape[0]
                    
                    # If tensorboard logs empty or insufficient, provide fallback message
                    if max_iterations == 0:
                        logging.warning(f"Iteration {iter}: Code Run {response_id} no tensorboard metrics found")
                        content += "Training completed but no tensorboard metrics were found. This could mean the training didn't record metrics properly.\n"
                        successes.append(DUMMY_FAILURE)
                        reward_correlations.append(DUMMY_FAILURE)
                        continue
                        
                    epoch_freq = max(int(max_iterations // 10), 1)
                    
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                    # Initialize success value 
                    success_found = False
                    
                    # Compute Correlation between Human-Engineered and GPT Rewards
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs and len(tensorboard_logs["gt_reward"]) > 0 and len(tensorboard_logs["gpt_reward"]) > 0:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                        reward_correlations.append(reward_correlation)
                    else:
                        logging.warning(f"Iteration {iter}: Code Run {response_id} missing reward metrics")
                        reward_correlations.append(DUMMY_FAILURE)

                    # Add reward components log to the feedback
                    for metric in tensorboard_logs:
                        if "/" not in metric and len(tensorboard_logs[metric]) > 0:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if "consecutive_successes" == metric:
                                successes.append(metric_cur_max)
                                success_found = True
                            metric_cur_min = min(tensorboard_logs[metric])
                            if metric != "gt_reward" and metric != "gpt_reward":
                                if metric != "consecutive_successes":
                                    metric_name = metric 
                                else:
                                    metric_name = "task_score"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                            else:
                                # Provide ground-truth score when success rate not applicable
                                if "consecutive_successes" not in tensorboard_logs:
                                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"  
                    
                    # If no success metric was found, add a dummy failure
                    if not success_found:
                        successes.append(DUMMY_FAILURE)
                else:
                    # Even though there's no traceback, we couldn't find the tensorboard directory
                    logging.info(f"Iteration {iter}: Code Run {response_id} couldn't find tensorboard directory")
                    successes.append(DUMMY_FAILURE)
                    reward_correlations.append(DUMMY_FAILURE)
                    content += execution_error_feedback.format(traceback_msg="Training completed but no tensorboard logs were generated.")
                
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 
        
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
            
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    shutil.copy(max_reward_code_path, output_file)


if __name__ == "__main__":
    main()