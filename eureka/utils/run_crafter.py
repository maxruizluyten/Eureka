#!/usr/bin/env python3
import os
import sys
import argparse
import importlib.util
import numpy as np
import torch
import time
import logging
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Set up logging first, before any imports that might use it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add crafter to the path
CRAFTER_PATH = "/home/mr971/strategist/data/crafter"
if CRAFTER_PATH not in sys.path:
    sys.path.append(CRAFTER_PATH)
    logger.info(f"Added Crafter path: {CRAFTER_PATH}")

# Add strategist to path for importing the RLAgent and config
STRATEGIST_PATH = "/home/mr971/strategist"
if STRATEGIST_PATH not in sys.path:
    sys.path.append(STRATEGIST_PATH)
    logger.info(f"Added Strategist path: {STRATEGIST_PATH}")

# Import required modules from strategist
logger.info("Importing modules from strategist...")
from strategist.config import load_config
from strategist.rl_agent import RLAgent
try:
    from strategist.env_utils import get_env  # Import get_env for consistent environment setup
    logger.info("Successfully imported get_env from strategist.env_utils")
    get_env_available = True
except ImportError:
    logger.warning("Could not import get_env from strategist.env_utils")
    get_env = None
    get_env_available = False

import crafter
logger.info("Successfully imported crafter")

def load_reward_function(file_path):
    """Load the reward function from a file"""
    logger.info(f"Loading reward function from: {file_path}")
    try:
        # First, check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"Reward function file does not exist: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the module
        logger.info(f"Loading module from file: {file_path}")
        spec = importlib.util.spec_from_file_location("reward_module", file_path)
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        
        # Check what functions are available in the module
        module_functions = [f for f in dir(reward_module) if callable(getattr(reward_module, f)) and not f.startswith("__")]
        logger.info(f"Available functions in the module: {module_functions}")
        
        # Try to get the compute_reward_gpt function
        if hasattr(reward_module, 'compute_reward'):
            logger.info("Found compute_reward function")
            return reward_module.compute_reward
        else:
            logger.error(f"Not compute_reward found in module. Available attributes: {dir(reward_module)}")
            raise AttributeError("Required reward function not found in module")
    except Exception as e:
        logger.error(f"Error loading reward function: {e}")
        logger.error(traceback.format_exc())
        raise

def load_config(config_path):
    """Load configuration from YAML file"""
    logger.info(f"Loading config from: {config_path}")
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config path not provided or does not exist: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")
    return config

def load_prompt(config):
    """Load prompt from config's game_prompt_file"""
    if not config or 'game_prompt_file' not in config:
        logger.warning("No game_prompt_file specified in config")
        return None
    
    prompt_path = f"/home/mr971/strategist/prompts/{config['game_prompt_file']}"
    if not os.path.exists(prompt_path):
        logger.warning(f"Prompt file not found: {prompt_path}")
        return None
    
    logger.info(f"Loading prompt from: {prompt_path}")
    with open(prompt_path, 'r') as f:
        prompt_content = yaml.safe_load(f)
    return prompt_content

def create_output_dirs(reward_file):
    """Create output directories for logs and videos"""
    base_name = Path(reward_file).stem
    log_dir = Path(f"./logs/{base_name}")
    video_dir = Path(f"./videos/{base_name}")
    export_dir = Path(f"./export/{base_name}")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directories: logs={log_dir}, videos={video_dir}, export={export_dir}")
    return log_dir, video_dir, export_dir

def train_agent_with_eureka_reward(reward_fn, agent_config, env_name="crafter", difficulty="normal", seed=None, use_wandb=False, export_path="./export"):
    """
    Train an agent using the Eureka-generated reward function.
    Based on the train_agent function from strategist_agent.py
    
    Args:
        reward_fn: The custom reward function to use
        agent_config: Configuration for the agent
        env_name: Name of the environment
        difficulty: Game difficulty (easy, normal, hard)
        seed: Random seed for the environment
        use_wandb: Whether to use Weights & Biases for logging
        export_path: Directory to export model checkpoints
        
    Returns:
        The trained agent
    """
    # Similar to strategist_agent.py's train_agent function
    logger.info(f"Training agent with Eureka reward. Env: {env_name}, Difficulty: {difficulty}, Seed: {seed}")
    
    # Test the reward function with a dummy observation
    if get_env_available:
        try:
            # Try to use get_env from strategist.env_utils for consistent environment setup
            logger.info(f"Creating environment using get_env with difficulty={difficulty}, seed={seed}")
            env = get_env(env_name, reward_func=None, 
                         difficulty=difficulty,
                         seed=seed)
            logger.info(f"Created environment using get_env with difficulty={difficulty}, seed={seed}")
        except Exception as e:
            logger.warning(f"Error using get_env: {e}, falling back to direct env creation")
            logger.error(traceback.format_exc())
            env = crafter.Env(reward=True)
            logger.info("Created environment directly with crafter.Env(reward=True)")
    else:
        # Fall back to direct creation if get_env is not available
        logger.warning("get_env not available, falling back to direct env creation")
        env = crafter.Env(reward=True)
        logger.info("Created environment directly with crafter.Env(reward=True)")
    
    # Log environment properties
    logger.info(f"Environment: {env}")
    logger.info(f"Environment action space: {env.action_space}")
    logger.info(f"Environment observation space: {env.observation_space}")
    
    # Initialize the RL agent with custom reward function
    logger.info(f"Initializing RLAgent with config: {agent_config}")
    agent = RLAgent(env, agent_config, use_wandb=use_wandb, 
                    export_path=export_path, custom_reward_func=reward_fn)
    logger.info("RLAgent initialized successfully")
    
    # Train the agent
    logger.info(f"Starting training for {agent_config['train_ts']} iterations")
    start_time = time.time()
    agent.train()
    logger.info(f"Training completed in {time.time() - start_time:.2f}s")
    logger.info(f"Total steps: {agent_config['train_ts']}")
    
    return agent

def run_training(reward_fn, iterations, difficulty="normal", seed=None, checkpoint=None, use_wandb=False, export_path="./export", video_dir=None):
    """
    Run PPO training with the specified reward function
    
    Args:
        reward_fn: The custom reward function to use
        iterations: Number of training iterations
        difficulty: Game difficulty (easy, normal, hard)
        seed: Random seed for the environment
        checkpoint: Path to checkpoint file to load (optional)
        use_wandb: Whether to use Weights & Biases for logging
        export_path: Directory to export model checkpoints
        video_dir: Directory to save videos (optional)
        
    Returns:
        The trained agent and collected rewards
    """
    logger.info(f"Running training with iterations={iterations}, difficulty={difficulty}, seed={seed}")
    
    # Load PPO config
    ppo_config = load_config('/home/mr971/strategist/configs/rl_agent-ppo.yaml')
    
    # Override training steps
    ppo_config['train_ts'] = iterations
    logger.info(f"PPO config after overriding training steps: {ppo_config}")
    
    # Set up callback parameters
    if video_dir:
        # Modify log and checkpoint intervals to create more frequent visualizations
        ppo_config['log_interval'] = min(ppo_config.get('log_interval', 20000), 10000)
        ppo_config['checkpoint_interval'] = min(ppo_config.get('checkpoint_interval', 500000), 100000)
        logger.info(f"Modified intervals for video capture: log_interval={ppo_config['log_interval']}, checkpoint_interval={ppo_config['checkpoint_interval']}")
    
    # Use the train_agent function for consistency with strategist_agent.py
    agent = train_agent_with_eureka_reward(
        reward_fn=reward_fn,
        agent_config=ppo_config,
        difficulty=difficulty,
        seed=seed,
        use_wandb=use_wandb,
        export_path=export_path
    )
    
    # Load checkpoint if specified
    if checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint}")
        agent.load_from_checkpoint(checkpoint)
    
    # After training, run evaluation episodes to capture videos and collect rewards
    episode_gt_rewards = []
    episode_gpt_rewards = []
    
    # Always run at least 5 evaluation episodes
    eval_episodes = 10
    logger.info(f"Running {eval_episodes} evaluation episodes for reward collection...")
    
    for i in range(eval_episodes):
        logger.info(f"Running evaluation episode {i+1}/{eval_episodes}")
        try:
            trajectory, episode_gt_reward = agent.run_one_episode()
            logger.info(f"Episode {i+1} - Trajectory length: {len(trajectory)}, GT reward: {episode_gt_reward}")
            episode_gt_rewards.append(float(episode_gt_reward))
            
            # Calculate GPT reward
            episode_gpt_reward = 0
            for step_idx, (_, _, obs, _, _, _) in enumerate(trajectory):
                try:
                    # Log the observation shape for debugging
                    logger.debug(f"Episode {i+1}, Step {step_idx} - Observation shape: {obs.shape}, type: {type(obs)}")
                    
                    # Get reward from GPT reward function
                    obs_tensor = torch.tensor(obs).unsqueeze(0).float()
                    logger.debug(f"Tensor shape: {obs_tensor.shape}, tensor type: {obs_tensor.dtype}")
                    
                    reward_result = reward_fn(obs_tensor)
                    logger.debug(f"Raw reward result: {reward_result}, type: {type(reward_result)}")
                    
                    # Handle different return types (single value or tuple with dict)
                    if isinstance(reward_result, tuple):
                        reward_value, reward_dict = reward_result
                        logger.debug(f"Reward value from tuple: {reward_value}, dict keys: {reward_dict.keys() if isinstance(reward_dict, dict) else 'not a dict'}")
                    else:
                        reward_value = reward_result
                        logger.debug(f"Direct reward value: {reward_value}")
                    
                    # Ensure reward_value is a float
                    reward_value = float(reward_value)
                    logger.debug(f"Final reward value (after float conversion): {reward_value}")
                    
                    episode_gpt_reward += reward_value
                except Exception as e:
                    logger.error(f"Error calculating GPT reward at step {step_idx}: {e}")
                    logger.error(traceback.format_exc())
                    # Use a default reward of 0 for this step if calculation fails
                    logger.warning(f"Using default reward of 0 for this step due to error")
            
            episode_gpt_rewards.append(float(episode_gpt_reward))
            logger.info(f"Episode {i+1} completed - GT reward: {episode_gt_reward:.2f}, GPT reward: {episode_gpt_reward:.2f}")
            
            # Save frames if video capture is enabled
            if video_dir:
                for step, (_, _, obs, _, _, _) in enumerate(trajectory):
                    if step % 10 == 0:  # Save every 10th frame
                        plt.figure(figsize=(8, 8))
                        plt.imshow(obs)
                        plt.title(f"Episode {i+1}, Step {step}, GT: {episode_gt_reward:.2f}, GPT: {episode_gpt_reward:.2f}")
                        plt.savefig(Path(video_dir) / f"episode_{i+1}_step_{step:05d}.png")
                        plt.close()
        except Exception as e:
            logger.error(f"Error during episode {i+1}: {e}")
            logger.error(traceback.format_exc())
            logger.warning(f"Skipping episode {i+1} due to error")
            # Add placeholder values for this episode
            episode_gt_rewards.append(0.0)
            episode_gpt_rewards.append(0.0)
    
    # Log the final reward lists for debugging
    logger.info(f"Collected GT rewards: {episode_gt_rewards}")
    logger.info(f"Collected GPT rewards: {episode_gpt_rewards}")
    
    # Return the agent and collected rewards
    return agent, episode_gt_rewards, episode_gpt_rewards

def main():
    parser = argparse.ArgumentParser(description="Run Crafter with a GPT-generated reward function")
    parser.add_argument('--reward_file', type=str, required=True, help='Path to the reward function file')
    parser.add_argument('--config_path', type=str, default='', help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=2000000, help='Number of training iterations')
    parser.add_argument('--capture_video', action='store_true', help='Whether to capture video')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of training runs')
    parser.add_argument('--difficulty', type=str, default='normal', help='Game difficulty (easy, normal, hard)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for the environment')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose logging')
    parser.add_argument('--evaluation_only', action='store_true', help='Skip training and only run evaluation (requires checkpoint)')
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # If evaluation_only is specified, a checkpoint must be provided
    if args.evaluation_only and not args.checkpoint:
        logger.error("--evaluation_only requires --checkpoint to be specified")
        return 1
    
    logger.info(f"Started run_crafter.py with arguments: {args}")
    
    # Load config
    game_config = load_config(args.config_path)
    
    # Load prompt if available
    prompt = load_prompt(game_config)
    if prompt:
        logger.info(f"Loaded prompt from {game_config.get('game_prompt_file')}")
        if 'goal_context' in prompt:
            logger.info(f"Task goal: {prompt['goal_context']}")
    
    # Load the reward function
    try:
        reward_fn = load_reward_function(args.reward_file)
        logger.info("Successfully loaded reward function")
    except Exception as e:
        logger.error(f"Failed to load reward function: {e}")
        logger.error(traceback.format_exc())
        # Print tensorboard directory even on failure to ensure eureka.py can find it
        return 1
    
    # Create output directories
    log_dir, video_dir, export_dir = create_output_dirs(args.reward_file)
    print(f"Tensorboard Directory: {log_dir}")
    logger.info(f"Tensorboard Directory: {log_dir}")
    
    # Create a tensorboard-like log file
    log_file = log_dir / "training_log.yaml"
    logs = {"gt_reward": [], "gpt_reward": []}
    
    # Write initial logs to ensure the file exists even if training fails
    with open(log_file, 'w') as f:
        yaml.dump(logs, f)
    
    # Run multiple training runs if specified
    for run in range(args.n_runs):
        logger.info(f"Starting run {run+1}/{args.n_runs}")
        
        # Set a different seed for each run if seed is specified
        run_seed = args.seed + run if args.seed is not None else None
        logger.info(f"Using seed: {run_seed}")
        
        # Run the training
        try:
            if args.evaluation_only:
                logger.info("Evaluation-only mode: skipping training")
                
                # Create an environment and agent without training
                logger.info("Setting up environment and agent for evaluation only")
                env = crafter.Env(reward=True)
                
                # Load PPO config
                ppo_config = load_config('/home/mr971/strategist/configs/rl_agent-ppo.yaml')
                
                # Initialize agent without training
                agent = RLAgent(env, ppo_config, use_wandb=False)
                
                # Load the specified checkpoint
                logger.info(f"Loading checkpoint from {args.checkpoint}")
                agent.load_from_checkpoint(args.checkpoint)
                
                # Run evaluation episodes
                gt_rewards = []
                gpt_rewards = []
                
                eval_episodes = 10
                logger.info(f"Running {eval_episodes} evaluation episodes...")
                
                for i in range(eval_episodes):
                    logger.info(f"Running evaluation episode {i+1}/{eval_episodes}")
                    try:
                        trajectory, episode_gt_reward = agent.run_one_episode()
                        logger.info(f"Episode {i+1} - Trajectory length: {len(trajectory)}, GT reward: {episode_gt_reward}")
                        gt_rewards.append(float(episode_gt_reward))
                        
                        # Calculate GPT reward
                        episode_gpt_reward = 0
                        for step_idx, (_, _, obs, _, _, _) in enumerate(trajectory):
                            try:
                                # Get reward from GPT reward function
                                obs_tensor = torch.tensor(obs).unsqueeze(0).float()
                                reward_result = reward_fn(obs_tensor)
                                
                                # Handle different return types
                                if isinstance(reward_result, tuple):
                                    reward_value, _ = reward_result
                                else:
                                    reward_value = reward_result
                                
                                episode_gpt_reward += float(reward_value)
                            except Exception as e:
                                logger.error(f"Error calculating GPT reward at step {step_idx}: {e}")
                                logger.error(traceback.format_exc())
                        
                        gpt_rewards.append(float(episode_gpt_reward))
                        logger.info(f"Episode {i+1} - GT reward: {episode_gt_reward:.2f}, GPT reward: {episode_gpt_reward:.2f}")
                        
                        # Save frames if video capture is enabled
                        if args.capture_video:
                            for step, (_, _, obs, _, _, _) in enumerate(trajectory):
                                if step % 10 == 0:  # Save every 10th frame
                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(obs)
                                    plt.title(f"Episode {i+1}, Step {step}, GT: {episode_gt_reward:.2f}, GPT: {episode_gpt_reward:.2f}")
                                    plt.savefig(Path(video_dir) / f"episode_{i+1}_step_{step:05d}.png")
                                    plt.close()
                    except Exception as e:
                        logger.error(f"Error during evaluation episode {i+1}: {e}")
                        logger.error(traceback.format_exc())
                        gt_rewards.append(0.0)
                        gpt_rewards.append(0.0)
                
                logger.info(f"Evaluation completed, GT rewards: {gt_rewards}")
                logger.info(f"Evaluation completed, GPT rewards: {gpt_rewards}")
            else:
                # Normal training and evaluation
                agent, gt_rewards, gpt_rewards = run_training(
                    reward_fn=reward_fn,
                    iterations=args.iterations,
                    difficulty=args.difficulty,
                    seed=run_seed,
                    checkpoint=args.checkpoint,
                    use_wandb=args.use_wandb,
                    export_path=str(export_dir),
                    video_dir=str(video_dir) if args.capture_video else None
                )
            
            # Update the logs with rewards from this run
            logs["gt_reward"].extend(gt_rewards)
            logs["gpt_reward"].extend(gpt_rewards)
            
            # Save the logs with detailed error handling
            logger.info(f"Saving logs to {log_file}, gt_rewards: {len(gt_rewards)}, gpt_rewards: {len(gpt_rewards)}")
            try:
                # First, ensure all values are basic Python types for YAML serialization
                clean_logs = {
                    "gt_reward": [float(r) for r in logs["gt_reward"]],
                    "gpt_reward": [float(r) for r in logs["gpt_reward"]]
                }
                
                # Log data types for debugging
                logger.debug(f"GT reward types: {[type(r) for r in clean_logs['gt_reward'][:5]]}")
                logger.debug(f"GPT reward types: {[type(r) for r in clean_logs['gpt_reward'][:5]]}")
                
                # Check for NaN or infinite values
                for key in clean_logs:
                    for i, val in enumerate(clean_logs[key]):
                        if np.isnan(val) or np.isinf(val):
                            logger.warning(f"Found {val} in {key}[{i}], replacing with 0.0")
                            clean_logs[key][i] = 0.0
                
                # Ensure the directory exists
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the logs
                with open(log_file, 'w') as f:
                    yaml.dump(clean_logs, f)
                logger.info(f"Successfully saved logs to {log_file}")
                
                # Also save as json for redundancy/debugging
                import json
                json_log_file = log_file.with_suffix('.json')
                with open(json_log_file, 'w') as f:
                    json.dump(clean_logs, f, indent=2)
                logger.info(f"Also saved logs as JSON to {json_log_file}")
                
            except Exception as e:
                logger.error(f"Error saving logs: {e}")
                logger.error(traceback.format_exc())
                # Try a fallback approach - save raw values
                try:
                    fallback_file = log_file.with_name(f"{log_file.stem}_fallback.txt")
                    with open(fallback_file, 'w') as f:
                        f.write(f"GT rewards: {logs['gt_reward']}\n")
                        f.write(f"GPT rewards: {logs['gpt_reward']}\n")
                    logger.info(f"Saved fallback log to {fallback_file}")
                except Exception as e2:
                    logger.error(f"Error even with fallback logging: {e2}")
            
            logger.info(f"Run {run+1} completed successfully")
            
        except Exception as e:
            logger.error(f"Error during run {run+1}: {e}")
            logger.error(traceback.format_exc())
            logger.info(f"Run {run+1} failed")
            continue
    
    logger.info(f"All runs completed")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Export Directory: {export_dir}")
    
    # Make sure wandb is properly cleaned up before exiting
    if args.use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                logger.info("Training complete. Stopping wandb logging.")
                print("Training complete. Stopped wandb logging.")
                # Force sync to ensure all data is uploaded before terminating
                wandb.finish(quiet=True, exit_code=0)
                
                # Kill any lingering wandb processes
                try:
                    import psutil
                    current_process = psutil.Process()
                    for child in current_process.children(recursive=True):
                        if "wandb" in " ".join(child.cmdline()):
                            logger.info(f"Terminating wandb child process: {child.pid}")
                            child.terminate()
                except Exception as e:
                    logger.error(f"Error terminating wandb child processes: {e}")
                    
                # Set environment variable to disable wandb on exit
                os.environ["WANDB_DISABLED"] = "true"
        except Exception as e:
            logger.error(f"Error while stopping wandb: {e}")
    
    # Force stdout/stderr to flush to ensure all logs are written
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Small delay to ensure logs are written
    time.sleep(1)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 