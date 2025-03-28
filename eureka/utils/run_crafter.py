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

# Add crafter to the path
CRAFTER_PATH = "/home/mr971/strategist/data/crafter"
if CRAFTER_PATH not in sys.path:
    sys.path.append(CRAFTER_PATH)

import crafter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_reward_function(file_path):
    """Load the reward function from a file"""
    try:
        spec = importlib.util.spec_from_file_location("reward_module", file_path)
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        return reward_module.compute_reward_gpt
    except Exception as e:
        logger.error(f"Error loading reward function: {e}")
        raise

def load_config(config_path):
    """Load configuration from YAML file"""
    if not config_path or not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_prompt(config):
    """Load prompt from config's game_prompt_file"""
    if not config or 'game_prompt_file' not in config:
        return None
    
    prompt_path = f"/home/mr971/strategist/prompts/{config['game_prompt_file']}"
    if not os.path.exists(prompt_path):
        logger.warning(f"Prompt file not found: {prompt_path}")
        return None
    
    with open(prompt_path, 'r') as f:
        prompt_content = yaml.safe_load(f)
    return prompt_content

def create_output_dirs(reward_file):
    """Create output directories for logs and videos"""
    base_name = Path(reward_file).stem
    log_dir = Path(f"./logs/{base_name}")
    video_dir = Path(f"./videos/{base_name}")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    return log_dir, video_dir

def main():
    parser = argparse.ArgumentParser(description="Run Crafter with a GPT-generated reward function")
    parser.add_argument('--reward_file', type=str, required=True, help='Path to the reward function file')
    parser.add_argument('--config_path', type=str, default='', help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=3000, help='Number of training iterations')
    parser.add_argument('--capture_video', action='store_true', help='Whether to capture video')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config_path)
    
    # Load prompt if available
    prompt = load_prompt(config)
    if prompt:
        logger.info(f"Loaded prompt from {config.get('game_prompt_file')}")
        if 'goal_context' in prompt:
            logger.info(f"Task goal: {prompt['goal_context']}")
    
    # Load the reward function
    reward_fn = load_reward_function(args.reward_file)
    
    # Create output directories
    log_dir, video_dir = create_output_dirs(args.reward_file)
    
    # Create a tensorboard-like log file
    log_file = log_dir / "training_log.yaml"
    logs = {"gt_reward": [], "gpt_reward": {}}
    
    # Create environment
    env = crafter.Env(reward=True, custom_reward_func=None)
    obs = env.reset()
    
    # Training loop
    total_reward = 0
    total_gpt_reward = 0
    episodes = 0
    frames = 0
    
    # Convert obs to tensor for the reward function
    def obs_to_tensor(obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float()
        return torch.tensor(obs).float()
    
    start_time = time.time()
    logger.info(f"Starting training for {args.iterations} iterations")
    
    for i in range(args.iterations):
        action = env.action_space.sample()  # Random policy for simplicity
        
        # Take a step in the environment
        next_obs, reward, done, info = env.step(action)
        
        # Calculate GPT reward
        obs_tensor = obs_to_tensor(obs)
        gpt_reward_tensor, reward_dict = reward_fn(obs_tensor.unsqueeze(0))
        gpt_reward = gpt_reward_tensor.item()
        
        # Log rewards
        logs["gt_reward"].append(reward)
        
        # Log reward components
        for key, value in reward_dict.items():
            if key not in logs["gpt_reward"]:
                logs["gpt_reward"][key] = []
            logs["gpt_reward"][key].append(value.item())
        
        total_reward += reward
        total_gpt_reward += gpt_reward
        frames += 1
        
        if args.capture_video and i % 100 == 0:
            # Save a frame
            plt.figure(figsize=(8, 8))
            plt.imshow(env.render())
            plt.title(f"Frame {frames}")
            plt.savefig(video_dir / f"frame_{frames:05d}.png")
            plt.close()
        
        # Print progress
        if i % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Iteration {i}/{args.iterations} | "
                      f"GT Reward: {total_reward/max(1, episodes):.2f} | "
                      f"GPT Reward: {total_gpt_reward/max(1, episodes):.2f} | "
                      f"FPS: {frames/elapsed:.2f}")
        
        if done:
            obs = env.reset()
            episodes += 1
        else:
            obs = next_obs
    
    # Save the logs
    with open(log_file, 'w') as f:
        yaml.dump(logs, f)
    
    # Print final stats
    logger.info(f"Training completed in {time.time() - start_time:.2f}s")
    logger.info(f"Total episodes: {episodes}")
    logger.info(f"Average GT reward: {total_reward/max(1, episodes):.2f}")
    logger.info(f"Average GPT reward: {total_gpt_reward/max(1, episodes):.2f}")
    
    # Save tensorboard-compatible log file
    logger.info(f"Tensorboard Directory: {log_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 