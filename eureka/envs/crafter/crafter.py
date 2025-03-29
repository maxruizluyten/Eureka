import sys
import os
import numpy as np
import torch
import gym

# Add the path to the crafter module
CRAFTER_PATH = "/home/mr971/strategist/data/crafter"
if CRAFTER_PATH not in sys.path:
    sys.path.append(CRAFTER_PATH)

import crafter

class CrafterTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = crafter.Env(reward=True, length=10000, seed=None)
        self.observation_space = gym.spaces.Box(0, 255, tuple(self.env._size) + (3,), np.uint8)
        self.action_space = gym.spaces.Discrete(len(self.env.action_names))
        self.device = "cuda:0"
        self.rew_buf = torch.zeros(1, device=self.device)
        self.rew_dict = {}
        self.reset()
        
    def reset(self):
        self.obs = self.env.reset()
        self.rew_buf = torch.zeros(1, device=self.device)
        self.rew_dict = {}
        return self.obs
        
    def step(self, action):
        action = action.item() if isinstance(action, torch.Tensor) else action
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        self.rew_buf = torch.tensor([reward], device=self.device)
        # Set rewards in dict - for Eureka compatibility
        self.rew_dict["environment_reward"] = self.rew_buf
        return obs, self.rew_buf, done, info
        
    def compute_reward(self):
        """This method will be replaced by GPT-generated reward functions"""
        self.rew_buf[:], self.rew_dict = compute_reward_gpt(self.obs)
        self.extras = {}
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict:
            self.extras[rew_state] = self.rew_dict[rew_state].mean()
        return self.rew_buf, self.extras
        
# Default reward function (to be replaced by GPT)
def compute_reward_gpt(obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # Default implementation - just return a constant reward
    reward = torch.tensor([0.0])
    rew_dict = {"default_reward": reward}
    return reward, rew_dict 