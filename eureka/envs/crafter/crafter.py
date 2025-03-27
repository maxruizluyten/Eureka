import os
import numpy as np
import torch
import crafter
import gym

class CrafterEnv(gym.Wrapper):
    """
    Crafter environment wrapper for Eureka.
    This adapts the Crafter environment to work with Eureka's reward function generation.
    The environment provides both RGB and semantic views to support different approaches to reward function generation.
    """
    
    def __init__(self, task, rl_device="cuda", sim_device=None, graphics_device_id=None, headless=True, 
                 virtual_screen_capture=False, force_render=False, reward_type="semantic"):
        self.cfg = task
        self.reward_type = reward_type  # Can be "semantic" or "rgb"
        self.rl_device = rl_device
        
        # Initialize the Crafter environment
        if self.cfg.get("custom_reward_func", None) is not None:
            self.env = crafter.Env(custom_reward_func=self.cfg["custom_reward_func"])
        else:
            self.env = crafter.Env()
        
        super().__init__(self.env)
        
        # Setup observation and action spaces to match Eureka's expectations
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Semantic metadata for better understanding of the environment
        self.semantic_metadata = {
            0: "map_edge",
            1: "water",
            2: "grass",
            3: "stone",
            4: "path",
            5: "sand",
            6: "tree",
            7: "lava",
            8: "coal",
            9: "iron",
            10: "diamond",
            11: "table",
            12: "furnace",
            13: "player",
            14: "cow",
            15: "zombie",
            16: "skeleton",
            17: "arrow",
            18: "plant"
        }
        
        # Initialize tracking for achievements and inventories
        self.num_envs = 1  # Crafter runs one environment at a time
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=rl_device)
        self.rew_dict = {}
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=rl_device)
        self.extras = {}
        self.episode_length = self.cfg.get("episodeLength", 10000)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=rl_device)
        
        # Initialize tracking variables for meat and saplings
        self.last_meat = 0
        self.last_saplings = 0
        self.total_meat_collected = 0
        self.total_saplings_collected = 0
        
    def reset(self):
        """Reset the environment and return the initial observation."""
        obs = self.env.reset()
        self.progress_buf[:] = 0
        self.reset_buf[:] = 0
        self.rew_buf[:] = 0
        self.rew_dict = {}
        self.extras = {}
        
        # Reset tracking variables
        self.last_meat = 0
        self.last_saplings = 0
        self.total_meat_collected = 0
        self.total_saplings_collected = 0
        
        # Return the observation based on the reward type
        if self.reward_type == "semantic" and "_sem_view" in dir(self.env):
            semantic_obs = self.env._sem_view()
            self.extras['semantic'] = semantic_obs
            return obs  # Still return RGB for the agent, semantic is used for reward only
        else:
            return obs
    
    def step(self, action):
        """Step the environment with the given action."""
        obs, reward, done, info = self.env.step(action)
        
        # Track meat and saplings collected
        current_meat = info.get('inventory', {}).get('meat', 0)
        current_saplings = info.get('inventory', {}).get('sapling', 0)
        
        if current_meat > self.last_meat:
            self.total_meat_collected += (current_meat - self.last_meat)
        if current_saplings > self.last_saplings:
            self.total_saplings_collected += (current_saplings - self.last_saplings)
            
        self.last_meat = current_meat
        self.last_saplings = current_saplings
        
        # Update Eureka's tracking variables
        self.progress_buf += 1
        self.rew_buf[0] = reward
        self.extras['env_reward'] = reward
        self.extras['meat_collected'] = self.total_meat_collected
        self.extras['saplings_collected'] = self.total_saplings_collected
        self.extras['inventory'] = info.get('inventory', {})
        self.extras['achievements'] = info.get('achievements', {})
        self.reset_buf[0] = done
        
        # Get semantic view if available and requested
        if self.reward_type == "semantic" and 'semantic' in info:
            self.extras['semantic'] = info['semantic']
        
        if done or self.progress_buf[0] >= self.episode_length:
            self.reset_buf[0] = True
        
        return obs, reward, done, info
    
    def compute_reward(self):
        """Placeholder for Eureka to inject the reward function."""
        return self.rew_buf, self.rew_dict
    
    def get_state_for_reward_function(self):
        """
        Returns the current state in the format expected by the reward function.
        This can be either the RGB observation or the semantic view based on the reward_type.
        """
        if self.reward_type == "semantic" and 'semantic' in self.extras:
            # Convert semantic view to tensor for reward function
            semantic = self.extras['semantic']
            semantic_tensor = torch.tensor(semantic, dtype=torch.float32, device=self.rl_device).unsqueeze(0)
            return {
                'semantic': semantic_tensor,
                'inventory': self.extras.get('inventory', {}),
                'achievements': self.extras.get('achievements', {}),
                'meat_collected': self.extras.get('meat_collected', 0),
                'saplings_collected': self.extras.get('saplings_collected', 0)
            }
        else:
            # Use the RGB observation
            obs = self.env._obs()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.rl_device).permute(2, 0, 1).unsqueeze(0)
            return obs_tensor 