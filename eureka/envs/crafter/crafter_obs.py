import sys
import os
import torch
import numpy as np

# Add the path to the crafter module
CRAFTER_PATH = "/home/mr971/strategist/data/crafter"
if CRAFTER_PATH not in sys.path:
    sys.path.append(CRAFTER_PATH)

import crafter

def get_obs_dict():
    """
    Returns a dictionary of observation components for the Crafter environment.
    This will be used by the GPT model to understand what's available in observations.
    """
    obs_dict = {
        "obs": "Observation tensor with shape (64, 64, 3) containing the RGB pixel values of the game state",
        "inventory": "Dictionary containing the current player's inventory (wood, stone, iron, etc.)",
        "achievements": "Dictionary containing the player's achievements and their counts",
        "semantic": "Semantic map of the environment showing entities",
        "player_pos": "Position of the player in the world as (x, y) coordinates",
        "player_facing": "Direction the player is facing",
    }
    return obs_dict

class CrafterObservation:
    """Example class showing observation structure for the Crafter environment"""
    
    def __init__(self):
        # Create a dummy environment to get sample observations
        self.env = crafter.Env()
        self.obs = self.env.reset()
        self.action_space = self.env.action_space
        _, _, _, self.info = self.env.step(0)  # Take a dummy step to get info dict
    
    def get_sample_obs(self):
        """Return a sample observation from the environment"""
        return {
            "obs": self.obs,  # RGB image of the game state
            "inventory": self.info["inventory"],
            "achievements": self.info["achievements"],
            "semantic": self.info["semantic"],
            "player_pos": self.info["player_pos"],
            "player_facing": self.info["player_facing"],
        }
    
    def get_action_space(self):
        """Return the action space of the environment"""
        return self.env.action_names 