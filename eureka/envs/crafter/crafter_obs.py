import torch
import numpy as np

"""
Crafter is a survival crafting game where the player must collect resources to survive.
The observation is a 64x64x3 RGB image of the game state.

The player can perform the following actions:
0: noop
1: move left
2: move right
3: move up
4: move down
5: do (interact with objects)
6: sleep
7: place stone
8: place table
9: place furnace
10: place plant
11: make wood pickaxe
12: make stone pickaxe
13: make iron pickaxe 
14: make wood sword
15: make stone sword
16: make iron sword

The player's inventory includes:
- health: player's health (0-9)
- food: player's food level (0-9)
- drink: player's water level (0-9)
- energy: player's energy level (0-9)
- sapling: number of saplings collected
- wood: number of wood pieces collected
- stone: number of stones collected
- coal: number of coal pieces collected
- iron: number of iron pieces collected
- diamond: number of diamonds collected
- wood_pickaxe: whether player has a wood pickaxe (0 or 1)
- stone_pickaxe: whether player has a stone pickaxe (0 or 1)
- iron_pickaxe: whether player has an iron pickaxe (0 or 1)
- wood_sword: whether player has a wood sword (0 or 1)
- stone_sword: whether player has a stone sword (0 or 1)
- iron_sword: whether player has an iron sword (0 or 1)
- table: whether player has a table (0 or 1)
- furnace: whether player has a furnace (0 or 1)
- plant: whether player has a plant (0 or 1)
- meat: number of meat pieces collected

Achievements that can be unlocked:
- collect_coal: collected coal
- collect_diamond: collected diamond
- collect_drink: obtained water
- collect_iron: collected iron
- collect_sapling: collected sapling
- collect_stone: collected stone
- collect_wood: collected wood
- eat_cow: obtained meat from cow
- make_furnace: crafted furnace
- make_iron_pickaxe: crafted iron pickaxe
- make_iron_sword: crafted iron sword
- make_stone_pickaxe: crafted stone pickaxe
- make_stone_sword: crafted stone sword
- make_table: crafted table
- make_wood_pickaxe: crafted wood pickaxe
- make_wood_sword: crafted wood sword
- place_furnace: placed furnace
- place_plant: placed plant
- place_stone: placed stone
- place_table: placed table
- wake_up: woke up after sleeping
- defeat_skeleton: defeated a skeleton
- defeat_zombie: defeated a zombie
- eat_plant: ate a plant
- sapling_to_wood: converted sapling to wood
- wood_to_coal: converted wood to coal in furnace
- wood_to_iron: converted wood to iron in furnace
- sapling_to_coal: converted sapling to coal
- sapling_to_iron: converted sapling to iron
- sapling_to_cow: converted sapling to cow

The goal is to collect meat from cows, which requires a complex sequence of actions:
1. Collect saplings
2. Convert saplings to wood
3. Craft a wood pickaxe
4. Mine stone
5. Craft a stone sword
6. Plant saplings to grow cows
7. Hunt cows for meat

Eureka can create reward functions using two different approaches:

1. RGB-Based Reward Functions:
   These operate directly on the 64x64x3 RGB image observation.
   Function signature: compute_crafter_reward_rgb(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict]
   Where obs_buf is a batch of RGB images with shape [batch_size, 3, 64, 64]

2. Semantic-Based Reward Functions:
   These operate on a semantic observation containing object IDs and inventory information.
   Function signature: compute_crafter_reward_semantic(state: Dict) -> Tuple[torch.Tensor, Dict]
   Where state is a dictionary containing:
   - 'semantic': A tensor with shape [batch_size, 64, 64] containing object IDs
   - 'inventory': Dictionary of inventory items and counts
   - 'achievements': Dictionary of achievements and their status
   - 'meat_collected': Total meat collected in this episode
   - 'saplings_collected': Total saplings collected in this episode

   The semantic view uses these object IDs:
   0: map_edge
   1: water
   2: grass
   3: stone
   4: path
   5: sand
   6: tree
   7: lava
   8: coal
   9: iron
   10: diamond
   11: table
   12: furnace
   13: player
   14: cow
   15: zombie
   16: skeleton
   17: arrow
   18: plant

A good reward function should encourage the necessary steps to collect meat efficiently.
"""

# Example RGB-based reward function
def compute_crafter_reward_rgb(
    obs_buf: torch.Tensor
) -> torch.Tensor:
    """
    Compute reward for the Crafter environment based on RGB observation.
    
    Args:
        obs_buf: Batch of RGB observations with shape [batch_size, 3, 64, 64]
        
    Returns:
        Reward tensor and reward components dictionary
    """
    # Example implementation: Look for brown pixels (potential cows/animals)
    # and green pixels (potential plants/saplings)
    batch_size = obs_buf.shape[0]
    reward = torch.zeros(batch_size, 1, device=obs_buf.device)
    
    # Normalize pixel values to 0-1 range if needed
    if obs_buf.max() > 1.0:
        obs_buf = obs_buf / 255.0
    
    # Look for brown pixels (potential cows/animals)
    brown_mask = (obs_buf[:, 0] > obs_buf[:, 1]) & (obs_buf[:, 0] < 0.8)
    brown_reward = brown_mask.float().mean(dim=[1, 2]).unsqueeze(1) * 3.0
    
    # Look for green pixels (potential plants/saplings)
    green_mask = (obs_buf[:, 1] > obs_buf[:, 0]) & (obs_buf[:, 1] > obs_buf[:, 2])
    green_reward = green_mask.float().mean(dim=[1, 2]).unsqueeze(1) * 1.0
    
    # Combine rewards
    reward = brown_reward + green_reward
    
    reward_dict = {
        'brown_reward': brown_reward,
        'green_reward': green_reward
    }
    
    return reward, reward_dict

# Example semantic-based reward function
def compute_crafter_reward_semantic(
    state: dict
) -> torch.Tensor:
    """
    Compute reward for the Crafter environment based on semantic observations.
    
    Args:
        state: Dictionary containing semantic observations and inventory information
        
    Returns:
        Reward tensor and reward components dictionary
    """
    semantic = state['semantic']
    inventory = state['inventory']
    achievements = state['achievements']
    meat_collected = state.get('meat_collected', 0)
    saplings_collected = state.get('saplings_collected', 0)
    
    batch_size = semantic.shape[0]
    device = semantic.device
    reward = torch.zeros(batch_size, 1, device=device)
    
    # Reward for having meat in inventory
    meat_reward = torch.tensor(inventory.get('meat', 0), device=device).float() * 10.0
    
    # Reward for having saplings in inventory
    sapling_reward = torch.tensor(inventory.get('sapling', 0), device=device).float() * 1.0
    
    # Reward for having tools needed to collect meat
    tool_reward = torch.zeros(1, device=device)
    if inventory.get('wood_pickaxe', 0) > 0:
        tool_reward += 2.0
    if inventory.get('stone_pickaxe', 0) > 0:
        tool_reward += 3.0
    if inventory.get('wood_sword', 0) > 0:
        tool_reward += 3.0
    if inventory.get('stone_sword', 0) > 0:
        tool_reward += 5.0
    
    # Reward for achievements related to meat collection
    achievement_reward = torch.zeros(1, device=device)
    if achievements.get('sapling_to_cow', 0) > 0:
        achievement_reward += 5.0
    if achievements.get('eat_cow', 0) > 0:
        achievement_reward += 5.0
    
    # Count cows in the semantic view (ID 14)
    cow_count = (semantic == 14).float().sum(dim=[1, 2]).unsqueeze(1)
    cow_reward = cow_count * 2.0
    
    # Combine all rewards
    reward = meat_reward + sapling_reward + tool_reward + achievement_reward + cow_reward
    
    reward_dict = {
        'meat_reward': meat_reward,
        'sapling_reward': sapling_reward,
        'tool_reward': tool_reward,
        'achievement_reward': achievement_reward,
        'cow_reward': cow_reward
    }
    
    return reward, reward_dict 