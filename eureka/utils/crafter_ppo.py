"""
Custom PPO implementation for Crafter environment that works with Eureka-generated reward functions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import wandb
from typing import Dict, List, Tuple, Optional, Callable

class EurekaCrafterPPO:
    """
    PPO algorithm for Crafter environment that integrates with Eureka-generated reward functions.
    
    This implementation is designed to:
    1. Track metrics specific to the Crafter task (meat and saplings collected)
    2. Use the Eureka-generated reward function
    3. Log results to wandb
    """
    
    def __init__(
        self,
        env: gym.Env,
        reward_function: Callable = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_wandb: bool = True,
        update_epochs: int = 10,
        batch_size: int = 64,
        history_length: int = 5000,
    ):
        self.env = env
        self.reward_function = reward_function
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Initialize the actor-critic network
        self._initialize_networks()
        
        # Initialize buffers for storing trajectories
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.custom_rewards = []
        self.values = []
        self.dones = []
        self.meat_collected = []
        self.saplings_collected = []
        
        # Track metrics
        self.total_steps = 0
        self.episode_count = 0
        self.max_history_length = history_length
        self.episode_rewards = []
        self.episode_lengths = []
        self.meat_counts = []
        self.sapling_counts = []
    
    def _initialize_networks(self):
        """Initialize the policy and value networks."""
        obs_shape = self.env.observation_space.shape
        action_dim = self.env.action_space.n
        
        # Simple CNN architecture for Crafter's 64x64x3 RGB observations
        self.actor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action from the policy distribution.
        
        Args:
            state: The current observation
            
        Returns:
            action: The selected action
            action_prob: The probability of the selected action
            value: The value estimate of the current state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # Sample an action
            action = dist.sample()
            action_prob = probs[0, action.item()].item()
            
        return action.item(), action_prob, value.item()
    
    def compute_custom_reward(self, state: np.ndarray) -> float:
        """
        Compute reward using the Eureka-generated reward function.
        
        Args:
            state: The current observation
            
        Returns:
            reward: The custom reward value
        """
        if self.reward_function is None:
            return 0.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            reward = self.reward_function(state_tensor).item()
        return reward
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        action_prob: float,
        reward: float,
        custom_reward: float,
        value: float,
        done: bool,
        meat: int,
        saplings: int
    ):
        """Store a transition in the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.custom_rewards.append(custom_reward)
        self.values.append(value)
        self.dones.append(done)
        self.meat_collected.append(meat)
        self.saplings_collected.append(saplings)
        
        # Limit buffer size
        if len(self.states) > self.max_history_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.action_probs.pop(0)
            self.rewards.pop(0)
            self.custom_rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)
            self.meat_collected.pop(0)
            self.saplings_collected.pop(0)
    
    def compute_returns_and_advantages(
        self,
        next_value: float,
        use_custom_rewards: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Compute the returns and advantages for PPO update.
        
        Args:
            next_value: The value estimate of the next state
            use_custom_rewards: Whether to use Eureka-generated rewards
            
        Returns:
            returns: The discounted returns
            advantages: The advantages for each transition
        """
        rewards = self.custom_rewards if use_custom_rewards else self.rewards
        
        returns = []
        advantages = []
        
        next_return = next_value
        next_advantage = 0
        
        for i in reversed(range(len(rewards))):
            if self.dones[i]:
                next_return = 0
                next_advantage = 0
                
            current_return = rewards[i] + self.gamma * next_return * (1 - self.dones[i])
            current_advantage = current_return - self.values[i]
            
            returns.insert(0, current_return)
            advantages.insert(0, current_advantage)
            
            next_return = current_return
            next_advantage = current_advantage
            
        return returns, advantages
    
    def update(self, next_value: float = 0.0, use_custom_rewards: bool = True):
        """Update the policy and value networks."""
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value, use_custom_rewards)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_action_probs = torch.FloatTensor(self.action_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update in mini-batches
        batch_size = min(self.batch_size, len(states))
        indices = np.arange(len(states))
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_action_probs = old_action_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                logits = self.actor(batch_states)
                values = self.critic(batch_states).squeeze()
                
                # Calculate action probabilities
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                # Get log probabilities of the actions
                log_probs = dist.log_prob(batch_actions)
                
                # Calculate the entropy
                entropy = dist.entropy().mean()
                
                # Calculate the ratio (policy / old_policy)
                ratio = torch.exp(log_probs - torch.log(batch_old_action_probs + 1e-10))
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass and update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        # Clear buffers after update
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.custom_rewards = []
        self.values = []
        self.dones = []
        self.meat_collected = []
        self.saplings_collected = []
    
    def train(self, num_episodes: int = 1000, update_frequency: int = 2048, log_frequency: int = 10):
        """Train the agent for a specified number of episodes."""
        total_env_rewards = 0
        total_custom_rewards = 0
        episode_length = 0
        total_meat = 0
        total_saplings = 0
        
        obs = self.env.reset()
        current_meat = 0
        current_saplings = 0
        
        for episode in range(num_episodes):
            done = False
            episode_env_reward = 0
            episode_custom_reward = 0
            episode_length = 0
            episode_meat = 0
            episode_saplings = 0
            
            while not done:
                # Get action from the policy
                action, action_prob, value = self.get_action(obs)
                
                # Take a step in the environment
                next_obs, env_reward, done, info = self.env.step(action)
                
                # Compute custom reward using Eureka function
                custom_reward = self.compute_custom_reward(next_obs)
                
                # Track meat and saplings
                prev_meat = current_meat
                prev_saplings = current_saplings
                current_meat = info.get('inventory', {}).get('meat', 0)
                current_saplings = info.get('inventory', {}).get('sapling', 0)
                
                if current_meat > prev_meat:
                    episode_meat += (current_meat - prev_meat)
                if current_saplings > prev_saplings:
                    episode_saplings += (current_saplings - prev_saplings)
                
                # Store the transition
                self.store_transition(
                    obs, action, action_prob, env_reward, custom_reward, 
                    value, done, current_meat, current_saplings
                )
                
                # Update the current observation
                obs = next_obs
                
                # Update counters
                self.total_steps += 1
                episode_length += 1
                episode_env_reward += env_reward
                episode_custom_reward += custom_reward
                
                # Update the policy if we have collected enough steps
                if self.total_steps % update_frequency == 0:
                    # Get value estimate for the next state (for bootstrapping)
                    _, _, next_value = self.get_action(next_obs)
                    self.update(next_value, use_custom_rewards=True)
            
            # Episode completed
            self.episode_count += 1
            
            # If episode terminated with done flag, reset environment
            if done:
                obs = self.env.reset()
                current_meat = 0
                current_saplings = 0
            
            # Store episode statistics
            self.episode_rewards.append(episode_env_reward)
            self.episode_lengths.append(episode_length)
            self.meat_counts.append(episode_meat)
            self.sapling_counts.append(episode_saplings)
            
            # Log episode metrics
            if episode % log_frequency == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(self.episode_rewards[-log_frequency:])
                avg_length = np.mean(self.episode_lengths[-log_frequency:])
                avg_meat = np.mean(self.meat_counts[-log_frequency:])
                avg_saplings = np.mean(self.sapling_counts[-log_frequency:])
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Avg Meat: {avg_meat:.2f} | "
                      f"Avg Saplings: {avg_saplings:.2f}")
                
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "avg_reward": avg_reward,
                        "avg_custom_reward": np.mean(self.custom_rewards[-update_frequency:]) if self.custom_rewards else 0,
                        "avg_episode_length": avg_length,
                        "avg_meat_collected": avg_meat,
                        "avg_saplings_collected": avg_saplings,
                        "total_steps": self.total_steps
                    })
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "meat_counts": self.meat_counts,
            "sapling_counts": self.sapling_counts
        }
    
    def save(self, path: str):
        """Save the model to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the model from the specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Model loaded from {path}") 