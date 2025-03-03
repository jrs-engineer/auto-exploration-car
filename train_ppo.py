import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from car_env import AutoCarEnv
import os
import torch
import torch.nn as nn

# --- Configuration ---
class TrainingConfig:
    def __init__(self):
        self.num_cpu = 4
        self.image_size = (84, 84)
        self.total_timesteps = 100000
        self.batch_size = 64 
        self.policy_kwargs = {
            'features_extractor_class': CustomCNN,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[64, 64], vf=[64, 64]),
            'activation_fn': nn.ReLU
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CNN Architecture ---
class CustomCNN(nn.Module):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__()
        n_input_channels = observation_space.shape[2]  # Assuming (H, W, C)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Reduces spatial size to 1x1
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Rearrange from (batch, H, W, C) to (batch, C, H, W)
        observations = observations.permute(0, 3, 1, 2)
        features = self.cnn(observations)
        return self.linear(features)

# --- Environment Creation ---
def create_envs(config, eval=False):
    def make_env():
        return AutoCarEnv(render=False, image_size=config.image_size)
    
    if eval:
        env = DummyVecEnv([lambda: Monitor(make_env(), filename="./logs/eval")])
        return VecNormalize(env, norm_obs=True, norm_reward=True, training=False)
    
    env = DummyVecEnv([make_env for _ in range(config.num_cpu)])
    return VecNormalize(env, norm_obs=True, norm_reward=True)

# --- Model Management ---
def create_model(env, config, hyperparams=None):
    model_path = "ppo_autocar_model"
    
    if os.path.exists(f"{model_path}.zip"):
        return PPO.load(model_path, env=env, device=config.device)
    
    # Merge config defaults with hyperparameters
    params = {
        "batch_size": config.batch_size,
        "policy_kwargs": config.policy_kwargs,
        "device": config.device
    }
    
    if hyperparams:
        params.update(hyperparams)

    return PPO(
        "CnnPolicy",
        env,
        verbose=1,
        **params
    )

# --- Training Execution ---
def train_model(hyperparams=None):
    config = TrainingConfig()
    
    # Create environments
    train_env = create_envs(config)
    eval_env = create_envs(config, eval=True)

    # Create model
    model = create_model(train_env, config, hyperparams)

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10,
        deterministic=True,
        render=False
    )

    # Train
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)
    model.save("ppo_autocar_model")
    
    # Cleanup
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    train_model()