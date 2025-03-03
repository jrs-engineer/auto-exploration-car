import time
import pybullet as p
from stable_baselines3 import PPO
from car_env import AutoCarEnv
# Import your custom policy kwargs from your training module.
from train_ppo import policy_kwargs

# Create the environment in render mode (for visualization)
env = AutoCarEnv(render=True, image_size=(84, 84))

# Provide custom objects to correctly deserialize the model.
custom_objects = {"policy_kwargs": policy_kwargs}

# Make sure to use the proper extension if needed (e.g., ".zip")
model = PPO.load("logs/best_model.zip", env=env, custom_objects=custom_objects)
print("Model loaded successfully!")

# Reset the environment to start a new episode.
obs = env.reset()
done = False

while not done:
    # Predict the next action (deterministic for evaluation)
    action, _ = model.predict(obs, deterministic=True)
    # import random
    # action = random.randint(3, 8)
    print("Action:", action)
    # Take a step in the environment.
    obs, reward, done, info = env.step(action)
    print("Reward:", reward)
    # Use a short sleep to keep the UI responsive.

env.close()
