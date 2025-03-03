import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from train_ppo import create_envs, create_model, TrainingConfig
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

def objective(trial):
    config = TrainingConfig()
    
    # Hyperparameter search space
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "n_epochs": trial.suggest_int("n_epochs", 3, 10),
        "net_arch": trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    }

    # Network architecture mapping
    arch_config = {
        "small": dict(pi=[64], vf=[64]),
        "medium": dict(pi=[128, 64], vf=[128, 64]),
        "large": dict(pi=[256, 128], vf=[256, 128])
    }
    config.policy_kwargs['net_arch'] = arch_config[hyperparams.pop("net_arch")]

    # Create environments
    train_env = create_envs(config)
    eval_env = create_envs(config, eval=True)

    # Create model with trial parameters
    model = create_model(train_env, config, hyperparams)

    # Create unique trial directory first
    trial_dir = f"./logs/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=trial_dir,
        log_path=trial_dir,
        eval_freq=100,
        deterministic=True,
        render=False
    )

    try:
        model.learn(
            total_timesteps=100000,  # Reduced for faster trials
            callback=eval_callback,
            progress_bar=False
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("-inf")

    # Load evaluation rewards from trial-specific file
    eval_file = os.path.join(trial_dir, "eval.monitor.csv")
    if os.path.exists(eval_file):
        eval_rewards = np.loadtxt(eval_file, skiprows=2, delimiter=",")[-10:, 0]
    else:
        eval_rewards = [0]  # Handle missing file case
    
    mean_reward = np.mean(eval_rewards)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return mean_reward

def optimize_hyperparameters():
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage="sqlite:///logs/ppo_study.db"
    )

    study.optimize(objective, n_trials=50, n_jobs=4, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    with open("best_params.txt", "w") as f:
        f.write(str(trial.params))

if __name__ == "__main__":
    optimize_hyperparameters()