import gymnasium as gym
import snake_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = gym.make("Snake-v0", render_mode="rgb_array")
eval_env = gym.make("Snake-v0", render_mode="rgb_array")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{models_dir}/best_model",
    log_path="./logs/eval",
    eval_freq=10000,
    deterministic=True,
)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 100000
NUM_ITERATIONS = 1  # Adjust according to your needs

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=True, # usado para seguir o no el training con el num de timesteps global
        tb_log_name=ALGORITHM,
        callback=eval_callback,
    )
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
