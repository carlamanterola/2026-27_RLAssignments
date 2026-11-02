import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import snake_env  # noqa: F401 - Registers Snake environments via gymnasium.register

ALGORITHM = "PPO"
ENV_ID = "Snake-v3"
N_ENVS = 8
TIMESTEPS = 2_000_000

models_dir = f"models/{ALGORITHM}_{ENV_ID}"   # keeps checkpoints of different variants apart
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


class ApplesCallback(BaseCallback):
    """Logs the mean apples eaten per episode to TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None and "apples" in episode:
                self.logger.record_mean("rollout/apples_per_episode", episode["apples"])
        return True


monitor_kwargs = dict(info_keywords=("apples",))
env = make_vec_env(ENV_ID, n_envs=N_ENVS, monitor_kwargs=monitor_kwargs)
eval_env = make_vec_env(ENV_ID, n_envs=1, monitor_kwargs=monitor_kwargs)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{models_dir}/best_model",
    log_path=f"{log_dir}/eval",
    eval_freq=max(10_000 // N_ENVS, 1),  # counted per env step
    n_eval_episodes=20,
    deterministic=True,
)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=256,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    learning_rate=lambda progress_remaining: 3e-4 * progress_remaining,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    device="cpu",
    tensorboard_log=log_dir,
    verbose=1,
)

model.learn(
    total_timesteps=TIMESTEPS,
    tb_log_name=ALGORITHM,
    callback=[eval_callback, ApplesCallback()],
)
model.save(f"{models_dir}/{TIMESTEPS}")

env.close()
eval_env.close()