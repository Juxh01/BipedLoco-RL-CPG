import os

os.environ["MUJOCO_GL"] = "glfw"


import shutil
from datetime import datetime
from pathlib import Path

import hydra
from myosuite.utils import gym
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

timestamp_load = None


def play():
    # Geht bi mir (Max) nur mit nvidia: __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python source/experiments/exp.py

    # Einzelne Env (kein VecEnv)
    env = gym.make("myoLegWalk-v0", reset_type="random")

    # Modell laden (achte auf die .zip-Endung)
    model = PPO.load("ppo_walker.zip", device="cpu")

    obs, _ = env.reset()
    while True:
        # deterministic=True für stabilere Demo
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Render ruft der Env-Treiber selbst auf, aber explizit geht auch:
        env.unwrapped.mj_render()

        if terminated or truncated:
            obs, _ = env.reset()


CONFIG_DIR = Path(__file__).parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # environments
    # env = gym.make('myoLegWalk-v0', reset_type='random')
    env = make_vec_env("myoLegWalk-v0", n_envs=12)

    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=cfg.agent.learning_rate,
        n_steps=cfg.agent.n_steps,
        batch_size=cfg.agent.batch_size,
        n_epochs=cfg.agent.n_epochs,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_range=cfg.agent.clip_range,
        verbose=1,
    )

    if timestamp_load is not None:
        load_dir = os.path.join(
            hydra.utils.get_original_cwd(),
            "RAW_Data",
            timestamp_load,
            "models",
            f"model_{timestamp_load}.zip",
        )
        model = PPO.load(load_dir, env=env, device="cpu")

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(cfg.train.num_frames), progress_bar=True)
    # Save the agent
    model.save("ppo_walker")

    working_dir = os.path.join(hydra.utils.get_original_cwd(), "RAW_Data", timestamp)
    training_data_dir = os.path.join(working_dir, "training_data")
    model_dir = os.path.join(working_dir, "models")

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # save yaml config to working directory
    try:
        # Originale config.yaml kopieren
        original_config = os.path.join(
            hydra.utils.get_original_cwd(), "configs/example.yaml"
        )
        config_destination = os.path.join(working_dir, "config.yaml")
        shutil.copy2(original_config, config_destination)
        print(f"Config gespeichert unter: {config_destination}")
    except FileNotFoundError:
        print("Config-Datei nicht gefunden")

    # save model

    save_path = os.path.join(model_dir, f"model_{timestamp}")
    model.save(save_path)
    print(f"Modell gespeichert unter: {save_path}")

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")
    play()


if __name__ == "__main__":
    main()
