from pathlib import Path

import hydra
from omegaconf import DictConfig

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    env = EnvironmentHandler.create_environment(cfg)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Environment Obs: {env.observation_space}")
    print(f"Environment Act: {env.action_space}")


    while True:
            # deterministic=True für stabilere Demo
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Render ruft der Env-Treiber selbst auf, aber explizit geht auch:
            env.unwrapped.mj_render()

            if terminated or truncated:
                obs, _ = env.reset()


    return 0.0





if __name__ == "__main__":
    main()
