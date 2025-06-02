from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from parking_config import config
import highway_env
import imageio
import os


def make_gif(model_save_path, obs_name, algo, diff, device):
    model_class = {"TD3": TD3, "SAC": SAC, "PPO": PPO}[algo]
    config["vehicles_count"] = {"empty": 0, "normal": 3, "almost_full": 10}[diff]

    config["manual_control"] = False
    config["offscreen_rendering"] = True
    config["render_agent"] = True

    env = gym.make("parking-v0", config=config, render_mode="rgb_array")
    env.reset()
    model = model_class.load(model_save_path, env=env, device=device)

    images = []
    num_episodes = 5

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # render and append image
            frame = env.render()
            images.append(frame)

    env.close()

    os.makedirs("gifs", exist_ok=True)
    gif_path = f"gifs/{obs_name}_{algo}_{diff}.gif"
    imageio.mimsave(gif_path, images, fps=15)
    print(f"Saved gif to {gif_path}")


if __name__ == "__main__":
    model_save_path = "./models/kine_TD3_normal.zip"
    obs_name = "kine"
    algo = "TD3"
    diff = "normal"
    device = "cpu"
    make_gif(model_save_path, obs_name, algo, diff, device)
