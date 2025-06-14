from stable_baselines3 import TD3, SAC, PPO
import gymnasium as gym
from parking_config_4_image import config
import HighwayEnv.highway_env
import imageio.v2 as imageio  # use v2 to avoid warning
import os


def save_last_frame_png(model_save_path, obs_name, algo, diff, device):
    model_class = {"TD3": TD3, "SAC": SAC, "PPO": PPO}[algo]
    config["vehicles_count"] = {"empty": 0, "normal": 3, "almost_full": 10}[diff]

    config["manual_control"] = False
    config["offscreen_rendering"] = True
    config["render_agent"] = True

    env = gym.make("parking-v0", config=config, render_mode="rgb_array")
    env.reset()
    model = model_class.load(model_save_path, env=env, device=device)

    obs, _ = env.reset()
    done = False
    last_frame = None

    env.render()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # env.unwrapped.world_surface.move_display_window_to(config["centering_position"])
        # env.unwrapped.viewer.reset_window_position()
        last_frame = env.render()

    env.close()

    os.makedirs("images", exist_ok=True)
    img_path = f"images/2/{obs_name}_{algo}_{diff}.png"
    imageio.imwrite(img_path, last_frame)
    print(f"Saved final frame to {img_path}")


if __name__ == "__main__":
    list_observations = ["kine", "lidar", "vision"]
    list_observations = list_observations[:1]
    list_algos = ["TD3", "SAC", "PPO"]
    list_diffs = ["empty", "normal", "almost_full"]
    for obs_name in list_observations:
        for algo in list_algos:
            for diff in list_diffs:
                model_save_path = f"./models/{obs_name}_{algo}_{diff}.zip"
                if not os.path.exists(model_save_path):
                    continue
                save_last_frame_png(model_save_path, obs_name, algo, diff, device="cpu")
                
    # obs_name = "kine"  # kine (radar), lidar, vision
    # algo = "TD3"       # TD3, SAC, PPO
    # diff = "normal"    # empty, normal, almost_full
    # device = "cpu"
    # model_save_path = f"./models/{obs_name}_{algo}_{diff}.zip"
    # save_last_frame_png(model_save_path, obs_name, algo, diff, device)
