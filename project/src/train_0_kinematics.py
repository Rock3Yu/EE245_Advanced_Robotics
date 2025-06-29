import gymnasium as gym
import highway_env
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch
from parking_config import config
import argparse

TOTAL_TIMESTEPS = 500_000


def train(algo, diff, device):
    name = f"kine_{algo}_{diff}"
    model_class = {"TD3": TD3, "SAC": SAC, "PPO": PPO}[algo]
    config["vehicles_count"] = {"empty": 0, "normal": 3, "almost_full": 10}[diff]
    env = gym.make("parking-v0", render_mode="rgb_array", config=config)
    env.reset()
    model_save_path = f"./models/{name}"
    if algo != "PPO":
        model = model_class(
            policy="MultiInputPolicy",
            env=env,
            buffer_size=10_000,
            verbose=1,
            tensorboard_log=f"./tensorboard_logs/{name}/",
            device=device,
            learning_starts=1000,
            batch_size=256,
            learning_rate=1e-3,
        )
    else:
        model = model_class(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            device=device,
            learning_rate=1e-3,
            tensorboard_log=f"./tensorboard_logs/{name}/",
        )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_save_path)
    return model_save_path, model, env


def test(model_save_path, algo, diff, device):
    model_class = {"TD3": TD3, "SAC": SAC, "PPO": PPO}[algo]
    config["vehicles_count"] = {"empty": 0, "normal": 3, "almost_full": 10}[diff]
    env = gym.make("parking-v0", render_mode="rgb_array", config=config)
    env.reset()
    model = model_class.load(model_save_path, env=env, device=device)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"Avg reward: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=int, default=0, help="Algorithm index: 0=TD3, 1=SAC, 2=PPO")
    parser.add_argument("-d", type=int, default=0, help="Difficulty index: 0=empty, 1=normal, 2=almost_full")
    parser.add_argument("--test", action="store_true", default=False, help="Test the trained model after training")
    args = parser.parse_args()
    
    algos = ["TD3", "SAC", "PPO"][args.a]
    env_difficulties = ["empty", "normal", "almost_full"][args.d]
    algo, diff = algos, env_difficulties
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{device}] - Training {algo} with kinematics observation in {diff} difficulty...")

    model_save_path, model, env = train(algo, diff, device)

    if args.test:
        test(model_save_path, algo, diff, device)
