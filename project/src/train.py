import gymnasium as gym
import highway_env
# from stable_baselines3 import DQN
from stable_baselines3 import TD3
# from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch
import matplotlib.pyplot as plt


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device  # For Apple Silicon Macs
print(f"Using device: {device}")
env = gym.make("parking-v0", render_mode="rgb_array")
env.reset()
model_save_path = "./models/parking_td3_model"
# model = TD3(
#     policy="MultiInputPolicy",
#     env=env,
#     verbose=1,
#     tensorboard_log="./tensorboard_logs/",
#     device=device,
#     learning_starts=1000,
#     batch_size=256,
#     learning_rate=1e-3,
# )
# model = TD3.load(model_save_path, env=env) if os.path.exists(model_save_path) else model

# train
# model.learn(total_timesteps=200_0)
# model.save(model_save_path)

# test
model = TD3.load(model_save_path, env=env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    frame = env.render()

    # 可选：可视化某一帧
    plt.imshow(frame)
    plt.axis('off')
    plt.pause(0.05)

env.close()