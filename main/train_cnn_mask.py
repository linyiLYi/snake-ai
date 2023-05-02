import os
import sys
import random

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

NUM_ENV = 32
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    # lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    # clip_range_schedule = linear_schedule(0.15, 0.025)

    # Instantiate a PPO agent
    # model = MaskablePPO(
    #     "CnnPolicy", 
    #     env, 
    #     device="cuda",
    #     verbose=1,
    #     n_steps=2048,
    #     batch_size=512,
    #     n_epochs=4,
    #     gamma=0.94,
    #     learning_rate=lr_schedule,
    #     clip_range=clip_range_schedule,
    #     tensorboard_log=LOG_DIR
    # )

    # finetune 01
    # lr_schedule = linear_schedule(1.25e-5, 2.5e-6)
    # clip_range_schedule = linear_schedule(0.03, 0.025)
    
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule
    # }
    
    # model_path = "trained_models_cnn/ppo_snake_100000000_steps.zip"
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # finetune 02 & 03 & 04
    lr_schedule = linear_schedule(5e-5, 2.5e-6)
    clip_range_schedule = linear_schedule(0.075, 0.025)

    # finetune 04
    # n_steps = 1024
    
    custom_objects = {
        "learning_rate": lr_schedule,
        "clip_range": clip_range_schedule,
        # "n_steps": n_steps # finetune 04
    }
    
    # finetune 01 & 02 & 03 & 04
    # model_path = "trained_models_cnn_finetuned_03/ppo_snake_26000000_steps.zip"
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    model_path = "trained_models_cnn_mask_finetuned_01/ppo_snake_80000000_steps.zip"
    model = MaskablePPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set the save directory
    save_dir = "trained_models_cnn_mask_finetuned_02"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()
