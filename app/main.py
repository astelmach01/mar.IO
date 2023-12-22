import datetime
import os
from pathlib import Path

import gym_super_mario_bros
import gymnasium as gym
import torch
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from .agent import DiscreteAgent
from .metrics import MetricLogger
from .verify_device import verify_device
from .wrappers import ResizeObservation, SkipFrame
from .neural_net import ConvNet, FeedForwardNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = verify_device()
torch.set_default_device(device)

SHOULD_RENDER = True
NUM_EPISODES = 40000

CHECKPOINTS_DIR = Path("./checkpoints").resolve()


def get_mario_environment():
    # Initialize Super Mario environment
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb" if not SHOULD_RENDER else "human",
        apply_api_compatibility=True,
    )

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.0)
    env = FrameStack(env, num_stack=4)

    env.reset()

    return env


def get_mario_agent(action_dim: int):
    save_dir = Path("./checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = (
        CHECKPOINTS_DIR / "2023-11-30T22-29-08" / "mario_net_5.chkpt"
    ) 
    mario = DiscreteAgent(
        state_dim=(4, 84, 84),
        action_dim=action_dim,
        net=ConvNet((4,84,84), action_dim, device),
        save_dir=save_dir,
        device=device,
        checkpoint=checkpoint,
    )

    return mario


def setup_mario():
    env = get_mario_environment()
    agent = get_mario_agent(env.action_space.n)

    return env, agent


def get_frozen_lake_environment():
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human")

    return env


def setup_frozen_lake():
    env = get_frozen_lake_environment()
    
    net = FeedForwardNet(env.observation_space.n, env.action_space.n, device)
    agent = DiscreteAgent(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        net=net,
        save_dir=None,
        device=device,
        checkpoint=None,
    )

    return env, agent


def run_agent(env, agent, logger, episodes=NUM_EPISODES):
    for e in range(episodes):
        current_state, _ = env.reset()

        # Play the game!
        while True:
            # 3. Show environment (the visual) [WIP]
            if SHOULD_RENDER:
                env.render()

            # 4. Run agent on the state
            action = agent.act(current_state)

            # 5. Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # 6. Remember
            agent.cache(current_state, next_state, action, reward, done)

            # 7. Learn
            q, loss = agent.learn()

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            current_state = next_state

            # 10. Check if end of game
            if done or ("flag_get" in info.keys() and info["flag_get"]):
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e, epsilon=agent.exploration_rate, step=agent.curr_step
            )


def main():
    env, agent = setup_frozen_lake()
    
    save_dir = Path("./checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = MetricLogger(save_dir)

    run_agent(env, agent, logger)
