import datetime
import os
from pathlib import Path

import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from .agent import Mario
from .metrics import MetricLogger
from .verify_device import verify_device
from .wrappers import ResizeObservation, SkipFrame

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = verify_device()
torch.set_default_device(device)

SHOULD_RENDER = False

CHECKPOINTS_DIR = Path("./checkpoints").resolve()


def main():
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

    save_dir = Path("./checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = (
        CHECKPOINTS_DIR / "2023-11-30T22-29-08" / "mario_net_5.chkpt"
    )  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        device=device,
        checkpoint=checkpoint,
    )

    logger = MetricLogger(save_dir)

    episodes = 40000

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):
        current_state, _ = env.reset()

        # Play the game!
        while True:
            # 3. Show environment (the visual) [WIP]
            if SHOULD_RENDER:
                env.render()

            # 4. Run agent on the state
            action = mario.act(current_state)

            # 5. Agent performs action
            next_state, reward, done, trunc, info = env.step(action)
            done = done or trunc

            # 6. Remember
            mario.cache(current_state, next_state, action, reward, done)

            # 7. Learn
            q, loss = mario.learn()

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            current_state = next_state

            # 10. Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
