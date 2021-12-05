from collections import deque
import torch
from tqdm import tqdm
from itertools import count
from statistics import mean


def fit(alg, visual_env):
    episode_durations = deque([], maxlen=100)

    screen, state = visual_env.get_screen()
    if screen is not None:
        alg.writer.add_image('Model view', screen)

    alg.load_last_state()
    _, state = visual_env.get_screen()

    while alg.epoch:
        # Initialize the environment and state
        state = visual_env.reset()
        for tttttt in count():
            # Select and perform an action
            action = alg.select_action(state)
            new_state, reward, done, _ = visual_env.step(action.item())
            if done:
                reward = -10

            alg.optimization_step(state, action, reward, new_state)

            state = new_state
            if done:
                episode_durations.append(tttttt + 1)
                alg.writer.add_scalars('Duration', {
                    'current': episode_durations[-1],
                    'mean': mean(episode_durations)
                }, alg.epoch)
                break
        alg.process_metric(episode_durations)
        alg.save_model_state()
    print('Complete')
