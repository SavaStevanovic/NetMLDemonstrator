from collections import deque
from statistics import mean
from itertools import count

from algorithams.rl_alg import ReinforcmentAlgoritham
from environment.environments import Environment


def fit(algoritham: ReinforcmentAlgoritham, environment: Environment):
    episode_durations = deque([], maxlen=100)

    screen, state = environment.get_screen()
    if screen is not None:
        algoritham.writer.add_image('Model view', screen)

    algoritham.load_last_state()

    while algoritham.epoch:
        # Initialize the environment and state
        state = environment.reset()
        for duration in count():
            # Select and perform an action
            action = algoritham.preform_action(state)
            new_state, reward, done, _ = environment.step(action.item())
            algoritham.optimization_step(state, action, reward, new_state)

            state = new_state
            if done:
                episode_durations.append(duration)
                algoritham.writer.add_scalars('Duration', {
                    'current': episode_durations[-1],
                    'mean': mean(episode_durations)
                }, algoritham.epoch)
                break
        algoritham.process_metric(episode_durations)
        algoritham.save_model_state()
    print('Complete')
