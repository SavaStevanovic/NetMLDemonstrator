from collections import deque
from statistics import mean

from algorithams.rl_alg import ReinforcmentAlgoritham
from environment.playgrounds import Playground


def fit(algoritham: ReinforcmentAlgoritham, environment: Playground):
    episode_metric = deque([], maxlen=100)

    screen, _ = environment.get_screen()
    if screen is not None:
        algoritham.writer.add_image('Model view', screen)

    algoritham.load_last_state()

    while algoritham.epoch:
        # Initialize the environment and state
        state = environment.reset()
        done = False

        while not done:
            # Select and perform an action
            action, log_prob = algoritham.preform_action(state)
            new_state, reward, done, _ = environment.step(action)
            algoritham.optimization_step(
                state, action, log_prob, reward, new_state)

            state = new_state
            if done:
                episode_metric.append(environment.metric)
                algoritham.writer.add_scalars('Duration', {
                    'current': episode_metric[-1],
                    'mean': mean(episode_metric)
                }, algoritham.epoch)
        algoritham.process_metric(episode_metric)
        algoritham.save_model_state()
    print('Complete')
