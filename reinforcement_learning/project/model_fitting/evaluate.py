from algorithams.rl_alg import ReinforcmentAlgoritham
from environment.playgrounds import Playground


def run(algoritham: ReinforcmentAlgoritham, environment: Playground):
    # Initialize the environment and state
    algoritham.load_best_state()
    state = environment.reset()
    done = False
    while not done:
        # Select and perform an action
        action, _ = algoritham.preform_action(state)
        state, _, done, _ = environment.step(action)
