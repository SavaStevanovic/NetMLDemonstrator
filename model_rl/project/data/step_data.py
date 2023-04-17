from dataclasses import dataclass

import numpy as np


@dataclass
class StepDescriptor:
    current_state: np.array
    next_state: np.array
    action: np.array
    reward: float
    done: bool
