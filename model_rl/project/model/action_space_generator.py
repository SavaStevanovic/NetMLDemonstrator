import itertools
from typing import Any


from gym.spaces import Box
import numpy as np

class ActionSpaceProducer:
    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
    
    @property
    def horizon(self) -> int:
        return self._horizon

    def __call__(self, space_box: Box) -> np.array:
        pass

class RandomSpaceProducer(ActionSpaceProducer):
    def __init__(self, horizon, num_samples) -> None:
        self._num_samples = num_samples
        super().__init__(horizon)

    def __call__(self, space_box: Box) -> np.array:
        np.random.seed(42)
        return np.random.uniform(
            low=space_box.low, high=space_box.high, size=(self._horizon, self._num_samples, len(space_box.low))
        )
    
class EvenlyspacedSpaceProducer(ActionSpaceProducer):
    def __init__(self, horizon, size) -> None:
        self._size = size
        super().__init__(horizon)

    @staticmethod
    def _cartesian_product_itertools(arrays):
        return np.array(list(itertools.product(*arrays)))

    def __call__(self, space_box: Box) -> np.array:
        dim_options = [np.linspace(space_box.low[i], space_box.high[i], num=self._size) for i in range(len(space_box.low))]    
        time_options = EvenlyspacedSpaceProducer._cartesian_product_itertools(dim_options)
        options = EvenlyspacedSpaceProducer._cartesian_product_itertools([time_options for _ in range(self._horizon)])

        return np.moveaxis(options, 0, 1)
        