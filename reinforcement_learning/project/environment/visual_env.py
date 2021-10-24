
import torchvision.transforms as T
from data_loader.augmentation import ResizeTransform
import numpy as np
import torch

class VisualEnv():
    def __init__(self, env) -> None:
        self.env = env
        self._resize = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                ResizeTransform((40, 30))
            ]
        )

    def _get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen_orig = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen_orig.shape
        screen = screen_orig[:, int(screen_orig*0.4):int(screen_orig * 0.8)]
        view_width = int(screen_width * 0.2)
        cart_location = self._get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return screen_orig, self._resize(screen)

class ParameterEnv():
    def __init__(self, env) -> None:
        self.env = env

    def get_screen(self):
        screen_orig = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        return screen_orig, torch.tensor(self.env.state).float().unsqueeze(0)