from collections import defaultdict
import cv2
from typing import Union


class Grid:
    def __init__(
            self,
            width=5,
            height=5,
            image_size=(150, 150)
    ):
        self.width = width
        self.height = height
        self.d_x = int(image_size[0]/width) + 1
        self.d_y = int(image_size[1]/height) + 1

    def get_cell(self, kp: [cv2.KeyPoint, tuple, list]):
        if isinstance(kp, cv2.KeyPoint):
            pt = kp.pt
        else:
            pt = kp
        grid_x = int(pt[0]/self.d_x)
        grid_y = int(pt[1]/self.d_y)
        return grid_x, grid_y

    def get_cell_id(self, kp: [cv2.KeyPoint, tuple, list]):
        if isinstance(kp, cv2.KeyPoint):
            pt = kp.pt
        else:
            pt = kp

        grid_x = int(pt[0] / self.d_x)
        grid_y = int(pt[1] / self.d_y)
        if grid_x >= self.width:
            grid_x = self.width - 1
        if grid_x >= self.height:
            grid_x = self.height - 1

        return grid_y*self.width + grid_x

    def __len__(self):
        return self.width * self.height
