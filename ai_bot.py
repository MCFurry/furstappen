from typing import Tuple

from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform


class FurStAIppen(Bot):
    @property
    def name(self):
        return "FurStAIppen"

    @property
    def contributor(self):
        return "Ferry"

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        target = self.track.lines[next_waypoint]
        # calculate the target in the frame of the robot
        target = position.inverse() * target
        # calculate the angle to the target
        angle = target.as_polar()[1]

        throttle = 1
        steer = 1
        return throttle, steer
