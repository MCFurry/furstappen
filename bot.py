from typing import Tuple

from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math

# Tuuuuuunings
LOOKAHEAD_DISTANCE = 7.74
STEERING_GAIN = 29.09
THROTTLE_GAIN = 137.60
MAX_VELOCITY = 386.0
MIN_VELOCITY = 145.0


class FurStappen(Bot):
    @property
    def name(self):
        return "FurStappen"

    @property
    def contributor(self):
        return "Ferry"

    def __init__(self, track):
        super().__init__(track)

        self.max_position_idx = len(self.track.lines) - 1

    def calculate_curvature(self, current_position, next_waypoint, following_waypoint):
        Ax, Ay = current_position
        Bx, By = next_waypoint
        Cx, Cy = following_waypoint

        numerator = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))
        denominator = math.sqrt((Ax - Bx)**2 + (Ay - By)**2) * math.sqrt((Bx - Cx)**2 + (By - Cy)**2) * math.sqrt((Cx - Ax)**2 + (Cy - Ay)**2)

        if denominator == 0:
            return 0  # Avoid division by zero

        curvature = numerator / denominator
        return curvature

    def adjust_velocity(self, curvature):
        if curvature == 0:
            return MAX_VELOCITY

        velocity = MAX_VELOCITY / (1 + abs(curvature*THROTTLE_GAIN))
        return max(MIN_VELOCITY, min(MAX_VELOCITY, velocity))

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        target = self.track.lines[next_waypoint]
        next_target = self.track.lines[next_waypoint+1] if next_waypoint < self.max_position_idx else self.track.lines[0]

        # Transform target to the car's local coordinate frame
        target_local = position.inverse() * target

        # Calculate the distance and angle to the target in local frame
        target_distance = math.sqrt(target_local.x**2 + target_local.y**2)
        target_angle = math.atan2(target_local.y, target_local.x)

        # If the target is too close, skip to the next waypoint
        if target_distance < LOOKAHEAD_DISTANCE:
            target_local = position.inverse() * next_target
            target_distance = math.sqrt(target_local.x**2 + target_local.y**2)
            target_angle = math.atan2(target_local.y, target_local.x)

        if target_distance > 0:
            curvature = (2 * target_local.y) / (LOOKAHEAD_DISTANCE**2)
        else:
            curvature = 0

        steer = curvature * STEERING_GAIN

        # Clamping steer value between -1 and 1
        steer = max(-1, min(1, steer))

        curvature = self.calculate_curvature(position.p, target, next_target)
        target_velocity = self.adjust_velocity(curvature)
        current_speed = velocity.length()

        if current_speed < target_velocity:
            throttle = 1  # Accelerate
        else:
            throttle = -1  # Decelerate

        return throttle, steer
