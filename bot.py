from typing import Tuple

import pygame
from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math
from scipy.special import comb
import numpy as np


def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(i)) * (1 - t)**(n-i)

def bezier_curve_with_derivatives(points, num_points):
    n_points = len(points)
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    curve_d1 = np.zeros((num_points, 2))
    curve_d2 = np.zeros((num_points, 2))

    for i in range(n_points):
        curve += np.outer(bernstein_poly(i, n_points - 1, t), points[i])
        if i < n_points - 1:
            curve_d1 += np.outer(bernstein_poly(i, n_points - 2, t), (n_points - 1) * (points[i+1] - points[i]))
        if i < n_points - 2:
            curve_d2 += np.outer(bernstein_poly(i, n_points - 3, t), (n_points - 1) * (n_points - 2) * (points[i+2] - 2*points[i+1] + points[i]))

    return curve, curve_d1, curve_d2

def calculate_curvature( d1, d2):
    numerator = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denominator = (d1[:, 0]**2 + d1[:, 1]**2)**(3/2)
    return numerator / denominator

class FurStappen(Bot):
    @property
    def name(self):
        return "FurStappen"

    @property
    def contributor(self):
        return "Ferry"

    def __init__(self, track):
        super().__init__(track)
        self.lookahead_distance = 2.0
        self.steering_gain = 1.2
        self.throttle_gain = 365.0
        self.max_velocity = 500.0
        self.min_velocity = 50.0
        self.target_idx = 0

        self.smooth_track, self.curvatures = self.smooth_path(self.track.lines, int(len(track.lines)*5))

    def draw(self, map_scaled, zoom):
        # Plot the smooth track
        idx = 0
        for point in self.smooth_track:
            if idx == self.target_idx:
                pygame.draw.circle(map_scaled, (255, 0, 0), (int(point[0]*zoom), int(point[1]*zoom)), 3)
            else:
                pygame.draw.circle(map_scaled, (0, 255, 0), (int(point[0]*zoom), int(point[1]*zoom)), 3)
            idx += 1

    def smooth_path(self, path, num_points=100):
        """
        Smooth the path using piecewise Bézier curves and calculate curvature.

        Parameters:
        path (list of tuples): List of (x, y) coordinates.
        num_points (int): Number of points in the smoothed path.

        Returns:
        list of tuples: Smoothed path with curvature (x, y, curvature).
        """
        if len(path) < 4:
            return path  # Not enough points to create a cubic Bézier curve

        num_segments = (len(path) - 1) // 3
        smoothed_path = []
        points_per_segment = num_points // num_segments

        for i in range(num_segments):
            control_points = path[i*3:i*3+4]

            curve, curve_d1, curve_d2 = bezier_curve_with_derivatives(control_points, points_per_segment)
            curvature = calculate_curvature(curve_d1, curve_d2)

            segment_with_curvature = [(x, y, k) for (x, y), k in zip(curve, curvature)]
            smoothed_path.extend(segment_with_curvature[:-1])  # Exclude last point to avoid duplication

        # Add the last point
        smoothed_path.append((path[-1][0], path[-1][1], 0))  # Assuming zero curvature at the end point

        # Convert to Vector2 objects
        return [Vector2(x, y) for x, y, _ in smoothed_path], [curvature for _, _, curvature in smoothed_path]

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
            return self.max_velocity

        velocity = self.max_velocity / (1 + abs(curvature*self.throttle_gain))
        return max(self.min_velocity, min(self.max_velocity, velocity))

    def find_target(self, position):
        target = self.smooth_track[self.target_idx]
        if math.sqrt((position.p.x - target.x)**2 + (position.p.y - target.y)**2) < 50.0:
            self.target_idx += 1
            if self.target_idx >= len(self.smooth_track):
                self.target_idx = 0

    def get_curvature_idx(self, target_idx, look_ahead=5):
        return target_idx + look_ahead if target_idx < len(self.smooth_track) - look_ahead else target_idx + look_ahead - len(self.smooth_track)

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:

        self.find_target(position)

        target = self.smooth_track[self.target_idx]
        next_target = self.smooth_track[self.target_idx+1] if self.target_idx < len(self.smooth_track) - 1 else self.smooth_track[0]

        # Transform target to the car's local coordinate frame
        target_local = position.inverse() * target

        # Calculate the distance and angle to the target in local frame
        target_distance = math.sqrt(target_local.x**2 + target_local.y**2)
        target_angle = math.atan2(target_local.y, target_local.x)

        # # If the target is too close, skip to the next waypoint
        # if target_distance < self.lookahead_distance:
        #     target_local = position.inverse() * next_target
        #     target_distance = math.sqrt(target_local.x**2 + target_local.y**2)
        #     target_angle = math.atan2(target_local.y, target_local.x)

        if target_distance > 0:
            curvature = (2 * target_local.y) / (self.lookahead_distance**2)
        else:
            curvature = 0

        steer = curvature * self.steering_gain

        # Clamping steer value between -1 and 1
        steer = max(-1, min(1, steer))

        target_velocity = self.adjust_velocity(self.curvatures[self.get_curvature_idx(self.target_idx)])

        print(f'target_velocity: {target_velocity}, target_idx: {self.target_idx}, curvature_idx: {self.get_curvature_idx(self.target_idx)}')

        current_speed = velocity.length()

        if current_speed < target_velocity:
            throttle = 1  # Accelerate
        else:
            throttle = -1  # Decelerate

        return throttle, steer
