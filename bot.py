from typing import Tuple

import pygame
from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math
from scipy.interpolate import interp1d
import numpy as np


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
        self.min_velocity = 140.0
        self.target_idx = 0

        self.smooth_track = self.smooth_path(self.track.lines, int(len(track.lines)*3))

        self.distances = [math.sqrt((self.smooth_track[i].x - self.smooth_track[i-1].x)**2 + (self.smooth_track[i].y - self.smooth_track[i-1].y)**2) for i in range(1, len(self.smooth_track))]
        self.distances.insert(0, math.sqrt((self.smooth_track[0].x - self.smooth_track[-1].x)**2 + (self.smooth_track[0].y - self.smooth_track[-1].y)**2))
        self.max_distance = max(self.distances)
        self.min_distance = min(self.distances)

        self.velocities = [self.min_velocity + (distance - self.min_distance) / self.max_distance * self.max_velocity for distance in self.distances]

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
        Smooth the path using PCHIP interpolation.
        
        Parameters:
        path (list of tuples): List of (x, y) coordinates.
        num_points (int): Number of points in the smoothed path.
        
        Returns:
        list of tuples: Smoothed path.
        """
        if len(path) < 2:
            return path  # Not enough points to interpolate

        # Extract x and y coordinates from the path
        x = [point[0] for point in path]
        y = [point[1] for point in path]

        # Create a parameter t for the original path
        t = np.linspace(0, 1, len(path))

        # Create a new parameter t for the smoothed path
        t_approx = np.linspace(0, 1, num_points)

        # Use PCHIP interpolation for smoothing
        pchip_x = interp1d(t, x)
        pchip_y = interp1d(t, y)

        # Generate the smoothed path
        x_smooth = pchip_x(t_approx)
        y_smooth = pchip_y(t_approx)

        smoothed_path = [Vector2(x, y) for x, y in zip(x_smooth, y_smooth)]
        return smoothed_path
        
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

        # curvature = self.calculate_curvature(position.p, target, next_target)
        # target_velocity = self.adjust_velocity(curvature)

        target_velocity = self.velocities[self.target_idx]
        print(f'target_velocity: {target_velocity}')

        current_speed = velocity.length()
        
        if current_speed < target_velocity:
            throttle = 1  # Accelerate
        else:
            throttle = -1  # Decelerate

        return throttle, steer
