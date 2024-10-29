from typing import Tuple

import pygame
from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math
import numpy as np
import os
from scipy import interpolate

pygame.font.init()
font = pygame.font.Font(None, 24)  # None uses default font, 24 is the font size

# Tuuuuuunings
LOOKAHEAD_DISTANCE = 11.64
PACMAN_DISTANCE = 70.18
STEERING_GAIN = 45.0
MAX_VELOCITY = 348.0
MIN_VELOCTIY = 94.5
ACCELERATION = 124.4
MULTIPLIER = 3

DEBUG = False
GIMMICKS = True

def calculate_curvature(points):
    curvatures = []
    for i in range(len(points)):
        prev = points[i-1]
        curr = points[i]
        nextt = points[(i+1) % len(points)]

        v1 = curr - prev
        v2 = nextt - curr

        cross_product = abs(v1.x * v2.y - v1.y * v2.x)
        denominator = (v1.length() * v2.length())

        if denominator > 1e-6:
            curvature = cross_product / denominator
        else:
            curvature = 0

        curvatures.append(curvature)

    return curvatures

def optimize_racing_line(points, num_interpolated_points, smoothing_factor):
    # Convert Vector2 objects to numpy array
    points_array = np.array([(p.x, p.y) for p in points])

    # Create a parameter array for interpolation
    t = np.linspace(0, 1, len(points_array), endpoint=False)

    # Duplicate the first point at the end to ensure closure
    points_array = np.vstack((points_array, points_array[0]))
    t = np.append(t, 1)

    # Create periodic spline with smoothing
    tck, _ = interpolate.splprep([points_array[:, 0], points_array[:, 1]], s=smoothing_factor, per=1)

    # Interpolate
    t_interpolated = np.linspace(0, 1, num_interpolated_points, endpoint=False)
    interpolated_points = interpolate.splev(t_interpolated, tck)

    # Convert back to Vector2 objects
    return [Vector2(x, y) for x, y in zip(*interpolated_points)]

def get_speed_setpoints(track, curvatures, min_velocity, max_velocity, acceleration, deceleration):
    max_curvature = max(curvatures)
    speed_setpoints = []

    # First pass: calculate initial speed setpoints based on curvature
    # for curvature in curvatures:
    #     if curvature <= 0.1:
    #         speed_setpoints.append(max_velocity)
    #     else:
    #         curvature_factor = 1 - (curvature / max_curvature)
    #         speed_setpoints.append(min_velocity + (max_velocity - min_velocity) * curvature_factor)
    for curvature in curvatures:
        if curvature <= 0.2:
            speed_setpoints.append(max_velocity)
        else:
            curvature_factor = 1 - (curvature / max_curvature)
            # Apply exponential function to curvature factor
            aggressiveness=2
            adjusted_factor = curvature_factor ** aggressiveness
            speed_setpoints.append(min_velocity + (max_velocity - min_velocity) * adjusted_factor)

    # Second pass: adjust speeds based on acceleration and deceleration limits
    for i in range(len(speed_setpoints) - 1, 0, -1):
        current_speed = speed_setpoints[i]
        prev_speed = speed_setpoints[i-1]
        distance = (track[i] - track[i-1]).length()

        # Calculate the maximum speed the car can have at the previous point
        # to be able to slow down to the current speed
        max_prev_speed = math.sqrt(current_speed**2 + 2 * deceleration * distance)

        if prev_speed > max_prev_speed:
            speed_setpoints[i-1] = max_prev_speed

    # Third pass: adjust speeds based on acceleration limits (forward)
    for i in range(1, len(speed_setpoints)):
        current_speed = speed_setpoints[i]
        prev_speed = speed_setpoints[i-1]
        distance = (track[i] - track[i-1]).length()

        # Calculate the maximum speed the car can reach from the previous point
        max_current_speed = math.sqrt(prev_speed**2 + 2 * acceleration * distance)

        if current_speed > max_current_speed:
            speed_setpoints[i] = max_current_speed

    return speed_setpoints


class Schummi(Bot):
    @property
    def name(self):
        return "Schummi"

    @property
    def contributor(self):
        return "Ferry"

    def __init__(self, track):
        super().__init__(track)
        self.target_idx = 0

        # Calculate the optimized racing line
        self.smooth_track = optimize_racing_line(self.track.lines, MULTIPLIER*len(self.track.lines), smoothing_factor=123)

        # Calculate curvatures for the optimized racing line
        self.curvatures = calculate_curvature(self.smooth_track)
        self.max_curvature = max(self.curvatures)
        self.speed_setpoints = get_speed_setpoints(self.smooth_track, self.curvatures, MIN_VELOCTIY, MAX_VELOCITY, ACCELERATION, ACCELERATION)

        self.iter = 0
        self.banana = pygame.image.load(
                os.path.dirname(__file__) + '/Banana.png')
        self.draw_banana = False
        self.oil = pygame.image.load(
                os.path.dirname(__file__) + '/oil_puddle.png')
        self.draw_oil_iter = None

    def draw(self, map_scaled, zoom):
        if GIMMICKS:
            if self.iter % 900 == 0:
                self.draw_banana = True
                self.banana_pos = self.car_position.p
                self.banana_rot = self.car_position.M.angle + 45
            elif self.iter % 1400 == 0:
                self.draw_banana = False
                self.iter = 0
            if self.draw_banana:
                if self.iter % 900 < 35:
                    banana_zoom = max(0.1 + (self.iter % 900)*0.0029, 0.1) * zoom
                else:
                    banana_zoom = max(0.15 - (self.iter % 935)*0.0029, 0.1) * zoom
                _image = pygame.transform.rotozoom(
                    self.banana, -math.degrees(self.banana_rot) - 45, banana_zoom)
                _rect = _image.get_rect(
                    center=self.banana_pos * zoom)
                map_scaled.blit(_image, _rect)

            if self.draw_oil_iter is not None:
                iter_diff = abs(self.iter - self.draw_oil_iter)
                if iter_diff > 100:
                    self.draw_oil_iter = None
                oil_zoom = 0.2 * zoom
                _image = pygame.transform.rotozoom(
                    self.oil, -math.degrees(self.oil_rot), oil_zoom)
                _image.set_alpha(255-iter_diff*2)
                _rect = _image.get_rect(
                    center=self.oil_pos * zoom)
                map_scaled.blit(_image, _rect)

        if not DEBUG:
            return
        # Plot the smooth track
        for i, (point, speed) in enumerate(zip(self.smooth_track, self.speed_setpoints)):
            # Draw the point
            pygame.draw.circle(map_scaled, (0, 255, 0), (int(point.x*zoom), int(point.y*zoom)), 3)

            # Render the speed text
            text = font.render(f'{speed:.1f}', True, (255, 255, 255))

            # Calculate position for the text (offset by 5 pixels in both x and y)
            text_pos = (int(point.x*zoom) + 5, int(point.y*zoom) + 5)

            # Blit the text onto the surface
            map_scaled.blit(text, text_pos)

        # Highlight the target point
        target = self.smooth_track[self.target_idx]
        pygame.draw.circle(map_scaled, (255, 0, 0), (int(target.x*zoom), int(target.y*zoom)), 3)

        # Draw current speed next to the car
        if hasattr(self, 'current_speed'):
            speed_text = font.render(f'Speed: {self.current_speed:.1f}', True, (255, 0, 0))
            car_pos = (int(self.car_position.p.x * zoom), int(self.car_position.p.y * zoom))
            speed_pos = (car_pos[0] + 20, car_pos[1] + 20)  # Offset from car position
            map_scaled.blit(speed_text, speed_pos)


    def adjust_velocity(self, target_idx, target_distance, current_speed):
        # Speed we need to reach at the target point
        target_velocity = self.speed_setpoints[target_idx]
        # Speed up case
        if target_velocity >= current_speed:
            new_vel = target_velocity
        # Possibly slowdown case
        else:
            # Based on current speed, target speed and acceleration, calculate time needed to reach target speed
            time_to_reach_target_speed = (current_speed - target_velocity) / ACCELERATION
            # Calculate time to next target
            time_to_next_target = target_distance / current_speed

            if time_to_next_target <= time_to_reach_target_speed:
                new_vel = target_velocity
            else:
                new_vel = MAX_VELOCITY

        return new_vel

    def find_target(self, position):
        target = self.smooth_track[self.target_idx]
        if math.sqrt((position.p.x - target.x)**2 + (position.p.y - target.y)**2) < PACMAN_DISTANCE:
            self.target_idx += 1
            if self.target_idx >= len(self.smooth_track):
                self.target_idx = 0

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        self.find_target(position)
        # Recovery behavior in case we miss next_waypoint
        game_target = position.inverse() * self.track.lines[next_waypoint]
        if game_target[0] < 0:
            target = self.track.lines[next_waypoint]
            if self.draw_oil_iter is None:
                self.oil_pos = self.track.lines[next_waypoint]
                self.oil_rot = position.M.angle
            self.draw_oil_iter = self.iter
        else:
            target = self.smooth_track[self.target_idx]

        self.car_position = position

        # Transform target to the car's local coordinate frame
        target_local = position.inverse() * target

        # Calculate the distance and angle to the target in local frame
        target_distance = math.sqrt(target_local.x**2 + target_local.y**2)

        if target_distance > 0:
            curvature = (2 * target_local.y) / (LOOKAHEAD_DISTANCE**2)
        else:
            curvature = 0

        steer = curvature * STEERING_GAIN
        # Clamping steer value between -1 and 1
        steer = max(-1, min(1, steer))

        self.current_speed = velocity.length()
        target_velocity = self.adjust_velocity(self.target_idx, target_distance, self.current_speed)

        if self.current_speed < target_velocity:
            throttle = 1  # Accelerate
        else:
            throttle = -1  # Decelerate
        self.iter += 1
        return throttle, steer
