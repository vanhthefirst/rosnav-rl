import math

import numpy as np

# Pre-allocated 3x3 buffer to avoid per-call allocation in get_relative_pos_to_robot
_TRANSFORM_BUFFER = np.empty((3, 3), dtype=np.float64)
_TRANSFORM_BUFFER[2, :] = [0, 0, 1]  # last row is constant


def get_relative_pos_to_robot(
    robot_pose: np.ndarray, distant_poses: np.ndarray, output_buffer: np.ndarray = None
) -> np.ndarray:
    """Transforms distant poses from map frame to robot frame coordinates.

    This function calculates the relative positions of a set of points with respect to
    the robot's current pose. It creates a homogeneous transformation matrix from map
    to robot coordinates, inverts it, and applies it to the distant poses.

    Args:
        robot_pose: A numpy array or dictionary containing the robot's pose with keys:
            - "x": x-coordinate of the robot in the map frame
            - "y": y-coordinate of the robot in the map frame
            - "yaw": orientation of the robot in the map frame (in radians)
        distant_poses: A numpy array of shape (N, 3) where each row represents a pose
            in homogeneous coordinates [x, y, 1] in the map frame
        output_buffer: Optional numpy array of shape (N, 2) to store the result in-place.

    Returns:
        A numpy array of shape (N, 2) containing the x and y coordinates of the distant
        poses in the robot's frame of reference (excluding the homogeneous component).
        If output_buffer is provided, the result is written in-place and returned.
    """
    x = robot_pose["x"]
    y = robot_pose["y"]
    yaw = robot_pose["yaw"]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Fill the pre-allocated inverse transformation matrix robot_T_map
    _TRANSFORM_BUFFER[0, 0] = cos_yaw
    _TRANSFORM_BUFFER[0, 1] = sin_yaw
    _TRANSFORM_BUFFER[0, 2] = -x * cos_yaw - y * sin_yaw
    _TRANSFORM_BUFFER[1, 0] = -sin_yaw
    _TRANSFORM_BUFFER[1, 1] = cos_yaw
    _TRANSFORM_BUFFER[1, 2] = x * sin_yaw - y * cos_yaw

    # Apply the transformation to the distant poses using einsum.
    # Return the transformed poses, excluding the homogeneous component.
    result = np.einsum("ij,kj->ki", _TRANSFORM_BUFFER, distant_poses)[:, :2]
    if output_buffer is not None:
        output_buffer[: result.shape[0], :2] = result
        return output_buffer[: result.shape[0], :2]
    return result


def get_relative_vel_to_robot(
    robot_pose: np.ndarray,
    pedestrian_vel_vector: np.ndarray,
) -> np.ndarray:
    """Transforms pedestrian velocity vectors from a global map frame to the robot's local frame.

    This function transforms the velocity vectors of pedestrians from the map coordinate
    frame to the robot's local coordinate frame using a rotation matrix derived from the
    robot's orientation (yaw).

    Parameters:
        robot_pose (np.ndarray): A numpy array containing the robot's pose information,
                                 which must include a 'yaw' key representing the robot's
                                 orientation in radians.
        pedestrian_vel_vector (np.ndarray): A numpy array of shape (n, 2) containing
                                            pedestrian velocity vectors [vx, vy] in
                                            the map coordinate frame.
    Returns:
        np.ndarray: A numpy array of shape (n, 2) containing the pedestrian velocity
                    vectors transformed to the robot's coordinate frame. Returns an
                    empty array with shape (0, 2) if the input pedestrian_vel_vector
                    is empty.
    """
    if len(pedestrian_vel_vector) == 0:
        return np.empty((0, 2))

    yaw = robot_pose["yaw"]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # robot_r_map = map_r_robot.T = [[cos, sin], [-sin, cos]]
    # Apply as matmul: (2x2) @ (2xN) -> (2xN) -> transpose to (Nx2)
    vx = pedestrian_vel_vector[:, 0]
    vy = pedestrian_vel_vector[:, 1]
    rel_vx = cos_yaw * vx + sin_yaw * vy
    rel_vy = -sin_yaw * vx + cos_yaw * vy

    return np.column_stack((rel_vx, rel_vy))


def time_to_contact(
    relative_positions: np.ndarray,
    relative_velocities: np.ndarray,
    contact_radius: float,
    horizon: float,
) -> np.ndarray:
    """Constant-velocity time until each relative disc closes to contact_radius, capped at horizon, 0 when already in contact."""
    if len(relative_positions) == 0:
        return np.empty(0, dtype=np.float32)
    px, py = relative_positions[:, 0], relative_positions[:, 1]
    vx, vy = relative_velocities[:, 0], relative_velocities[:, 1]
    a = vx * vx + vy * vy
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - contact_radius * contact_radius
    disc = b * b - 4.0 * a * c
    hits = (a > 1e-9) & (disc >= 0.0)
    t = np.where(hits, (-b - np.sqrt(np.maximum(disc, 0.0))) / np.where(hits, 2.0 * a, 1.0), horizon)
    t = np.where(hits & (t > 0.0), np.minimum(t, horizon), horizon)
    return np.where(c <= 0.0, 0.0, t).astype(np.float32)


def nearest_group(
    relative_positions: np.ndarray,
    velocities: np.ndarray,
    labels: np.ndarray,
    robot_velocity: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """[count, centroid x, y, centroid vx, vy relative to the robot, lateral spread] of the labelled group with the nearest centroid, zeros without one."""
    out = np.zeros(6, dtype=np.float32)
    best = math.inf
    for label in np.unique(labels[labels >= 0]):
        members = labels == label
        if np.count_nonzero(members) < 2:
            continue
        centroid = relative_positions[members].mean(axis=0)
        distance = float(np.hypot(*centroid))
        if distance > max_range or distance >= best:
            continue
        best = distance
        lateral = relative_positions[members, 1]
        out[:] = [
            np.count_nonzero(members),
            *centroid,
            *(velocities[members].mean(axis=0) - robot_velocity[:2]),
            lateral.max() - lateral.min(),
        ]
    return out


def proximity_clusters(
    positions: np.ndarray, velocities: np.ndarray, distance: float, velocity_tolerance: float
) -> np.ndarray:
    """Cluster label per pedestrian, linking pairs closer than distance whose velocities differ by less than velocity_tolerance."""
    n = len(positions)
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    close = np.hypot(*(positions[:, None, :] - positions[None, :, :]).transpose(2, 0, 1)) < distance
    agree = np.hypot(*(velocities[:, None, :] - velocities[None, :, :]).transpose(2, 0, 1)) < velocity_tolerance
    for i, j in zip(*np.nonzero(np.triu(close & agree, k=1)), strict=True):
        parent[root(int(i))] = root(int(j))
    return np.array([root(i) for i in range(n)], dtype=np.int64)


def straight_line_subgoal(
    robot_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    speed: float,
    min_lookahead: float,
    max_lookahead: float,
    lookahead_time: float,
) -> tuple[float, float]:
    """Point on the robot-to-goal line `speed * lookahead_time` ahead, clamped to [min_lookahead, max_lookahead], the goal itself when closer."""
    dx = goal_xy[0] - robot_xy[0]
    dy = goal_xy[1] - robot_xy[1]
    dist = math.hypot(dx, dy)
    lookahead = max(min_lookahead, min(max_lookahead, speed * lookahead_time))
    if dist <= lookahead or dist < 1e-6:
        return goal_xy[0], goal_xy[1]
    scale = lookahead / dist
    return robot_xy[0] + dx * scale, robot_xy[1] + dy * scale
