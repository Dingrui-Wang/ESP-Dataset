import json
import math
import os

import numpy as np


def read_json(data_path):
    with open(data_path) as fn:
        _dict = json.load(fn)
    return _dict


def get_file_list(root_path):
    if os.path.isfile(root_path):
        return [root_path]
    file_lists = []
    for file in os.listdir(root_path):
        if not os.path.isdir(file):
            file_lists.append(os.path.join(root_path, file))
    return file_lists


def wcs_to_vcs(point: np.ndarray, ego_position, ego_pose, translation=True):
    if point.shape[0] == 0:
        return point
    ego_x = ego_position["x"]
    ego_y = ego_position["y"]
    shift = np.array([ego_x, ego_y])
    qx = ego_pose["qx"]
    qy = ego_pose["qy"]
    qz = ego_pose["qz"]
    qw = ego_pose["qw"]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_cos = math.cos(yaw)
    yaw_sin = math.sin(yaw)
    rot = np.array([[yaw_cos, -yaw_sin], [yaw_sin, yaw_cos]])
    vcs_point = np.matmul(point - shift, rot) if translation else np.matmul(point, rot)
    return vcs_point
