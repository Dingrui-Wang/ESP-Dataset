import copy
import json
import math
import os
import pickle
from io import BytesIO
from pathlib import Path
import multiprocessing
from functools import partial
import re
import uuid

import cv2
import geopandas as gpd
import imageio
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from shapely import affinity
from shapely.geometry import box
from tqdm import tqdm

from tools.road_model_operator import RoadModelOperator
from tools.utils import wcs_to_vcs


# Define is_valid_uuid function
def is_valid_uuid(uuid_string):
    """
    Check if the provided string is a valid UUID.
    
    Args:
        uuid_string (str): The string to check
        
    Returns:
        bool: True if the string is a valid UUID, False otherwise
    """
    try:
        # Try to parse the string as a UUID
        uuid_obj = uuid.UUID(uuid_string)
        # Check if the string representation matches the original
        return str(uuid_obj) == uuid_string
    except (ValueError, AttributeError):
        return False


# Define the multiprocessing functions directly in this file
def process_map(func, args_list, process_name="", num_works=None):
    """Process a list of arguments with the given function using multiple processes."""
    if num_works is None:
        num_works = multiprocessing.cpu_count() // 2

    with multiprocessing.Pool(processes=num_works) as pool:
        if process_name:
            results = list(tqdm(pool.imap(func, args_list), total=len(args_list), desc=process_name))
        else:
            results = list(pool.imap(func, args_list))
    return results


def multiprocessing_func(func, args_list, process_name="", num_works=None):
    """Process a list of argument pairs with the given function using multiple processes."""
    if num_works is None:
        num_works = multiprocessing.cpu_count() // 2
    
    # Handle the case where args_list contains pairs of arguments
    def wrapper(args):
        return func(*args)
    
    with multiprocessing.Pool(processes=num_works) as pool:
        if process_name:
            results = list(tqdm(pool.imap(wrapper, args_list), total=len(args_list), desc=process_name))
        else:
            results = list(pool.imap(wrapper, args_list))
    return results


class TokenBevGenerator:
    def __init__(self, path, token) -> None:
        token_path = Path(path, token)
        objs_path = str(token_path) + "_obj.json"
        rm_path = str(token_path) + "_road_model.json"
        frame_data = pd.read_json(objs_path, orient="records", convert_dates=False)
        road_model = json.load(open(rm_path, "rb"))
        rm_op = RoadModelOperator(road_model)
        local_lanes = rm_op.get_vis_lane(wcs=False)
        if local_lanes is None or len(local_lanes) == 0:
            self.token_video = None
        else:
            groupped_data = frame_data.groupby("timestamp")
            bev_data = []
            for ts, frame in tqdm(groupped_data):
                mat = self.draw_frame(frame, rm_op, local_lanes)
                mat_np = np.asarray(mat)
                bev_data.append(np.expand_dims(mat_np, axis=0))
            self.token_video = np.concatenate(bev_data, axis=0)

    def get_video(self):
        return self.token_video

    def write_video(self, path):
        imageio.mimwrite(
            path,
            self.token_video,
            fps=10,
        )

    def draw_frame(self, frame, rm_op: RoadModelOperator, lanes):
        local_lanes = copy.deepcopy(lanes)
        ego_info = frame[frame.anno == "ego"].iloc[0]

        def get_plot_blackground():
            plt.style.use("dark_background")
            fig, ax = plt.subplots(dpi=300)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("gray")
            ax.spines["left"].set_color("gray")
            ax.xaxis.label.set_color("gray")
            ax.yaxis.label.set_color("gray")
            ax.tick_params(axis="x", colors="gray")
            ax.tick_params(axis="y", colors="gray")
            return fig, ax

        fig, ax = get_plot_blackground()
        local_lanes.plot(ax=ax, color="white", linewidth=2)

        def get_vertex(x, y, pose, length, width, custom_shift=None):
            vertex_result = []
            obs_center = np.array([x, y, 0])
            rotation_matrix = R.from_quat(
                [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]
            ).as_matrix()
            if custom_shift is None:
                delta_x = length / 2.0
                delta_y = width / 2.0
                custom_shift = [
                    [delta_x, delta_y, 0.0],
                    [delta_x, -delta_y, 0.0],
                    [-delta_x, -delta_y, 0],
                    [-delta_x, delta_y, 0],
                ]
            for point in custom_shift:
                np_point = np.array(point)
                mat_point = np.reshape(np_point, (3, -1))
                obs_wcs = np.matmul(rotation_matrix, mat_point) + np.reshape(
                    obs_center, (3, -1)
                )
                vertex_result.append([obs_wcs[0][0], obs_wcs[1][0]])
            return vertex_result

        ego_shift = [
            [6, 1.25, 0],
            [6, -1.25, 0],
            [-2.5, -1.25, 0],
            [-2.5, 1.25, 0],
            [6, 1.25, 0],
        ]
        ego_vertex = get_vertex(
            ego_info["position"]["x"],
            ego_info["position"]["y"],
            ego_info["pose"],
            None,
            None,
            ego_shift,
        )
        vcs_ego_vertex = rm_op.wcs_to_vcs(np.array(ego_vertex))
        all_polygons = []
        ego_rect = mp.Polygon(
            vcs_ego_vertex, edgecolor="green", facecolor="green", alpha=1.0,zorder=2
        )
        all_polygons.append(ego_rect)
        for i in range(len(frame)):
            obj_info = frame.iloc[i]
            if obj_info.anno == "ego":
                continue
            color = "yellow" if obj_info.anno == "tv" else "blue"
            obj_vertex = get_vertex(
                obj_info["position"]["x"],
                obj_info["position"]["y"],
                obj_info["pose"],
                obj_info["length"],
                obj_info["width"],
            )
            vcs_obj_vertex = rm_op.wcs_to_vcs(np.array(obj_vertex))
            obj_rect = mp.Polygon(
                vcs_obj_vertex, edgecolor=color, facecolor=color, alpha=1.0,zorder=2
            )
            all_polygons.append(obj_rect)

        for rect in all_polygons:
            ax.add_patch(rect)
        ego_pos = rm_op.wcs_to_vcs(
            np.array([ego_info["position"]["x"], ego_info["position"]["y"]])
        )
        ax.set_xlim([ego_pos[0] - 40, ego_pos[0] + 100])
        ax.set_ylim([ego_pos[1] - 12, ego_pos[1] + 12])
        # fig.tight_layout()
        fig.canvas.draw()
        fig.subplots_adjust(left=0.06, bottom=0, right=1, top=1)
        # img = np.array(fig.canvas.renderer.buffer_rgba())
        # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img2 = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
        buffer.close()
        plt.close()
        return img2


class MomentBevGenerator:
    def __init__(self, mon_id):
        self.mon_id = mon_id
        self.path = "./mons" + "/" + mon_id
        files = os.listdir(self.path)
        self.tokens = []
        for f in files:
            if is_valid_uuid(f[: min(len(f), 36)]) and f[:36] not in self.tokens:
                self.tokens.append(f[:36])
        self.video = []
        for token in self.tokens:
            token_bev = TokenBevGenerator(self.path, token)
            token_bev.write_video(self.path + "/" + token + "_bev3.mp4")

    def get_videos(self):
        video_dict = dict()
        for i in range(len(self.tokens)):
            video_dict.update({self.tokens[i], self.video[i]})
        return video_dict


if __name__ == "__main__":
    import os

    files = os.listdir("./mons")
    existed_files = os.listdir("/mnt/data_cpfs/zheyuan/esp_dataset_process/video_data/")
    existed_token = []
    for f in existed_files:
        existed_token.append(f[:36])
    tokens = []
    token_args = []
    for f in files:
        if is_valid_uuid(f[: min(len(f), 36)]) and f[:36] not in tokens and f[:36] not in existed_token:
            tokens.append(f[:36])
            token_args.append(
                ["mons", f[:36]]
            )
    prp = TokenBevGenerator("/mnt/data_cpfs/zheyuan/esp_dataset_process/json_data", tokens[0])
    prp.write_video("test5.mp4")
    def process_mon(path,token):
        try:
            prp = TokenBevGenerator(path,token)
        except:
            print(token)
        if prp.token_video is not None:
            prp.write_video("/mnt/data_cpfs/zheyuan/esp_dataset_process/video_data/"+token+".mp4")

    multiprocessing_func(process_mon, token_args, process_name="batch dl mon", num_works=8)

    pass
