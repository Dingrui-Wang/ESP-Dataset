import copy
import json
import math
import os
import os.path as osp
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import (
    RoadModelOperator,
    get_file_list,
    get_logger,
    multiprocessing_func,
    print_log,
    read_json,
)


def preprocess_esp_feature(esp_feature):
    esp_feature = pd.DataFrame(esp_feature)
    esp_feature[
        [
            "tv_dist_to_ev",
            "tv_dist_to_cipv",
            "ego_dist_to_tv",
            "ego_dist_to_cipv",
            "ego_dist_to_ev",
        ]
    ] = esp_feature[
        [
            "tv_dist_to_ev",
            "tv_dist_to_cipv",
            "ego_dist_to_tv",
            "ego_dist_to_cipv",
            "ego_dist_to_ev",
        ]
    ].fillna(
        200  # self.esp_config["dist_padding"]
    )
    esp_feature[
        [
            "tv_speed_to_ev",
            "tv_speed_to_cipv",
            "ego_speed_to_tv",
            "ego_speed_to_cipv",
            "ego_speed_to_ev",
        ]
    ] = esp_feature[
        [
            "tv_speed_to_ev",
            "tv_speed_to_cipv",
            "ego_speed_to_tv",
            "ego_speed_to_cipv",
            "ego_speed_to_ev",
        ]
    ].fillna(
        0  # self.esp_config["speed_padding"]
    )
    esp_feature[
        [
            "tv_ttc_to_ev",
            "tv_ttc_to_cipv",
            "ego_ttc_to_tv",
            "ego_ttc_to_cipv",
            "ego_ttc_to_ev",
        ]
    ] = esp_feature[
        [
            "tv_ttc_to_ev",
            "tv_ttc_to_cipv",
            "ego_ttc_to_tv",
            "ego_ttc_to_cipv",
            "ego_ttc_to_ev",
        ]
    ].fillna(
        200  # self.esp_config["ttc_padding"]
    )

    def clamp_dist(x):
        return (
            200  # self.esp_config["dist_padding"]
            if x > 200  # self.esp_config["dist_padding"]
            else x
        )

    def clamp_ttc(x):
        return (
            200  # self.esp_config["ttc_padding"]
            if x > 200  # self.esp_config["ttc_padding"]
            else x
        )

    esp_feature.loc[:, "tv_dist_to_ev"] = esp_feature["tv_dist_to_ev"].apply(clamp_dist)
    esp_feature.loc[:, "tv_dist_to_cipv"] = esp_feature["tv_dist_to_cipv"].apply(
        clamp_dist
    )
    esp_feature.loc[:, "ego_dist_to_tv"] = esp_feature["ego_dist_to_tv"].apply(
        clamp_dist
    )
    esp_feature.loc[:, "ego_dist_to_cipv"] = esp_feature["ego_dist_to_cipv"].apply(
        clamp_dist
    )
    esp_feature.loc[:, "ego_dist_to_ev"] = esp_feature["ego_dist_to_ev"].apply(
        clamp_dist
    )
    esp_feature.loc[:, "tv_ttc_to_ev"] = esp_feature["tv_ttc_to_ev"].apply(clamp_ttc)
    esp_feature.loc[:, "tv_ttc_to_cipv"] = esp_feature["tv_ttc_to_cipv"].apply(
        clamp_ttc
    )
    esp_feature.loc[:, "ego_ttc_to_tv"] = esp_feature["ego_ttc_to_tv"].apply(clamp_ttc)
    esp_feature.loc[:, "ego_ttc_to_cipv"] = esp_feature["ego_ttc_to_cipv"].apply(
        clamp_ttc
    )
    esp_feature.loc[:, "ego_ttc_to_ev"] = esp_feature["ego_ttc_to_ev"].apply(clamp_ttc)
    return esp_feature.to_numpy()


def uniform_linear_smooth(lane: np.array, interval):
    ret = []
    ret.append(lane[0])
    for i in range(1, len(lane)):
        pt = lane[i]
        start = ret[-1]
        dis = np.linalg.norm(pt - start)
        if dis > interval:
            num = math.floor(dis / interval)
            piece_num = math.ceil(dis / 0.1)
            incre_piece = math.ceil(interval / 0.1)
            pt_diff = pt - start
            for j in range(1, num):
                iter_pt = start + pt_diff * (float(j * incre_piece) / float(piece_num))
                ret.append(iter_pt)
    ret.append(lane[-1])
    ret = np.array(ret)
    return ret


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = (
            torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
        )
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = (
            torch.stack(
                (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1
            )
            .view(-1, 3, 3)
            .float()
        )
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def transform_trajs_to_center_coords(
    obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None
):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = (
        obj_trajs.clone()
        .view(1, num_objects, num_timestamps, num_attrs)
        .repeat(num_center_objects, 1, 1, 1)
    )
    obj_trajs[:, :, :, 0 : center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
        angle=-center_heading,
    ).view(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
            angle=-center_heading,
        ).view(num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs


def generate_batch_polylines_from_map(
    polylines,
    point_sampled_interval=1,
    vector_break_dist_thresh=1.0,
    num_points_each_polyline=20,
):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate(
        (sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1
    )  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (
        np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1)
        > vector_break_dist_thresh
    ).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[: len(new_polyline)] = new_polyline
        cur_valid_mask[: len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(
                polyline_list[k][idx : idx + num_points_each_polyline]
            )

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    # # CHECK the results
    # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
    # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
    # assert center_dist.max() < 10
    return ret_polylines, ret_polylines_mask


def create_map_data_for_center_objects(center_objects, map_infos, center_offset):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(
        neighboring_polylines, neighboring_polyline_valid_mask
    ):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6],
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6],
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = torch.from_numpy(map_infos["all_polylines"].copy())
    center_objects = torch.from_numpy(center_objects)

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(),
        point_sampled_interval=1,
        vector_break_dist_thresh=5.0,
        num_points_each_polyline=10,
    )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 90

    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(
            batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0
        )
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[
            None, :
        ].repeat(num_center_objects, 1)
        center_offset_rot = rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=center_objects[:, 6],
        ).view(num_center_objects, 2)

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(
            dim=-1
        )  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[
            topk_idxs
        ]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[
            topk_idxs
        ]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines = batch_polylines[None, :, :, :].repeat(
            num_center_objects, 1, 1, 1
        )
        map_polylines_mask = batch_polylines_mask[None, :, :].repeat(
            num_center_objects, 1, 1
        )

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask,
    )

    temp_sum = (
        map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()
    ).sum(
        dim=-2
    )  # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / torch.clamp_min(
        map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0
    )  # (num_center_objects, num_polylines, 3)

    map_polylines = map_polylines.numpy()
    map_polylines_mask = map_polylines_mask.numpy()
    map_polylines_center = map_polylines_center.numpy()

    return map_polylines, map_polylines_mask, map_polylines_center


def process_agent_data(token_dict):
    sdc_track_index = token_dict["sdc_track_index"]
    current_time_index = token_dict["current_time_index"]
    timestamps = np.array(
        token_dict["timestamps_seconds"][: current_time_index + 1], dtype=np.float32
    )
    track_infos = token_dict["track_infos"]
    track_index_to_predict = np.array(token_dict["tracks_to_predict"]["track_index"])
    obj_types = np.array(track_infos["object_type"])
    obj_ids = np.array(track_infos["object_id"])
    obj_trajs_full = track_infos["trajs"]  # (num_objects, num_timestamp, 10)
    obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
    obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]
    # obj_trajs_past = track_infos['all_his_trajs']
    # tv_gt = track_infos['tv_gt_traj']
    # get_interested_agents
    center_objects = np.array([obj_trajs_full[0][current_time_index]])
    track_index_to_predict = np.array([0])

    # generate_centered_trajs_for_agents
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape
    # transform to cpu torch tensor
    center_objects = torch.from_numpy(center_objects).float()
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
    timestamps = torch.from_numpy(timestamps)

    # transform coordinates to the centered objects
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6,
        rot_vel_index=[7, 8],
    )

    ## generate the attributes for each object
    object_onehot_mask = torch.zeros(
        (num_center_objects, num_objects, num_timestamps, 5)
    )
    object_onehot_mask[:, obj_types == "TYPE_VEHICLE", :, 0] = 1
    object_onehot_mask[
        :, obj_types == "TYPE_PEDESTRAIN", :, 1
    ] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == "TYPE_CYCLIST", :, 2] = 1
    object_onehot_mask[
        torch.arange(num_center_objects), track_index_to_predict, :, 3
    ] = 1
    object_onehot_mask[:, sdc_track_index, :, 4] = 1

    object_time_embedding = torch.zeros(
        (num_center_objects, num_objects, num_timestamps, num_timestamps + 1)
    )
    object_time_embedding[
        :, :, torch.arange(num_timestamps), torch.arange(num_timestamps)
    ] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

    object_heading_embedding = torch.zeros(
        (num_center_objects, num_objects, num_timestamps, 2)
    )
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[
        :, :, :, 7:9
    ]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (
        vel - vel_pre
    ) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat(
        (
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ),
        dim=-1,
    )

    ret_obj_valid_mask = obj_trajs[
        :, :, :, -1
    ]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    ##  generate label for future trajectories
    obj_trajs_future = torch.from_numpy(obj_trajs_future).float()  # 1*50*10
    obj_trajs_future = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_future,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6,
        rot_vel_index=[7, 8],
    )  # 1*1*50*10
    ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    ret_obj_valid_mask_future = obj_trajs_future[
        :, :, :, -1
    ]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0
    (obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask) = (
        ret_obj_trajs.numpy(),
        ret_obj_valid_mask.numpy(),
        ret_obj_trajs_future.numpy(),
        ret_obj_valid_mask_future.numpy(),
    )

    center_obj_idxs = np.arange(len(track_index_to_predict))
    center_gt_trajs = obj_trajs_future_state[
        center_obj_idxs, track_index_to_predict
    ]  # (num_center_objects, num_future_timestamps, 4)
    center_gt_trajs_mask = obj_trajs_future_mask[
        center_obj_idxs, track_index_to_predict
    ]  # (num_center_objects, num_future_timestamps)
    center_gt_trajs[center_gt_trajs_mask == 0] = 0

    # filter invalid past trajs
    valid_past_mask = np.logical_not(
        obj_trajs_past[:, :, -1].sum(axis=-1) == 0
    )  # (num_objects (original))

    obj_trajs_mask = obj_trajs_mask[
        :, valid_past_mask
    ]  # (num_center_objects, num_objects (filtered), num_timestamps)
    obj_trajs_data = obj_trajs_data[
        :, valid_past_mask
    ]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
    obj_trajs_future_state = obj_trajs_future_state[
        :, valid_past_mask
    ]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
    obj_trajs_future_mask = obj_trajs_future_mask[
        :, valid_past_mask
    ]  # (num_center_objects, num_objects, num_timestamps_future):
    obj_types = obj_types[valid_past_mask]
    obj_ids = obj_ids[valid_past_mask]
    valid_index_cnt = valid_past_mask.cumsum(axis=0)
    track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
    sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

    # generate the final valid position of each object
    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros(
        (num_center_objects, num_objects, 3), dtype=np.float32
    )
    for k in range(num_timestamps):
        cur_valid_mask = (
            obj_trajs_mask[:, :, k] > 0
        )  # (num_center_objects, num_objects)
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

    center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
    for k in range(center_gt_trajs_mask.shape[1]):
        cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
        center_gt_final_valid_idx[cur_valid_mask] = k

    return (
        obj_trajs_data,
        obj_trajs_mask > 0,
        obj_trajs_pos,
        obj_trajs_last_pos,
        obj_trajs_future_state,
        obj_trajs_future_mask,
        center_gt_trajs,
        center_gt_trajs_mask,
        center_gt_final_valid_idx,
        track_index_to_predict_new,
        sdc_track_index_new,
        obj_types,
        obj_ids,
    )


def process_map_data(token_dict, center_objects):
    road_model_path = (
        os.path.join(
            "/mnt/data_cpfs/zheyuan/esp_public_dataset/maps", token_dict["map_id"]
        )
        + ".json"
    )
    road_model = read_json(road_model_path)
    rm_op = RoadModelOperator(road_model)
    tv_x, tv_y = token_dict["track_infos"]["trajs"][0][
        token_dict["current_time_index"]
    ][:2]
    yaw = token_dict["track_infos"]["trajs"][0][token_dict["current_time_index"]][6]
    local_map = list()
    polygons = []
    for lane in rm_op.lanes:
        if lane.lane_id != "LANE_ID_INVALID":
            left, right = lane.get_boundary()
            left = rm_op.vcs_to_wcs(left)
            right = rm_op.vcs_to_wcs(right)
            local_map.append({"left_boundary": left, "right_boundary": right})
    for branching_lane in rm_op.branching_lanes:
        left, right = branching_lane.lane.get_boundary()
        left = rm_op.vcs_to_wcs(left)
        right = rm_op.vcs_to_wcs(right)
        local_map.append({"left_boundary": left, "right_boundary": right})

    def composite_polygon(lane, start_idx, end_idx):
        return np.array(
            [
                lane["centerline"][start_idx:end_idx],
                lane["left_boundary"][start_idx:end_idx],
                lane["right_boundary"][start_idx:end_idx],
            ]
        )

    for i in range(len(local_map)):
        for k in ["left_boundary", "right_boundary"]:
            local_map[i][k] = uniform_linear_smooth(local_map[i][k], 2)

        min_num = min(
            len(local_map[i]["left_boundary"]),
            len(local_map[i]["right_boundary"]),
        )
        local_map[i]["left_boundary"] = local_map[i]["left_boundary"][:min_num]
        local_map[i]["right_boundary"] = local_map[i]["right_boundary"][:min_num]
        local_map[i]["centerline"] = (
            local_map[i]["right_boundary"] + local_map[i]["left_boundary"]
        ) / 2.0
        for k in range(int(min_num / 10)):
            polygons.append(
                composite_polygon(
                    local_map[i],
                    k * 10,
                    (k + 1) * 10,
                )
            )

    polygons.sort(
        key=lambda x: np.sum(np.linalg.norm(np.array([tv_x, tv_y]) - x[0], axis=1))
    )
    polygons = polygons[: min(30, len(polygons))]
    all_polylines = []

    def expand_single(line):
        z = np.zeros([10, 1])
        line = np.hstack([line, z])
        polyline_pre = np.roll(line, shift=1, axis=0)
        polyline_pre[0] = line[0]
        diff = line - polyline_pre
        polyline_dir = diff / np.clip(
            np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000
        )
        cur_polyline = np.concatenate(
            (line[:, 0:3], polyline_dir, np.ones([10, 1])), axis=-1
        )
        return cur_polyline

    for poly in polygons:
        all_polylines.append(expand_single(poly[0]))
        all_polylines.append(expand_single(poly[1]))
        all_polylines.append(expand_single(poly[2]))
    all_polylines = np.concatenate(all_polylines, axis=0)
    map_infos = {"all_polylines": all_polylines}
    return create_map_data_for_center_objects(center_objects, map_infos, (30.0, 0))


def get_sample_from_token(mon, token):
    token_path = (
        "/mnt/data_cpfs/zheyuan/esp_public_dataset/mons/"
        + mon
        + "/pruned_tokens/"
        + token
        + ".json"
    )
    token = json.load(open(token_path, "rb"))
    tv_info = token["TvInformation"]
    tv_his = tv_info["historical_trajectories"]
    tv_gt = tv_info["gt_trajectories"]
    esp_features = preprocess_esp_feature(
        token["ExtroSpectivePredictionFeatures"]
    )  # (30, 15)

    def get_tv_his_gt():
        tv_his_traj = []
        for i in range(len(tv_his)):
            center_x = tv_info["historical_trajectories"][i]["x"]
            center_y = tv_info["historical_trajectories"][i]["y"]
            length = tv_info["vehicle_box"]["length"]
            width = tv_info["vehicle_box"]["width"]
            qw = tv_info["historical_poses"][i]["qw"]
            qx = tv_info["historical_poses"][i]["qx"]
            qy = tv_info["historical_poses"][i]["qy"]
            qz = tv_info["historical_poses"][i]["qz"]
            # heading = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
            heading = np.arctan2(
                2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qy**2 + qz**2)
            )
            velocity_x = tv_info["historical_velocities"][i]["vx"]
            velocity_y = tv_info["historical_velocities"][i]["vy"]
            state = np.array(
                [
                    center_x,
                    center_y,
                    1.0,
                    length,
                    width,
                    1.0,
                    heading,
                    velocity_x,
                    velocity_y,
                    1.0,
                ],
                dtype=np.float32,
            )
            tv_his_traj.append(state)
        tv_his_traj = np.stack(tv_his_traj, axis=0)  # 30*10
        tv_gt_traj = []
        for i in range(len(tv_gt)):
            center_x = tv_info["gt_trajectories"][i]["x"]
            center_y = tv_info["gt_trajectories"][i]["y"]
            length = tv_info["vehicle_box"]["length"]
            width = tv_info["vehicle_box"]["width"]
            qw = tv_info["historical_poses"][-1]["qw"]
            qx = tv_info["historical_poses"][-1]["qx"]
            qy = tv_info["historical_poses"][-1]["qy"]
            qz = tv_info["historical_poses"][-1]["qz"]
            # heading = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
            heading = np.arctan2(
                2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qy**2 + qz**2)
            )
            velocity_x = tv_info["historical_velocities"][-1]["vx"]
            velocity_y = tv_info["historical_velocities"][-1]["vy"]
            state = np.array(
                [
                    center_x,
                    center_y,
                    1.0,
                    length,
                    width,
                    1.0,
                    heading,
                    velocity_x,
                    velocity_y,
                    1.0,
                ],
                dtype=np.float32,
            )
            tv_gt_traj.append(state)
        tv_gt_traj = np.stack(tv_gt_traj, axis=0)  # 50*10
        return tv_his_traj, tv_gt_traj

    tv_his_traj, tv_gt_traj = get_tv_his_gt()
    tv_all_traj = np.concatenate([tv_his_traj, tv_gt_traj], axis=0)

    ego_his_traj = []
    ego_info = token["EgoVehicleInformation"]
    for i in range(len(ego_info["historical_timestamp"])):
        center_x = ego_info["historical_trajectories"][i]["x"]
        center_y = ego_info["historical_trajectories"][i]["y"]
        length = ego_info["vehicle_box"]["length"]
        width = ego_info["vehicle_box"]["width"]
        qw = ego_info["historical_poses"][i]["qw"]
        qx = ego_info["historical_poses"][i]["qx"]
        qy = ego_info["historical_poses"][i]["qy"]
        qz = ego_info["historical_poses"][i]["qz"]
        # heading = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
        heading = np.arctan2(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qy**2 + qz**2))
        velocity_x = ego_info["historical_velocities"][i]["vx"]
        velocity_y = ego_info["historical_velocities"][i]["vy"]
        state = np.array(
            [
                center_x,
                center_y,
                1.0,
                length,
                width,
                1.0,
                heading,
                velocity_x,
                velocity_y,
                1.0,
            ],
            dtype=np.float32,
        )
        ego_his_traj.append(state)
    ego_his_traj = np.stack(ego_his_traj, axis=0)  # 30*10
    ego_future_traj = np.array(
        [
            [
                center_x,
                center_y,
                1.0,
                length,
                width,
                1.0,
                heading,
                velocity_x,
                velocity_y,
                1.0,
            ]
            for i in range(len(tv_gt_traj))
        ]
    )
    ego_all_traj = np.concatenate([ego_his_traj, ego_future_traj], axis=0)

    evs_traj = []
    for veh in token["OtherVehiclesInformation"]:
        if len(veh["historical_timestamp"]) == len(tv_his):
            ev_traj = []
            for i in range(len(veh["historical_timestamp"])):
                center_x = veh["historical_trajectories"][i]["x"]
                center_y = veh["historical_trajectories"][i]["y"]
                length = veh["vehicle_box"]["length"]
                width = veh["vehicle_box"]["width"]
                qw = veh["historical_poses"][i]["qw"]
                qx = veh["historical_poses"][i]["qx"]
                qy = veh["historical_poses"][i]["qy"]
                qz = veh["historical_poses"][i]["qz"]
                # heading = np.arctan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
                heading = np.arctan2(
                    2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qy**2 + qz**2)
                )
                velocity_x = veh["historical_velocities"][i]["vx"]
                velocity_y = veh["historical_velocities"][i]["vy"]
                state = np.array(
                    [
                        center_x,
                        center_y,
                        1.0,
                        length,
                        width,
                        1.0,
                        heading,
                        velocity_x,
                        velocity_y,
                        1.0,
                    ],
                    dtype=np.float32,
                )
                ev_traj.append(state)
            ev_traj = np.stack(ev_traj, axis=0)  # 30*10
            ev_future = np.array(
                [
                    [
                        center_x,
                        center_y,
                        1.0,
                        length,
                        width,
                        1.0,
                        heading,
                        velocity_x,
                        velocity_y,
                        1.0,
                    ]
                    for i in range(len(tv_gt_traj))
                ]
            )
            ev_all_traj = np.concatenate([ev_traj, ev_future], axis=0)
            evs_traj.append(ev_all_traj)
    all_vehilce_traj = [tv_all_traj, ego_all_traj]
    all_vehilce_traj.extend(evs_traj)
    all_vehilce_traj = np.array(all_vehilce_traj)  # num*30*10

    preprocess_dict = dict()
    preprocess_dict["scenario_id"] = token["TokenId"]
    preprocess_dict["map_id"] = token["MapId"]
    preprocess_dict["timestamps_seconds"] = [
        0.1 * i for i in range(len(tv_his) + len(tv_gt))
    ]
    preprocess_dict["current_time_index"] = len(tv_his) - 1
    preprocess_dict["sdc_track_index"] = 1  # ego
    preprocess_dict["objects_of_interest"] = []
    preprocess_dict["tracks_to_predict"] = {
        "track_index": [0],
        "difficulty": [0],
        "object_type": ["TYPE_VEHICLE"],
    }
    preprocess_dict["esp_feature"] = token["ExtroSpectivePredictionFeatures"]
    preprocess_dict["track_infos"] = dict()
    preprocess_dict["track_infos"]["trajs"] = all_vehilce_traj
    agents_num = all_vehilce_traj.shape[0]
    old_time_stamps = all_vehilce_traj.shape[1]
    preprocess_dict["track_infos"]["object_type"] = [
        "TYPE_VEHICLE" for i in range(agents_num)
    ]
    preprocess_dict["track_infos"]["object_id"] = [i for i in range(agents_num)]
    preprocess_dict["timestamps_seconds"] = [
        round(0.1 * i, 1) for i in range(old_time_stamps)
    ]
    preprocess_dict["tv_bbox"] = token["TvInformation"]["vehicle_box"]
    preprocess_dict["gt_cut_in"] = token["TvInformation"]["gt_bbox_cut_in"]
    obj_trajs_past = all_vehilce_traj[:, : len(tv_his)]
    # obj_trajs_past = track_infos['all_his_trajs']
    # tv_gt = track_infos['tv_gt_traj']
    # get_interested_agents
    center_objects = np.array([obj_trajs_past[0][-1]])
    (
        obj_trajs_data,
        obj_trajs_mask,
        obj_trajs_pos,
        obj_trajs_last_pos,
        obj_trajs_future_state,
        obj_trajs_future_mask,
        center_gt_trajs,
        center_gt_trajs_mask,
        center_gt_final_valid_idx,
        track_index_to_predict_new,
        sdc_track_index_new,
        obj_types,
        obj_ids,
    ) = process_agent_data(copy.deepcopy(preprocess_dict))
    map_polylines_data, map_polylines_mask, map_polylines_center = process_map_data(
        preprocess_dict, center_objects
    )
    track_index_to_predict = np.array([0])
    ret_dict = {
        "scenario_id": np.array([token["TokenId"]] * len(track_index_to_predict)),
        "token_id": token["TokenId"],
        "map_id": token["MapId"],
        "tv_bbox": token["TvInformation"]["vehicle_box"],
        "gt_cut_in": token["TvInformation"]["gt_bbox_cut_in"],
        "obj_trajs": obj_trajs_data,
        "obj_trajs_mask": obj_trajs_mask,
        "track_index_to_predict": track_index_to_predict_new,  # used to select center-features
        "obj_trajs_pos": obj_trajs_pos,
        "obj_trajs_last_pos": obj_trajs_last_pos,
        "obj_types": obj_types,
        "obj_ids": obj_ids,
        "center_objects_world": center_objects,
        "center_objects_id": np.array(preprocess_dict["track_infos"]["object_id"])[
            track_index_to_predict
        ],
        "center_objects_type": np.array(preprocess_dict["track_infos"]["object_type"])[
            track_index_to_predict
        ],
        "obj_trajs_future_state": obj_trajs_future_state,
        "obj_trajs_future_mask": obj_trajs_future_mask,
        "center_gt_trajs": center_gt_trajs,
        "center_gt_trajs_mask": center_gt_trajs_mask,
        "center_gt_final_valid_idx": center_gt_final_valid_idx,
        "center_gt_trajs_src": all_vehilce_traj[track_index_to_predict],
        "map_polylines": map_polylines_data,
        "map_polylines_mask": (map_polylines_mask > 0),
        "map_polylines_center": map_polylines_center,
        "esp_features": esp_features,
    }
    return ret_dict


if __name__ == "__main__":
    mode = "test_new"
    path = "/mnt/data_cpfs/zheyuan/esp_public_dataset"
    table_dir = Path(path, mode)
    table_path = str(table_dir) + ".csv"
    table_data = pd.read_csv(table_path)
    args = []
    # for i in range(int(len(table_data) / 2), len(table_data)):
    #     sample = table_data.iloc[i]
    #     args.append([sample.mon_id, sample.token_id])
    # all_process_data = multiprocessing_func(
    #     get_sample_from_token, args, "process", num_works=12
    # )
    preprocess_dict = get_sample_from_token(
        table_data.iloc[0].mon_id, table_data.iloc[0].token_id
    )
    # pickle.dump(
    #     all_process_data, open("/mnt/data_cpfs/zheyuan/esp_mtr/data/val/val.pkl", "wb")
    # )
    pass
