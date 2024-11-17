import copy
import math
from cmath import inf
from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
import numpy as np
from multipledispatch import dispatch
from shapely.geometry import LineString, Polygon

EPS = 0.15
INVALID = 255


@dataclass
class Point2d:
    x: float = inf
    y: float = inf


@dataclass
class FrenetPointd:
    s_: float = inf
    l_: float = inf


@dataclass
class LaneIdEx:
    lane_id: str = "LANE_ID_INVALID"
    branching_index: int = -1


@dataclass
class LaneFrenet:
    lane_id_ex: LaneIdEx = LaneIdEx("LANE_ID_INVALID", -1)
    frenet: FrenetPointd = FrenetPointd(inf, inf)


@dataclass
class FrenetInfo:
    frenet: FrenetPointd = FrenetPointd(inf, inf)
    from_left_boundary_m: float = 0
    from_right_boundary_m: float = 0


@dataclass
class LaneSegmentInfo:
    left_boundary_offset_m: float = inf
    left_boundary_type: str = "LANE_BOUNDARY_SEGMENT_TYPE_UNKNOWN"
    left_boundary_color: str = "LANE_BOUNDARY_SEGMENT_COLOR_UNKNOWN"
    right_boundary_offset_m: float = inf
    right_boundary_type: str = "LANE_BOUNDARY_SEGMENT_TYPE_UNKNOWN"
    right_boundary_color: str = "LANE_BOUNDARY_SEGMENT_COLOR_UNKNOWN"
    curvature: float = inf
    slope_rad: float = inf
    super_elevation: str = "SUPER_ELEVATION_CLASS_UNKNOWN"
    lane_type: str = "LANE_TYPE_UNKNOWN"
    lane_transition_type: str = "LANE_TRANSITION_TYPE_UNKNOWN"
    road_type: str = "ROAD_TYPE_UNKNOWN"
    additional_layer: str = "LAYER_TYPE_UNKNOWN"


@dataclass
class LaneSpeed:
    min_speed_mps: float = 0
    max_speed_mps: float = 0
    conditional_speed_mps: float = 0


def to_positional(lane_id: str) -> int:
    if lane_id == "LANE_ID_EGO_LANE":
        return 0
    if lane_id == "LANE_ID_LEFT_LANE":
        return 1
    if lane_id == "LANE_ID_RIGHT_LANE":
        return -1
    if lane_id == "LANE_ID_NEXT_LEFT_LANE":
        return 2
    if lane_id == "LANE_ID_NEXT_RIGHT_LANE":
        return -2
    return 0


class Lane:
    def __init__(self, rm_lane: dict):
        self.additional_layers = rm_lane["additional_layers"]
        self.arc_length_m = np.array(rm_lane["arc_length_m"])
        self.center_line = np.array(
            [[pt["x"], pt["y"]] for pt in rm_lane["center_line"]]
        )
        self.conditional_speeds = rm_lane["conditional_speeds"]
        self.curvature = rm_lane["curvature"]
        self.fixed_speeds = rm_lane["fixed_speeds"]
        self.global_lane_id = rm_lane["global_lane_id"]
        self.lane_id = rm_lane["lane_id"]
        self.lane_segments = rm_lane["lane_segments"]
        self.lane_transition_types = rm_lane["lane_transition_types"]
        self.lane_types = rm_lane["lane_types"]
        self.on_opposite_road = rm_lane["on_opposite_road"]
        self.road_types = rm_lane["road_types"]
        self.slope_rad = rm_lane["slope_rad"]
        self.super_elevation = rm_lane["super_elevation"]
        self.total_length_m = rm_lane["total_length_m"]

    def inner(self, info: FrenetInfo) -> bool:
        return (
            (info.frenet.s_ <= self.arc_length_m[-1] + EPS)
            and (info.frenet.s_ >= self.arc_length_m[0] - EPS)
            and (info.from_left_boundary_m <= EPS)
            and (info.from_right_boundary_m >= -EPS)
        )

    def get_nearest_segment_by_arc_length(
        self, arc_length_m: float
    ) -> Tuple[int, float]:
        idx, res = -1, inf
        if (
            len(self.arc_length_m) == 0
            or arc_length_m < self.arc_length_m[0]
            or arc_length_m > self.arc_length_m[-1]
        ):
            return idx, res
        idx = np.abs(self.arc_length_m - arc_length_m).argmin()
        res = arc_length_m - self.arc_length_m[idx]
        return idx, res

    def get_lane_segment_by_arc_length(self, arc_length_m: float) -> LaneSegmentInfo:
        segment = LaneSegmentInfo(
            left_boundary_type="LANE_BOUNDARY_SEGMENT_TYPE_UNKNOWN",
            right_boundary_type="LANE_BOUNDARY_SEGMENT_TYPE_UNKNOWN",
            lane_type="LANE_TYPE_UNKNOWN",
        )
        if (
            len(self.arc_length_m) == 0
            or arc_length_m < self.arc_length_m[0]
            or arc_length_m > self.arc_length_m[-1]
        ):
            return segment
        idx, res = self.get_nearest_segment_by_arc_length(arc_length_m)
        if idx != -1:
            segment.left_boundary_offset_m = self.lane_segments[idx]["left_boundary"][
                "offset_to_center_line"
            ]
            segment.left_boundary_type = self.lane_segments[idx]["left_boundary"][
                "boundary_type"
            ]
            segment.left_boundary_color = self.lane_segments[idx]["left_boundary"][
                "color"
            ]
            segment.right_boundary_offset_m = self.lane_segments[idx]["right_boundary"][
                "offset_to_center_line"
            ]
            segment.right_boundary_type = self.lane_segments[idx]["right_boundary"][
                "boundary_type"
            ]
            segment.right_boundary_color = self.lane_segments[idx]["right_boundary"][
                "color"
            ]
            if self.curvature:
                segment.curvature = self.curvature[idx]
            if self.slope_rad:
                segment.slope_rad = self.slope_rad[idx]
        for lane_type in self.lane_types:
            if (
                lane_type["start_arc_length_m"] <= arc_length_m
                and arc_length_m < lane_type["end_arc_length_m"]
            ):
                segment.lane_type = lane_type["type"]
                break
        for lane_transition_type in self.lane_transition_types:
            if (
                lane_transition_type["start_arc_length_m"] <= arc_length_m
                and arc_length_m < lane_transition_type["end_arc_length_m"]
            ):
                segment.lane_transition_type = lane_transition_type["type"]
                break
        for road_type in self.road_types:
            if (
                road_type["start_arc_length_m"] <= arc_length_m
                and arc_length_m < road_type["end_arc_length_m"]
            ):
                segment.road_type = road_type["type"]
                break
        for additional_layer in self.additional_layers:
            if (
                additional_layer["start_arc_length_m"] <= arc_length_m
                and arc_length_m < additional_layer["end_arc_length_m"]
            ):
                segment.additional_layer = additional_layer["layer_type"]
                break
        return segment

    def get_lane_speed_by_arc_length(self, arc_length_m: float) -> LaneSpeed:
        rst = LaneSpeed()
        for fix_speed in self.fixed_speeds:
            if (
                fix_speed["start_arc_length_m"] <= arc_length_m
                and arc_length_m < fix_speed["end_arc_length_m"]
            ):
                rst.min_speed_mps = fix_speed["min_speed_mps"]
                rst.max_speed_mps = fix_speed["max_speed_mps"]
                break
        for conditional_speed in self.conditional_speeds:
            if (
                conditional_speed["start_arc_length_m"] <= arc_length_m
                and arc_length_m < conditional_speed["end_arc_length_m"]
            ):
                rst.conditional_speed_mps = conditional_speed["conditional_speed_mps"]
                break
        return rst

    def is_in_routing(self, arc_length_m: float) -> bool:
        idx, _ = self.get_nearest_segment_by_arc_length(arc_length_m)
        if idx != -1:
            return self.lane_segments[idx]["on_routing"]
        return False

    def uniform_linear_smooth(self, points: np.array, interval=5):
        ret = []
        ret.append(points[0])
        for i in range(1, len(points)):
            pt = points[i]
            start = ret[-1]
            dis = np.linalg.norm(pt - start)
            if dis > interval:
                num = math.floor(dis / interval)
                piece_num = math.ceil(dis / 0.1)
                incre_piece = math.ceil(interval / 0.1)
                pt_diff = pt - start
                for j in range(1, num):
                    iter_pt = start + pt_diff * (
                        float(j * incre_piece) / float(piece_num)
                    )
                    ret.append(iter_pt)
        ret.append(points[-1])
        ret = np.array(ret)
        return ret

    def get_boundary_with_type(self):
        if len(self.center_line) == 0:
            return [], []
        l_b, r_b = self.get_boundary()
        l_t = []
        r_t = []
        for i in range(len(self.lane_segments)):
            segment = self.lane_segments[i]
            is_left_dashed = (
                1 if "DASHED" in segment["left_boundary"]["boundary_type"] else 0
            )
            l_t.append(is_left_dashed)
            is_right_dashed = (
                1 if "DASHED" in segment["right_boundary"]["boundary_type"] else 0
            )
            r_t.append(is_right_dashed)

        def split_list(lst):
            result = []
            current_list = []
            current_index = 0
            for index, item in enumerate(lst):
                if len(current_list) == 0 or item == current_list[-1]:
                    current_list.append(item)
                else:
                    result.append((current_list, current_index))
                    current_list = [item]
                    current_index = index
            if len(current_list) >= 2:
                result.append((current_list, current_index))
            return result

        def process_boundary_by_type(boundary_points, boundary_types):
            split = split_list(boundary_types)
            result = []
            for i in range(len(split) - 1):
                result.append(
                    {
                        "is_dash": split[i][0][0] == 1,
                        "points": self.uniform_linear_smooth(
                            boundary_points[split[i][1] : split[i + 1][1]]
                        ),
                    }
                )
            result.append(
                {
                    "is_dash": split[-1][0][0] == 1,
                    "points": self.uniform_linear_smooth(
                        boundary_points[split[-1][1] :]
                    ),
                }
            )
            return result

        return process_boundary_by_type(l_b, l_t), process_boundary_by_type(r_b, r_t)

    def get_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        left_boundary = list()
        right_boundary = list()
        total_points = len(self.center_line)
        for i in range(total_points):
            point_vec = (
                (self.center_line[i] - self.center_line[i - 1])
                if i == total_points - 1
                else (self.center_line[i + 1] - self.center_line[i])
            )
            normalized_vec = point_vec / np.sqrt(np.sum(point_vec**2))
            left_offset = self.lane_segments[i]["left_boundary"][
                "offset_to_center_line"
            ]
            right_offset = self.lane_segments[i]["right_boundary"][
                "offset_to_center_line"
            ]
            left_boundary.append(
                self.center_line[i]
                + np.array([-normalized_vec[1], normalized_vec[0]]) * left_offset
            )
            right_boundary.append(
                self.center_line[i]
                + np.array([normalized_vec[1], -normalized_vec[0]]) * right_offset
            )
        return np.stack(left_boundary), np.stack(right_boundary)


class BranchingLane:
    def __init__(self, rm_branching_lane: dict):
        self.branching_type = rm_branching_lane["branch_type"]
        self.connect_index = rm_branching_lane["connect_index"]
        self.lane = Lane(rm_branching_lane["lane"])
        self.target_lane_id = rm_branching_lane["target_lane_id"]


class RoadModelOperator:
    def __init__(self, road_model_dict: dict):
        self.lanes = [Lane(x) for x in road_model_dict["lanes"]]
        self.branching_lanes = [
            BranchingLane(x) for x in road_model_dict["branching_lanes"]
        ]
        self.num_lane = len(self.lanes) + len(self.branching_lanes)
        self.shift = np.array(
            [
                road_model_dict["ego_pose"]["translation"]["x"],
                road_model_dict["ego_pose"]["translation"]["y"],
            ]
        )
        self.rot = np.array(road_model_dict["ego_pose"]["rotation"]["data"]).reshape(
            3, 3
        )[:2, :2]

    @dispatch(Point2d)
    def wcs_to_vcs(self, point: Point2d) -> Point2d:
        vcs_pt = np.matmul([point.x, point.y] - self.shift, self.rot)
        return Point2d(vcs_pt[0], vcs_pt[1])

    @dispatch(Point2d)
    def vcs_to_wcs(self, point: Point2d) -> Point2d:
        wcs_pt = np.matmul([point.x, point.y], self.rot.T) + self.shift
        return Point2d(wcs_pt[0], wcs_pt[1])

    @dispatch(np.ndarray)
    def wcs_to_vcs(self, point: np.ndarray) -> np.ndarray:
        vcs_pt = np.matmul(point - self.shift, self.rot)
        return vcs_pt

    @dispatch(np.ndarray)
    def vcs_to_wcs(self, point: np.ndarray) -> np.ndarray:
        wcs_pt = np.matmul(point, self.rot.T) + self.shift
        return wcs_pt

    def frenet_to_vcs_cartesian(
        self, frenet_point: FrenetPointd, lane: Lane
    ) -> Point2d:
        point = Point2d(inf, inf)
        idx, res = lane.get_nearest_segment_by_arc_length(frenet_point.s_)
        if idx == -1:
            return point
        vec = np.array([1.0, 0.0])
        if (idx == 0 or res >= 0) and (idx + 1) < len(lane.arc_length_m):
            vec = lane.center_line[idx + 1] - lane.center_line[idx]
        elif idx - 1 >= 0:
            vec = lane.center_line[idx] - lane.center_line[idx - 1]
        else:
            pass

        lat = frenet_point.l_
        if lat >= inf:
            lat = lane.lane_segments[idx]["left_boundary"]["offset_to_center_line"]
        elif lat <= -inf:
            lat = -lane.lane_segments[idx]["right_boundary"]["offset_to_center_line"]
        else:
            pass

        def move_in_direction(
            x: float, y: float, dir: np.ndarray, s: float
        ) -> np.ndarray:
            dir_norm = np.linalg.norm(dir)
            if dir_norm == 0:
                return np.array([x, y])
            ratio = s / dir_norm
            return np.array([x + ratio * dir[0], y + ratio * dir[1]])

        tmp = move_in_direction(
            lane.center_line[idx][0], lane.center_line[idx][1], vec, res
        )
        pt = move_in_direction(tmp[0], tmp[1], np.array([-vec[1], vec[0]]), lat)
        return Point2d(pt[0], pt[1])

    def frenet_to_wcs_cartesian(
        self, frenet_point: FrenetPointd, lane: Lane
    ) -> Point2d:
        return self.vcs_to_wcs(self.frenet_to_vcs_cartesian(frenet_point, lane))

    def get_frenet_info(self, point: Point2d, lane: Lane, is_vcs=True) -> FrenetInfo:
        vcs_point = copy.deepcopy(point) if is_vcs else self.wcs_to_vcs(point)
        frenet_info = FrenetInfo(FrenetPointd(inf, inf), 0, 0)
        lane_center_line_size = len(lane.center_line)
        if lane_center_line_size < 2:
            return frenet_info
        idx = np.argmax(lane.arc_length_m >= 0)
        if idx == 0 and lane.arc_length_m[-1] < 0:
            idx = lane_center_line_size - 1
        dis_square = np.linalg.norm(lane.center_line[idx] - [vcs_point.x, vcs_point.y])
        if vcs_point.x >= lane.center_line[idx][0]:
            i = idx + 1
            while (
                i < lane_center_line_size
                and lane.center_line[i][0] >= lane.center_line[i - 1][0]
            ):
                tmp = np.linalg.norm(lane.center_line[i] - [vcs_point.x, vcs_point.y])
                if tmp < dis_square:
                    dis_square = tmp
                    idx = i
                i += 1
        else:
            i = idx - 1
            while i >= 0 and lane.center_line[i][0] <= lane.center_line[i + 1][0]:
                tmp = np.linalg.norm(lane.center_line[i] - [vcs_point.x, vcs_point.y])
                if tmp < dis_square:
                    dis_square = tmp
                    idx = i
                i -= 1
        frenet_info.frenet = self.__get_frenet_pointd(vcs_point, lane.center_line, idx)
        frenet_info.frenet.s_ += lane.arc_length_m[idx]
        frenet_info.from_left_boundary_m = (
            frenet_info.frenet.l_
            - lane.lane_segments[idx]["left_boundary"]["offset_to_center_line"]
        )
        frenet_info.from_right_boundary_m = (
            frenet_info.frenet.l_
            + lane.lane_segments[idx]["right_boundary"]["offset_to_center_line"]
        )
        return frenet_info

    def get_frenet_point(
        self, point: Point2d, lane: Lane, ignore_boundary=True, is_vcs=True
    ) -> FrenetPointd:
        """
        when point is out of range, will extrapolation if ignore_boundary is True, else output (inf,inf).
        """
        frenet_info = self.get_frenet_info(point, lane, is_vcs=is_vcs)
        if frenet_info.frenet.s_ < inf and not ignore_boundary:
            if not lane.inner(frenet_info):
                frenet_info.frenet = FrenetPointd()
        return copy.deepcopy(frenet_info.frenet)

    def get_lane_by_point(self, point: Point2d, is_vcs=True) -> LaneFrenet:
        vcs_point = copy.deepcopy(point) if is_vcs else self.wcs_to_vcs(point)
        lane_frenet = LaneFrenet(
            LaneIdEx("LANE_ID_INVALID", -1), FrenetPointd(inf, inf)
        )
        quick_skip_left = INVALID
        quick_skip_right = INVALID

        def update_quick_skip(lane_id: str, info: FrenetInfo) -> None:
            nonlocal quick_skip_left
            nonlocal quick_skip_right
            if info.from_left_boundary_m > 0:
                quick_skip_right = to_positional(lane_id)
            elif info.from_right_boundary_m < 0:
                quick_skip_left = to_positional(lane_id)

        def lane_quick_skip(lane_id: str) -> bool:
            if lane_id == "LANE_ID_INVALID":
                return True
            positional = to_positional(lane_id)
            return (quick_skip_left != INVALID and positional >= quick_skip_left) or (
                quick_skip_right != INVALID and positional <= quick_skip_right
            )

        def branching_quick_skip(target_lane_id: str, type: str) -> bool:
            if target_lane_id == "LANE_ID_INVALID":
                return True
            positional = to_positional(target_lane_id)
            return (
                quick_skip_left != INVALID
                and positional >= quick_skip_left
                and type == "BRANCHING_TYPE_LEFT"
            ) or (
                quick_skip_right != INVALID
                and positional <= quick_skip_right
                and type == "BRANCHING_TYPE_RIGHT"
            )

        for lane in self.lanes:
            if lane_quick_skip(lane.lane_id):
                continue
            info = self.get_frenet_info(vcs_point, lane, is_vcs=True)
            if info.frenet.s_ < inf:
                if lane.inner(info):
                    lane_frenet.lane_id_ex.lane_id = lane.lane_id
                    lane_frenet.frenet = info.frenet
                    return copy.deepcopy(lane_frenet)
                update_quick_skip(lane.lane_id, info)

        for i in range(len(self.branching_lanes)):
            if branching_quick_skip(
                self.branching_lanes[i].target_lane_id,
                self.branching_lanes[i].branching_type,
            ):
                continue
            lane_frenet.frenet = self.get_frenet_point(
                vcs_point, lane, ignore_boundary=False, is_vcs=True
            )
            if lane_frenet.frenet.s_ < inf:
                lane_frenet.lane_id_ex.branching_index = i
                return copy.deepcopy(lane_frenet)
        return copy.deepcopy(lane_frenet)

    def get_lane_by_id_ex(self, lane_id_ex: LaneIdEx) -> Lane:
        result = self.__get_lane_by_id(lane_id_ex.lane_id)
        if not result:
            result = self.__get_branch_lane_by_index(lane_id_ex.branching_index)
        return result

    def get_adjoined_lane(
        self, lane_id_ex: LaneIdEx, arc_length_m: float, is_left: bool
    ) -> LaneIdEx:
        result = LaneIdEx("LANE_ID_INVALID", -1)
        boundary_pot = Point2d(inf, inf)
        frenet = (
            FrenetPointd(arc_length_m, inf)
            if is_left
            else FrenetPointd(arc_length_m, -inf)
        )

        ipt_branch = None
        if lane_id_ex.lane_id != "LANE_ID_INVALID":
            boundary_pot = self.frenet_to_vcs_cartesian(
                frenet, self.__get_lane_by_id(lane_id_ex.lane_id)
            )
        else:
            ipt_branch = self.__get_branch_lane_by_index(lane_id_ex.branching_index)
            if ipt_branch is not None:
                boundary_pot = self.frenet_to_vcs_cartesian(frenet, ipt_branch.lane)
        if boundary_pot.x >= inf:
            return result

        def is_on_boundary(lane: Lane) -> bool:
            info = self.get_frenet_info(boundary_pot, lane)
            return (
                info.frenet.s_ < lane.arc_length_m[-1] + EPS
                and info.frenet.s_ > lane.arc_length_m[0] - EPS
                and (
                    abs(info.from_right_boundary_m)
                    if is_left
                    else abs(info.from_left_boundary_m)
                )
                < EPS
            )

        ipt_positional = 0
        ipt_branch_type = "BRANCHING_TYPE_UNKNOWN"
        if lane_id_ex.lane_id != "LANE_ID_INVALID":
            ipt_positional = to_positional(lane_id_ex.lane_id)
        else:
            ipt_positional = to_positional(ipt_branch.target_lane_id)
            ipt_branch_type = ipt_branch.branching_type
        for des_lane in self.lanes:
            if des_lane.lane_id == "LANE_ID_INVALID":
                continue
            des_positional = to_positional(des_lane.lane_id)
            if lane_id_ex.lane_id != "LANE_ID_INVALID":
                if (
                    des_positional <= ipt_positional
                    if is_left
                    else des_positional >= ipt_positional
                ):
                    continue
            elif ipt_branch_type == "BRANCHING_TYPE_LEFT":
                if (
                    des_positional <= ipt_positional
                    if is_left
                    else des_positional > ipt_positional
                ):
                    continue
            elif ipt_branch_type == "BRANCHING_TYPE_RIGHT":
                if (
                    des_positional < ipt_positional
                    if is_left
                    else des_positional >= ipt_positional
                ):
                    continue

            if is_on_boundary(des_lane):
                return LaneIdEx(des_lane.lane_id, -1)

        for i in range(len(self.branching_lanes)):
            des_positional = to_positional(self.branching_lanes[i].target_lane_id)
            if lane_id_ex.lane_id != "LANE_ID_INVALID":
                if (
                    (
                        des_positional <= ipt_positional
                        and self.branching_lanes[i].branching_type
                        == "BRANCHING_TYPE_RIGHT"
                    )
                    if is_left
                    else (
                        des_positional >= ipt_positional
                        and self.branching_lanes[i].branching_type
                        == "BRANCHING_TYPE_LEFT"
                    )
                ):
                    continue
            else:
                if (
                    des_positional < ipt_positional
                    if is_left
                    else des_positional > ipt_positional
                ):
                    continue
            if is_on_boundary(self.branching_lanes[i].lane):
                return LaneIdEx("LANE_ID_INVALID", i)
        return result

    def get_vcs_local_map(self) -> list:
        local_map = list()
        for lane in self.lanes:
            if lane.lane_id != "LANE_ID_INVALID":
                local_map.extend(lane.get_boundary())
        for branching_lane in self.branching_lanes:
            local_map.extend(branching_lane.lane.get_boundary())
        return local_map

    def get_wcs_local_map(self) -> list:
        vcs_local_map = self.get_vcs_local_map()
        return [self.vcs_to_wcs(lane) for lane in vcs_local_map]

    # vis
    def get_vis_map(self, wcs=True) -> gpd.GeoDataFrame:
        geometry = []

        def get_lane_polygon(lane: Lane, wcs=True) -> Polygon:
            if len(lane.center_line) == 0:
                return None
            l, r = lane.get_boundary()
            if wcs:
                l = self.vcs_to_wcs(l)
                r = self.vcs_to_wcs(r)
            l_pt = l.tolist()
            r_pt = r.tolist()
            r_pt.reverse()
            l_pt.extend(r_pt)
            return Polygon(l_pt)

        for lane in self.lanes:
            poly = get_lane_polygon(lane, wcs=wcs)
            if poly is not None:
                geometry.append(poly)
        for br_lane in self.branching_lanes:
            poly = get_lane_polygon(br_lane.lane, wcs=wcs)
            if poly is not None:
                geometry.append(poly)
        return gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    def get_vis_lane(self, wcs=True) -> gpd.GeoDataFrame:
        lane_info = {"geometry": []}

        def append_lane(lane: Lane):
            l_r, r_r = lane.get_boundary_with_type()
            if wcs:
                for i in range(len(l_r)):
                    l_r[i]["points"] = self.vcs_to_wcs(l_r[i]["points"])
                for i in range(len(r_r)):
                    r_r[i]["points"] = self.vcs_to_wcs(r_r[i]["points"])
            l_r.extend(r_r)
            for seg in l_r:
                if len(seg["points"]) >= 2:
                    if seg["is_dash"]:
                        for i in range(0, len(seg["points"]), 2):
                            if i + 2 < len(seg["points"]):
                                lane_info["geometry"].append(
                                    LineString(seg["points"][i : i + 2])
                                )
                    else:
                        lane_info["geometry"].append(LineString(seg["points"]))

        for lane in self.lanes:
            append_lane(lane)
        for br_lane in self.branching_lanes:
            append_lane(br_lane.lane)
        return gpd.GeoDataFrame(lane_info, crs="EPSG:32633")

    def get_lane_nums(self) -> int:
        return self.num_lane

    def __get_lane_by_id(self, lane_id: str) -> Lane:
        if lane_id == "LANE_ID_INVALID":
            return None
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None

    def __get_branch_lane_by_index(self, index: int) -> BranchingLane:
        if index >= 0 and index < len(self.branching_lanes):
            return self.branching_lanes[index]
        return None

    def __get_frenet_pointd(
        self, point: Point2d, point_sequence: np.ndarray, nearest_index: int
    ) -> FrenetPointd:
        if len(point_sequence) == 0 or nearest_index >= len(point_sequence):
            return FrenetPointd(inf, inf)
        vec_target_nearest = [point.x, point.y] - point_sequence[nearest_index]
        norm_target_nearest = np.linalg.norm(vec_target_nearest)
        if len(point_sequence) == 1 or norm_target_nearest == 0:
            return FrenetPointd(0, norm_target_nearest)

        def segment_frenet(vec_segment: np.ndarray, frenet: FrenetPointd) -> bool:
            norm_segment = np.linalg.norm(vec_segment)
            if norm_segment == 0:
                return False
            cos_theta = np.dot(vec_segment, vec_target_nearest) / (
                norm_segment * norm_target_nearest
            )
            sin_theta = (
                np.sqrt(1 - cos_theta * cos_theta) if abs(cos_theta) < 1 else 0.0
            )
            sin_dis = norm_target_nearest * sin_theta
            frenet.s_ = norm_target_nearest * cos_theta
            frenet.l_ = (
                sin_dis if np.cross(vec_segment, vec_target_nearest) >= 0 else -sin_dis
            )
            return frenet.s_ >= 0 and frenet.s_ <= norm_segment

        backward_flag = False
        backward_frenet = FrenetPointd(inf, inf)
        if nearest_index > 0:
            backward_flag = segment_frenet(
                point_sequence[nearest_index - 1] - point_sequence[nearest_index],
                backward_frenet,
            )
            backward_frenet.s_ = -backward_frenet.s_
            backward_frenet.l_ = -backward_frenet.l_

        forward_flag = False
        forward_frenet = FrenetPointd(inf, inf)
        if nearest_index + 1 < len(point_sequence):
            forward_flag = segment_frenet(
                point_sequence[nearest_index + 1] - point_sequence[nearest_index],
                forward_frenet,
            )

        result = FrenetPointd(inf, inf)
        if not backward_flag and forward_flag:
            result = forward_frenet
        elif backward_flag and not forward_flag:
            result = backward_frenet
        elif backward_flag and forward_flag:
            result = (
                backward_frenet
                if abs(backward_frenet.l_) < abs(forward_frenet.l_)
                else forward_frenet
            )
        else:
            if nearest_index == 0:
                result = forward_frenet
            elif nearest_index + 1 == len(point_sequence):
                result = backward_frenet
            else:
                result.s_ = 0.0
                result.l_ = (
                    -norm_target_nearest
                    if (backward_frenet.l_ < 0 or forward_frenet.l_ < 0)
                    else norm_target_nearest
                )
        return result


if __name__ == "__main__":
    rm_operator = RoadModelOperator(road_model_result_data.data[1663218524107166000])
    lane = rm_operator.get_lane_by_id_ex(LaneIdEx("LANE_ID_EGO_LANE", -1))
    frenet_info = rm_operator.get_frenet_info(Point2d(1, 0), lane)
    vcs_pos = rm_operator.frenet_to_vcs_cartesian(frenet_info.frenet, lane)
    wcs_pos = rm_operator.frenet_to_wcs_cartesian(frenet_info.frenet, lane)
    assert abs(vcs_pos.x - 1.0) < 1e-4 and abs(vcs_pos.y) < 1e-4

    lane_frenet = rm_operator.get_lane_by_point(Point2d(0, 0), is_vcs=True)
    lane = rm_operator.get_lane_by_id_ex(lane_frenet.lane_id_ex)
    left_boundary, right_boundary = lane.get_boundary()
    seg_info = lane.get_lane_segment_by_arc_length(0)
    left_lane_id_ex = rm_operator.get_adjoined_lane(lane_frenet.lane_id_ex, 0, True)
    left_lane = rm_operator.get_lane_by_id_ex(left_lane_id_ex)

    test_pt = Point2d(-32781.16, 25093.5)
    vcs_pt = rm_operator.wcs_to_vcs(test_pt)
    wcs_pt = rm_operator.vcs_to_wcs(vcs_pt)
    assert abs(test_pt.x - wcs_pt.x) < 1e-4 and abs(test_pt.y - wcs_pt.y) < 1e-4

    local_map = rm_operator.get_wcs_local_map()
    pass
