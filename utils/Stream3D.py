import numpy as np
import os
import torch
import open3d as o3d
import heapq
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from utils.geometry import judge_bbox_overlay
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from itertools import chain


def merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio):
    '''
        Merge objects that have larger than 0.8 overlapping ratio.
    '''
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)

    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i+1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            if judge_bbox_overlay(bbox_i, bbox_j):
                intersect = len(point_ids_i.intersection(point_ids_j))
                if intersect / len(point_ids_i) > overlapping_ratio:
                    invalid_object[i] = True
                elif intersect / len(point_ids_j) > overlapping_ratio:
                    invalid_object[j] = True

    valid_point_ids_list = []
    valid_pcld_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_pcld_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_pcld_mask_list


def filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args, flag=0):
    '''
        Following OVIR-3D, we filter the points that hardly appear in this cluster (node), i.e. the detection ratio is lower than a threshold.
        Specifically, detection ratio = #frames that the point appears in this cluster (node) / #frames that the point appears in the whole video.
    '''
    def count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list):
        '''
            For all points in the cluster, compute #frames that the point appears in the whole video.
            Initialize #frames that the point appears in this cluster as 0.
        '''
        point_appear_in_video_nums, point_appear_in_node_matrixs = [], []
        for point_ids in point_ids_list:
            point_appear_in_video_matrix = point_frame_matrix[point_ids, ]
            point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
            point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
            
            point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool) # initialize as False
            point_appear_in_node_matrixs.append(point_appear_in_node_matrix)
        return point_appear_in_video_nums, point_appear_in_node_matrixs

    def count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs):
        '''
            Fillin the point_appear_in_node_matrixs by iterating the masks in this cluster (node).
            Meanwhile, since we split the disconnected point cloud into different objects, we also decide which object this mask belongs to.
            Besides, for each mask, we compute the coverage of this mask of the object it belongs to for furture use in OpenMask3D.
        '''
        object_mask_list = [[] for _ in range(len(point_ids_list))]

        for frame_id, mask_id in mask_list:
            frame_id_in_list = np.where(node_frame_id_list == frame_id)[0][0]
            mask_point_ids = list(mask_point_clouds[f'{frame_id}_{mask_id}'])

            object_id_with_largest_intersect, largest_intersect, coverage = -1, 0, 0
            for i, point_ids in enumerate(point_ids_list):
                point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
                point_appear_in_node_matrixs[i][point_ids_within_object, frame_id_in_list] = True
                if len(point_ids_within_object) > largest_intersect:
                    object_id_with_largest_intersect, largest_intersect = i, len(point_ids_within_object)
                    coverage = len(point_ids_within_object) / len(point_ids)
            if largest_intersect == 0:
                continue
            object_mask_list[object_id_with_largest_intersect] += [(frame_id, mask_id, coverage)]
        return object_mask_list, point_appear_in_node_matrixs

    node_global_frame_id_list = torch.where(node.visible_frame)[0].cpu().numpy()
    node_frame_id_list = np.array(frame_list)[node_global_frame_id_list]
    mask_list = node.mask_list
    # print(mask_list)
    point_appear_in_video_nums, point_appear_in_node_matrixs = count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list)
    object_mask_list, point_appear_in_node_matrixs = count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs)

    # filter points
    filtered_point_ids, filtered_mask_list, filtered_bbox_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(zip(point_appear_in_video_nums, point_appear_in_node_matrixs)):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        valid_point_ids = np.where(detection_ratio > args.point_filter_threshold[flag])[0]
        # print(detection_ratio, args.point_filter_threshold[flag], len(valid_point_ids), len(object_mask_list[i]))
        if (len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2):
        # if len(valid_point_ids) == 0:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        filtered_bbox_list.append([np.amin(pcld_list[i].points, axis=0), np.amax(pcld_list[i].points, axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def filter_point_new(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args, flag=0):
    '''
        Following OVIR-3D, we filter the points that hardly appear in this cluster (node), i.e. the detection ratio is lower than a threshold.
        Specifically, detection ratio = #frames that the point appears in this cluster (node) / #frames that the point appears in the whole video.
    '''
    def count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list):
        '''
            For all points in the cluster, compute #frames that the point appears in the whole video.
            Initialize #frames that the point appears in this cluster as 0.
        '''
        point_appear_in_video_nums, point_appear_in_node_matrixs = [], []
        for point_ids in point_ids_list:
            point_appear_in_video_matrix = point_frame_matrix[point_ids, ]
            point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
            point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
            
            point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool) # initialize as False
            point_appear_in_node_matrixs.append(point_appear_in_node_matrix)
        return point_appear_in_video_nums, point_appear_in_node_matrixs

    def count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs):
        '''
            Fillin the point_appear_in_node_matrixs by iterating the masks in this cluster (node).
            Meanwhile, since we split the disconnected point cloud into different objects, we also decide which object this mask belongs to.
            Besides, for each mask, we compute the coverage of this mask of the object it belongs to for furture use in OpenMask3D.
        '''
        object_mask_list = [[] for _ in range(len(point_ids_list))]

        for frame_id, mask_id in mask_list:
            frame_id_in_list = np.where(node_frame_id_list == frame_id)[0][0]
            mask_point_ids = list(mask_point_clouds[f'{frame_id}_{mask_id}'])

            object_id_with_largest_intersect, largest_intersect, coverage = -1, 0, 0
            for i, point_ids in enumerate(point_ids_list):
                point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
                point_appear_in_node_matrixs[i][point_ids_within_object, frame_id_in_list] = True
                if len(point_ids_within_object) > largest_intersect:
                    object_id_with_largest_intersect, largest_intersect = i, len(point_ids_within_object)
                    coverage = len(point_ids_within_object) / len(point_ids)
            if largest_intersect == 0:
                continue
            object_mask_list[object_id_with_largest_intersect] += [(frame_id, mask_id, coverage)]
        return object_mask_list, point_appear_in_node_matrixs

    node_global_frame_id_list = torch.where(node.visible_frame)[0].cpu().numpy()
    node_frame_id_list = np.array(frame_list)[node_global_frame_id_list]
    mask_list = node.mask_list
    # print(mask_list)
    point_appear_in_video_nums, point_appear_in_node_matrixs = count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list)
    object_mask_list, point_appear_in_node_matrixs = count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs)

    # filter points
    filtered_point_ids, filtered_mask_list, filtered_bbox_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(zip(point_appear_in_video_nums, point_appear_in_node_matrixs)):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        # valid_point_ids = np.where(detection_ratio > args.point_filter_threshold[flag])[0]
        valid_point_ids = np.where(detection_ratio > 0.0)[0]
        # print(detection_ratio, args.point_filter_threshold[flag], len(valid_point_ids), len(object_mask_list[i]))
        # if (len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2):
        if len(valid_point_ids) == 0:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        filtered_bbox_list.append([np.amin(pcld_list[i].points, axis=0), np.amax(pcld_list[i].points, axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1, min_points=4):
    '''
        Following OVIR-3D, we use DBSCAN to split the disconnected point cloud into different objects.
    '''
    
    labels = np.array(pcld.cluster_dbscan(eps=DBSCAN_THRESHOLD, min_points=min_points)) + 1 # -1 for noise
    count = np.bincount(labels)

    # split disconnected point cloud into different objects
    pcld_list, point_ids_list = [], []
    pcld_ids_list = np.array(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        new_pcld = pcld.select_by_index(remain_index)
        point_ids = pcld_ids_list[remain_index]
        pcld_list.append(new_pcld)
        point_ids_list.append(point_ids)
    return pcld_list, point_ids_list


def find_represent_mask(mask_info_list):
    mask_info_list.sort(key=lambda x: x[2], reverse=True)
    return mask_info_list[:5]


def export_class_agnostic_mask(args, class_agnostic_mask_list):
    pred_dir = os.path.join('data/prediction', args.config)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    pred_masks = np.stack(class_agnostic_mask_list, axis=1)
    pred_dict = {
        "pred_masks": pred_masks, 
        "pred_score":  np.ones(num_instance),
        "pred_classes" : np.zeros(num_instance, dtype=np.int32)
    }
    class_agnostic_pred_dir = os.path.join('data/prediction', args.config + '_class_agnostic')
    os.makedirs(class_agnostic_pred_dir, exist_ok=True)
    np.savez(os.path.join(class_agnostic_pred_dir, f'{args.seq_name}.npz'), **pred_dict)
    return


def export_new(dataset, total_point_ids_list, total_mask_list, detected_points, args):
    '''
        Export class agnostic masks in standard evaluation format 
        and object dict with corresponding mask lists for semantic instance segmentation.
        Node that after clustering, a node = a cluster of masks = an object.
    '''

    flat_unique = detected_points
    total_point_num = dataset.get_scene_points().shape[0]
    class_agnostic_mask_list = []
    object_dict = {}
    for i, (point_ids, mask_list) in enumerate(zip(total_point_ids_list, total_mask_list)):
        # print(len(point_ids))
        object_dict[i] = {
            'point_ids': point_ids,
            'mask_list': mask_list,
            'repre_mask_list': find_represent_mask(mask_list),
        }
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        class_agnostic_mask_list.append(binary_mask)

    export_class_agnostic_mask(args, class_agnostic_mask_list)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    # print(dataset.object_dict_dir, args.config)
    np.save(os.path.join(dataset.object_dict_dir, args.config, 'object_dict.npy'), object_dict, allow_pickle=True)

    array = np.array(flat_unique)
    title = '............/TMP/'
    import re
    path = dataset.object_dict_dir
    # print(path)
    if args.config == 'scannet':
        match = re.search(r"scene\d{4}_\d{2}", path)
        scene_id = match.group()
    if args.config == 'scannetpp':
        match = re.search(r"data/([^/]+)", path)
        scene_id = match.group(1)
    if args.config == 'matterport3d':
        match = re.search(r"scans/([^/]+)", path)
        scene_id = match.group(1)

    os.makedirs(os.path.join(title + args.config), exist_ok=True)
    np.save(os.path.join(title + args.config + '/' + scene_id + '_pre_points.npy'), array)

def export(dataset, total_point_ids_list, total_mask_list, args):
    '''
        Export class agnostic masks in standard evaluation format 
        and object dict with corresponding mask lists for semantic instance segmentation.
        Node that after clustering, a node = a cluster of masks = an object.
    '''
    total_point_num = dataset.get_scene_points().shape[0]
    class_agnostic_mask_list = []
    object_dict = {}
    for i, (point_ids, mask_list) in enumerate(zip(total_point_ids_list, total_mask_list)):
        object_dict[i] = {
            'point_ids': point_ids,
            'mask_list': mask_list,
            'repre_mask_list': find_represent_mask(mask_list),
        }
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        class_agnostic_mask_list.append(binary_mask)

    export_class_agnostic_mask(args, class_agnostic_mask_list)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    np.save(os.path.join(dataset.object_dict_dir, args.config, 'object_dict.npy'), object_dict, allow_pickle=True)


def compute_bounding_boxes(points, masks):
    '''
    Compute axis-aligned cubic bounding boxes for each mask based on the point cloud and mask list.

    Parameters:
        points (np.ndarray/list): An N*3 array of point cloud coordinates
        masks (list): A list of K sublists, each containing point indices belonging to the same mask

    Returns:
        list: A list of K cube coordinates, each represented as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    '''
    points = np.asarray(points)
    bounding_boxes = []
    
    for mask in masks:
        if len(mask) == 0:  # Handle empty mask cases (optional, depending on requirements)
            continue

        subset = points[mask]
        
        min_coords = np.min(subset, axis=0)
        max_coords = np.max(subset, axis=0)
        
        bounding_boxes.append([min_coords.tolist(), max_coords.tolist()])
    
    return bounding_boxes


def find_overlapping_boxes(boxes, query_box):
    '''
    Find the indices of all bounding boxes that intersect with the query bounding box.

    Parameters:
        boxes (list): A list of multiple bounding boxes, each formatted as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        query_box (list): The query bounding box, with the same format as above

    Returns:
        list: A list of indices of all intersecting bounding boxes
    '''
    # print(query_box)
    q_min, q_max = query_box
    q_min_x, q_min_y, q_min_z = q_min
    q_max_x, q_max_y, q_max_z = q_max
    
    overlapping_indices = []
    # print(boxes)
    for idx, box in enumerate(boxes):

        c_min, c_max = box
        c_min_x, c_min_y, c_min_z = c_min
        c_max_x, c_max_y, c_max_z = c_max

        # print(c_min_x, c_min_y, c_min_z, c_max_x, c_max_y, c_max_z)
        
        # 检查三个轴向的投影是否重叠  Check whether the projections of the three axes overlap each other.
        overlap_x = (c_min_x <= q_max_x) and (c_max_x >= q_min_x)
        overlap_y = (c_min_y <= q_max_y) and (c_max_y >= q_min_y)
        overlap_z = (c_min_z <= q_max_z) and (c_max_z >= q_min_z)
        
        # 三个轴向均重叠则判定为相交  If all three axes overlap, it is determined that they intersect.
        if overlap_x and overlap_y and overlap_z:
            overlapping_indices.append(idx)
    
    return overlapping_indices


def post_process(dataset, node_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args, flag=0):
    if args.debug:
        print('start exporting')
    
    # For each cluster, MaskClustering follows OVIR-3D to i) use DBScan to split the disconnected point cloud into different objects
    # ii) filter the points that hardly appear within this cluster, i.e. the detection ratio is lower than a threshold
    total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
    for node in (node_list):
        if len(node.mask_list) < 2: # objects merged from less than 2 masks are ignored
            continue
        
        pcld, point_ids = node.get_point_cloud(scene_points)
        pcld_list, point_ids_list = dbscan_process(pcld, point_ids) # split the disconnected point cloud into different objects
        point_ids_list, bbox_list, mask_list = filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args, flag=flag)

        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    # merge objects that have larger than 0.8 overlapping ratio
    total_point_ids_list, total_mask_list = merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)
    export(dataset, total_point_ids_list, total_mask_list, args)
    return



def Seq3D_MC(dataset, node_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args, flags=[0, 1]):
    if args.debug:
        print('start exporting')
    frames = node_list[0]

    N = len(scene_points)
    point_mask_list = []
    for i in range(N):
        point_mask_list.append([])

    def point_to_masks(point_mask_list, cur_section, begin = 0):
        for i in range(len(cur_section)):
            for j in cur_section[i]:
                point_mask_list[j].append(i + begin)
        return point_mask_list
        
    import random
    import pandas as pd
    random.seed(10)
    min_num = 10

    def remove_connection_points_in_smaller_masks(points, masks, a):
        """
        Remove connecting points from masks with fewer points.

        Parameters:
            points (np.ndarray): An (N, 3) NumPy array of point cloud coordinates
            masks (list of lists): Each sublist contains point indices belonging to one mask
            a (float): Distance threshold

        Returns:
            list: The updated list of masks
        """
        # 1. 创建点云到 mask 的映射  Create a mapping from point clouds to masks
        point_to_mask = np.full(len(points), -1, dtype=int)
        mask_sizes = [len(mask) for mask in masks]
        
        for mask_idx, mask in enumerate(masks):
            for point_idx in mask:
                point_to_mask[point_idx] = mask_idx
        
        # 2. 收集所有 mask 点并构建 KDTree  Collect all the mask points and build a KDTree
        all_mask_points = []
        all_mask_indices = []
        
        for mask in masks:
            all_mask_points.extend(points[mask])
            all_mask_indices.extend(mask)
        
        all_mask_points = np.array(all_mask_points)
        tree = KDTree(all_mask_points)
        
        # 3. 查找所有距离小于 a 的点对  Find all pairs of points whose distance is less than a.
        connection_pairs = tree.query_pairs(a)
        
        # 4. 标记需要移除的点（在较小 mask 中的连接点） Mark the points that need to be removed (the connecting points within the smaller mask)
        points_to_remove = set()
        
        for i, j in connection_pairs:
            idx_i = all_mask_indices[i]
            idx_j = all_mask_indices[j]
            
            mask_i = point_to_mask[idx_i]
            mask_j = point_to_mask[idx_j]
            
            # 只处理不同 mask 之间的点  Only process the points between different masks
            if mask_i != mask_j:
                size_i = mask_sizes[mask_i]
                size_j = mask_sizes[mask_j]
                
                # 将较小 mask 中的点标记为需要移除  Mark the points in the smaller mask as those that need to be removed
                if size_i > size_j:
                    points_to_remove.add(idx_i)
                elif size_j > size_i:
                    points_to_remove.add(idx_j)
        
        # 5. 从 masks 中移除标记的点  Remove the marked points from the masks
        new_masks = []
        for mask in masks:
            new_mask = [idx for idx in mask if idx not in points_to_remove]
            new_masks.append(new_mask)
        
        return new_masks, list(points_to_remove)

    indices = []
    T = 1
    for i in range(T):
        indices.append([])

    all_masks = []
    all_masks_frames = []
    all_masks_points = []
    all_boxes = []

    lastframe_point_ids_list = []
    from scipy.spatial import KDTree
    all_detected_points = []
    max_iter = len(frames)
    for iter in range(max_iter):
        currentframe_point_ids_list = []
        currentframe_masks_ids_list = []
        currentframe_point_ids_list_full = []
        
        for mask_idx in range(len(point_frame_matrix)):

            total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
            for node in node_list[mask_idx][iter]:
                for t in range(len(indices)):
                    if len(node.mask_list) < 2: # objects merged from less than 2 masks are ignored
                        continue
                    pcld, point_ids = node.get_point_cloud(scene_points)                       
                    pcld_list, point_ids_list = dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1, min_points=4) # split the disconnected point cloud into different objects
                    point_ids_list, bbox_list, mask_list = filter_point(point_frame_matrix[mask_idx][iter], node, pcld_list, point_ids_list, mask_point_clouds[mask_idx][iter], frame_list[iter], args, flag=flags[0])
                    total_point_ids_list.extend(point_ids_list)
                    total_bbox_list.extend(bbox_list)
                    total_mask_list.extend(mask_list)
            total_point_ids_list, total_mask_list = merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)

            for node in mask_point_clouds[mask_idx][iter]:
                for t in range(len(indices)):
                    point_ids_list = list(mask_point_clouds[mask_idx][iter][node])
                    all_detected_points.extend(point_ids_list)

        currentframe_point_ids_list_full.extend(total_point_ids_list)
        currentframe_point_ids_list.extend(total_point_ids_list)
        currentframe_masks_ids_list.extend(total_mask_list)
        all_masks_points.extend(total_point_ids_list)
        all_masks_frames.extend(total_mask_list)

        if len(currentframe_point_ids_list) != 0:
            currentframe_point_ids_list, remove_set = remove_connection_points_in_smaller_masks(scene_points, currentframe_point_ids_list, 0.1)

        if iter == 0 and len(node_list[mask_idx][iter]) != 0:
            lastframe_point_ids_list.extend(currentframe_point_ids_list)
            cur_section = currentframe_point_ids_list
            point_mask_list = point_to_masks(point_mask_list, currentframe_point_ids_list_full, len(all_masks))
            cur_boxes = compute_bounding_boxes(scene_points, cur_section)
            all_masks.extend(cur_section)
            all_boxes.extend(cur_boxes)
            continue

        cur_section = currentframe_point_ids_list
        cur_boxes = compute_bounding_boxes(scene_points, cur_section)
        
        tmp_masks = []
        tmp_len = len(all_masks)

        overlapped_boxes_idx_pair = []
        for idx in range(len(cur_boxes)):
            overlapped_boxes_idx = find_overlapping_boxes(all_boxes, cur_boxes[idx])
            if len(overlapped_boxes_idx) == 0:
                if len(cur_section[idx]) > min_num:
                    all_masks.append(cur_section[idx])
                    all_boxes.append(cur_boxes[idx])
                    tmp_masks.append(cur_section[idx])
            else:
                overlapped_boxes_idx_pair.append([overlapped_boxes_idx, idx])

        point_mask_list = point_to_masks(point_mask_list, tmp_masks, tmp_len)
        
        tmp_masks = []
        tmp_len = len(all_masks)

        for box_idx_pair in overlapped_boxes_idx_pair:
            overlapped_boxes_idx = box_idx_pair[0]
            cur_idx = box_idx_pair[1]
            mask_cur = cur_section[cur_idx]

            max_overlap_part = 0
            max_idx = 0
            idxs = []
            nums = []
            for mask_A_idx in overlapped_boxes_idx:
                mask_pre = all_masks[mask_A_idx]

                overlap_part = list(set(mask_pre) & set(mask_cur))
                difference_pre = list(set(mask_pre) - set(overlap_part))

                if len(overlap_part) == 0:
                    continue                    

                if len(overlap_part) > max_overlap_part:
                    max_overlap_part = len(overlap_part)
                    max_idx = mask_A_idx
                
                nums.append(len(overlap_part))
                idxs.append(mask_A_idx)

                if len(difference_pre) > min_num:
                    all_masks[mask_A_idx] = difference_pre
                    all_boxes[mask_A_idx] = compute_bounding_boxes(scene_points, [difference_pre])[0]

                else:
                    mask_cur = list(set(mask_cur) | set(mask_pre))
                    all_masks[mask_A_idx] = []
                    all_boxes[mask_A_idx] = [[0, 0, 0], [0, 0, 0]]

            if max_overlap_part == 0:
                if len(cur_section[cur_idx]) > min_num:
                    # add
                    all_masks.append(cur_section[cur_idx])
                    all_boxes.append(compute_bounding_boxes(scene_points, [cur_section[cur_idx]])[0])
                    tmp_masks.append(cur_section[cur_idx])
            else:
                # merging
                point_mask_list = point_to_masks(point_mask_list, [cur_section[cur_idx]], max_idx)

                mask_cur = list(set(mask_cur) | set(all_masks[max_idx]))
                all_masks[max_idx] = mask_cur
                all_boxes[max_idx] = compute_bounding_boxes(scene_points, [mask_cur])[0]

        point_mask_list = point_to_masks(point_mask_list, tmp_masks, tmp_len)

        lastframe_point_ids_list = currentframe_point_ids_list

    ##############################
    from itertools import chain
    merged_set = list(dict.fromkeys(chain.from_iterable(all_masks)))
    all_detected_points = list(set(all_detected_points))
    difference_pre = list(set(all_detected_points) - set(merged_set))
    from scipy.spatial import KDTree

    def merge_unclassified_points(points, A, B):
        """
        Merge unclassified point clouds into the nearest mask.

        Parameters:
            points (np.ndarray): An (N, 3) NumPy array representing point cloud coordinates
            A (list of lists): Each sublist contains point IDs belonging to a mask
            B (list): A list containing unclassified point IDs

        Returns:
            list: The updated mask list A
        """
        if len(B) == 0:
            return A
            
        A = [list(mask) for mask in A]
        
        all_points = []
        mask_labels = []
        
        for mask_idx, mask_ids in enumerate(A):
            if len(mask_ids) > 0:  
                
                mask_points = points[np.array(mask_ids)]
                all_points.append(mask_points)
                # 为每个点记录其所属的掩码索引  Record the mask index to which each point belongs
                mask_labels.extend([mask_idx] * len(mask_ids))
        
        # 处理没有有效掩码点的情况 Handling the situation where there are no valid mask points
        if len(all_points) == 0:
            if len(A) > 0:  # A中有掩码但都为空  There is a mask in A, but all of them are empty.
                # 将所有未分类点加入第一个掩码  Add all unclassified points to the first mask
                A[0].extend(B)
            else:  # A为空 A is empty
                # 创建新掩码包含所有未分类点  Creating a new mask includes all unclassified points
                A.append(B)
            return A
        
        # 合并所有点并构建KD树  Merge all the points and construct a KD tree
        all_points = np.vstack(all_points)
        tree = KDTree(all_points)
        
        # 处理其他点  Handling other points
        for point_id in B:
            point = points[point_id]
            # 查询最近点（k=1） Query the nearest point (k = 1)
            dist, idx = tree.query(point, k=1)
            # 获取最近点对应的掩码索引  Obtain the mask index corresponding to the nearest point
            mask_idx = mask_labels[idx]
            # 将当前点添加到该掩码  Add the current point to this mask
            A[mask_idx].append(point_id)
            
        return A
    
    all_masks = merge_unclassified_points(scene_points, all_masks, difference_pre)  

    total_mask_list = []
    total_point_ids_list = []
    import heapq
    def find_max_k_intersecting_masks(A, B, K):
        result = []
        B_sets = [set(mask) for mask in B]
        
        for a_mask in A:
            a_set = set(a_mask)
            heap = []  # 使用最小堆来维护前K个最大的交集  # Use a minimum heap to maintain the top K largest intersections
            
            for idx, b_set in enumerate(B_sets):
                intersection_size = len(a_set & b_set)
                
                # 如果交集大小大于0，才考虑加入堆  If the size of the intersection is greater than 0, then it will be considered for adding to the heap.
                if intersection_size > 0:
                    # 如果堆未满或当前交集大于堆顶元素，则加入堆  If the heap is not full or the current intersection is greater than the top element of the heap, then add it to the heap.
                    if len(heap) < K:
                        heapq.heappush(heap, (intersection_size, idx))
                    else:
                        if intersection_size > heap[0][0]:
                            heapq.heapreplace(heap, (intersection_size, idx))
            
            # 处理没有交集的情况  Handling the situation where there is no intersection
            if len(heap) == 0:
                # 如果没有交集，返回K个-1  If there is no intersection, return K -1s.
                result.append([-1] * K)
            else:
                # 从堆中提取索引并按交集大小降序排序  Extract the indices from the heap and sort them in descending order based on the size of the intersection.
                sorted_indices = sorted(heap, key=lambda x: x[0], reverse=True)
                indices = [idx for _, idx in sorted_indices]
                
                # 如果找到的交集数量少于K，用-1填充剩余位置  If the number of found intersections is less than K, fill the remaining positions with -1.
                if len(indices) < K:
                    indices.extend([-1] * (K - len(indices)))
                
                result.append(indices)
        
        return result

    output_k = find_max_k_intersecting_masks(all_masks, all_masks_points, K=5)

    for idx in range(len(all_masks)):
        total_mask_list.append([])
        total_point_ids_list.append(all_masks[idx])
        for x in output_k[idx]:
            if x != -1:
                total_mask_list[idx].extend(all_masks_frames[x])
    export_new(dataset, total_point_ids_list, total_mask_list, all_detected_points, args)
    return


def cluster_points_with_dbscan(point_cloud, point_ids, eps=0.5, min_samples=4):
    """
    Cluster 3D point clouds using the DBSCAN algorithm.

    Parameters:
        point_cloud (np.ndarray): An N×3 NumPy array containing 3D coordinates
        point_ids (list): A list of length N, providing a unique identifier for each point
        eps (float): DBSCAN neighborhood radius, default is 0.5
        min_samples (int): Minimum number of points required to form a core point, default is 5

    Returns:
        list: Clustering results, where each sublist contains the point IDs belonging to the same cluster.
            Noise points are excluded.
    """

    assert len(point_cloud) == len(point_ids), "The point cloud data does not match the number of IDs."

    mask_pcld = o3d.geometry.PointCloud()
    mask_pcld.points = o3d.utility.Vector3dVector(point_cloud)
    labels = np.array(mask_pcld.cluster_dbscan(eps=eps, min_points=min_samples))

    clusters = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label != -1:

            cluster_indices = np.where(labels == label)[0]

            cluster_ids = [point_ids[i] for i in cluster_indices]

            if len(cluster_ids) > 10:
                clusters.append(np.array(cluster_ids))
    
    return clusters


def Stream3D(
            dataset, 
            min_num,
            node_list, 
            mask_point_clouds, 
            scene_points, 
            point_frame_matrix, 
            frame_list, 
            frame_node_list,
            args, 
            flags=[0, 1],
            para=[10, 0.01, 0.1, 0.3, 0.05]
            ):
    
    if args.debug:
        print('start exporting')
    frames = node_list[0]
        
    def remove_connection_points_in_smaller_masks(points, masks, a):
        """
        Remove connecting points from masks with fewer points.

        Parameters:
            points (np.ndarray): An (N, 3) NumPy array of point cloud coordinates
            masks (list of lists): Each sublist contains point indices belonging to one mask
            a (float): Distance threshold

        Returns:
            list: The updated list of masks
        """
        # 1. 创建点云到 mask 的映射 Create a mapping from point clouds to masks
        point_to_mask = np.full(len(points), -1, dtype=int)
        mask_sizes = [len(mask) for mask in masks]
        
        for mask_idx, mask in enumerate(masks):
            for point_idx in mask:
                point_to_mask[point_idx] = mask_idx
        
        # 2. 收集所有 mask 点并构建 KDTree  Collect all the mask points and build a KDTree
        all_mask_points = []
        all_mask_indices = []
        
        for mask in masks:
            all_mask_points.extend(points[mask])
            all_mask_indices.extend(mask)
        
        all_mask_points = np.array(all_mask_points)
        tree = KDTree(all_mask_points)
        
        # 3. 查找所有距离小于 a 的点对  Find all pairs of points whose distance is less than a.
        connection_pairs = tree.query_pairs(a)
        
        # 4. 标记需要移除的点（在较小 mask 中的连接点） Mark the points that need to be removed (the connecting points within the smaller mask)
        points_to_remove = set()
        
        for i, j in connection_pairs:
            idx_i = all_mask_indices[i]
            idx_j = all_mask_indices[j]
            
            mask_i = point_to_mask[idx_i]
            mask_j = point_to_mask[idx_j]
            
            # 只处理不同 mask 之间的点  Only process the points between different masks
            if mask_i != mask_j:
                size_i = mask_sizes[mask_i]
                size_j = mask_sizes[mask_j]
                
                # 将较小 mask 中的点标记为需要移除  Mark the points in the smaller mask as those that need to be removed
                if size_i > size_j:
                    points_to_remove.add(idx_i)
                elif size_j > size_i:
                    points_to_remove.add(idx_j)
        
        # 5. 从 masks 中移除标记的点  Remove the marked points from the masks
        new_masks = []
        for mask in masks:
            new_mask = [idx for idx in mask if idx not in points_to_remove]
            new_masks.append(new_mask)
        
        return new_masks, list(points_to_remove)
    

    def merge_unclassified_points(points, A, B):
        """
        Merge unclassified point clouds into the nearest mask.

        Parameters:
            points (np.ndarray): An (N, 3) NumPy array representing point cloud coordinates
            A (list of lists): Each sublist contains point IDs belonging to a mask
            B (list): A list containing unclassified point IDs

        Returns:
            list: The updated mask list A
        """
        
        if len(B) == 0:
            return A
            
        A = [list(mask) for mask in A]
        
        all_points = []
        mask_labels = []
        
        for mask_idx, mask_ids in enumerate(A):
            if len(mask_ids) > 0: 
                mask_points = points[np.array(mask_ids)]
                all_points.append(mask_points)
                mask_labels.extend([mask_idx] * len(mask_ids))
        
        if len(all_points) == 0:
            if len(A) > 0:  
                A[0].extend(B)
            else: 
                A.append(B)
            return A
        
        all_points = np.vstack(all_points)
        tree = KDTree(all_points)
        
        for point_id in B:
            point = points[point_id]
            dist, idx = tree.query(point, k=1)
            mask_idx = mask_labels[idx]
            A[mask_idx].append(point_id)
            
        return A


    all_masks = []
    all_masks_frames = []
    all_masks_points = []
    all_boxes = []
    lastframe_point_ids_list = []
    
    all_detected_points = []
    step_frames_masks = []
    max_iter = len(frames)
    count = 0

    # min_num = 0

    Manifold_Refining = True
    # Manifold_Refining = False

    Local_MV_Mask_Filtering = True
    # Local_MV_Mask_Filtering = False

    Fast_BBox_Retrieval = True
    # Fast_BBox_Retrieval = False

    local_detected_set = []
    
    for iter in range(max_iter):
        currentframe_point_ids_list = []
        currentframe_point_ids_list_full = []
        
        for mask_idx in range(len(point_frame_matrix)):

            total_point_ids_list, _, total_mask_list = [], [], []
            points_in_a_frame = []

            for node_idx in range(len(frame_node_list[iter])):
                single_node = frame_node_list[iter][node_idx]
                pcld, point_ids = single_node.get_point_cloud(scene_points) 
                point_ids_list_, _, mask_list = filter_point_new(point_frame_matrix[mask_idx][iter], single_node, [pcld], [np.array(point_ids)], mask_point_clouds[mask_idx][iter], frame_list[iter], args, flag=flags[0])
                all_masks_points.extend(point_ids_list_)
                all_masks_frames.extend(mask_list)

            for node in mask_point_clouds[mask_idx][iter]:
                point_ids_list = list(mask_point_clouds[mask_idx][iter][node])
                if len(point_ids_list) == 0:
                    continue
                all_detected_points.extend(point_ids_list)

                local_detected_set.extend(point_ids_list)

                points_in_a_frame.extend(point_ids_list)
                # --------------------------------------------
                if Manifold_Refining:
                    point_ids_list_tmp = cluster_points_with_dbscan(scene_points[point_ids_list], point_ids_list, eps=para[4], min_samples=1)
                    if len(point_ids_list_tmp) > 0:
                        point_ids_list = point_ids_list_tmp
                        point_ids_list.sort(key=len, reverse=True)
                    else:
                        point_ids_list = [point_ids_list]
                    total_point_ids_list.append(point_ids_list[0])
                else:
                    total_point_ids_list.append(point_ids_list)
                # --------------------------------------------
            step_frames_masks.extend(total_point_ids_list)

        # --------------------------------------------
        if Local_MV_Mask_Filtering:
            Step = para[0]  
        else:
            Step = 1  # 1
        if (Local_MV_Mask_Filtering == False) and (Manifold_Refining == False):
            Step = 1000000000 # MAX
        if iter % Step != 0 and iter != (max_iter - 1):  
            count += 1
            continue
        else:
            count += 1
            total_point_ids_list = step_frames_masks
            step_frames_masks = []

        # local_merged_set = list(dict.fromkeys(chain.from_iterable(total_point_ids_list)))
        # local_merged_set = list(set(local_merged_set))
        # local_detected_set = list(set(local_detected_set))
        # local_difference_pre = list(set(local_detected_set) - set(local_merged_set))
        # local_detected_set = []
        # --------------------------------------------

        def fps_downsample_masks_open3d(A, B, sample_ratio=0.01):
            """
            Perform mask point downsampling using Open3D's FPS implementation.

            Parameters:
                A (list of lists): Mask list, e.g. [[id1, id2, ...], ...]
                B (np.ndarray): Point cloud coordinate matrix of shape (N, 3)
                sample_ratio (float): Sampling ratio, default is 0.01 (1%)

            Returns:
                selected_ids (list): List of selected point IDs
            """

            # 步骤1: 收集所有mask中的点ID（去重）  Step 1: Collect all the point IDs in the masks (without duplicates)
            mask_point_ids = set()
            for mask in A:
                mask_point_ids.update(mask)
            
            mask_point_ids = list(mask_point_ids)
            
            if not mask_point_ids:
                return []
            
            # 步骤2: 获取mask点的坐标  Step 2: Obtain the coordinates of the mask points
            mask_points = B[mask_point_ids]
            
            # 步骤3: 计算需要采样的点数  Step 3: Calculate the number of points to be sampled
            n_samples = max(1, int(len(mask_point_ids) * sample_ratio))
            
            if n_samples >= len(mask_point_ids):
                return mask_point_ids
            
            # 步骤4: 使用Open3D进行FPS采样  Step 4: Conduct FPS sampling using Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mask_points)
            
            downsampled_pcd = pcd.farthest_point_down_sample(n_samples)
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # 步骤5: 使用KD树找到采样点在mask_points中的索引  Step 5: Use the KD tree to find the index of the sampling points in the mask_points
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(mask_points)
            _, indices = nbrs.kneighbors(downsampled_points)
            
            # 步骤6: 将采样索引映射回原始点ID  Step 6: Map the sampling index back to the original point ID
            selected_ids = [mask_point_ids[i] for i in indices.flatten()]
            
            return selected_ids
        
        def merge_masks(masks=[], Manifold_Refining=True, a=0.2):
            n = len(masks)
            if n == 0:
                return []
            
            mask_sets = [set(mask) for mask in masks]
            
            # 步骤1: 计算所有mask对的IOU并构建图（邻接列表）Step 1: Calculate the Intersection over Union (IOU) for all mask pairs and construct a graph (adjacency list)
            graph = defaultdict(list)
            for i in range(n):
                for j in range(i + 1, n):
                    set_i = mask_sets[i]
                    set_j = mask_sets[j]
                    union = set_i | set_j
                    if len(union) == 0:  
                        iou = 0.0
                    else:
                        intersection = set_i & set_j
                        iou = len(intersection) / len(union)
                    if iou > a:  # IOU大于阈值，添加边  If the IOU is greater than the threshold, add the edge.
                        graph[i].append(j)
                        graph[j].append(i)
            
            # 步骤2: 使用BFS找到连通分量（待合并的mask集合）  Step 2: Use BFS to find the connected components (the set of masks to be merged)
            visited = [False] * n
            components = []  
            for i in range(n):
                if not visited[i]:
                    comp = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        comp.append(node)
                        for neighbor in graph.get(node, []):  # 处理无边的节点  Handling infinite nodes
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
                    components.append(comp)
            
            # 步骤3: 合并每个连通分量中的mask（取点并集）  Step 3: Merge the masks within each connected component (take the union of the sets of points)
            merged_masks = []  
            for comp in components:
                if len(comp) == 1:  # single mask, directly add
                    idx = comp[0]
                    merged_masks.append(list(mask_sets[idx]))  
                else:  # multiple masks, merge
                    point_set = set()
                    for idx in comp:
                        point_set |= mask_sets[idx]  # union
                    merged_masks.append(list(point_set))  
            
            # 步骤4: 互斥处理（确保每个点只存在于一个mask中）  Step 4: Mutual exclusion processing (ensuring that each point exists only in one mask)
            point_to_mask_indices = defaultdict(list)
            for i, mask in enumerate(merged_masks):
                for point in mask:
                    point_to_mask_indices[point].append(i)
            
            # 为每个点分配目标mask索引（最小索引）  Assign the target mask index (the smallest index) to each point
            point_assignment = {}
            for point, indices in point_to_mask_indices.items():
                assigned_idx = min(indices)  # 保留在索引最小的mask中  Keep in the mask with the smallest index
                point_assignment[point] = assigned_idx
            
            # 构建互斥处理后的mask列表：每个mask只包含分配的点  Construct a list of masks after mutual exclusion processing: Each mask only contains the allocated points
            new_merged_masks = [[] for _ in range(len(merged_masks))]
            for point, assigned_idx in point_assignment.items():
                new_merged_masks[assigned_idx].append(point)
            
            # 过滤空mask并返回  Filter out the empty masks and return
            if Manifold_Refining:
                final_masks = [mask for mask in new_merged_masks if mask]
            else:
                final_masks = [mask for mask in merged_masks if mask]
            return final_masks

        def select_masks_large(mask_list, point_list_A):
            
            set_A = set(point_list_A)
            
            # 步骤1: 过滤mask_list，只保留与点列表A有交集的mask。同时，为每个mask预计算其点集，以提高效率 
            # Step 1: Filter the mask_list, retaining only the masks that have intersections with the point list A. Additionally, pre-compute the point sets for each mask to enhance efficiency.
            masks_with_sets = []
            for mask in mask_list:
                mask_set = set(mask)
                if mask_set & set_A:  # null ?
                    masks_with_sets.append((mask, mask_set))
            
            if not masks_with_sets:
                return []
            
            # 步骤2: 根据mask的大小（点数）降序排序，大小相同则保持原始顺序  Step 2: Sort the masks in descending order based on their size (number of points). If the sizes are the same, maintain the original order.
            masks_with_sets.sort(key=lambda x: len(x[0]), reverse=True)
            # masks_with_sets.sort(key=lambda x: len(x[0]), reverse=False)
            uncovered = set(set_A)  
            selected_masks = []    
            
            # 步骤3: 遍历排序后的mask列表，贪心选择  Step 3: Traverse the sorted list of masks and make greedy selections
            for mask, mask_set in masks_with_sets:
                if not uncovered:  # 所有点已覆盖，提前终止  All points have been covered. Terminating early.
                    break
                # 计算当前mask与未覆盖点的交集  Calculate the intersection of the current mask and the uncovered points
                intersection = mask_set & uncovered
                if intersection:  # 如果交集非空，选择该mask  If the intersection is not empty, select this mask
                    selected_masks.append(mask)   # add
                    uncovered -= intersection     # Remove the covered points from the un-covered set
            
            return selected_masks
        
        # --------------------------------------------
        beforlen_1 = len(total_point_ids_list)
        if Local_MV_Mask_Filtering:
            tmp_id= fps_downsample_masks_open3d(total_point_ids_list, scene_points, sample_ratio=para[1]) 
            total_point_ids_list = select_masks_large(total_point_ids_list, tmp_id)
        beforlen_2 = len(total_point_ids_list)
        if iter != 0 and beforlen_1 != 0:
            total_point_ids_list = merge_masks(total_point_ids_list, Manifold_Refining, a=para[3])   
            # print(beforlen_1, "-->", beforlen_2, "-->", len(total_point_ids_list))
            count = 0
        # --------------------------------------------
        local_merged_mask_list = []
        if Manifold_Refining:
            for mask in total_point_ids_list:
                point_ids_list_tmp = cluster_points_with_dbscan(scene_points[mask], mask, eps=para[4], min_samples=1)  #
                if len(point_ids_list_tmp) > 1:
                    point_ids_list_tmp.sort(key=len, reverse=True)
                else:
                    point_ids_list_tmp = [mask]
                local_merged_mask_list.extend(point_ids_list_tmp)
        else:
            local_merged_mask_list = total_point_ids_list
        # --------------------------------------------
        currentframe_point_ids_list_full.extend(local_merged_mask_list)
        currentframe_point_ids_list.extend(local_merged_mask_list)

        if Manifold_Refining and len(currentframe_point_ids_list) != 0:
            currentframe_point_ids_list, remove_set = remove_connection_points_in_smaller_masks(scene_points, currentframe_point_ids_list, para[2])

        if iter == 0 and len(currentframe_point_ids_list) != 0:
            lastframe_point_ids_list.extend(currentframe_point_ids_list)
            cur_section = currentframe_point_ids_list
            cur_boxes = compute_bounding_boxes(scene_points, cur_section)
            all_masks.extend(cur_section)
            all_boxes.extend(cur_boxes)
            continue

        cur_section = currentframe_point_ids_list
        cur_boxes = compute_bounding_boxes(scene_points, cur_section)
        
        tmp_masks = []

        overlapped_boxes_idx_pair = []
        for idx in range(len(cur_boxes)):
            if Fast_BBox_Retrieval:
                overlapped_boxes_idx = find_overlapping_boxes(all_boxes, cur_boxes[idx])
                if len(overlapped_boxes_idx) == 0:
                    if len(cur_section[idx]) > min_num:
                        all_masks.append(cur_section[idx])
                        all_boxes.append(cur_boxes[idx])
                        tmp_masks.append(cur_section[idx])
                else:
                    overlapped_boxes_idx_pair.append([overlapped_boxes_idx, idx])
            else:
                overlapped_boxes_idx_pair.append([list(range(len(all_boxes))), idx])

        tmp_masks = []

        for box_idx_pair in overlapped_boxes_idx_pair:
            overlapped_boxes_idx = box_idx_pair[0]
            cur_idx = box_idx_pair[1]
            mask_cur = cur_section[cur_idx]

            max_overlap_part = 0
            max_idx = 0
            idxs = []
            nums = []
            for mask_A_idx in overlapped_boxes_idx:
                mask_pre = all_masks[mask_A_idx]
                overlap_part = list(set(mask_pre) & set(mask_cur))
                if len(overlap_part) == 0:
                    continue 
                difference_pre = list(set(mask_pre) - set(overlap_part))

                if len(overlap_part) > max_overlap_part:
                    max_overlap_part = len(overlap_part)
                    max_idx = mask_A_idx
                
                nums.append(len(overlap_part))
                idxs.append(mask_A_idx)

                if len(difference_pre) > min_num:
                    if Manifold_Refining:
                        all_masks[mask_A_idx] = difference_pre
                        all_boxes[mask_A_idx] = compute_bounding_boxes(scene_points, [difference_pre])[0]
                else:
                    mask_cur = list(set(mask_cur) | set(mask_pre))
                    all_masks[mask_A_idx] = []
                    all_boxes[mask_A_idx] = [[0, 0, 0], [0, 0, 0]]

            if max_overlap_part == 0:
                if len(cur_section[cur_idx]) > min_num:
                    # add
                    all_masks.append(cur_section[cur_idx])
                    all_boxes.append(compute_bounding_boxes(scene_points, [cur_section[cur_idx]])[0])
                    tmp_masks.append(cur_section[cur_idx])
            else:
                # merge
                mask_cur = list(set(mask_cur) | set(all_masks[max_idx]))
                all_masks[max_idx] = mask_cur
                all_boxes[max_idx] = compute_bounding_boxes(scene_points, [mask_cur])[0]

        lastframe_point_ids_list = currentframe_point_ids_list
        # lastframe_point_ids_list = merge_unclassified_points(scene_points, currentframe_point_ids_list, local_difference_pre) 
    ##############################

    all_masks_tmp = []
    for m in all_masks:
        if len(m) > min_num:
            all_masks_tmp.append(m)
    all_masks = all_masks_tmp

    merged_set = list(dict.fromkeys(chain.from_iterable(all_masks)))
    all_detected_points = list(set(all_detected_points))
    difference_pre = list(set(all_detected_points) - set(merged_set))

    # neighbor point merging
    all_masks = merge_unclassified_points(scene_points, all_masks, difference_pre) 

    total_mask_list = []
    total_point_ids_list = []


    def find_max_k_intersecting_masks(A, B, K):
        result = []
        B_sets = [set(mask) for mask in B]
        
        for a_mask in A:
            a_set = set(a_mask)
            heap = []  # 使用最小堆来维护前K个最大的交集  Use a minimum heap to maintain the top K largest intersections
            
            for idx, b_set in enumerate(B_sets):
                intersection_size = len(a_set & b_set)
                
                # 如果交集大小大于0，才考虑加入堆  If the size of the intersection is greater than 0, then it will be considered for adding to the heap.
                if intersection_size > 0:
                    # 如果堆未满或当前交集大于堆顶元素，则加入堆  If the heap is not full or the current intersection is greater than the top element of the heap, then add it to the heap.
                    if len(heap) < K:
                        heapq.heappush(heap, (intersection_size, idx))
                    else:
                        if intersection_size > heap[0][0]:
                            heapq.heapreplace(heap, (intersection_size, idx))
            
            if len(heap) == 0:
                # 如果没有交集，返回K个-1  If there is no intersection, return K -1s.
                result.append([-1] * K)
            else:
                # 从堆中提取索引并按交集大小降序排序  Extract the indices from the heap and sort them in descending order based on the size of the intersection.
                sorted_indices = sorted(heap, key=lambda x: x[0], reverse=True)
                indices = [idx for _, idx in sorted_indices]
                
                # 如果找到的交集数量少于K，用-1填充剩余位置  If the number of found intersections is less than K, fill the remaining positions with -1.
                if len(indices) < K:
                    indices.extend([-1] * (K - len(indices)))
                
                result.append(indices)
        
        return result
    
    output_k = find_max_k_intersecting_masks(all_masks, all_masks_points, K=20) # para[0]

    for idx in range(len(all_masks)):
        total_mask_list.append([])
        total_point_ids_list.append(all_masks[idx])
        for x in output_k[idx]:
            if x != -1:
                total_mask_list[idx].append(all_masks_frames[x][0])
    export_new(dataset, total_point_ids_list, total_mask_list, all_detected_points, args)
    return
