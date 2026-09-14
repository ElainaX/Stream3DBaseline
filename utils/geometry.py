import numpy as np

def judge_bbox_overlay(bbox_1, bbox_2):
    for i in range(3):
        if bbox_1[0][i] > bbox_2[1][i] or bbox_2[0][i] > bbox_1[1][i]:
            return False
    return True

def denoise(pcd, eps=0.04, min_points=4, component=0.2, nb_neighbors=20, std_ratio=0.5):

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points)) + 1 # -1 for noise
    mask = np.ones(len(labels), dtype=bool)
    count = np.bincount(labels)
    # remove component with less than 20% points
    for i in range(len(count)):
        if count[i] < component * len(labels):  # component
            mask[labels == i] = False
    remain_index = np.where(mask)[0]
    pcd = pcd.select_by_index(remain_index)
    
    pcd, index = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # pcd, index = pcd.remove_radius_outlier(nb_points=6, radius=0.05)
    remain_index = remain_index[index]

    return pcd, []

def filter_boundary(depth, delta=0.05):
    remove_mask = np.zeros(depth.shape).astype(bool)
    delta_depth_1 = np.abs(depth[1:, :] - depth[:-1, :])
    delta_depth_2 = np.abs(depth[:, 1:] - depth[:, :-1])
    remove_mask[1:, :] = remove_mask[1:, :] | (delta_depth_1 > delta)
    remove_mask[:-1, :] = remove_mask[:-1, :] | (delta_depth_1 > delta)
    remove_mask[:, 1:] = remove_mask[:, 1:] | (delta_depth_2 > delta)
    remove_mask[:, :-1] = remove_mask[:, :-1] | (delta_depth_2 > delta)
    depth[remove_mask] = 0
    return depth