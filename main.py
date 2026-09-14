import torch
from utils.config import get_dataset, get_args
from utils.Stream3D import Seq3D_MC, Stream3D, post_process
from graph.construction import mask_graph_construction
from graph.iterative_clustering import iterative_clustering
from tqdm import tqdm
import os


def main(args, para):
    seg_method_idx = [0]
    backbone = args.backbone
    # print(backbone)
    dataset_tmp = [get_dataset(args, backbone)]
    datasets = []
    for idx in seg_method_idx:
        # print('2D method: ', seg_methods[idx])
        datasets.append(dataset_tmp[idx])

    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    with torch.no_grad():

        multiple_object_lists = []

        multiple_mask_point_clouds = []

        multiple_point_frame_matrixs = []

        node_list = []

        # frame_id = dataset.get_frame_list(1)
        
        for idx in range(len(datasets)):
            multiple_object_lists.append([])
            multiple_mask_point_clouds.append([])
            multiple_point_frame_matrixs.append([])

        scene_points = datasets[0].get_scene_points()
        # frame_id = datasets[0].get_frame_list(1)
        # print(len(frame_id), frame_id)

        overlap_frame = 0
        node_num_subgraph = 1 
        frame_step_2D_mask = 1

        if args.dataset == 'scannet':
            frame_id = datasets[0].get_frame_list(10) # [0, 10, 20, 30, 40...]
            Point_ratio = 0.05   # 
            Isolated_dis = 0.05  # 
            Overlap_ratio = 0.2  # 
            Cluster_dis = 0.05   # 
            Step_num = 20        # 

        if args.dataset == 'matterport3d':
            frame_id = datasets[0].get_frame_list(1)  # [0, 1, 2, 3, 4...]
            Point_ratio = 0.05   # 
            Isolated_dis = 0.05  # 
            Overlap_ratio = 0.2  # 
            Cluster_dis = 0.05   # 
            Step_num = 20        # 

        if args.dataset == 'scannetpp':
            frame_id = datasets[0].get_frame_list(1)  # [0, 10, 20, 30, 40...]
            Point_ratio = 0.05   # 
            Isolated_dis = 0.05  # 
            Overlap_ratio = 0.2  # 
            Cluster_dis = 0.05   # 
            Step_num = 20        #  
        
        # print(frame_id)
        
        Point_ratio = 0.05
        
        Step_num = 20
        # Step_num = para

        Overlap_ratio = 0.2
        # Overlap_ratio = para

        Isolated_dis = 0.05
        # Isolated_dis = para
        Cluster_dis = Isolated_dis
        min_num = 10

        sparse = False
        if sparse:            
            frame_step_2D_mask = 10   # Sparse-ScanNet200  Sparse-ScanNet++
            Step_num = 2
        
        MaskC = False    # MaskClustering*
        if MaskC:
            node_num_subgraph = Step_num

        total_frame_num_subgraph = frame_step_2D_mask * node_num_subgraph
        Runs = int(len(frame_id)/total_frame_num_subgraph) + 1
        # Runs = 9 for visualization

        frame_lists = []
        flags = []
        flags.append(0)

        import time
        start_time_0 = time.time()

        for real_time_begin in range(Runs):
            start_time = time.time()
            begin = real_time_begin*total_frame_num_subgraph
            end = (real_time_begin+1)*total_frame_num_subgraph

            end = end + overlap_frame

            if end > len(frame_id):
                end = len(frame_id)
            frame_lists.append(list(range(begin, end, frame_step_2D_mask)))
            
            if len(frame_lists[real_time_begin]) == 0:
                break

            frame_lists[real_time_begin] = [frame_id[i] for i in frame_lists[real_time_begin]]

            for dataset_idx in range(len(datasets)):

                dataset = datasets[dataset_idx]
                
                nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix, _= mask_graph_construction(args, scene_points, frame_lists[real_time_begin], dataset, flag=flags[0], 
                                                                                                                  depth_max_pre=0.03)
                node_list.append(nodes)
                object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold[0], args.debug)

                multiple_object_lists[dataset_idx].append(object_list)

                multiple_mask_point_clouds[dataset_idx].append(mask_point_clouds)

                multiple_point_frame_matrixs[dataset_idx].append(point_frame_matrix)

            end_time = time.time()
            
        end_time = time.time()
        # print('Projection total time:', '{:.1f}s'.format(end_time - start_time_0))

        start_time_1 = time.time()

        if MaskC:   # MaskClustering*
            Seq3D_MC(datasets[0], 
                     multiple_object_lists, 
                     multiple_mask_point_clouds, 
                     scene_points, 
                     multiple_point_frame_matrixs, 
                     frame_lists, 
                     args, 
                     flags)
        else:
            Stream3D(
                datasets[0], 
                min_num,
                multiple_object_lists, 
                multiple_mask_point_clouds, 
                scene_points, 
                multiple_point_frame_matrixs, 
                frame_lists, 
                node_list, 
                args, 
                flags, 
                para=[Step_num, Point_ratio, Isolated_dis, Overlap_ratio, Cluster_dis]
                )

        end_time = time.time()
        with open(args.dataset + "_frames.txt", "a", encoding="utf-8") as f:
            f.write(f"{len(frame_id)}\n")

def Maskclustering_main(args):
    # seg_methods = ['Cropformer', 'SAM2']
    dataset = get_dataset(args, 'Cropformer')
    scene_points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list(args.step[0])
    print(frame_list)
    flags = []
    flags.append(0)
    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix, depth_max_pre= mask_graph_construction(args, scene_points, frame_list, dataset, flag=flags[0], depth_max_pre=20)

        object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold[0], args.debug)

        post_process(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args)
    
        with open(args.dataset + "_frames.txt", "a", encoding="utf-8") as f:
                f.write(f"{len(frame_list)}\n")


if __name__ == '__main__':

    cpu_num = 4
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    args = get_args()
    seq_name_list = args.seq_name_list.split('+')
    para = args.para
    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        # print(args.seq_name)
        main(args, para)
        # Maskclustering_main(args)