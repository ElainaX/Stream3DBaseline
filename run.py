import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from tqdm import tqdm
import time
from utils.config import get_args
import os
from multiprocessing import Pool

CUDA_LIST = [0, 1, 2, 3]
# for i in range(torch.cuda.device_count()):
#     try:
#         torch.cuda.set_device(i)
#         torch.cuda.get_device_name(i)
#         # print(f"GPU {i} ok")
#         CUDA_LIST.append(i)
#     except Exception as e:
#         print(f"GPU {i} fault: {e}")

CPU_CORE = 4
print('GPUs:', CUDA_LIST)

# nohup python run.py --config matterport3d >my.log_matterport3d &
# python -m visualize.vis_scene --config matterport3d --seq_name 2t7WUuJeko7
# nohup python run.py --config scannet >my.log_scannet &
# python -m visualize.vis_scene --config scannet --seq_name scene0050_00

def execute_commands(commands_list, command_type, process_num):
    print('====> Start', command_type)
    from multiprocessing import Pool
    pool = Pool(process_num)
    for _ in tqdm(pool.imap_unordered(os.system, commands_list), total=len(commands_list)):
        pass
    pool.close()
    pool.join()
    pool.terminate()
    print('====> Finish', command_type)

def execute_commands_2(commands_list, command_type, process_num, core_num=4):
    import os
    import psutil
    from multiprocessing import Pool
    from tqdm import tqdm

    print('====> Start', command_type)

    def get_least_busy_cores(total_cores_needed):
        # 获取当前各 CPU 核的使用率  Obtain the usage rate of each current CPU core
        cpu_percents = psutil.cpu_percent(percpu=True)
        # 按照使用率升序排列核编号   Sort the kernel numbers in ascending order of usage rate
        sorted_cores = sorted(range(len(cpu_percents)), key=lambda i: cpu_percents[i])
        # 返回使用率最低的核编号  Return the kernel number with the lowest usage rate
        return sorted_cores[:total_cores_needed]

    def group_cores(core_ids, group_size):
        return [set(core_ids[i:i + group_size]) for i in range(0, len(core_ids), group_size)]

    # 获取最空闲的 32 个核（8组，每组4核）  Obtain the 32 most idle cores (8 groups, 4 cores per group)
    available_core_ids = get_least_busy_cores(process_num * core_num)
    cpu_groups = group_cores(available_core_ids, core_num)

    def init_worker():
        pid = os.getpid()
        idx = pid % len(cpu_groups)  # 可替换为更精准索引  Can be replaced by a more precise index
        os.sched_setaffinity(0, cpu_groups[idx])
        # print(f"[PID {pid}] on available CPUs {sorted(cpu_groups[idx])}")

    pool = Pool(processes=process_num, initializer=init_worker)

    for _ in tqdm(pool.imap_unordered(os.system, commands_list), total=len(commands_list)):
        pass

    pool.close()
    pool.join()
    pool.terminate()

    print('====> Finish', command_type)


def get_seq_name_list(dataset):
    if dataset == 'scannet':
        file_path = 'splits/scannet.txt'
    elif dataset == 'scannetpp':
        file_path = 'splits/scannetpp.txt'
    elif dataset == 'matterport3d':
        file_path = 'splits/matterport3d.txt'
        
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    return seq_name_list

def parallel_compute(general_command, command_name, resource_type, cuda_list, seq_name_list):
    cuda_num = len(cuda_list)
    
    if resource_type == 'cuda':
        commands = []
        for i, cuda_id in enumerate(cuda_list):
            process_seq_name = seq_name_list[i::cuda_num]
            if len(process_seq_name) == 0:
                continue
            process_seq_name = '+'.join(process_seq_name)
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command % process_seq_name}'
            commands.append(command)
            # print(process_seq_name)
        # execute_commands(commands, command_name, cuda_num)
        execute_commands_2(commands, command_name, cuda_num, CPU_CORE)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)

def get_label_text_feature(cuda_id, dataset, config):
    if dataset == 'scannet':
        label_text_feature_path = 'data/text_features/scannet.npy'
    if dataset == 'scannetpp':
        label_text_feature_path = 'data/text_features/scannetpp.npy'
    if dataset == 'matterport3d':
        label_text_feature_path = 'data/text_features/matterport3d.npy'
    # if os.path.exists(label_text_feature_path):
    #     return
    command = f'CUDA_VISIBLE_DEVICES={cuda_id} python -m semantics.extract_label_featrues --config {config}'
    os.system(command)

def main(args):
    dataset = args.dataset
    config = args.config
    para = args.para
    backbone = args.backbone
    cropformer_path = args.cropformer_path

    if dataset == 'scannet':
        root = 'data/scannet/processed'
        image_path_pattern = 'color/*0.jpg' # stride = 10
        # image_path_pattern = 'color/*.jpg' 
        gt = 'data/scannet/gt'
    elif dataset == 'scannetpp':
        root = 'data/scannetpp/data'
        image_path_pattern = 'iphone/rgb/*0.jpg'  # stride = 10
        # image_path_pattern = 'iphone/rgb/*.jpg'
        gt = 'data/scannetpp/gt'
    elif dataset == 'matterport3d':
        root = 'data/matterport3d/scans'
        image_path_pattern = '*/undistorted_color_images/*.jpg'
        gt = 'data/matterport3d/gt'

    open(dataset + "_frames.txt", "w", encoding="utf-8").close()

    seq_name_list = get_seq_name_list(dataset)

    tmp = [
    '2t7WUuJeko7',  # 666
    'gYvKGZ5eRqb',  # 684
    'YVUC4YcDtcY',  # 828
    'RPmz2sHmrrY',  # 1062
    'gxdoqLR6rwA',  # 3042
    'ARNzJeq3xxb',  # 1764
    'YFuZgdQ5vWj',  # 1602
    'WYY7iVyf5p8',  # 1404
    ]
    if dataset == 'matterport3d':
        seq_name_list = tmp

    print('There are %d scenes' % len(seq_name_list))

    t0 = time.time()

    # Step 1: use Cropformer to get 2D instance masks for all sequences.
    parallel_compute(f'python third_party/detectron2/projects/CropFormer/demo_cropformer/Cropformer.py --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --seq_name_list %s --opts MODEL.WEIGHTS {cropformer_path}', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)
    # parallel_compute(f'python third_party/sam2/SAM2.py --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --seq_name_list %s', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)
    
    t1 = time.time()
    print('Total 2D segmentation time', (t1 - t0)//60, 'min')

    # # Step 2: Mask clustering using our proposed method.
    parallel_compute(f'python main.py --config {config} --para {para} --seq_name_list %s', 'Stream3D process', 'cuda', CUDA_LIST, seq_name_list)
    print('Total 3D segmentation time', (time.time() - t0)//60, 'min')
    all_frames = []
    with open(dataset + "_frames.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    num = int(line) 
                    all_frames.append(num)
                except ValueError:
                    print(f"skip: {line}")
    if sum(all_frames) != 0:
        print('Average time', (time.time() - t0) / sum(all_frames), 'sec. / frame', len(seq_name_list), sum(all_frames))

    # Step 3: Evaluate the class-agnostic results.
    os.system(f'python -m evaluation.evaluate --pred_path data/prediction/{config}_class_agnostic --gt_path {gt} --dataset {dataset} --no_class')

    # Step 4: Get the open-vocabulary semantic features for each 2D masks.
    parallel_compute(f'python -m semantics.get_open-voc_features --config {config}  --seq_name_list %s', 'get open-vocabulary semantic features using CLIP', 'cuda', CUDA_LIST, seq_name_list)
    
    # Step 5: Get the text CLIP features for each label.
    get_label_text_feature(CUDA_LIST[0], args.dataset, config)

    # Step 6: Get labels for each 3D instances.
    parallel_compute(f'python -m semantics.open-voc_query --config {config}', 'get text labels', 'cpu', CUDA_LIST, seq_name_list)
    
    # Step 7: Evaluate the class-aware results.
    os.system(f'python -m evaluation.evaluate --pred_path data/prediction/{config} --gt_path {gt} --dataset {dataset}')

if __name__ == '__main__':
    args = get_args()
    for para in [1]:
        args.para = para
        # print("Parameter: ", args.para)
        main(args)

    # os.system(f'python -m visualize.vis_scene --config matterport3d --seq_name ???????')
    