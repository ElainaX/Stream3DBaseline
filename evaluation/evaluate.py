import os, sys, argparse
from copy import deepcopy
import numpy as np
import torch
from evaluation.utils_3d import get_instances

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', required=True, help='path to directory of predicted .txt files')
parser.add_argument('--gt_path', required=True, help='path to directory of ground truth .txt files')
# parser.add_argument('--seq_name_list', default=[], help='')
parser.add_argument('--dataset', required=True, help='type of dataset, e.g. matterport3d, scannet, etc.')
parser.add_argument('--output_file', default='', help='path to output file')
parser.add_argument('--no_class', action='store_true', help='class agnostic evaluation')
opt = parser.parse_args()

# ---------- Label info ---------- #
from evaluation.constants import MATTERPORT_LABELS, MATTERPORT_IDS, SCANNET_LABELS, SCANNET_IDS, SCANNETPP_LABELS, SCANNETPP_IDS

if opt.dataset == 'matterport3d':
    CLASS_LABELS = MATTERPORT_LABELS
    VALID_CLASS_IDS = MATTERPORT_IDS
elif opt.dataset == 'scannet':
    CLASS_LABELS = SCANNET_LABELS
    VALID_CLASS_IDS = SCANNET_IDS
elif opt.dataset == 'scannetpp':
    CLASS_LABELS = SCANNETPP_LABELS
    VALID_CLASS_IDS = SCANNETPP_IDS


if opt.output_file == '':
    opt.output_file = os.path.join(f'data/evaluation/{opt.dataset}', opt.pred_path.split('/')[-1] + '.txt')
    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
if opt.no_class:
    if 'class_agnostic' not in opt.output_file:
        opt.output_file = opt.output_file.replace('.txt', '_class_agnostic.txt')

ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt.overlaps             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
opt.min_region_sizes     = np.array( [ 100 ] )
# distance thresholds [m]
opt.distance_threshes    = np.array( [  float('inf') ] )
# distance confidences
opt.distance_confs       = np.array( [ -float('inf') ] )


def evaluate_matches(matches):
    overlaps = opt.overlaps
    min_region_sizes = [ opt.min_region_sizes[0] ]
    dist_threshes = [ opt.distance_threshes[0] ]
    dist_confs = [ opt.distance_confs[0] ]
    
    # results: class x overlap
    ap = np.zeros( (len(dist_threshes) , len(CLASS_LABELS) , len(overlaps)) , float )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=bool )
                    # collect matches
                    for (gti,gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true,0)
                                    cur_score = np.append(cur_score,min_score)
                                    cur_match = np.append(cur_match,True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                                

                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true,0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score,confidence)
                    # append to overall results
                    y_true  = np.append(y_true,cur_true)
                    y_score = np.append(y_score,cur_score)
                
                # compute average precision
                if has_gt and has_pred:
                    if len(y_score) == 0:
                        ap_current = 0.0
                    else:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort      = np.argsort(y_score)
                        y_score_sorted      = y_score[score_arg_sort]
                        y_true_sorted       = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples      = len(y_score_sorted)
                        num_true_examples = y_true_sorted_cumsum[-1]
                        precision         = np.zeros(num_prec_recall)
                        recall            = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                        # deal with remaining
                        for idx_res,idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores-1]
                            tp = num_true_examples - cumsum
                            fp = num_examples      - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p  = float(tp)/(tp+fp)
                            r  = float(tp)/(tp+fn)
                            precision[idx_res] = p
                            recall   [idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall   [-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                
                ap[di,li,oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(opt.overlaps,0.5))
    o25   = np.where(np.isclose(opt.overlaps,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(opt.overlaps,0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict

def read_pridiction_npz(path, idx):
    pred_info = {}
    # print(path)
    pred = np.load(path)

    num_instance = len(pred['pred_score'])
    # mask = torch.from_numpy(pred['pred_masks']).cuda()
    mask = pred['pred_masks']
    # print(mask.shape)
    mask = mask[idx]
    # print(mask.shape)
    for i in range(num_instance):
        
        pred_info[path.split('/')[-1] + '_' +str(i)] = { # unique id of instance in all scenes
            # 'mask': mask[:, i].cpu().numpy(),
            'mask': mask[:, i],
            'label_id': pred['pred_classes'][i],
            'conf': pred['pred_score'][i]
        }
    return pred_info

def get_gt_tensor(gt_ids, gt_instances):
    '''
        return a dict of gt_tensor
    '''
    gt_tensor_dict = {}
    point_num = len(gt_ids)
    for label in gt_instances:
        gt_instance_num = len(gt_instances[label])
        gt_tensor = torch.zeros((point_num, gt_instance_num), dtype=torch.bool).cuda()
        for i, gt_instance_info in enumerate(gt_instances[label]):
            gt_tensor[:, i] = torch.from_numpy(gt_ids == gt_instance_info['instance_id'])
        gt_tensor_dict[label] = gt_tensor
    return gt_tensor_dict

def get_gt_tensor_optimized(gt_ids, gt_instances):
    '''Memory-optimized version'''
    gt_tensor_dict = {}
    point_num = len(gt_ids)

    for label in gt_instances:
        instance_masks = []
        for gt_instance_info in gt_instances[label]:
            mask_np = gt_ids == gt_instance_info['instance_id']
            mask_tensor = torch.from_numpy(mask_np).to(torch.bool).cuda(non_blocking=True)
            instance_masks.append(mask_tensor.unsqueeze(1))  
        if instance_masks:
            gt_tensor = torch.cat(instance_masks, dim=1)  
        else:
            gt_tensor = torch.empty((point_num, 0), dtype=torch.bool).cuda()
        gt_tensor_dict[label] = gt_tensor
        torch.cuda.empty_cache()  # ⚠️ Explicitly release the allocated but unused video memory

    return gt_tensor_dict

def get_gt_tensor_multi_gpu(gt_ids, gt_instances):
    '''
        Multi-GPU version: returns a dict of gt_tensor distributed across available GPUs
    '''
    gt_tensor_dict = {}
    point_num = len(gt_ids)

    available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if not available_devices:
        raise RuntimeError("No CUDA devices available.")

    labels = list(gt_instances.keys())
    num_devices = len(available_devices)

    # 将 labels 均匀分配到多个 GPU  Distribute the labels evenly across multiple GPUs
    label_chunks = [labels[i::num_devices] for i in range(num_devices)]

    # 分 GPU 并行构建 gt_tensor  Build gt_tensor using GPU parallel processing
    for device, label_group in zip(available_devices, label_chunks):
        for label in label_group:
            gt_instance_num = len(gt_instances[label])
            gt_tensor = torch.zeros((point_num, gt_instance_num), dtype=torch.bool, device=device)
            for i, gt_instance_info in enumerate(gt_instances[label]):
                mask_np = gt_ids == gt_instance_info['instance_id']
                mask_tensor = torch.from_numpy(mask_np).to(device)
                gt_tensor[:, i] = mask_tensor
            
            gt_tensor_dict[label] = gt_tensor

    return gt_tensor_dict

def assign_instances_for_scan(pred_file, gt_file):
    '''
        if intersection > 0, then the prediction is considered a match
    '''

    title = '......../TMP/'
    import re
    path = gt_file
    # print(gt_file)
    if 'scannet/' in path:
        config = 'scannet'
        match = re.search(r"scene\d{4}_\d{2}", path)
        scene_id = match.group()
    if 'scannetpp/' in path:
        config = 'scannetpp'
        match = re.search(r"gt/([^/]+)\.txt$", path)
        scene_id = match.group(1)
    if 'matterport3d/' in path:
        config = 'matterport3d'
        match = re.search(r"gt/([^/]+)\.txt$", path)
        scene_id = match.group(1)
    
    loaded_array = np.load(title + config + '/' + scene_id + '_pre_points.npy')


    pred_info = read_pridiction_npz(os.path.join(pred_file), loaded_array)

    gt_ids = np.loadtxt(gt_file)

    gt_ids = gt_ids[loaded_array]

    if opt.no_class:
        gt_ids = gt_ids % 1000 + VALID_CLASS_IDS[0] * 1000

    # get gt instances
    gt_instances = get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids//1000, VALID_CLASS_IDS))

    gt_tensor_dict = get_gt_tensor(gt_ids, gt_instances)
    # gt_tensor_dict = get_gt_tensor_multi_gpu(gt_ids, gt_instances)
    # gt_tensor_dict = get_gt_tensor_optimized(gt_ids, gt_instances)

    # go thru all prediction masks
    i = 1
    for pred_mask_file in (pred_info):
        # print('\r', i, "/", len(pred_info), end='               ')
        i += 1
        if opt.no_class:
            label_id = VALID_CLASS_IDS[0]
        else:
            label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[pred_mask_file]['mask']
        # print(len(pred_mask))

        if len(pred_mask) != len(gt_ids):
            print('wrong number of lines in ' + pred_mask_file + '(%d) vs #mesh vertices (%d), please double check and/or re-download the mesh' % (len(pred_mask), len(gt_ids)))
            raise NotImplementedError

        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt.min_region_sizes[0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        gt_tensor = gt_tensor_dict[label_name]

        def compute_intersection_async(gt_tensor, pred_mask_np):
            """
            多 GPU 异步流式并行计算交集  Multi-GPU Asynchronous Streaming Parallel Computing Intersection
            """
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
            if len(devices) < 2:
                # 如果只有一张 GPU，则使用串行版本  If there is only one GPU, then use the serial version.
                return compute_intersection_fallback(gt_tensor, pred_mask_np)

            N, K = gt_tensor.shape
            chunk_size = (K + len(devices) - 1) // len(devices)

            # 预处理预测掩码，复制到各 GPU  Preprocess the prediction masks and copy them to each GPU
            pred_mask_cpu = torch.from_numpy(pred_mask_np).to(torch.bool)
            pred_mask_copies = [pred_mask_cpu.to(dev).view(-1, 1) for dev in devices]

            streams = [torch.cuda.Stream(device=dev) for dev in devices]
            results = []

            for i, device in enumerate(devices):
                start = i * chunk_size
                end = min(K, start + chunk_size)
                if start >= end:
                    continue

                with torch.cuda.stream(streams[i]):
                    gt_chunk = gt_tensor[:, start:end].to(device, non_blocking=True)
                    pred_chunk = pred_mask_copies[i].expand(-1, gt_chunk.shape[1])
                    inter = torch.sum(gt_chunk & pred_chunk, dim=0)
                    results.append(inter.to(devices[0], non_blocking=True))

            for s in streams:
                torch.cuda.synchronize(s.device)

            return torch.cat(results, dim=0)


        def compute_intersection_fallback(gt_tensor, pred_mask_np):
            """
            显存优化的逐列处理（单 GPU 使用）  Column-by-column processing with graphics memory optimization (using a single GPU)
            """
            pred_mask = torch.from_numpy(pred_mask_np).to(torch.bool, device=gt_tensor.device, non_blocking=True)
            K = gt_tensor.shape[1]
            intersection = torch.empty(K, dtype=torch.int32, device=gt_tensor.device)
            for i in range(K):
                intersection[i] = torch.sum(gt_tensor[:, i] & pred_mask)
            return intersection


        def compute_intersection_smart(gt_tensor, pred_mask_np, chunk_threshold=512):
            """
            智能选择策略：根据 GPU 数量和数据规模自动分派交集计算方式  Intelligent selection strategy: Automatically allocate intersection calculation methods based on the number of GPUs and data size.
            """
            assert gt_tensor.device.type == "cuda", "gt_tensor must be on CUDA"

            N, K = gt_tensor.shape
            num_gpus = torch.cuda.device_count()

            if num_gpus >= 2:
                return compute_intersection_async(gt_tensor, pred_mask_np)

            if K > chunk_threshold:
                return compute_intersection_fallback(gt_tensor, pred_mask_np)

            # 原始简洁版本（无显存优化，适用于小型场景）  Original version (without GPU memory optimization, suitable for small-scale scenarios)
            pred_mask_tensor = torch.from_numpy(pred_mask_np).to(dtype=torch.bool, device=gt_tensor.device, non_blocking=True)
            return torch.sum(gt_tensor & pred_mask_tensor.view(-1, 1), dim=0)


        def compute_intersection_safe(gt_tensor, pred_mask_np):
            """
            GPU 优先计算交集点数，若 CUDA OOM，则回退用 CPU 计算  The GPU prioritizes the calculation of the intersection point count. If CUDA OOM occurs, it will then switch to using the CPU for the calculation.
            """
            try:
                pred_mask = torch.from_numpy(pred_mask_np).to(torch.bool).cuda(non_blocking=True).view(-1, 1)
                # ✅ Attempt to calculate on the GPU
                return torch.sum(gt_tensor & pred_mask, dim=0)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("⚠️ The CUDA memory is insufficient, so it automatically switches to CPU processing.")

                    # 🔄 CPU fallback
                    gt_cpu = gt_tensor.cpu()
                    pred_mask_cpu = torch.from_numpy(pred_mask_np).to(torch.bool).view(-1, 1)
                    inter = torch.sum(gt_cpu & pred_mask_cpu, dim=0)
                    return inter.cuda()  # if needed, back to GPU
                else:
                    raise  

        def compute_intersection_with_auto_gpu(gt_tensor, pred_mask_np):
            """
            First, attempt to perform the calculation using the main GPU. If there is additional video memory, then implement distributed parallel processing.
            - gt_tensor: CUDA Tensor of shape [N, K]
            - pred_mask_np: numpy array of shape [N]
            - return: CUDA Tensor of shape [K], on the main GPU
            """
            try:
                # 🚀 GPU fast trace
                pred_mask_tensor = torch.from_numpy(pred_mask_np).to(dtype=torch.bool, device=gt_tensor.device, non_blocking=True)
                return torch.sum(gt_tensor & pred_mask_tensor.view(-1, 1), dim=0)

            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise e  

                # print("⚠️ Insufficient video memory, automatically switching to multi-GPU parallel mode.")

                # multi-GPU parallel mode
                deviceall = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
                devices = deviceall[1:]
                # print(devices)
                N, K = gt_tensor.shape
                chunk_size = (K + len(devices) - 1) // len(devices)

                pred_mask_cpu = torch.from_numpy(pred_mask_np).to(torch.bool)
                pred_mask_copies = [pred_mask_cpu.to(dev).view(-1, 1) for dev in devices]
                results = []

                for i, device in enumerate(devices):
                    start = i * chunk_size
                    end = min(K, start + chunk_size)
                    if start >= end:
                        continue

                    gt_chunk = gt_tensor[:, start:end].to(device, non_blocking=True)
                    pred_chunk = pred_mask_copies[i].expand(-1, gt_chunk.shape[1])
                    inter = torch.sum(gt_chunk & pred_chunk, dim=0)
                    results.append(inter.to(devices[0], non_blocking=True))  # collecting by the main GPU

                # 🔄 Synchronize and return
                for dev in devices:
                    torch.cuda.synchronize(dev)

                return torch.cat(results, dim=0)


        # intersection = torch.sum(gt_tensor & torch.from_numpy(pred_mask).cuda().reshape(-1, 1), dim=0)  # 1.0 min
        # intersection = compute_intersection(gt_tensor, pred_mask)
        # intersection = compute_intersection_async(gt_tensor, pred_mask)   # 8.0 min
        # intersection = compute_intersection_smart(gt_tensor, pred_mask)
        # intersection = compute_intersection_safe(gt_tensor, pred_mask)
        intersection = compute_intersection_with_auto_gpu(gt_tensor, pred_mask)


        intersect_ids = torch.nonzero(intersection).cpu().numpy().reshape(-1)
        for gt_id in intersect_ids:
            gt_copy = gt_instances[label_name][gt_id].copy()
            pred_copy = pred_instance.copy()
            intersection_num = intersection[gt_id].item()
            gt_copy['intersection']   = intersection_num
            pred_copy['intersection'] = intersection_num
            matched_gt.append(gt_copy)
            gt2pred[label_name][gt_id]['matched_pred'].append(pred_copy)
        
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def print_results(avgs):
    sep     = "" 
    col1    = ":"
    lineLen = 64

    print ("")
    print ("#"*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    print (line)
    print ("#"*lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_avg  = avgs["classes"][label_name]["ap"]
        if np.isnan(ap_avg):
            continue
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        print (line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    print ("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1 
    line += "{:>15.3f}".format(all_ap_avg)  + sep 
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    print (line)
    print ("")
    return all_ap_avg, all_ap_50o, all_ap_25o

def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')    
        f.write(_SPLITTER.join([str(x) for x in [avgs["all_ap"], avgs["all_ap_50%"], avgs["all_ap_25%"]]]) + '\n')    

def evaluate(pred_files, gt_files, pred_path, output_file):
    print ('evaluating', len(pred_files), 'scans...')
    matches = {}
    for i in range(len(pred_files)):
        matches_key = os.path.abspath(gt_files[i])

        gt2pred, pred2gt = assign_instances_for_scan(pred_files[i], gt_files[i])
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt

        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    all_ap_avg, all_ap_50o, all_ap_25o = print_results(avgs)
    write_result_file(avgs, output_file)
    return all_ap_avg, all_ap_50o, all_ap_25o

def main():

    def get_seq_name_list(dataset):
        if dataset == 'scannet':
            file_path = 'splits/scannet.txt'
        elif dataset == 'scannetpp':
            file_path = 'splits/scannetpp.txt'
        elif dataset == 'matterport3d':
            file_path = 'splits/matterport3d.txt'
        with open(file_path, 'r') as f:
            seq_name_list = f.readlines()
        seq_name_list = [seq_name.strip() + '.npz' for seq_name in seq_name_list]
        return seq_name_list

    print('start evaluating:', opt.pred_path.split('/')[-1])
    pred_files = [f for f in sorted(os.listdir(opt.pred_path)) if f.endswith('.npz') and not f.startswith('semantic_instance_evaluation')]

    if opt.dataset == 'matterport3d':
        pred_files = [
        '2t7WUuJeko7.npz',  
        'ARNzJeq3xxb.npz', 
        'WYY7iVyf5p8.npz', 
        'YFuZgdQ5vWj.npz', 
        'YVUC4YcDtcY.npz', 
        'gxdoqLR6rwA.npz', 
        'gYvKGZ5eRqb.npz', 
        'RPmz2sHmrrY.npz'
        ]

    gt_files = []

    for i in range(len(pred_files)):
        gt_file = os.path.join(opt.gt_path, pred_files[i].replace('.npz', '.txt'))
        if not os.path.isfile(gt_file):
            print('Result file {} does not match any gt file'.format(pred_files[i]))
            raise NotImplementedError

        gt_files.append(gt_file)
        pred_files[i] = os.path.join(opt.pred_path, pred_files[i])

        # tmp_gt_files = [gt_file]
        # tmp_pred_files = [pred_files[i]]
        # all_ap_avg, all_ap_50o, all_ap_25o = evaluate(tmp_pred_files, tmp_gt_files, opt.pred_path, opt.output_file)
        # print(pred_files[i][-16:-4], all_ap_avg, all_ap_50o, all_ap_25o)

    evaluate(pred_files, gt_files, opt.pred_path, opt.output_file)
    print('save results to', opt.output_file)

if __name__ == '__main__':
    main()
