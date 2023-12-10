import numpy
import gudhi as gd
from pylab import *
import torch
import math
from ripser import ripser
import cripser as cr
import os
from gudhi.wasserstein import wasserstein_distance

def getCriticalPoints_cr(likelihood, threshold):
        
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")
    pd_arr_lh = pd[pd[:, 0] == 0] # 0-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    valid_idx = np.where(pd_pers > threshold)[0]
    noisy_idx = np.where(pd_pers <= threshold)[0]

    pd_lh_filtered = pd_lh[valid_idx]
    bcp_lh_filtered = bcp_lh[valid_idx]
    dcp_lh_filtered = dcp_lh[valid_idx]

    #return pd_lh_filtered, bcp_lh_filtered, dcp_lh_filtered, pairs_lh_pa
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx


def get_matchings(lh_stu, lh_tea):
    
    cost, matchings = wasserstein_distance(lh_stu, lh_tea, matching=True)

    #print(f"Wasserstein distance value = {cost:.2f}")
    dgm1_to_diagonal = matchings[matchings[:,1] == -1, 0]
    dgm2_to_diagonal = matchings[matchings[:,0] == -1, 1]
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

    return dgm1_to_diagonal, off_diagonal_match


def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    """
    Compute the persistent diagram of the image

    Args:
        stu_lh_dgm: likelihood persistent diagram of student model.
        tea_lh_dgm: likelihood persistent diagram of teacher model.

    Returns:
        idx_holes_to_remove: The index of student persistent points that require to remove for the following training process
        off_diagonal_match: The index pairs of persistent points that requires to fix in the following training process
    
    """
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    
    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        idx_holes_to_remove, off_diagonal_match = get_matchings(stu_lh_dgm, tea_lh_dgm)
    
    return idx_holes_to_remove, off_diagonal_match


def getTopoLoss(stu_tensor, tea_tensor, topo_size=100, loss_mode="mse"):

    stu_pd_threshold = 0.7
    tea_pd_threshold = 0.7

    if stu_tensor.ndim != 2:
        print("incorrct dimension")
    
    likelihood = stu_tensor.clone()
    gt = tea_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):
            #print("Loop itr (x,y) = {},{}".format(x,y))
            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue
            
            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx_lh, noisy_idx_lh = getCriticalPoints_cr(lh_patch, threshold=stu_pd_threshold)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_gt, noisy_idx_gt = getCriticalPoints_cr(gt_patch, threshold=tea_pd_threshold)

            # select pd with high threshold to match
            pd_lh_for_matching = pd_lh[valid_idx_lh]
            pd_gt_for_matching = pd_gt[valid_idx_gt]

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh_for_matching, pd_gt_for_matching)

            idx_holes_to_remove = []
            off_diagonal_match = []

            if (len(idx_holes_to_remove_for_matching) > 0):
                for i in idx_holes_to_remove_for_matching:
                    index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
                    idx_holes_to_remove.append(index_pd_lh_removed)
            
            for k in noisy_idx_lh:
                idx_holes_to_remove.append(k)
            
            if len(off_diagonal_for_matching) > 0:
                for idx, (i, j) in enumerate(off_diagonal_for_matching):
                    index_pd_lh = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
                    index_pd_gt = np.where(np.all(pd_gt == pd_gt_for_matching[j], axis=1))[0][0]
                    off_diagonal_match.append((index_pd_lh, index_pd_gt))

            if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
                for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = pd_gt[j][0]
                    
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = pd_gt[j][1]
                
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((stu_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    return loss_topo

