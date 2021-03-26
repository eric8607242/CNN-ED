import numpy as np

import torch

def evaluate(query_out, base_out, eval_distance, k=10):
    """
    Evaluate recall by threshold searching (Algorithm 2 in the paper)
    """
    average_recall = 0
    for i, query in enumerate(query_out):
        distance = torch.norm(query-base_out, dim=1)
        gt_distance = eval_distance[i]

        candidate_set = distance.cpu().numpy()
        result_set = gt_distance.cpu().numpy()

        recall = calculate_recall(candidate_set, result_set, k)
        average_recall += recall
    average_recall /= len(query_out)
    return average_recall


def calculate_recall(candidate_set, result_set, k):
    candidate_argsort = candidate_set.argsort()[:k]
    result_argsort = result_set.argsort()[:k]

    tp = len(candidate_argsort[np.in1d(candidate_argsort, result_argsort)])
    #fn = len(candidate_argsort[np.logical_not(np.in1d(candidate_argsort, result_argsort))])

    recall = tp / k
    return recall


