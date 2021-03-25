import numpy as np

import torch

def evaluate(query_out, base_out, eval_distance, k=100):
    """
    Evaluate recall by threshold searching (Algorithm 2 in the paper)
    """
    average_recall = 0
    for i, query in enumerate(query_out):
        candidate_set = []
        result_set = []

        distance = torch.norm(query-base_out, dim=1)
        candidate_set.append(distance)

        gt_distance = eval_distance[i]
        result_set.append(gt_distance)

        candidate_set = torch.cat(candidate_set)
        result_set = torch.cat(result_set)

        candidate_set = candidate_set.cpu().numpy()
        result_set = result_set.cpu().numpy()

        recall = calculate_recall(candidate_set, result_set, k)
        average_recall += recall
    average_recall /= len(query_out)
    return average_recall


def calculate_recall(candidate_set, result_set, k):
    candidate_argsort = candidate_set.argsort()[:k]
    result_argsort = result_set.argsort()[:k]

    tp = len(candidate_argsort[np.in1d(candidate_argsort, result_argsort)])
    fn = len(candidate_argsort[np.logical_not(np.in1d(candidate_argsort, result_argsort))])

    recall = tp / (tp+fn)
    return recall


