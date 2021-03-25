import numpy as np

def evaluate(query_out, base_out, eval_distance, K=10):
    """
    Evaluate recall by threshold searching (Algorithm 2 in the paper)
    """
    average_recall = 0
    for i, query in enumerate(query_out):
        candidate_set = []
        result_set = []
        for j, base in enumerate(base_out):
            distance = torch.norm(query - base, dim=1)
            candidate_set.append(distance.item())

            gt_distance = eval_distance[i, j]
            result_set.append(distance.item())

        recall = calculate_recall(candidate_set, result_set, k)
        average_recall += recall
    average_recall /= len(query_out)
    return average_recall


def calculate_recall(candidate_set, result_set, k):
    candidate_set = np.array(candidate_set)
    result_set = np.array(result_set)

    candidate_argsort = candidate_set.argsort()[:k]
    result_argsort = result_set.argsort()[:k]

    tp = candidate_set[np.in1d(candidate_argsort, result_argsort)]
    fn = candidate_set[np.logical_not(np.in1d(candidate_argsort, result_argsort))]

    recall = tp / (tp+fn)

    return recall


