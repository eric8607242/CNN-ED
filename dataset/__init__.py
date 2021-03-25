from .dataset import Dataset, SplitDataset, EvalDataset

def get_dataset(path_to_dataset, training_set_num, query_set_num, neighbor_num=100):
    """
    Load data and process the data into training, query, and base set.
    Calulate the distance and the nearest martrix for the training data and query data.
    """
    split_dataset = SplitDataset(path_to_dataset, training_set_num, query_set_num)

    train_data, train_nearest_info, train_distance_info = split_dataset.get_train_data()
    query_data, base_data, eval_nearest_info, eval_distance_info = split_dataset.get_eval_data()

    alphabet_table = split_dataset.get_alphabet_table()
    alphabet_len = len(alphabet_table)
    max_str_len = split_dataset.get_max_str_len()

    train_dataset = Dataset(train_data, train_data, train_nearest_info, train_distance_info, alphabet_table, max_str_len, neighbor_num)
    #val_dataset = Dataset(query_data, base_data, eval_nearest_info, eval_distance_info, alphabet_table, max_str_len, neighbor_num)
    query_dataset = EvalDataset(query_data, alphabet_table, max_str_len, eval_distance_info)
    base_dataset = EvalDataset(base_data, alphabet_table, max_str_len)


    #return train_dataset, val_dataset, alphabet_len, max_str_len
    return train_dataset, query_dataset, base_dataset, alphabet_len, max_str_len
