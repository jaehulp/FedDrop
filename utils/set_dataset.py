import numpy as np
import ujson
import os


def allocate_data(xs, ys, dataidx_map, num_clients):

    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = xs[idxs]
        Y[client] = ys[idxs]

        for i in np.unique(Y[client]):
            statistic[client].append((int(i), int(sum(Y[client]==i))))
                
    return X, Y, statistic


def separate_iid(xs, ys, num_classes, num_clients):

    dataidx_map = {}

    idxs = np.array(range(len(ys)))
    idx_for_each_class = []

    for i in range(num_classes):
        idx_for_each_class.append(idxs[ys == i])


    class_num_per_client = [num_classes for _ in range(num_clients)]
    for i in range(num_classes):

        np.random.shuffle(idx_for_each_class[i])
           
        num_per_client = int(len(idx_for_each_class[i]) / num_clients)

        idx = 0

        for client in range(num_clients):
            if client not in dataidx_map.keys():
                dataidx_map[client] = idx_for_each_class[i][idx:idx+num_per_client]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_per_client], axis=0)
            idx += num_per_client
    
    return dataidx_map


def separate_noniid(xs, ys, num_classes, num_clients):

    dataidx_map = {}

    idxs = np.array(range(len(ys)))
    idx_for_each_class = []

    for i in range(num_classes):
        idx_for_each_class.append(idxs[ys == i])
        np.random.shuffle(idx_for_each_class[i])

    if num_classes > num_clients:
        class_assignment = np.arange(num_classes) % num_clients
        np.random.shuffle(class_assignment)

        for class_id, client_id in enumerate(class_assignment):
            if client_id not in dataidx_map.keys():
                dataidx_map[client_id] = idx_for_each_class[class_id]
            else:
                dataidx_map[client_id] = np.append(dataidx_map[client_id], idx_for_each_class[class_id], axis=0)
        
    else:
        client_assignment = np.arange(num_clients) % num_classes
        np.random.shuffle(client_assignment)
        num_samples = len(ys)//num_clients

        for client_id, class_id in enumerate(client_assignment):
            if client_id not in dataidx_map.keys():
                dataidx_map[client_id] = idx_for_each_class[class_id][num_samples*(client_id//num_classes):num_samples*(client_id//num_classes+1)]
            else:
                dataidx_map[client_id] = np.append(dataidx_map[client_id], idx_for_each_class[class_id][num_samples*(client_id//num_classes):num_samples*(client_id//num_classes+1)], axis=0)

    return dataidx_map


def separate_dirichlet(xs, ys, num_classes, num_clients):

    dataidx_map = {}

    K = num_classes
    N = len(ys)

    idx_batch = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(ys == i)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
        proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
        proportions = proportions/proportions.sum()
        proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for i in range(num_clients):
        dataidx_map[i] = idx_batch[i]
    return dataidx_map

def separate_shard(xs, ys, num_classes, num_clients):
    dataidx_map = {}
    shard=200
    idxs = np.array(range(len(ys)))
    idx_for_each_class = []

    for i in range(num_classes):
        idx_for_each_class.append(idxs[ys == i])
        np.random.shuffle(idx_for_each_class[i])

    data_shard = list(range(int(len(ys)/shard)))
    data_shard = np.array(data_shard)

    num_shard_per_client = int(len(data_shard) / num_clients)
    
    for i in range(num_clients):
        shard_idxs = np.random.choice(data_shard, num_shard_per_client, replace=False).tolist()
        data_shard = [x for x in data_shard if x not in shard_idxs]
        client_data = []
        for j in shard_idxs:

            shard_class = j % num_classes
            shard_p = j // num_classes
            data = idx_for_each_class[shard_class][(shard_p*shard):((shard_p+1)*shard)].tolist()
            client_data = client_data + data
        dataidx_map[i] = client_data
    
    # Check if assignment has overlap

    check_idx = []
    for i in range(num_clients):
        check_idx += dataidx_map[i]
    check_idx = np.unique(check_idx)
    assert len(check_idx) == len(ys)
    
    return dataidx_map

def separate_data(xs, ys, num_classes, num_clients, dist):

    if dist == 'iid':
        dataidx_map = separate_iid(xs, ys, num_classes, num_clients)
    elif dist == 'non_iid':
        dataidx_map = separate_noniid(xs, ys, num_classes, num_clients)
    elif dist == 'dirichlet':
        dataidx_map = separate_dirichlet(xs, ys, num_classes, num_clients)
    elif dist == 'shard':
        dataidx_map = separate_shard(xs, ys, num_classes, num_clients)
    else:
        raise NotImplementedError('Check ./set_dataset.py separate_data func')


    X, Y, statistic = allocate_data(xs, ys, dataidx_map, num_clients)

    train_data = []

    for i in range(len(Y)):
        
        train_data.append({'x': X[i], 'y': Y[i]})
    
    del X, Y

    return train_data, statistic


def save_dataset(dir_path, train_data, statistic, num_classes, num_clients):

    train_path = os.path.join(dir_path, 'train')

    for idx, train_dict in enumerate(train_data):
        with open(train_path + '_' + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'Size of samples for labels in clients': statistic, 
    }

    config_path = os.path.join(dir_path, 'config.json')

    with open(config_path, 'w') as f:
        ujson.dump(config, f)
    
    return
    

    