import random, json, sys
from collections import OrderedDict


def split(dataset):
    targets = json.load(open(f'{dataset}/targets.txt'), object_pairs_hook=OrderedDict)

    target_count = 0
    target_dict = {}
    for target in targets:
        if targets[target] not in target_dict:
            target_dict[targets[target]] = [target_count]
        else:
            target_dict[targets[target]].append(target_count)
        target_count += 1
    
    target_list = []
    target_list_bak = []
    for target in target_dict:
        if len(target_dict[target]) > 1:
            target_list_bak.extend(target_dict[target])
        else:
            target_list.append(target_dict[target][0])

    random.seed(1)
    random.shuffle(target_list)
    
    k_folds = 6
    CV_size = int(len(target_list) / k_folds)
    print(target_count, len(target_list), len(target_list_bak), CV_size)
    test_fold = target_list[:CV_size]
    print("Num of test fold:", len(test_fold))
    train_folds = []
    for i in range(1, k_folds - 1):
        train_folds.append(target_list[i * CV_size: (i + 1) * CV_size])
        print(f"Num of train fold {i}:", len(train_folds[-1]))
    train_folds.append(target_list[(i + 1) * CV_size:])
    print(f"Num of train fold {i + 1}:", len(train_folds[-1]))
    json.dump(train_folds, open(f'{dataset}/S3_train_set.txt', "w"))
    json.dump(test_fold, open(f'{dataset}/S3_test_set.txt', "w"))
    json.dump(target_list_bak, open(f'{dataset}/S3_mask_set.txt', "w"))


if __name__ == '__main__':
    dataset = sys.argv[1]
    split(dataset)
