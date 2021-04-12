import glob
import os
import sys
from collections import defaultdict

HOME_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HOME_DIR)
from config import configs
from data_utils import data_generator_tslide, get_padded_tensor, get_y_probs_from_y_inds


def get_batch_max_label_counts(config_dict):
    #
    print('-----------------------------', flush=True)
    n_train = config_dict['n_train']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']
    train_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_train']))
    test_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_test']))
    train_data_generator = data_generator_tslide(train_files, batch_size)
    test_data_generator = data_generator_tslide(test_files, batch_size)
    train_steps = n_train//batch_size
    test_steps = n_test//batch_size

    print('train_files =', train_files, flush=True)
    print('test_files =', test_files, flush=True)
    print('batch_size =', batch_size, flush=True)
    print('train_steps =', train_steps, flush=True)
    print('test_steps =', test_steps, flush=True)
    print(flush=True)
    print('train batch max true label lengths', flush=True)
    train_max = 0
    for i in range(train_steps):
        x_inds, x_vals, y_inds = next(train_data_generator)
        # print(max(len(y_ind) for y_ind in y_inds), flush=True)
        train_max = max(train_max, max(len(y_ind) for y_ind in y_inds))
    print(flush=True)
    test_max = 0
    print('test batch max true label lengths', flush=True)
    for i in range(test_steps):
        x_inds, x_vals, y_inds = next(test_data_generator)
        # print(max(len(y_ind) for y_ind in y_inds), flush=True)
        test_max = max(test_max, max(len(y_ind) for y_ind in y_inds))
    print('train_max =', train_max)
    print('test_max =', test_max)

def get_label_count_frequencies(config_dict):
    print('-----------------------------', flush=True)
    n_train = config_dict['n_train']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']
    train_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_train']))
    test_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_test']))
    train_data_generator = data_generator_tslide(train_files, batch_size)
    test_data_generator = data_generator_tslide(test_files, batch_size)
    train_steps = n_train//batch_size
    test_steps = n_test//batch_size

    print('train_files =', train_files, flush=True)
    print('test_files =', test_files, flush=True)
    print('batch_size =', batch_size, flush=True)
    print('train_steps =', train_steps, flush=True)
    print('test_steps =', test_steps, flush=True)
    print(flush=True)

    counts = defaultdict(lambda: 0)
    for i in range(train_steps):
        x_inds, x_vals, y_inds = next(train_data_generator)
        for y_ind in y_inds:
            counts[len(y_ind)] += 1
    print('train label length frequencies', flush=True)
    for k,v in sorted(counts.items()):
        print(k, v, flush=True)
    print(flush=True)

    counts = defaultdict(lambda: 0)
    for i in range(test_steps):
        x_inds, x_vals, y_inds = next(test_data_generator)
        for y_ind in y_inds:
            counts[len(y_ind)] += 1
    print('test label length frequencies', flush=True)
    for k,v in sorted(counts.items()):
        print(k, v, flush=True)
    print(flush=True)

def get_feature_count_frequencies(config_dict):
    print('-----------------------------', flush=True)
    n_train = config_dict['n_train']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']
    train_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_train']))
    test_files = glob.glob(os.path.join(HOME_DIR, config_dict['data_path_test']))
    train_data_generator = data_generator_tslide(train_files, batch_size)
    test_data_generator = data_generator_tslide(test_files, batch_size)
    train_steps = n_train//batch_size
    test_steps = n_test//batch_size

    print('train_files =', train_files, flush=True)
    print('test_files =', test_files, flush=True)
    print('batch_size =', batch_size, flush=True)
    print('train_steps =', train_steps, flush=True)
    print('test_steps =', test_steps, flush=True)
    print(flush=True)

    counts = defaultdict(lambda: 0)
    for i in range(train_steps):
        x_inds, x_vals, y_inds = next(train_data_generator)
        for x_ind in x_inds:
            counts[len(x_ind)] += 1
    print('train feature length frequencies', flush=True)
    for k,v in sorted(counts.items()):
        print(k, v, flush=True)
    print(flush=True)

    counts = defaultdict(lambda: 0)
    for i in range(test_steps):
        x_inds, x_vals, y_inds = next(test_data_generator)
        for x_ind in x_inds:
            counts[len(x_ind)] += 1
    print('test feature length frequencies', flush=True)
    for k,v in sorted(counts.items()):
        print(k, v, flush=True)
    print(flush=True)


if __name__ == '__main__':
    # get_batch_max_label_counts(configs['amazon670k'])
    # get_label_count_frequencies(configs['amazon670k'])
    # get_batch_max_label_counts(configs['delicious200k'])
    # get_label_count_frequencies(configs['delicious200k'])
    # get_batch_max_label_counts(configs['amazon3m'])
    # get_label_count_frequencies(configs['amazon3m'])
    get_feature_count_frequencies(configs['amazon670k'])
