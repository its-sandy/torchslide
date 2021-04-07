
configs = {
    'delicious200k' : {
        'data_path_train' : 'data/delicious200k/shuffled_deliciousLarge_train.txt',
        'data_path_test' : 'data/delicious200k/shuffled_deliciousLarge_test.txt',
        'feature_dim' : 782585,
        'n_classes' : 205443,
        'n_train' : 196606,
        'n_test' : 100095,
        
        'hidden_dim' : 128,
        'batch_size' : 128,
        'lr' : 0.0001,
        # 'lr' : 0.00025,
        # 'GPUs' : '0', # empty string uses only CPU
        'GPUs' : '', # empty string uses only CPU
        'num_threads' : 48, # Only used when GPUs is empty string

        'val_freq' : 50,
        # 'val_freq' : 100,
        'num_val_batches' : 50, # -1 for full test
        # 'num_val_batches' : -1, # -1 for full test
        'n_epochs' : 7,
        'log_file' : 'log_delicious200k',
        'model_save_file_prefix' : 'delicious200k',

        # for TorchSLIDE
        'last_K' : 9,
        'last_L' : 50,
        'n_label_samples' : 0,
        # 'n_label_samples' : 2048,
        # 'n_label_samples' : 1024,
        # 'n_label_samples' : -1, # for dense
        'rehash_freq' : 50,
        'repermute_freq' : 1000,

        # for sampled softmax
        'n_samples' : 205443//10,
        # choose the max_labels per training sample. 
        # If the number of true labels is < max_label,
        # we will pad the rest of them with a dummy class (see data_generator_ss in util.py)
        'max_label' : 1,
    },

    'amazon670k' : {
        'data_path_train' : 'data/amazon670k/Amazon/shuffled_amazon_train.txt',
        'data_path_test' : 'data/amazon670k/Amazon/shuffled_amazon_test.txt',
        'feature_dim' : 135909,
        'n_classes' : 670091,
        'n_train' : 490449,
        'n_test' : 153025,
        
        'hidden_dim' : 128,
        'batch_size' : 128,
        'n_epochs' : 15,
        'lr' : 0.0001,
        # 'lr' : 0.00025,
        'GPUs' : '', # empty string uses only CPU
        'num_threads' : 48, # Only used when GPUs is empty string

        'val_freq' : 100,
        'num_val_batches' : 50, # -1 for full test
        'log_file' : 'log_amazon670k',
        'model_save_file_prefix' : 'amazon670k',

        # for TorchSLIDE
        'hash_fn' : 'srp',
        'last_K' : 14,
        'last_L' : 50,
        # 'n_label_samples' : 0,
        'n_label_samples' : 4096,
        # 'n_label_samples' : 1024,
        # 'n_label_samples' : -1, # for dense
        'rehash_freq' : 50,
        'repermute_freq' : 4000,
        'bucket_size' : 128,
        'fill_mode' : 'reservoir_sampling',
        'sample_mode' : 'vanilla',
        'perm_size' : 4,

        # for sampled softmax
        'n_samples' : 670091//10,
        # choose the max_labels per training sample. 
        # If the number of true labels is < max_label,
        # we will pad the rest of them with a dummy class (see data_generator_ss in util.py)
        'max_label' : 1,
    },
}