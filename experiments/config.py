
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
        # 'n_epochs' : 2,
        'n_epochs' : 5,
        'lr' : 0.0001,
        # 'lr' : 0.00025,
        'GPUs' : '0', # empty string uses only CPU
        'num_threads' : 48, # Only used when GPUs is empty string

        'log_file' : 'log_delicious200k',

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
        'n_epochs' : 25,
        'lr' : 0.0001,
        # 'lr' : 0.00025,
        'GPUs' : '0', # empty string uses only CPU
        'num_threads' : 48, # Only used when GPUs is empty string

        'log_file' : 'log_amazon670k',

        # for sampled softmax
        'n_samples' : 670091//10,
        # choose the max_labels per training sample. 
        # If the number of true labels is < max_label,
        # we will pad the rest of them with a dummy class (see data_generator_ss in util.py)
        'max_label' : 1,
    },
}