import numpy as np
import time
import math
import os
import glob
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import configs
from data_utils import data_generator_train_tf, data_generator_test_tf

## Training Params
def main(config_dict):
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    float_dtype = tf.float32

    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dict['GPUs']
    if config_dict['GPUs'] == '':
        num_threads = config_dict['num_threads']
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #
    test_files = glob.glob(config_dict['data_path_test'])
    #
    x_idxs = tf.placeholder(tf.int64, shape=[None,2])
    x_vals = tf.placeholder(dtype = float_dtype, shape=[None])
    x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])
    y = tf.placeholder(dtype = float_dtype, shape=[None,n_classes])
    #
    W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)), dtype = float_dtype)
    b1 = tf.Variable(tf.truncated_normal([hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)), dtype = float_dtype)
    layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal([hidden_dim,n_classes], stddev=2.0/math.sqrt(hidden_dim+n_classes)), dtype = float_dtype)
    b2 = tf.Variable(tf.truncated_normal([n_classes], stddev=2.0/math.sqrt(n_classes+hidden_dim)), dtype = float_dtype)
    logits = tf.matmul(layer_1,W2)+b2
    #
    k=1
    if k==1:
        top_idxs = tf.argmax(logits, axis=1)
    else:
        top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]
    #
    #
    if config_dict['GPUs'] == '':
        Config = tf.ConfigProto(inter_op_parallelism_threads=num_threads, intra_op_parallelism_threads=num_threads)
    else:
        Config = tf.ConfigProto()
        Config.gpu_options.allow_growth = True
    #
    sess = tf.Session(config=Config)
    sess.run(tf.global_variables_initializer())
    
    #
    with open(config_dict['log_file'], 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('test_files =', test_files, file=out, flush=True)
        print('float_dtype =', float_dtype, file=out, flush=True)
        print('random seed =', seed, file=out, flush=True) 
        print(device_lib.list_local_devices(), file=out, flush=True)
        print(file=out, flush=True)
        
        begin_time = time.time()
        num_val_batches = n_test//batch_size  # precision on entire test data
        test_data_generator = data_generator_test_tf(test_files, batch_size)
        p_at_k = 0
        for l in range(num_val_batches):
            if l % 100 == 0:
                print(l)
            idxs_batch, vals_batch, labels_batch = next(test_data_generator)
            top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
            p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
        #
        print('inference_time=',time.time() - begin_time,
                'num_val_batches',num_val_batches,
                'p_at_1=',p_at_k/num_val_batches,
                file=out, flush=True)
        #

if __name__ == "__main__":
    # execute only if run as a script
    main(configs['amazon670k'])