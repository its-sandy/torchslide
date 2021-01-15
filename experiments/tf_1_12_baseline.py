import numpy as np
import time
import math
import os
import glob
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import configs
from itertools import islice
from data_utils import data_generator, data_generator_tst

## Training Params
def main(config_dict):
    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    n_train = config_dict['n_train']
    n_test = config_dict['n_test']
    n_epochs = config_dict['n_epochs']
    batch_size = config_dict['batch_size']
    lr = config_dict['lr']
    #
    if config_dict['GPUs'] == '':
        num_threads = config_dict['num_threads']
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config_dict['GPUs']
    #
    train_files = glob.glob(config_dict['data_path_train'])
    test_files = glob.glob(config_dict['data_path_test'])
    #
    x_idxs = tf.placeholder(tf.int64, shape=[None,2])
    x_vals = tf.placeholder(tf.float32, shape=[None])
    x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])
    y = tf.placeholder(tf.float32, shape=[None,n_classes])
    #
    W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)))
    b1 = tf.Variable(tf.truncated_normal([hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)))
    layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal([hidden_dim,n_classes], stddev=2.0/math.sqrt(hidden_dim+n_classes)))
    b2 = tf.Variable(tf.truncated_normal([n_classes], stddev=2.0/math.sqrt(n_classes+hidden_dim)))
    logits = tf.matmul(layer_1,W2)+b2
    #
    k=1
    if k==1:
        top_idxs = tf.argmax(logits, axis=1)
    else:
        top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]
    #
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    #
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
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
    training_data_generator = data_generator(train_files, batch_size, n_classes)
    steps_per_epoch = n_train//batch_size
    n_steps = n_epochs*steps_per_epoch
    n_check = 50
    #
    begin_time = time.time()
    total_time = 0
    #
    with open(config_dict['log_file'], 'a') as out:
        print('--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('train_files =', train_files, file=out, flush=True)
        print('test_files =', test_files, file=out, flush=True)
        print(device_lib.list_local_devices(), file=out, flush=True)
        print(file=out, flush=True)
        for i in range(n_steps):
            # train
            idxs_batch, vals_batch, labels_batch = next(training_data_generator)
            sess.run(train_step, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})

            # validate
            if i%steps_per_epoch==steps_per_epoch-1 or i%n_check==0:
                total_time+=time.time()-begin_time

                if i%steps_per_epoch==steps_per_epoch-1:
                    n_steps_val = n_test//batch_size  # precision on entire test data
                else:
                    n_steps_val = 20 # precision on first x batches

                test_data_generator = data_generator_tst(test_files, batch_size)
                p_at_k = 0
                for l in range(n_steps_val):
                    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                    top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
                    p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
                #
                print('step=',i,
                      'time=',total_time,
                      'num_val_batches',n_steps_val,
                      'p_at_1=',p_at_k/n_steps_val,
                      file=out, flush=True)
                #
                begin_time = time.time()


if __name__ == "__main__":
    # execute only if run as a script
    main(configs['delicious200k'])