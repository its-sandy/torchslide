# # import tensorflow as tf
# import tensorflow.compat.v1 as tf
# from tensorflow.python.client import timeline

# tf.disable_eager_execution()
# x = tf.random_normal([128, 128])
# y = tf.random_normal([128, 524288])
# res = tf.matmul(x, y)

# # tf.config.threading.set_inter_op_parallelism_threads(48)
# # Run the graph with full trace option
# with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=48)) as sess:
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     sess.run(res, options=run_options, run_metadata=run_metadata)

#     # Create the Timeline object, and write it to a json
#     tl = timeline.Timeline(run_metadata.step_stats)
#     ctf = tl.generate_chrome_trace_format()
#     with open('timeline.json', 'w') as f:
#         f.write(ctf)

import tensorflow as tf
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.threading.set_inter_op_parallelism_threads(48)
tf.config.threading.set_intra_op_parallelism_threads(48)
num_iter = 100
total_time = 0
for iter in range(1, num_iter+1):
    x = tf.random.normal([128, 128])
    y = tf.random.normal([128, 524288])
    t1 = time.time()
    res = tf.matmul(x, y)
    print(res[0][0])
    t2 = time.time()
    if iter>num_iter/2:
        # warm start
        total_time += t2-t1
print((total_time/(num_iter/2))*1000)