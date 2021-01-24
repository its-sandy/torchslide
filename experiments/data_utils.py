from itertools import islice
import numpy as np
import torch

torch_to_numpy_dtype_dict = {
    torch.bool       : np.bool,
    torch.uint8      : np.uint8,
    torch.int8       : np.int8,
    torch.int16      : np.int16,
    torch.int32      : np.int32,
    torch.int64      : np.int64,
    torch.float16    : np.float16,
    torch.float32    : np.float32,
    torch.float64    : np.float64,
    torch.complex64  : np.complex64,
    torch.complex128 : np.complex128,
}

def data_generator_train_tf(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline()  # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = []
                    vals = []
                    ##
                    y_inds = []
                    y_batch = np.zeros([batch_size,n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_inds = [int(itm) for itm in itms[0].split(',')]
                        for i in range(len(y_inds)):
                            y_batch[count,y_inds[i]] = 1.0/len(y_inds)
                            # y_batch[count,y_inds[i]] = 1.0
                        ##
                        inds += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)

def data_generator_test_tf(files, batch_size):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline() # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = []
                    vals = []
                    ##
                    y_batch = [None for i in range(len(lines))]
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_batch[count] = [int(itm) for itm in itms[0].split(',')]
                        ##
                        inds += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)

###############################################################################
def data_generator_train_pth(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline()  # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = [[], []]
                    vals = []
                    ##
                    y_inds = []
                    y_batch = np.zeros([batch_size,n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_inds = [int(itm) for itm in itms[0].split(',')]
                        for i in range(len(y_inds)):
                            y_batch[count,y_inds[i]] = 1.0/len(y_inds)
                            # y_batch[count,y_inds[i]] = 1.0
                        ##
                        inds[0] += [count]*len(itms[1:])
                        inds[1] += [int(itm.split(':')[0]) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)

def data_generator_test_pth(files, batch_size):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline() # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = [[], []]
                    vals = []
                    ##
                    y_batch = [None for i in range(len(lines))]
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_batch[count] = [int(itm) for itm in itms[0].split(',')]
                        ##
                        inds[0] += [count]*len(itms[1:])
                        inds[1] += [int(itm.split(':')[0]) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)

###############################################################################
def data_generator_tslide(files, batch_size):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline()  # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = []
                    vals = []
                    y_inds = []
                    ##
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_inds.append([int(itm) for itm in itms[0].split(',')])
                        inds.append([int(itm.split(':')[0]) for itm in itms[1:]])
                        vals.append([float(itm.split(':')[1]) for itm in itms[1:]])
                    lines = []
                    yield (inds, vals, y_inds)

def get_padded_tensor(lili, dtype, padded_length=None, pad_zero=True):
    max_length = max(len(li) for li in lili)
    if padded_length is None:
        padded_length = max_length
    elif padded_length < max_length:
        padded_length = max_length
        # raise ValueError('padded_length (%d) < max_length (%d)' % (padded_length, max_length))
    
    dtype = torch_to_numpy_dtype_dict[dtype]
    if pad_zero:
        ret = np.zeros((len(lili), padded_length), dtype=dtype)
    else:
        ret = np.empty((len(lili), padded_length), dtype=dtype)
    for i, li in enumerate(lili):
        ret[i,:len(li)] = li

    return torch.from_numpy(ret)

def get_y_probs_from_y_inds(y_inds, n_classes, dtype=torch.float32):
    y_probs = torch.zeros([len(y_inds),n_classes], dtype=dtype)
    for sample, labels in enumerate(y_inds):
        for ind in labels:
            y_probs[sample,ind] = 1.0/len(labels)
            # y_probs[sample,ind] = 1.0
        ##
    return y_probs

###############################################################################
def data_generator_ss(files, batch_size, n_classes, max_label):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline() # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    inds = []
                    vals = []
                    ##
                    y_batch = [None for i in range(len(lines))]
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_batch[count] = [int(itm) for itm in itms[0].split(',')]
                        if max_label>=len(y_batch[count]): # 
                            y_batch[count] += [n_classes for i in range(max_label-len(y_batch[count]))]
                        else:
                            y_batch[count] = np.random.choice(y_batch[count], max_label, replace=False)
                        ##
                        inds += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (inds, vals, y_batch)
