from __future__ import print_function
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

# bruteforce search
def searchsorted(sorted_array, values_to_search):
    # example of execution:
    #  sorted_array = [10, 20, 30]
    #  values_to_search = [12, 17, 27, 1, 34]
    #
    #  >>>  search_for, search_in = tf.meshgrid(values_to_search, sorted_array)
    #  search_for = [[12, 17, 27, 1, 34],
    #                [12, 17, 27, 1, 34],
    #                [12, 17, 27, 1, 34]]
    #  search_in =  [[10, 10, 10, 10, 10],
    #                [20, 20, 20, 20, 20],
    #                [30, 30, 30, 30, 30]]
    #
    #  >>> mask = tf.to_int32(tf.greater(search_for, search_in))
    #  mask =       [[1, 1, 1, 0, 1],
    #                [0, 0, 1, 0, 1],
    #                [0, 0, 0, 0, 1]]
    #
    #  >>> indices = tf.reduce_sum(mask, axis=0)
    #  indices =     [1, 1, 2, 0, 3]

    # use broadcasting to achieve all permutations between [sorted_array] x [values_to_search]
    # result is tensor with dimensions: [values_to_search.shape[0], sorted_array.shape[0]]

    # analogous, but faster than:
    # search_for = tf.tile(tf.expand_dims(values_to_search, axis=0), [tf.shape(sorted_array)[0], 1])
    # search_in = tf.tile(tf.expand_dims(sorted_array, axis=1), [1, tf.shape(values_to_search)[0]])

    # mask out values that are greater than the value we are looking for
    mask = tf.greater_equal(values_to_search, tf.expand_dims(sorted_array, axis=1))

    # convert mask to _float_ instead of int to speed up reduce_sum()
    mask = tf.to_float(mask)

    # sum up all the 1s in the mask to calculate array index
    indices = tf.reduce_sum(mask, axis=0)
    return indices

def searchsorted2(sorted_array, values_to_search):
    mask = tf.less(tf.expand_dims(sorted_array, axis=1), values_to_search)
    mask = tf.to_float(mask)
    indices = tf.reduce_sum(mask, axis=0)
    return indices

# takes more memory than searchsortedN
def searchsortedT(sorted_array, values_to_search):
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    
    indices = array_ops.zeros_like(values_to_search, dtype=tf.int32)        
    n = array_ops.shape(sorted_array)[0]
    
    def take_branch(n, indices):
        n = n / 2
        idxL = indices 
        idxR = indices + n
        pred = tf.less(values_to_search, tf.gather(sorted_array, idxR))
        indices = tf.where(pred, idxL, idxR)
        return [n, indices]

    _, indices = control_flow_ops.while_loop(
        lambda n, indices: n >= 1, take_branch, [n, indices])

    pred = tf.less(values_to_search, sorted_array[0])
    indices = tf.where(pred, indices, indices + 1)
    return indices

def searchsortedN(sorted_array, values_to_search, n):
    from tensorflow.python.ops import array_ops
    
    indices = array_ops.zeros_like(values_to_search, dtype=tf.int32)
    
    while n > 1:
        n = n / 2

        idxL = indices
        idxR = indices + n

        pred = tf.less(values_to_search, tf.gather(sorted_array, idxR))
        indices = tf.where(pred, idxL, idxR)

    pred = tf.less(values_to_search, sorted_array[0])
    indices = tf.where(pred, indices, indices + 1)
    return indices

def searchsortedNF(sorted_array, values_to_search, n):
    from tensorflow.python.ops import array_ops
    
    #indices = array_ops.zeros_like(values_to_search, dtype=tf.int32)
    indices = array_ops.zeros_like(values_to_search, dtype=tf.float32)
    
    while n > 1:
        n = n / 2

        idxL = indices
        idxR = indices + tf.to_float(n)

        pred = tf.less(values_to_search, tf.gather(sorted_array, tf.to_int32(idxR)))
        indices = tf.where(pred, idxL, idxR)

    pred = tf.less(values_to_search, sorted_array[0])
    indices = tf.where(pred, indices, indices + 1)
    return indices


# based on  _call_linear(self, x_new) from https://github.com/scipy/scipy/blob/v0.19.0/scipy/interpolate/interpolate.py
def interp_linear(x_new, x, y, nbins):
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops

    # Find where in the orignal data, the values to interpolate
    # would be inserted.
    # Note: If x_new[n] == x[m], then m is returned by searchsorted.

    x_new_indices = searchsortedNF(x, x_new, nbins)
    #x_new_indices = searchsortedT(x, x_new)
    #x_new_indices = searchsorted(x, x_new)
    #x_new_indices = tf.py_func(np.searchsorted, [x, x_new], tf.int64, stateful=False)
    #x_new_indices = xx_new_indices

    '''values_to_search = x_new
    sorted_array = x
    search_for = tf.tile(tf.expand_dims(values_to_search, axis=0), [tf.shape(sorted_array)[0], 1])
    search_in = tf.tile(tf.expand_dims(sorted_array, axis=1), [1, tf.shape(values_to_search)[0]])
    # mask out values that are greater than the value we are looking for
    mask = tf.to_float(tf.to_int32(tf.less(search_in, search_for)))
    # sum up all 1s in the mask to calculate array index
    indices = tf.reduce_sum(mask, axis=0)
    x_new_indices = indices
    '''

    lo = x_new_indices - 1
    hi = x_new_indices

    # Clip indices so that they are within the range
    hi = math_ops.cast(
        clip_ops.clip_by_value(hi, 0, nbins-1), dtypes.int32)
    lo = math_ops.cast(
        clip_ops.clip_by_value(lo, 0, nbins-1), dtypes.int32)

    x_lo = tf.gather(x, lo) #x_lo = x[lo]
    x_hi = tf.gather(x, hi) #x_hi = x[hi]
    y_lo = tf.gather(y, lo) #y_lo = y[lo]
    y_hi = tf.gather(y, hi) #y_hi = y[hi]

	# Calculate the slope of regions that each x_new value falls in.
    dx = (x_hi - x_lo)
    slope = (y_hi - y_lo) / dx

  	# Calculate the actual value for each entry in x_new.
    y_linear = slope*(x_new - x_lo) + y_lo
    y_nearest = y_lo

    # Protect against NaN (div-by-zero)
    p = tf.not_equal(dx, 0.0)
    y_new = tf.where(p, y_linear, y_nearest)

    return y_new

TRACE=True

n = 1024
steps = 10240000
xs = np.linspace(0.0, n, num=n)
#xs = np.concatenate(([0.0, 0.0, 0.0], xs))
ys = np.sin(xs)
new_xs = np.linspace(-1.0, n+1.0, num=steps)

t0 = time.time()
new_ys = np.interp(new_xs, xs, ys)
#res = np.mean(new_ys)
t1 = time.time()
#print(res)
print('Numpy run in %fs' % (t1-t0))

idx = np.searchsorted(xs, new_xs, side='left').astype(np.int32)
print(idx)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.placeholder(tf.float32)
    #d = tf.placeholder(tf.int32)
    #a = tf.constant(new_xs) # using constant here makes sess.run() roughly x10 slower! :(
    #b = tf.constant(xs)
    #c = tf.constant(ys)
    #d = tf.constant(idx)

    if TRACE:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    for i in range(3):
        t0 = time.time()
        if i > 0 and TRACE:
            res = sess.run(interp_linear(a,b,c,len(xs)), run_metadata=run_metadata, options=run_options, feed_dict={a: new_xs, b: xs, c: ys})
        else:
            res = sess.run(interp_linear(a,b,c,len(xs)), feed_dict={a: new_xs, b: xs, c: ys})
        t1 = time.time()
        print('Tensorflow run in %fs' % (t1-t0))

    '''print(len(xs))
    print(xs)
    print(new_xs)
    print(new_ys)
    '''
    print(res)
    
    print("bins: %d, steps: %d" % (n, steps))
    print("total error: %f" % np.sum(np.square(new_ys - res)))
    print("max error: %f" % np.max(np.abs(new_ys - res)))

    if TRACE:
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)