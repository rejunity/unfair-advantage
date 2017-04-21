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

    # mask out values that are greater than the value we are looking for
    mask = tf.greater_equal(values_to_search, tf.expand_dims(sorted_array, axis=1))

    # convert mask to _float_ instead of int to speed up reduce_sum()
    mask = tf.to_float(mask)

    # sum up all the 1s in the mask to calculate array index
    indices = tf.reduce_sum(mask, axis=0)
    return indices


''' prev impls
def searchsorted(sorted_array, values_to_search):
    search_for = tf.tile(tf.expand_dims(values_to_search, axis=0), [tf.shape(sorted_array)[0], 1])
    search_in = tf.tile(tf.expand_dims(sorted_array, axis=1), [1, tf.shape(values_to_search)[0]])
    mask = tf.to_int32(tf.less(search_in, search_for))
    mask = tf.to_float(mask)
    indices = tf.reduce_sum(mask, axis=0)
    return indices

def searchsorted(sorted_array, values_to_search):
    search_for, search_in = tf.meshgrid(values_to_search, sorted_array)
    mask = tf.to_int32(tf.less(search_in, search_for))
    indices = tf.cumsum(mask)
    shape = tf.shape(indices)
    indices = tf.squeeze(tf.slice(indices, [shape[0]-1,0], [1, shape[1]]))
    return indices - 1

def searchsorted(sorted_array, values_to_search):
    from tensorflow.python.ops import array_ops

    zeros = array_ops.zeros_like(values_to_search, dtype=tf.int32)
    ones = array_ops.ones_like(values_to_search)

    fn = lambda a, x: tf.where(tf.less(ones * x, values_to_search), a + 1, a)
    indices = tf.foldl(fn, sorted_array, initializer=zeros, parallel_iterations=1, back_prop=False)

    return indices - 1

def searchsorted(sorted_array, values_to_search):
    from tensorflow.python.ops import array_ops

    zeros_i32 = array_ops.zeros_like(values_to_search, dtype=tf.int32)
    ones_i32 = array_ops.zeros_like(values_to_search, dtype=tf.int32)
    ones = array_ops.ones_like(values_to_search)

    #fn = lambda a, x: a + tf.where(tf.less(ones * x, values_to_search), ones_i32, zeros_i32)
    fn = lambda a, x: a + tf.where(tf.greater(values_to_search, x), zeros_i32, ones_i32)
    indices = tf.foldl(fn, sorted_array, initializer=zeros_i32, back_prop=False)

    return indices
'''

# takes more memory than searchsortedN
def searchsortedT(sorted_array, values_to_search):
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    
    indices = array_ops.zeros_like(values_to_search, dtype=tf.float32)        
    n = array_ops.shape(sorted_array)[0]
    
    def take_branch(n, indices):
        n = n / 2
        idxL = indices 
        idxR = indices + tf.to_float(n)
        pred = tf.less(values_to_search, tf.gather(sorted_array, tf.to_int32(idxR)))
        indices = tf.where(pred, idxL, idxR)
        return [n, indices]

    _, indices = control_flow_ops.while_loop(
        lambda n, indices: n >= 1, take_branch, [n, indices])

    pred = tf.less(values_to_search, sorted_array[0])
    indices = tf.where(pred, indices, indices + 1)
    return indices

def searchsortedN(sorted_array, values_to_search, n):
    from tensorflow.python.ops import array_ops
    
    indices = array_ops.zeros_like(values_to_search, dtype=tf.float32)
    n = int(n)
    
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

    #x_new_indices = searchsortedN(x, x_new, nbins)
    #x_new_indices = searchsortedT(x, x_new)
    x_new_indices = searchsorted(x, x_new)
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


# bins: 1024, steps: 1024
# np            - 0.000018s
# searchsorted  - 0.000496s *
# searchsortedN - 0.000980s *
# searchsortedT - 0.001045s *
# * <- bottlenecked by feeding tasks to GPU (according to timeline graph)

# bins: 1024, steps: 10240
# np            - 0.000058s
# searchsorted  - 0.001481s !
# searchsortedN - 0.001066s *
# searchsortedT - 0.001087s *
# * <- bottlenecked by feeding tasks to GPU
# ! <- somewhat bottlenecked by feeding tasks to GPU

# bins: 1024, steps: 102400
# np            - 0.000484s
# data xfer     -             0.082ms+0.062ms = 0.144ms
# searchsorted  - 0.012689s
# searchsortedN - 0.001066s !
# ! <- somewhat bottlenecked by feeding tasks to GPU

# bins: 1024, steps: 1024000
# np            - 0.004236s
# data xfer     -             0.643ms+0.617ms = 1.260ms
# searchsorted  - 0.128820s
# searchsortedN - 0.006240s | 5.024ms
# searchsortedT - 0.006765s | 5.142ms


# bins: 16, steps: 10240
# np            - 0.000047s
# searchsorted  - 0.000570s !
# ! <- somewhat bottlenecked by feeding tasks to GPU

# bins: 16, steps: 1024000
# np            - 0.004024s
# data xfer     -             0.643ms+0.617ms = 1.260ms (~2*15.6MB => 23GB/s)
# searchsorted  - 0.005968s | 4.199ms
# searchsortedN - 0.005213s | 3.784ms
# searchsortedT - 0.005530s | 3.818ms


TRACE=False  

n = 256
steps = 256
xs = np.linspace(0.0, n, num=n)
#xs = np.concatenate(([0.0, 0.0, 0.0], xs))
ys = np.sin(xs)
new_xs = np.linspace(-1.0, n+1.0, num=steps)

t0 = time.time()
new_ys = np.interp(new_xs, xs, ys)
t1 = time.time()
print('Numpy run in %fs' % (t1-t0))

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.placeholder(tf.float32)
    #a = tf.constant(new_xs) # using constant here makes sess.run() run slower!
    #b = tf.constant(xs)
    #c = tf.constant(ys)

    if TRACE:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    for i in range(5):
        t0 = time.time()
        runnable = interp_linear(a,b,c,len(xs))
        feed = {a: new_xs, b: xs, c: ys}
        t1 = time.time()
        print('Runnable prepared in %fs' % (t1-t0))

        if i == 1 and TRACE:
            t0 = time.time()
            res = sess.run(runnable, run_metadata=run_metadata, options=run_options, feed_dict=feed)
            t1 = time.time()
            print('Trace run in %fs' % (t1-t0))
        else:
            t0 = time.time()            
            h = sess.partial_run_setup(runnable, [a,b,c])

            t1 = time.time()
            res = sess.partial_run(h, runnable, feed_dict=feed)
            t2 = time.time()
            print('Tensorflow prepared in %fs & run in %fs' % (t1-t0, t2-t1))

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