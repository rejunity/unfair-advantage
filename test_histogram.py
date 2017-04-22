from __future__ import print_function
import time

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave
from tensorflow.python.client import timeline


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

def interp_linear(x_new, x, y, nbins):
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops

    x_new_indices = searchsortedN(x, x_new, nbins)

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

def histogram_fixed_width(values, value_range, nbins=100):
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import array_ops

    nbins_float = math_ops.to_float(nbins)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(values - value_range[0],
                                     value_range[1] - value_range[0],
                                     name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    #counts = tf.Variable(...) <= array_ops.zeros_like(indices, dtype=dtypes.int32))
    #return tf.scatter_add(counts, indices, array_ops.ones_like(indices, dtype=dtypes.int32)), indices

    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtypes.float32),
        indices,
        nbins), indices

def feature_histogram_matching(source, template, value_range, nbins):

    s_value_range = [tf.reduce_min(source), tf.reduce_max(source)]
    t_value_range = [tf.reduce_min(template), tf.reduce_max(template)]
    #s_value_range = value_range
    #t_value_range = value_range
    t_values = tf.linspace(t_value_range[0], t_value_range[1], nbins)

    s_counts, indices = histogram_fixed_width(source, s_value_range, nbins)
    t_counts, _ = histogram_fixed_width(template, t_value_range, nbins)

    s_cdf = tf.to_float(s_counts)
    s_cdf = tf.cumsum(s_cdf)
    s_cdf /= s_cdf[-1]

    t_cdf = tf.to_float(t_counts)
    t_cdf = tf.cumsum(t_cdf)
    t_cdf /= t_cdf[-1]

    interp_t_values = interp_linear(s_cdf, t_cdf, t_values, nbins)
    interp_t_values = tf.maximum(interp_t_values, 0.0)
    values = tf.gather(interp_t_values, indices)
    return values

def histogram_matching(source, template, value_range, nbins):
    def wrap_feature_histogram_matching(x, template):
        return feature_histogram_matching(x, template, value_range, nbins)

    def map(fn, arrays, dtype=tf.float32):
        # assumes all arrays have same leading dim
        indices = tf.range(tf.shape(arrays[0])[0])
        out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype, back_prop=False)
        return out

    return map(wrap_feature_histogram_matching, [source, template])


def np_feature_histogram_matching(source, template, value_range, nbins):
    source = source.flatten()
    template = template.flatten()

    #source = np.clip(source, value_range[0], value_range[1], source)
    #template = np.clip(template, value_range[0], value_range[1], template)

    s_values, indices, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_cdf = np.cumsum(s_counts).astype(np.float32)
    s_cdf /= s_cdf[-1]

    t_cdf = np.cumsum(t_counts).astype(np.float32)
    t_cdf /= t_cdf[-1]

    interp_t_values = np.interp(s_cdf, t_cdf, t_values).astype(np.float32)
    interp_t_values = np.maximum(interp_t_values, 0.0)
    return interp_t_values[indices]

def np_feature_histogram_matching2(source, template, value_range, nbins):
    source = source.flatten()
    template = template.flatten()

    s_value_range = [np.min(source), np.max(source)]
    t_value_range = [np.min(template), np.max(template)]

    #source = np.clip(source, value_range[0], value_range[1], source)
    #template = np.clip(template, value_range[0], value_range[1], template)
    s_values = np.linspace(s_value_range[0], s_value_range[1], nbins)
    t_values = np.linspace(t_value_range[0], t_value_range[1], nbins)

    indices = np.digitize(source, s_values[0:-1])
    s_counts = np.bincount(indices, minlength=nbins)
    t_counts = np.bincount(np.digitize(template, t_values[0:-1]), minlength=nbins)

    s_cdf = np.cumsum(s_counts).astype(np.float32)
    s_cdf /= s_cdf[-1]

    t_cdf = np.cumsum(t_counts).astype(np.float32)
    t_cdf /= t_cdf[-1]

    print(s_cdf, t_cdf, t_values)
    interp_t_values = np.interp(s_cdf, t_cdf, t_values).astype(np.float32)
    interp_t_values = np.maximum(interp_t_values, 0.0)
    return interp_t_values[indices]

def report_feature_histogram(source, template, value_range, nbins):
    source = source.flatten()
    template = template.flatten()

    s_value_range = [np.min(source), np.max(source)]
    t_value_range = [np.min(template), np.max(template)]

    #source = np.clip(source, value_range[0], value_range[1], source)
    #template = np.clip(template, value_range[0], value_range[1], template)
    s_values = np.linspace(s_value_range[0], s_value_range[1], nbins)
    t_values = np.linspace(t_value_range[0], t_value_range[1], nbins)

    indices = np.digitize(source, s_values[0:-1])
    s_counts = np.bincount(indices, minlength=nbins)
    t_counts = np.bincount(np.digitize(template, t_values[0:-1]), minlength=nbins)

    #print(s_counts)
    #print(t_counts)
    print(s_counts - t_counts)
    return source

def np_histogram_matching(fn, source, template, value_range, nbins):
    o = []
    for s, t in zip(source, template):
        o.append(fn(s, t, value_range, nbins))

    return np.reshape(np.concatenate(o), np.shape(source))

def preproc_image(x):
    x = x.swapaxes(0,2) # cyx
    x = x.swapaxes(1,2) # cxy
    x = np.reshape(x, [np.shape(x)[0], x.size/np.shape(x)[0]]) # cflat
    print(np.shape(x))
    return x

def deproc_image(x):
    x = x.swapaxes(0,1)
    return x

TRACE=True


img = imread("../dev/Wall_512.jpg", mode='RGB')
shape = np.shape(img)
template = imread("../dev/StoneWall_512.png", mode='RGB')
#template = imread("../dev/Wall_512.jpg", mode='RGB')

img = preproc_image(img)
template = preproc_image(template)

value_range = [0.0, 256.0]
n = 4096

t0 = time.time()
new_img = np_histogram_matching(np_feature_histogram_matching, img, template, value_range, n)
print(np.shape(new_img))
new_img2 = np_histogram_matching(np_feature_histogram_matching2, img, template, value_range, n)
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
        runnable = histogram_matching(a,b,c,n)
        feed = {a: img, b: template, c: value_range}
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

    a = new_img.flatten()
    #b = new_img2.flatten()
    b = res.flatten()

    print("bins: %d" % (n))
    print("total error: %f" % np.sum(np.square(a - b)))
    print("max error: %f" % np.max(np.abs(a - b)))

    imsave("test_histogram_np.png", np.reshape(deproc_image(new_img), shape))
    imsave("test_histogram_np2.png", np.reshape(deproc_image(new_img2), shape))
    imsave("test_histogram_tf.png", np.reshape(deproc_image(res), shape))

    if TRACE:
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline2.json', 'w') as f:
            f.write(ctf)

        #new_img = np_histogram_matching(report_feature_histogram, new_img, res, value_range, n)
