'''Neural generator with Keras

Script Usage:
    # Arguments:
    ```
    --style-image:          image to learn style from
    --target-image-prefix:  path prefix for generated target images
    ```

    # Example:
    ```
    python neural_gen.py --style-image Wall_512.jpg --iterations 10 --phases 4 \
        --target-image-prefix generated/Wall
    ```

See:
[Gatys, 2015: A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1505.07376)
[Li 2016: Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](https://arxiv.org/abs/1601.04589)
[Risser, 2017: Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses](https://arxiv.org/abs/1701.08893)

Implementation details:
* https://github.com/fchollet/keras/blob/master/examples/neural_doodle.py
* https://github.com/alexjc/neural-doodle
* https://github.com/dsanno/chainer-neural-style
* https://github.com/awentzonline/image-analogies
[Picking an optimizer for Style Transfer: L-BFGS vs SGD](https://medium.com/slavv/picking-an-optimizer-for-style-transfer-86e7b8cba84b)
[Discussion on parameter tuning for Keras style transfer example](https://github.com/fchollet/keras/issues/3705)
'''
from __future__ import print_function
import time
import argparse
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imread, imsave, imresize

from keras import backend as K
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import custom_vgg19

# Command line arguments
parser = argparse.ArgumentParser(description='Keras neural doodle example')
add_arg = parser.add_argument
add_arg('--style-image',            type=str,                       help='Path to image to learn style from')
add_arg('--style-layers',           default='1,2,3,4,5', type=str,  help='VGG layers to learn style from')
add_arg('--prime-image',            default=None, type=str,         help='Path to image to prime optimization')
add_arg('--seed-range',             default='16:240', type=str,     help='Prime with random colors chosen in range, e.g. 0:255.')
add_arg('--target-image-prefix',    type=str,                       help='Path prefix for generated results')
add_arg('--target-size',            default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--phases',                 default=3, type=int,            help='Number of steps in texture pyramid.')
add_arg('--iterations',             default=50, type=int,           help='Number of iterations to run each phase/resolution.')
add_arg('--variation',              default=50, type=float,         help='Weight of total variational loss')
add_arg('--loss',                   default='gram', type=str,       help='Loss function to use for optimization')
add_arg('--loss-arg',               default=1.0, type=float,        help='Arbitrary argument to Loss function')
args = parser.parse_args()

num_iterations = args.iterations
num_colors = 3  # RGB
# determine image sizes based on style_image
target_size = imread(args.style_image).shape[:2]
if args.target_size:
    target_size = tuple([int(i) for i in args.target_size.split('x')])
img_nrows, img_ncols = target_size

total_variation_weight = args.variation

# To get better generation qualities, use more conv layers for style features
style_feature_layers = []
if args.style_layers is None:
    style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
else:
    for layer in args.style_layers.split(','):
        layer_indices = [int(i) for i in layer.split('_')]
        layer_indices.append(1)
        style_feature_layers.append('block%d_conv%d' % (layer_indices[0], layer_indices[1]))

# index constants for input images to VGG net 
STYLE, OUTPUT = 0, 1

# helper functions for reading/processing images
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def preprocess_image_partially(x):
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    if K.image_data_format() == 'channels_first':
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x

def deprocess_image_partially(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939 # 128.0
    x[:, :, 1] += 116.779 # 128.0
    x[:, :, 2] += 123.68 # 128.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# MRF patch functions
def patch(x, ksize=3, stride=1):
    assert K.ndim(x) == 3

    w = K.eye(ch * ksize * ksize, dtype=np.float32)
    if K.image_data_format() == 'channels_first':
        ch = K.shape(x)[0]
        w = w.reshape((ch * ksize * ksize, ch, ksize, ksize))
    else:
        ch = K.shape(x)[-1]
        w = w.reshape((ch * ksize * ksize, ksize, ksize, ch))
    
    return K.conv2d(x, kernel=w, strides=stride)

def make_patches(x, patch_size, patch_stride):
    '''Break image `x` up into a bunch of patches.'''

    # TODO: remove dependency on Theano
    import theano as T
    from theano.tensor.nnet.neighbours import images2neibs
    if K.image_data_format() == 'channels_first':
        ch = K.shape(x)[0]
        x = K.expand_dims(x, 0)
    else:
        ch = K.shape(x)[-1]
        x = K.permute_dimensions(x, (2, 0, 1))
        x = K.expand_dims(x, 0)

    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (ch, K.shape(patches)[0] // ch, patch_size, patch_size))
    if K.image_data_format() == 'channels_first':
        patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    else:
        patches = K.permute_dimensions(patches, (1, 2, 3, 0))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(1,2,3), keepdims=True))
    return patches, patches_norm

'''
def make_patches_grid(x, patch_size, patch_stride):
    #Break image `x` up into a grid of patches.
    #input shape: (channels, rows, cols)
    #output shape: (rows, cols, channels, patch_rows, patch_cols)

    from theano.tensor.nnet.neighbours import images2neibs  # TODO: all K, no T
    x = K.expand_dims(x, 0)
    xs = K.shape(x)
    num_rows = 1 + (xs[-2] - patch_size) // patch_stride
    num_cols = 1 + (xs[-1] - patch_size) // patch_stride
    num_channels = xs[-3]
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (num_channels, K.shape(patches)[0] // num_channels, patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = K.reshape(patches, (num_rows, num_cols, num_channels, patch_size, patch_size))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(2,3,4), keepdims=True))
    return patches, patches_norm
'''
def find_patch_matches(a, a_norm, b):
    '''For each patch in A, find the best matching patch in B'''
    convs = None
    if K.backend() == 'theano':
        # HACK: This was not being performed on the GPU for some reason.
        from theano.sandbox.cuda import dnn
        if dnn.dnn_available():
            convs = dnn.dnn_conv(
                img=a, kerns=b[:, :, ::-1, ::-1], border_mode='valid')
    if convs is None:
        if K.image_data_format() == 'channels_first':
            convs = K.conv2d(a, b[:, :, ::-1, ::-1])
        else:
            convs = K.conv2d(a, b[:, ::-1, ::-1, :])

    argmax = K.argmax(convs / a_norm, axis=1)
    return argmax

def mrf_loss(source, combination, patch_size=3, patch_stride=1):
    '''CNNMRF http://arxiv.org/pdf/1601.04589v1.pdf'''
    # extract patches from feature maps
    assert 3 == K.ndim(source) == K.ndim(combination)

    combination_patches, combination_patches_norm = make_patches(combination, patch_size, patch_stride)
    source_patches, source_patches_norm = make_patches(source, patch_size, patch_stride)
    # find best patches and calculate loss
    patch_ids = find_patch_matches(combination_patches, combination_patches_norm, source_patches / source_patches_norm)
    best_source_patches = K.reshape(source_patches[patch_ids], K.shape(combination_patches))
    loss = K.sum(K.square(best_source_patches - combination_patches)) / patch_size ** 2
    return loss

# Define loss functions
def gram_matrix(x, s=0):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x) + s
    gram = K.dot(features, K.transpose(features))
    return gram

def gram_loss(style_image, output_image):
    '''Calculate style loss between style_image and output_image
    '''
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]
    s = gram_matrix(style) / K.cast(num_channels, K.floatx())
    c = gram_matrix(output) / K.cast(num_channels, K.floatx())
    return K.mean(K.square(s - c))

def gram_offset_loss(style_image, output_image, offset):
    '''Calculate style loss between style_image and output_image
    '''
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]
    s = gram_matrix(style, offset) / K.cast(num_channels, K.floatx())
    c = gram_matrix(output, offset) / K.cast(num_channels, K.floatx())
    return K.mean(K.square(s - c))

def gram_sqrt_loss(style_image, output_image):
    return K.sqrt(gram_loss(style_image, output_image))

def L2_loss(style_image, output_image):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    s = K.batch_flatten(style_image)
    c = K.batch_flatten(output_image)
    return K.sum(K.square(s - c))

# TODO: moments and histogram losses, compare them

def var(x, axis=None, keepdims=False):
    # compute the axis-wise mean
    mean_input = K.mean(x, axis=axis, keepdims=True)

    # center the x
    centered_input = x - mean_input

    # return the mean sqr
    two = K.constant(2, dtype=centered_input.dtype)
    v = K.mean((centered_input ** two), axis=axis, keepdims=keepdims)
    return v

def moment(x, n, axis=None, keepdims=False):
    # compute the axis-wise mean
    m0 = K.mean(x, axis=axis, keepdims=True)

    # center the x
    c = x - m0

    # return the mean pow^n
    exp = K.constant(n, dtype=c.dtype)
    mn = K.mean((c ** exp), axis=axis, keepdims=keepdims)
    return mn

def moment_norm(x, n, axis=None, keepdims=False):
    v = K.var(x, axis=axis, keepdims=keepdims)
    v = K.maximum(K.abs(v), K.epsilon())
    norm = K.pow(v, n / 2.0)
    return moment(x, n, axis=axis, keepdims=keepdims) / norm

def stat_loss(style_image, output_image, n):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]

    #return K.sum(K.square(K.mean(style, axis=(1,2)) - K.mean(output, axis=(1,2)))) + gram_loss(style_image, output_image)
    return K.sum(K.square(moment(style, n, axis=(1,2)) - moment(output, n, axis=(1,2))))
    #return K.mean(K.square(K.mean(style, axis=(1,2)) - K.mean(output, axis=(1,2)))) + K.mean(K.square(K.std(style, axis=(1,2)) - K.std(output, axis=(1,2)))) + gram_loss(style_image, output_image)

def nstat_loss(style_image, output_image, n):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]

    return K.sum(K.square(moment_norm(style, n, axis=(1,2)) - moment_norm(output, n, axis=(1,2))))

def moments_loss(style_image, output_image, number_of_moments):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]

    loss = K.variable(0)
    for n in range(number_of_moments):
        s = moment_norm(style, n)
        c = moment_norm(output, n)
        loss += K.mean(K.square(s - c))
    #loss /= number_of_moments

    loss += gram_loss(style_image, output_image)

    return loss

def histogram(x, nbins):
    import tensorflow as tf
    
    mn = K.min(x)
    m = K.mean(x)
    s = K.std(x)

    def histogram_of_feature_map(v):
        v = K.flatten(v)
        #h = K.cast(tf.histogram_fixed_width(v, [m-s*2, m+s*2], nbins), K.floatx())
        h = K.cast(tf.histogram_fixed_width(v, [mn, m+s*2], nbins), K.floatx())
        #return h
        pdf = h / K.sum(h)
        #return pdf
        cdf = tf.cumsum(pdf)
        return cdf
        return K.square(cdf)
    return tf.map_fn(histogram_of_feature_map, x)
    #h = histogram_of_feature_map(K.flatten(x))
    #return h
    
    '''
    v = K.flatten(x)
    m = K.mean(v)
    s = K.std(v)
    h = tf.histogram_fixed_width(v, [m-s*2, m+s*2], nbins)
    return K.cast(h, K.floatx()) 
    '''

    '''hist = list()
    ta = tf.TensorArray(x.dtype)
    ta = ta.unpack(x)
    for v in tf.unpack(x):
    return tf.pack(hist)
    '''

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

    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtypes.int32),
        indices,
        nbins), indices
'''
def interp_normalized(normalized_values, xp, fp):
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import array_ops

    nbins_float = math_ops.to_float(nbins)

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * normalized_values, name='indices')
    fracs = math_ops.floor(nbins_float * normalized_values, name='indices')
    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    xp[indices] + xp[indices]


def interp_normalized2(x, xp, value_range, nbins):


# tf.fake_quant_with_min_max_args

# impl interp() via map_fn() + argmax()

# try with tf.py_func() first!

def np_match_histograms(A, B, rng=(0.0, 255.0), bins=64):
    (Ha, Xa), (Hb, Xb) = [np.histogram(i, bins=bins, range=rng, density=True) for i in [A, B]]
    Hpa, Hpb = [np.cumsum(i) * (rng[1] - rng[0]) ** 2 / float(bins) for i in [Ha, Hb]]
    
    X = np.linspace(rng[0], rng[1], bins, endpoint=True)

    inv_Ha = scipy.interpolate.interp1d(X, Hpa, bounds_error=False, fill_value='extrapolate')
    map_Hb = scipy.interpolate.interp1d(Hpb, X, bounds_error=False, fill_value='extrapolate')
    return map_Hb(inv_Ha(A).clip(0.0, 255.0))
'''

def histogram_matching(style_image, output_image, nbins):
    import tensorflow as tf
    
    mn = K.min(style_image)
    mx = K.max(style_image)

    # bruteforce search
    # 8 - 30s
    # 32 - 30s
    # 256 - 40s
    # 512 - 59s
    # 1024 - 163s
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
        #  >>> mask = tf.to_int32(tf.less(search_in, search_for))
        #  mask =       [[1, 1, 1, 0, 1],
        #                [0, 0, 1, 0, 1],
        #                [0, 0, 0, 0, 1]]
        #
        #  >>> indices = tf.foldl(lambda accum, mask: accum+mask, mask, back_prop=False)
        #  indices =     [1, 1, 2, 0, 3]

        # splat values into 2 tensors with dimensions: [values_to_search.shape[0], sorted_array.shape[0]]
        search_for, search_in = tf.meshgrid(values_to_search, sorted_array)
        # mask out values that are greater than the value we are looking for
        mask = tf.to_int32(tf.less(search_in, search_for))
        # sum up all 1s in the mask to calculate array index+1
        # NOTE: cumsum() with slicing turns out to be faster than foldl()
        # slower altenative: indices = tf.foldl(lambda accum, mask: accum+mask, mask, back_prop=False)
        indices = tf.cumsum(mask)
        shape = tf.shape(indices)
        indices = tf.squeeze(tf.slice(indices, [shape[0]-1,0], [1, shape[1]]))
        return indices - 1

    def searchsorted2(sorted_array, values_to_search):
        from tensorflow.python.ops import array_ops
        return tf.where(tf.less(sorted_array[1], values_to_search), 
            array_ops.zeros_like(values_to_search, dtype=tf.int32), 
            array_ops.ones_like(values_to_search, dtype=tf.int32))

    def searchsortedB(sorted_array, values_to_search):
        from tensorflow.python.ops import array_ops

        zeros = array_ops.zeros_like(values_to_search, dtype=tf.int32)
        ones = array_ops.ones_like(values_to_search)

        fn = lambda a, x: tf.where(tf.less(ones * x, values_to_search), a + 1, a)
        indices = tf.foldl(fn, sorted_array, initializer=zeros, parallel_iterations=1, back_prop=False)

        return indices - 1

    def searchsortedC(sorted_array, values_to_search):
        from tensorflow.python.ops import array_ops

        zeros_i32 = array_ops.zeros_like(values_to_search, dtype=tf.int32)
        ones_i32 = array_ops.zeros_like(values_to_search, dtype=tf.int32)
        ones = array_ops.ones_like(values_to_search)

        #fn = lambda a, x: a + tf.where(tf.less(ones * x, values_to_search), ones_i32, zeros_i32)
        fn = lambda a, x: a + tf.where(tf.greater(values_to_search, x), zeros_i32, ones_i32)
        indices = tf.foldl(fn, sorted_array, initializer=zeros_i32, back_prop=False)

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

    # 8 - 33s
    # 32 - 33s
    # 256 - 44s
    # 512 - 49s
    # 1024 - 48s
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

    # based on  _call_linear(self, x_new) from https://github.com/scipy/scipy/blob/v0.19.0/scipy/interpolate/interpolate.py
    def interp_linear(x_new, x, y, nbins):
        from tensorflow.python.framework import dtypes
        from tensorflow.python.ops import clip_ops
        from tensorflow.python.ops import math_ops

        # Find where in the orignal data, the values to interpolate
        # would be inserted.
        # Note: If x_new[n] == x[m], then m is returned by searchsorted.

        #x_new_indices = searchsortedN(x, x_new, nbins) # off by one
        #x_new_indices = searchsortedT(x, x_new) # off by one
        x_new_indices = searchsorted(x, x_new)
        #x_new_indices = tf.py_func(np.searchsorted, [x, x_new], tf.int64, stateful=False)

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

    def feature_histogram_matching(source, template, value_range, nbins):

        s_counts, indices = histogram_fixed_width(source, value_range, nbins)
        t_counts, _ = histogram_fixed_width(template, value_range, nbins)
        t_values = tf.linspace(value_range[0], value_range[1], nbins)

        s_cdf = tf.cumsum(s_counts)
        s_cdf = tf.to_float(s_cdf)
        s_cdf /= s_cdf[-1]

        t_cdf = tf.cumsum(t_counts)
        t_cdf = tf.to_float(t_cdf)
        t_cdf /= t_cdf[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        # interp_t_values = np.interp(s_cdf, t_cdf, t_values)

        #return interp_and_scatter(s_cdf, t_cdf, t_values, indices, nbins)
        #return tf.py_func(np_interp_and_scatter, [s_cdf, t_cdf, t_values, indices], tf.float32, stateful=False) #np.interp(s_cdf, t_cdf, t_values)
        #interp_t_values = tf.py_func(np_interp, [s_cdf, t_cdf, t_values], tf.float32, stateful=False) #np.interp(s_cdf, t_cdf, t_values)        
        interp_t_values = interp_linear(s_cdf, t_cdf, t_values, nbins)
        interp_t_values = tf.maximum(interp_t_values, 0.0)
        values = tf.gather(interp_t_values, indices)
        return values

    def wrap_feature_histogram_matching(x, template):
        return feature_histogram_matching(K.flatten(x), K.flatten(template), [tf.to_float(mn), tf.to_float(mx)], nbins)

    def map(fn, arrays, dtype=tf.float32):
        # assumes all arrays have same leading dim
        indices = tf.range(tf.shape(arrays[0])[0])
        out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype, back_prop=False)
        return out

    #return tf.map_fn(wrap_feature_histogram_matching, (output_image, style_image))
    return map(wrap_feature_histogram_matching, [output_image, style_image])
    #return tf.map_fn(wrap_feature_histogram_matching, style_image)

# based on http://stackoverflow.com/a/33047048/7873678
def np_histogram_matching(style_image, output_image, nbins):
    import tensorflow as tf
    
    mn = K.min(style_image)
    mx = K.max(style_image)

    def feature_histogram_matching(source, template, value_range, nbins):

        #print(np.shape(source))
        #print(np.shape(template))
        source = source.flatten()
        template = template.flatten()
        #_, indices, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        #t_values, t_counts = np.unique(template, return_counts=True)

        #s_counts, indices = histogram_fixed_width(source, value_range, nbins) #tf.unique_with_counts
        #t_counts, _ = histogram_fixed_width(template, value_range, nbins) #tf.unique_with_counts

        '''
        print(np.mean(source))
        print(np.max(source))

        n = np.max(source)
        s = np.sqrt((source / n))
        s *= n
        print(np.mean(s))
        print(np.max(s))

        print("---")

        print(np.mean(template))
        print(np.max(template))
        n = np.max(template)
        t = np.sqrt((template / n))
        t *= n
        print(np.mean(t))
        print(np.max(t))

        print("===")
        print("===")
        print("===")
        '''


        #print(nbins)
        #print(len(source))
        #print(len(template))

        indices = np.digitize(source, range(nbins-1), right=True)
        s_counts = np.bincount(indices, minlength=nbins)
        t_counts = np.bincount(np.digitize(template, range(nbins-1), right=True), minlength=nbins)
        t_values = np.linspace(value_range[0], value_range[1], nbins)

        #print(s_counts)
        #print(t_counts)

        #print(len(indices))
        #print(len(s_counts))
        #print(len(t_counts))
        #print(len(t_values))

        s_cdf = np.cumsum(s_counts).astype(np.float32)
        s_cdf /= s_cdf[-1]

        t_cdf = np.cumsum(t_counts).astype(np.float32)
        t_cdf /= t_cdf[-1]

        interp_t_values = np.interp(s_cdf, t_cdf, t_values).astype(np.float32)
        interp_t_values = np.maximum(interp_t_values, 0.0)
        return interp_t_values[indices]

    def wrap_feature_histogram_matching(x, template):
        value_range = [tf.to_float(mn), tf.to_float(mx)]
        return tf.py_func(feature_histogram_matching, [x, template, value_range, nbins], tf.float32, stateful=False)

    def map(fn, arrays, dtype=tf.float32):
        # assumes all arrays have same leading dim
        indices = tf.range(tf.shape(arrays[0])[0])
        out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, parallel_iterations=1, dtype=dtype)
        return out


    def unravel(output_image, style_image, value_range, nbins):
        #print("unravel")
        #print(np.shape(output_image))
        #print(np.shape(style_image))
        j = None
        for o, s in zip(output_image, style_image):
            x = feature_histogram_matching(o, s, value_range, nbins)
            if isinstance(j, np.ndarray):
                j = np.concatenate((j, x))
            else:
                j = x
        return j

    value_range = [tf.to_float(mn), tf.to_float(mx)]
    

    #return tf.py_func(unravel, [output_image, style_image, value_range, nbins], tf.float32, stateful=False)
    return map(wrap_feature_histogram_matching, [output_image, style_image])


def histogram_loss(style_image, output_image, nbins):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]

    output_remapped = histogram_matching(style, output, nbins)
    output_remapped = K.stop_gradient(output_remapped)
    return K.sum(K.square(K.flatten(output_image) - K.flatten(output_remapped)))

    a = K.flatten(output_image) / K.cast(num_channels, K.floatx())
    b = K.flatten(output_remapped) / K.cast(num_channels, K.floatx())
    return K.sum(K.square(a - b)) #* 0.5 * K.cast(K.shape(output)[0]*K.shape(output)[1], K.floatx()) + gram_loss(style_image, output_image)
    #return K.mean(K.square(s - c)) #+ gram_loss(style_image, output_image)

def np_histogram_loss(style_image, output_image, nbins):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        output = output_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        output = K.permute_dimensions(
            output_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]

    #output_remapped = np_histogram_matching(style, output, nbins)
    #output_remapped = K.stop_gradient(output_remapped)

    output_remapped = histogram_matching(style, output, nbins)
    output_remapped = K.stop_gradient(output_remapped)
    return K.sum(K.square(K.flatten(output_image) - K.flatten(output_remapped)))

    a = K.flatten(output_image) / K.cast(num_channels, K.floatx())
    b = K.flatten(output_remapped) / K.cast(num_channels, K.floatx())
    return K.sum(K.square(a - b))
    #return K.mean(K.square(s - c)) #+ gram_loss(style_image, output_image)

    #s = histogram(style_image, nbins)
    #c = histogram(output_image, nbins)
    #return K.max(c)
    #return K.sum(K.square(s - c))
    # + gram_loss(style_image, output_image)

def style_loss(style_image, output_image):
    if args.loss=='gram':
        return gram_loss(style_image, output_image)
    elif args.loss=='gram_offset':
        return gram_offset_loss(style_image, output_image, args.loss_arg)
    elif args.loss=='stat':
        return stat_loss(style_image, output_image, args.loss_arg)
    elif args.loss=='nstat':
        return nstat_loss(style_image, output_image, args.loss_arg)
    elif args.loss=='gram_sqrt':
        return gram_sqrt_loss(style_image, output_image)
    elif args.loss=='L2' or args.loss=='copy':
        return L2_loss(style_image, output_image)
    elif args.loss=='moments':
        return moments_loss(style_image, output_image, int(args.loss_arg))
    elif args.loss=='histogram':
        return histogram_loss(style_image, output_image, int(args.loss_arg))
    elif args.loss=='histogram_numpy':
        return np_histogram_loss(style_image, output_image, int(args.loss_arg))
    elif args.loss=='mrf':
        return mrf_loss(style_image, output_image)
    else:
        return gram_loss(style_image, output_image)

def total_variation_loss(x):
    assert 4 == K.ndim(x)
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


class VGG19FeatureExtractor(object):

    def __init__(self, style_image_in, img_nrows, img_ncols, num_colors):
        # Create tensor variables for images
        if K.image_data_format() == 'channels_first':
            shape = (1, num_colors, img_nrows, img_ncols)
        else:
            shape = (1, img_nrows, img_ncols, num_colors)

        self.style_image = K.variable(preprocess_image(style_image_in, (img_nrows, img_ncols)))
        self.output_image = K.placeholder(shape=shape)

        # see: STYLE, OUTPUT = 0, 1
        self.images = K.concatenate([self.style_image, self.output_image], axis=0)

        # Build image model and use layer outputs as features
        # image model as VGG19
        # self.model = vgg19.VGG19(include_top=False, input_tensor=self.images)
        self.model = custom_vgg19.myVGG19(include_top=False, input_tensor=self.images,
        #    weights=None, conv_activation='elu', conv_pooling='strided')
        #    weights=None, conv_activation='relu', conv_pooling='max')
            conv_activation='relu', conv_pooling='max')

        # Collect features from model
        self.features = {}
        for img_layer in self.model.layers:
            if 'conv' in img_layer.name:
                layer_name = img_layer.name
                img_feat = img_layer.output
                self.features[layer_name] = img_feat

    # TODO: separate Style (do it only once) and Output feature extraction - check performance
    # TODO: evaluate only necessary (or strip away unnecessary) layers for Evaluator (and their sources) - check performance

class Evaluator(object):

    def __init__(self, output_image, features):
        self.loss_value = None
        self.grads_values = None

        # Overall loss is the weighted sum of style_loss and tv_loss
        # Each individual loss uses features from image models.
        loss = K.variable(0)
        for layer in style_feature_layers:
            style_feat = features[layer][STYLE, :, :, :]
            output_feat = features[layer][OUTPUT, :, :, :]
            sl = style_loss(style_feat, output_feat)
            loss += (1. / len(style_feature_layers)) * sl

        # TODO: try tv_loss only on the first conv layer (suggested in E.Risser paper)
        loss += total_variation_weight * total_variation_loss(output_image)
        loss_grads = K.gradients(loss, output_image)

        outputs = [loss]
        # TODO: is this still necessary?
        if isinstance(loss_grads, (list, tuple)):
            outputs += loss_grads
        else:
            outputs.append(loss_grads)

        self.f_outputs = K.function([output_image], outputs)

    def _eval_loss_and_grads(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, img_nrows, img_ncols))
        else:
            x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    ###                 

    # TODO: investigate if separating loss and grads here makes any actual performance improvement
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self._eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

iter = int(0)
t0 = time.time()
lastX = None
for q in range(args.phases):
    t1 = time.time()
    scale = 1.0 / 2.0 ** (args.phases - 1 - q)

    img_nrows, img_ncols = target_size
    img_nrows = int(img_nrows*scale)
    img_ncols = int(img_ncols*scale)

    print("Pass:", (img_ncols, img_nrows))

    # def __init__(self, style_image_in, img_nrows, img_ncols, num_colors):
    model = VGG19FeatureExtractor(args.style_image, img_nrows, img_ncols, num_colors)

    #  def __init__(self, output_image, features):
    evaluator = Evaluator(model.output_image, model.features)

    # TODO: move noise initialization out of the loop
    if q == 0:
        # 1st pass
        # Generate images by iterative optimization
        if args.prime_image is None:
            seed_range = [int(i) for i in args.seed_range.split(':')]
            if K.image_data_format() == 'channels_first':
                x = np.random.uniform(seed_range[0], seed_range[1], (1, 3, img_nrows, img_ncols)) - 128.
            else:
                x = np.random.uniform(seed_range[0], seed_range[1],  (1, img_nrows, img_ncols, 3)) - 128.

        else:
            x = preprocess_image(args.prime_image, (img_nrows, img_ncols))

        print(x.shape)
    else:
        # Upscale
        print(x.shape)
        x = imresize(x, (img_nrows, img_ncols), interp='bicubic')
        x = preprocess_image_partially(x)

        img = deprocess_image(x.copy())
        fname = args.target_image_prefix + '_after_upscale_%d.png' % q
        imsave(fname, img)
        print('Image saved as', fname)

    # Optimization algorithm needs min and max bounds to prevent divergence.
    #data_bounds = np.zeros((np.product(x.shape), 2), dtype=np.float64)                             # ??? float64
    #data_bounds[:] = (0.0, 255.0)

    for w in range(num_iterations):
        if iter==0:
            print('Start of iteration %d' % iter)

        t2 = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         #bounds=data_bounds,
                                         fprime=evaluator.grads, maxfun=20)
        if w == 0 or w == num_iterations-1 or q == args.phases-1:
            # save current generated image
            img = deprocess_image(x.copy())
            fname = args.target_image_prefix + '_at_iteration_%d.png' % iter
            imsave(fname, img)
            print('Image saved as', fname)
        print('Iteration %d completed in %ds. Loss value: %f' % (iter, time.time() - t2, min_val))
        iter += 1

    print(x.shape)
    img = deprocess_image(x.copy())
    fname = args.target_image_prefix + '_after_pass%d.png' % q
    imsave(fname, img)
    print('Image saved as', fname)

    print('Phase %d completed in %ds' % (q, time.time() - t1))

    lastX = x.copy()
    x = deprocess_image_partially(x.copy())

print('Completed in %ds' % (time.time() - t0))

img = deprocess_image(lastX.copy())
fname = args.target_image_prefix + ".png"
imsave(fname, img)
print('Image saved as', fname)