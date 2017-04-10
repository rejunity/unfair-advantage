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
add_arg('--style-layers',           default='1_1,2_1,3_1,4_1,5_1',type=str,help='VGG layers to learn style from')
add_arg('--prime-image',            default=None, type=str,         help='Path to image to prime optimization')
add_arg('--seed-range',             default='16:240', type=str,     help='Prime with random colors chosen in range, e.g. 0:255.')
add_arg('--target-image-prefix',    type=str,                       help='Path prefix for generated results')
add_arg('--target-size',            default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--phases',                 default=3, type=int,            help='Number of steps in texture pyramid.')
add_arg('--iterations',             default=50, type=int,           help='Number of iterations to run each phase/resolution.')
add_arg('--variation',              default=50, type=float,         help='Weight of total variational loss')
add_arg('--loss',                   default='gramm', type=str,      help='Loss function to use for optimization')
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
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def gramm_loss(style_image, output_image):
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

def L2_loss(style_image, output_image):
    assert 3 == K.ndim(style_image) == K.ndim(output_image)
    s = K.batch_flatten(style_image)
    c = K.batch_flatten(output_image)
    return K.sum(K.square(s - c))

# TODO: moments and histogram losses, compare them
def moments_loss(style_image, output_image):
    return 0

def histogram_loss(style_image, output_image):
    return 0

def style_loss(style_image, output_image):
    if args.loss=='gramm':
        return gramm_loss(style_image, output_image)
    elif args.loss=='L2' or args.loss=='copy':
        return L2_loss(style_image, output_image)
    elif args.loss=='mrf':
        return mrf_loss(style_image, output_image)
    else:
        return gramm_loss(style_image, output_image)

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