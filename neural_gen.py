'''Neural generator with Keras

Script Usage:
    # Arguments:
    ```
    --style-image:          image to learn style from
    --target-image-prefix:  path prefix for generated target images
    ```

    # Example:
    ```
    python neural_doodle.py --style-image Wall.png \
    --target-image-prefix generated/wall
    ```

Base on:
[Dmitry Ulyanov's blog on fast-neural-doodle](http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
[Torch code for fast-neural-doodle](https://github.com/DmitryUlyanov/fast-neural-doodle)
[Torch code for online-neural-doodle](https://github.com/DmitryUlyanov/online-neural-doodle)
[Paper Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417)
[Discussion on parameter tuning](https://github.com/fchollet/keras/issues/3705)
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
add_arg('--prime-image',            default=None, type=str,         help='Path to image to prime optimization')
add_arg('--target-image-prefix',    type=str,                       help='Path prefix for generated results')
add_arg('--target-size',            default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--iterations',             default=50, type=int,           help='Number of iterations to run each resolution.')
add_arg('--seed-range',             default='16:240', type=str,     help='Random colors chosen in range, e.g. 0:255.')
add_arg('--phases',                 default=3, type=int,            help='Number of image scales to process in phases.')
add_arg('--variation',              default=50, type=float,         help='TODO')
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
style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
#style_feature_layers = ['block3_conv1', 'block4_conv1']

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
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68    
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


# Define loss functions
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style_image, output_image):
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

# TODO: moments and histogram losses, compare them
def moments_loss(style_image, output_image):
    return 0

def histogram_loss(style_image, output_image):
    return 0

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

    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
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