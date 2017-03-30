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

References:
[Dmitry Ulyanov's blog on fast-neural-doodle](http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
[Torch code for fast-neural-doodle](https://github.com/DmitryUlyanov/fast-neural-doodle)
[Torch code for online-neural-doodle](https://github.com/DmitryUlyanov/online-neural-doodle)
[Paper Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417)
[Discussion on parameter tuning](https://github.com/fchollet/keras/issues/3705)

Resources:
Example images can be downloaded from
https://github.com/DmitryUlyanov/fast-neural-doodle/tree/master/data
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
add_arg('--variation',              default=50, type=float,         help='todo')
args = parser.parse_args()

num_iterations = args.iterations
###style_mask_path = args.style_mask
###target_mask_path = args.target_mask
###content_img_path = args.content_image
###use_content_img = content_img_path is not None

###num_labels = args.nlabels
num_colors = 3  # RGB
# determine image sizes based on style_image
target_size = imread(args.style_image).shape[:2]
if args.target_size:
    target_size = tuple([int(i) for i in args.target_size.split('x')])
img_nrows, img_ncols = target_size

total_variation_weight = args.variation
###style_weight = 1.
###content_weight = 0.1 if use_content_img else 0
###content_feature_layers = ['block5_conv2']

# To get better generation qualities, use more conv layers for style features
style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
#style_feature_layers = ['block3_conv1', 'block4_conv1']

# index constants for images and tasks variables
STYLE, TARGET, CONTENT = 0, 1, 2


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

def rescale_image(self, img, scale):
    """Re-implementing skimage.transform.scale without the extra dependency. Saves a lot of space and hassle!
    """
    output = scipy.misc.toimage(img, cmin=0.0, cmax=255)
    output.thumbnail((int(output.size[0]*scale), int(output.size[1]*scale)), PIL.Image.ANTIALIAS)
    return np.asarray(output)
"""
def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype('float64'), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels():
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor: (1, m, nr, nc) for 'channels_first' or (1, nr, nc, m) for 'channels_last'
    '''
    target_mask_img = load_img(target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)
    if K.image_data_format() == 'channels_first':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                               target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)),
                               target_mask_img.reshape((-1, 3))])

    labels = kmeans(mask_vecs, num_labels)
    style_mask_label = labels[:img_nrows *
                              img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 if K.image_data_format() == 'channels_first' else -1
    style_mask = np.stack([style_mask_label == r for r in xrange(num_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in xrange(num_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))
"""

# Define loss functions
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

"""
def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    if K.image_data_format() == 'channels_first':
        masked_style = style_image * style_mask
        masked_target = target_image * target_mask
        num_channels = K.shape(style_image)[0]
    else:
        masked_style = K.permute_dimensions(
            style_image, (2, 0, 1)) * style_mask
        masked_target = K.permute_dimensions(
            target_image, (2, 0, 1)) * target_mask
        num_channels = K.shape(style_image)[-1]
    s = gram_matrix(masked_style) / K.mean(style_mask) / num_channels
    c = gram_matrix(masked_target) / K.mean(target_mask) / num_channels
    return K.mean(K.square(s - c))
"""

def region_style_loss(style_image, target_image):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    if K.image_data_format() == 'channels_first':
        style = style_image
        target = target_image
        num_channels = K.shape(style_image)[0]
    else:
        style = K.permute_dimensions(
            style_image, (2, 0, 1))
        target = K.permute_dimensions(
            target_image, (2, 0, 1))
        num_channels = K.shape(style_image)[-1]
    s = gram_matrix(style) / K.cast(num_channels, K.floatx())
    c = gram_matrix(target) / K.cast(num_channels, K.floatx())
    return K.mean(K.square(s - c))

"""
def style_loss(style_image, target_image, style_masks, target_masks):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = K.variable(0)
    for i in xrange(num_labels):
        if K.image_data_format() == 'channels_first':
            style_mask = style_masks[i, :, :]
            target_mask = target_masks[i, :, :]
        else:
            style_mask = style_masks[:, :, i]
            target_mask = target_masks[:, :, i]
        loss += region_style_loss(style_image,
                                  target_image, style_mask, target_mask)

    return loss
"""

def style_loss(style_image, target_image):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    loss = K.variable(0)
    loss += region_style_loss(style_image,
                              target_image)

    return loss

"""
def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))
"""

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


class VGG19Features(object):

    def __init__(self, style_image_in, img_nrows, img_ncols, num_colors):
        # Create tensor variables for images
        if K.image_data_format() == 'channels_first':
            shape = (1, num_colors, img_nrows, img_ncols)
        else:
            shape = (1, img_nrows, img_ncols, num_colors)

        self.style_image = K.variable(preprocess_image(style_image_in, (img_nrows, img_ncols)))
        self.target_image = K.placeholder(shape=shape)
        self.content_image = K.zeros(shape=shape)

        # see: STYLE, TARGET, CONTENT = 0, 1, 2
        self.images = K.concatenate([self.style_image, self.target_image, self.content_image], axis=0)

        # Build image model, mask model and use layer outputs as features
        # image model as VGG19
        # self.image_model = vgg19.VGG19(include_top=False, input_tensor=self.images)
        self.image_model = custom_vgg19.myVGG19(include_top=False, input_tensor=self.images)

        # Collect features from image_model
        self.image_features = {}
        for img_layer in self.image_model.layers:
            if 'conv' in img_layer.name:
                layer_name = img_layer.name
                img_feat = img_layer.output
                self.image_features[layer_name] = img_feat


class Evaluator(object):

    def __init__(self, target_image, image_features):
        self.loss_value = None
        self.grads_values = None

        # Overall loss is the weighted sum of content_loss, style_loss and tv_loss
        # Each individual loss uses features from image/mask models.
        loss = K.variable(0)
        """for layer in content_feature_layers:
            content_feat = image_features[layer][CONTENT, :, :, :]
            target_feat = image_features[layer][TARGET, :, :, :]
            loss += content_weight * content_loss(content_feat, target_feat)
        """
        for layer in style_feature_layers:
            style_feat = image_features[layer][STYLE, :, :, :]
            target_feat = image_features[layer][TARGET, :, :, :]
            ###style_masks = mask_features[layer][STYLE, :, :, :]
            ###target_masks = mask_features[layer][TARGET, :, :, :]
            ###sl = style_loss(style_feat, target_feat, style_masks, target_masks)
            sl = style_loss(style_feat, target_feat)
            ###loss += (style_weight / len(style_feature_layers)) * sl
            loss += len(style_feature_layers) * sl

        loss += total_variation_weight * total_variation_loss(target_image)
        loss_grads = K.gradients(loss, target_image)

        # Evaluator class for computing efficiency
        outputs = [loss]
        if isinstance(loss_grads, (list, tuple)):
            outputs += loss_grads
        else:
            outputs.append(loss_grads)

        self.f_outputs = K.function([target_image], outputs)   

    def eval_loss_and_grads(self, x):
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

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

###evaluator = Evaluator()

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
    model = VGG19Features(args.style_image, img_nrows, img_ncols, num_colors)

    #  def __init__(self, target_image, image_features):
    evaluator = Evaluator(model.target_image, model.image_features)

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

        """
        x = imresize(x.copy()-128, (img_nrows, img_ncols), interp='bicubic')
        #x = x.transpose((2, 0, 1))[np.newaxis]
        img = deprocess_image(img_to_array(x.copy()))
        fname = args.target_image_prefix + '_after_upscale_%d.png' % q
        imsave(fname, img)
        print('Image saved as', fname)
        """

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