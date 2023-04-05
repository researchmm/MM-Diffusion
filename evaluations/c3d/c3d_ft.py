from __future__ import print_function

import collections
import os

from chainer import link
from chainer.dataset import download
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_nd import max_pooling_nd
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.links.connection.convolution_nd import ConvolutionND
from chainer.links.connection.linear import Linear
from chainer.serializers import npz


class C3DVersion1(link.Chain):

    def __init__(self, pretrained_model='auto'):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            init = constant.Zero()
            conv_kwargs = {'initialW': init, 'initial_bias': init}
            fc_kwargs = conv_kwargs
        else:
            # employ default initializers used in the original paper
            conv_kwargs = {
                'initialW': normal.Normal(0.01),
                'initial_bias': constant.Zero(),
            }
            fc_kwargs = {
                'initialW': normal.Normal(0.005),
                'initial_bias': constant.One(),
            }
        super(C3DVersion1, self).__init__(
            conv1a=ConvolutionND(3, 3, 64, 3, 1, 1, **conv_kwargs),
            conv2a=ConvolutionND(3, 64, 128, 3, 1, 1, **conv_kwargs),
            conv3a=ConvolutionND(3, 128, 256, 3, 1, 1, **conv_kwargs),
            conv3b=ConvolutionND(3, 256, 256, 3, 1, 1, **conv_kwargs),
            conv4a=ConvolutionND(3, 256, 512, 3, 1, 1, **conv_kwargs),
            conv4b=ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs),
            conv5a=ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs),
            conv5b=ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs),
            fc6=Linear(512 * 4 * 4, 4096, **fc_kwargs),
            fc7=Linear(4096, 4096, **fc_kwargs),
            fc8=Linear(4096, 101, **fc_kwargs),
        )
        if pretrained_model == 'auto':
            _retrieve(
                'conv3d_deepnetA_ucf.npz',
                'http://vlg.cs.dartmouth.edu/c3d/'
                'c3d_ucf101_finetune_whole_iter_20000',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

        self.functions = collections.OrderedDict([
            ('conv1a', [self.conv1a, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2a', [self.conv2a, relu]),
            ('pool2', [_max_pooling_3d]),
            ('conv3a', [self.conv3a, relu]),
            ('conv3b', [self.conv3b, relu]),
            ('pool3', [_max_pooling_3d]),
            ('conv4a', [self.conv4a, relu]),
            ('conv4b', [self.conv4b, relu]),
            ('pool4', [_max_pooling_3d]),
            ('conv5a', [self.conv5a, relu]),
            ('conv5b', [self.conv5b, relu]),
            ('pool5', [_max_pooling_3d]),
            ('fc6', [self.fc6, relu, dropout]),
            ('fc7', [self.fc7, relu, dropout]),
            ('fc8', [self.fc8]),
            ('prob', [softmax]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As caffe_function uses shortcut symbols,
        # we import it here.
        from chainer.links.caffe import caffe_function
        caffe_pb = caffe_function.caffe_pb

        caffemodel = caffe_pb.NetParameter()
        with open(path_caffemodel, 'rb') as model_file:
            caffemodel.MergeFromString(model_file.read())
        chainermodel = cls(pretrained_model=None)
        _transfer(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def __call__(self, x, layers=['prob']):
        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


def _max_pooling_3d(x):
    # print(x.data.shape)
    return max_pooling_nd(x, ksize=2)
    # return max_pooling_nd(x, ksize=2, stride=2)


def _max_pooling_2d(x):
    return max_pooling_nd(x, ksize=(1, 2, 2))
    # return max_pooling_nd(x, ksize=(1, 2, 2), stride=(1, 2, 2))


def _transfer(caffemodel, chainermodel):

    def transfer_layer(src, dst):
        dst.W.data.ravel()[:] = src.blobs[0].diff
        dst.b.data.ravel()[:] = src.blobs[1].diff

    layers = {l.name: l for l in caffemodel.layers}
    print([l.name for l in caffemodel.layers])
    transfer_layer(layers['conv1a'], chainermodel.conv1a)
    transfer_layer(layers['conv2a'], chainermodel.conv2a)
    transfer_layer(layers['conv3a'], chainermodel.conv3a)
    transfer_layer(layers['conv3b'], chainermodel.conv3b)
    transfer_layer(layers['conv4a'], chainermodel.conv4a)
    transfer_layer(layers['conv4b'], chainermodel.conv4b)
    transfer_layer(layers['conv5a'], chainermodel.conv5a)
    transfer_layer(layers['conv5b'], chainermodel.conv5b)
    transfer_layer(layers['fc6'], chainermodel.fc6)
    transfer_layer(layers['fc7'], chainermodel.fc7)
    transfer_layer(layers['fc8'], chainermodel.fc8)


def _make_npz(path_npz, url, model):
    import pdb; pdb.set_trace()
    # path_caffemodel = "/mnt/sakura201/mattya/c3d/c3d_ucf101_finetune_whole_iter_20000"
    path_caffemodel = "models/nc3d_ucf101_finetune_whole_iter_20000"
    print('Now loading caffemodel (usually it may take few minutes)')
    C3DVersion1.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return n
    


def _retrieve(name, url, model):
    
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
