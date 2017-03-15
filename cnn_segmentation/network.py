from __future__ import print_function
import numpy as np
from keras.models import (
    Sequential,
    load_model,
)
from keras.layers.core import (
    Activation,
    Dense,
    Dropout,
    Lambda,
    Merge,
)
from keras.layers import (
    Convolution2D,
    Convolution3D,
    MaxPooling2D,
    MaxPooling3D,
    Flatten,
)
from keras.layers.advanced_activations import (
    PReLU,
)
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)
from keras.optimizers import (
    Adagrad,
)
from keras.constraints import maxnorm
from keras import backend as K
from utils.utils import (
    write_h5file,
    read_h5file,
    reshape_to_tf,
    reshape_to_th,
)


# My networks built on top of Keras
class Network():
    """Abstract class for all networks.

    Do not realize this class directly.
    """
    def __init__(self, out_size):
        self.out_size = out_size
        self.model = None
        self.network_type = None
        self.monitor = None
        self.n_train = 0
        self.n_vali = 0
        self.model_path = None
        self.logger_path = None
        self.stats_path = None
        self.loss = None
        self.optimizer = None
        # sanity check of backend and dim ordering
        if K.backend() == 'theano':
            K.set_image_dim_ordering('th')
            self.image_dim_ordering = 'tf'
        elif K.backend() == 'tensorflow':
            K.set_image_dim_ordering('tf')
            self.image_dim_ordering = 'th'

    def build(self):
        """Set default attributes of optional arguments.

        Early stopping
        Training logger
        Model path
        Training set statistics path
        Model optimizer
        Weight initialization

        Note:
            If you want to use default callbacks
            EarlyStopping, ModelCheckpoint, CSVLogger
            with different configuration than default,
            you should set these values BEFORE this method:

            monitor: the metric to monitor in EearlyStopping
            model_path: the best model path of checkpoint
            logger_path: the log file path
        """
        self.image_dim_ordering = K.image_dim_ordering()
        self.monitor = 'val_acc'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.init = 'glorot_uniform'  # keras default
        if not self.model_path:
            self.model_path = './best_model.h5'
        if not self.logger_path:
            self.logger_path = './training_log.csv'
        if not self.stats_path:
            self.stats_path = './training_stats.h5'

        earlyStopping = EarlyStopping(monitor=self.monitor,
                                      patience=50,
                                      verbose=0,
                                      mode='auto')
        checkpoint = ModelCheckpoint(self.model_path, monitor=self.monitor,
                                     verbose=0, save_best_only=True)
        logger = CSVLogger(self.logger_path)
        lr_reducer = ReduceLROnPlateau(monitor='val_acc',
                                       factor=0.5,
                                       patience=2,
                                       min_lr=1e-12)
        self.callbacks = [earlyStopping, checkpoint, lr_reducer, logger]
        self.optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

    def compile(self, **kwargs):
        """Compile the model.

        Compile the model, change different optimizers and callbacks if needed.
        Leave the argument empty for default settings.

        This method should be called after build()
        """
        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']
        if 'monitor' in kwargs:
            self.monitor = kwargs['monitor']
        if 'loss' in kwargs:
            self.loss = kwargs['loss']
        if 'metrics' in kwargs:
            self.metrics = kwargs['metrics']
        if 'init' in kwargs:
            self.init = kwargs['init']

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def set_train_stats(self, mu, sigma):
        pass

    def train(self,
              x_tr, y_tr,
              x_va, y_va,
              batch_size=200,
              max_epoch=20,
              callbacks=None):
        """Train model given data.

        All the prerocessing steps have to be
        done before calling this method.

        Arguments:
            x_tr: numpy array or list of numpy array (if multiple inputs)
                Training feature
            y_tr: numpy.ndarray (vector)
                Training label vector. Shape = n_voxel, n_classes
            n_train: int
                Number of all the train samples
            x_va: numpy array or list of numpy array (if multiple inputs)
                Validation feature
            y_va: numpy.ndarray (vector)
                Validation label vector. Shape = n_voxel, n_classes
            mu: numpy array or list of numpy array (if multiple inputs)
                Mean of all train samples
            sigma: numpy array or list of numpy array (if multiple inputs)
                Std of all train samples
            max_epoch: int
                Number of epochs to train
            callbacks: list of Callback
                The callback functions. Leave it empty to use default.

        Return:
            The trained model
        """
        self.save_stats()
        # If no callbacks is set, use these default callbacks
        if not callbacks:
            callbacks = self.callbacks

        # Train the model
        self.model.fit(x_tr, y_tr,
                       batch_size=batch_size,
                       nb_epoch=max_epoch,
                       verbose=1,
                       callbacks=callbacks,
                       validation_data=(x_va, y_va))
        return self.model

    def train_generator(self,
                        gen, n_train,
                        x_va, y_va,
                        max_epoch=20,
                        callbacks=None):
        """Train model given data generator.

        This is used when you really want to use a generator
        to feed the data. All the prerocessing steps have to be
        done before calling this method.

        Arguments:
            gen: Python generator
                Generator that yields (feature, label) tuples
                The features are normalized outside
                Note that label.shape = n_voxel, n_classes
            n_train: int
                Number of all the train samples
            x_va: numpy array or list of numpy array (if multiple inputs)
                Validation feature
            y_va: numpy.ndarray (vector)
                Validation label vector. Shape = n_voxel, n_classes
            mu: numpy array or list of numpy array (if multiple inputs)
                Mean of all train samples
            sigma: numpy array or list of numpy array (if multiple inputs)
                Std of all train samples
            max_epoch: int
                Number of epochs to train
            callbacks: list of Callback
                The callback functions. Leave it empty to use default.

        Return:
            The trained model
        """
        self.save_stats()

        # If no callbacks is set, use these default callbacks
        if not callbacks:
            callbacks = self.callbacks

        self.model.fit_generator(gen,
                                 n_train,
                                 nb_epoch=max_epoch,
                                 validation_data=(x_va, y_va),
                                 callbacks=callbacks)
        return self.model

    def predict(self, x_te, batch_size=20000):
        """Predict the probability.

        This function uses the default device, could be GPU or CPU.

        Args:
            x_te: numpy array or list of numpy array (if multiple inputs)
                The input feature, could be in Theano or Tensorflow.
            batch_size: int
                The number of samples fit into GPU at a time.

        Return:
            prob: numpy array
                The probability estimate of each class.
        """
        return self.model.predict(x_te, batch_size=batch_size)

    def predict_label(self, x_te, batch_size=20000):
        """Predict the label.

        This function uses the default device, could be GPU or CPU.

        Args:
            x_te: numpy array or list of numpy array (if multiple inputs)
                The input feature, could be in Theano or Tensorflow.
            batch_size: int
                The number of samples fit into GPU at a time.

        Return:
            label: numpy array
                The predicted label.
        """
        return np.argmax(self.predict(x_te, batch_size), axis=1)

    def predict_generator(self, gen, batch_size=20000):
        """Predict the probability given feature generator.

        Aruguments:
            gen: generator
                The generator that yields a batch of patch feature
            batch_size: int
                The batch of gpu batches

        Return:
            prob: numpy array
                The probability estimate of each class.
        """
        y_pred = []
        for batch in gen:
            y_pred.extend(self.predict(batch, batch_size))
        return np.asarray(y_pred)

    def predict_label_generator(self, gen, batch_size=20000):
        """Predict the predicted label given feature generator.

        Aruguments:
            gen: generator
                The generator that yields a batch of patch feature
            batch_size: int
                The batch of gpu batches

        Return:
            label: numpy array
                The predicted label.
        """
        pred_prob = self.predict_generator(gen, batch_size)
        return np.argmax(pred_prob, axis=1)

    def set_train_stats(self, mu, sigma, n_train, n_vali):
        self.n_train = n_train
        self.n_vali = n_vali

    def save_model(self, model_path=None):
        """Save the model into a HDF5.

        Arguments:
            model_path: string
                The path to store the model.
                Leave this empty to save to default path.
        """
        model_path = model_path if model_path else self.model_path
        self.model.save_weights(model_path)

    def load_model(self, model_path):
        """Load the model from a HDF5."""
        print("Load model from {}".format(model_path))
        self.build()
        self.model.load_weights(model_path)

    def save_stats(self, stats_path=None):
        """Save network and training statistics."""
        self.stats_path = stats_path if stats_path else self.stats_path

        # wrap general attributes
        self.attrs = {}
        self.attrs['network_type'] = self.network_type
        self.attrs['out_size'] = self.out_size
        self.attrs['monitor'] = self.monitor
        self.attrs['n_train'] = self.n_train
        self.attrs['n_vali'] = self.n_vali
        self.attrs['loss'] = self.loss
        self.attrs['optimizer'] = type(self.optimizer).__name__
        self.attrs['init'] = self.init
        self.attrs['image_dim_ordering'] = self.image_dim_ordering

        # wrap training statistics in subclass

    def load_stats(self, stats_path):
        """Load network and training statistics."""
        self.stats = read_h5file(stats_path)
        self.monitor = self.stats['monitor']
        self.n_train = self.stats['n_train']
        self.n_vali = self.stats['n_vali']

        stored_image_dim_ordering = self.stats['image_dim_ordering']
        assert self.image_dim_ordering == stored_image_dim_ordering, \
               "The model was trained using {}, and the current is {}". \
               format(stored_image_dim_ordering, self.image_dim_ordering)

        # load training statistics in subclass


class Network2DPatch(Network):
    """
    Abstract class for 2D patch input networks.
    Such networks accept one or more channels of 2D patches,
    The input patch sizee of all channels have to be the same.
    """
    def __init__(self, patch_size, n_channels, out_size):
        super(Network2DPatch, self).__init__(out_size)
        self.patch_size = patch_size
        self.n_channels = n_channels

    def set_train_stats(self, mu, sigma, n_train, n_vali):
        super(Network2DPatch, self).set_train_stats(mu,
                                                    sigma,
                                                    n_train,
                                                    n_vali)
        self.patch2d_mu = mu
        self.patch2d_sigma = sigma

    def save_stats(self, stats_path=None):
        super(Network2DPatch, self).save_stats(stats_path)
        self.stats = {}
        self.stats['patch2d_mu'] = self.patch2d_mu
        self.stats['patch2d_sigma'] = self.patch2d_sigma

        write_h5file(self.stats_path, self.stats, self.attrs)

    def load_stats(self, stats_path=None):
        super(Network2DPatch, self).load_stats(stats_path)
        self.patch2d_mu = self.stats['patch2d_mu']
        self.patch2d_sigma = self.stats['patch2d_sigma']


class NetworkDense(Network):
    """
    Abstract class with Dense (1-D) input.
    """
    def __init__(self, dense_size, out_size):
        super(NetworkDense, self).__init__(out_size)
        self.dense_size = dense_size

    def set_train_stats(self, mu, sigma, n_train, n_vali):
        super(NetworkDense, self).set_train_stats(mu,
                                                  sigma,
                                                  n_train,
                                                  n_vali)
        self.dense_mu = mu
        self.dense_sigma = sigma

    def save_stats(self, stats_path=None):
        super(NetworkDense, self).save_stats(stats_path)
        self.stats = {}
        self.stats['dense_mu'] = self.dense_mu
        self.stats['dense_sigma'] = self.dense_sigma

        write_h5file(self.stats_path, self.stats, self.attrs)

    def load_stats(self, stats_path=None):
        super(NetworkDense, self).load_stats(stats_path)
        self.dense_mu = self.stats['dense_mu']
        self.dense_sigma = self.stats['dense_sigma']


class Network2DPatchDense(Network):
    """
    Abstract class with two kinds of input
    - 2D patches input
    - Dense (1-D) input
    """
    def __init__(self, patch_size, n_channels, dense_size, out_size):
        super(Network2DPatchDense, self).__init__(out_size)
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.dense_size = dense_size

    def set_train_stats(self, mu, sigma, n_train, n_vali):
        super(Network2DPatchDense, self).set_train_stats(mu,
                                                         sigma,
                                                         n_train,
                                                         n_vali)
        self.patch2d_mu = mu[0]
        self.patch2d_sigma = sigma[0]
        self.dense_mu = mu[1]
        self.dense_sigma = sigma[1]

    def save_stats(self, stats_path=None):
        super(Network2DPatchDense, self).save_stats(stats_path)
        self.stats = {}
        self.stats['patch2d_mu'] = self.patch2d_mu
        self.stats['patch2d_sigma'] = self.patch2d_sigma
        self.stats['dense_mu'] = self.dense_mu
        self.stats['dense_sigma'] = self.dense_sigma

        write_h5file(self.stats_path, self.stats, self.attrs)

    def load_stats(self, stats_path=None):
        super(Network2DPatchDense, self).load_stats(stats_path)
        self.patch2d_mu = self.stats['patch2d_mu']
        self.patch2d_sigma = self.stats['patch2d_sigma']
        self.dense_mu = self.stats['dense_mu']
        self.dense_sigma = self.stats['dense_sigma']


# Real networks below
class MLP(NetworkDense):
    def __init__(self, in_dim, out_size):
        super(MLP, self).__init__(in_dim, out_size)
        self.network_type = 'MLP'

    def build(self):
        super(MLP, self).build()
        model = Sequential()
        model.add(Dense(512, input_shape=(self.in_dim,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.out_size))
        model.add(Activation('softmax'))

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)
        self.model = model


class CnnTriplanar(Network2DPatch):
    def __init__(self, patch_size, n_channels, out_size):
        super(CnnTriplanar, self).__init__(patch_size, n_channels, out_size)
        self.network_type = 'CnnTriplanar'

    def build(self):
        Network2DPatch.build(self)
        if self.image_dim_ordering == 'tf':
            input_shape = (self.patch_size, self.patch_size, self.n_channels)
        else:
            input_shape = (self.n_channels, self.patch_size, self.patch_size)

        n_kernels_0 = 20
        kernel_size_0 = (5, 5)
        pool_size_0 = (2, 2)

        model = Sequential()
        model.add(Convolution2D(n_kernels_0,
                                kernel_size_0[0], kernel_size_0[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=pool_size_0))

        n_kernels_1 = 50
        kernel_size_1 = (5, 5)
        pool_size_1 = (2, 2)

        model.add(Activation('relu'))
        model.add(Convolution2D(n_kernels_1,
                                kernel_size_1[0], kernel_size_1[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))

        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dense(self.out_size))
        model.add(Activation('softmax'))

        self.model = model


class CnnTriplanarMultiset(Network2DPatch):
    """
    Triplanar with multiple scales

    One should concatenate the patches into channels,
    each scale takes 3 channels.
    """
    def __init__(self, patch_size, n_channels, out_size):
        super(CnnTriplanarMultiset, self).__init__(patch_size,
                                                   n_channels,
                                                   out_size)
        self.network_type = 'CnnTriplanarMultiset'

    def build(self):
        Network.build(self)
        if self.image_dim_ordering == 'tf':
            input_shape = (self.patch_size, self.patch_size, self.n_channels)
        else:
            input_shape = (self.n_channels, self.patch_size, self.patch_size)

        model = Sequential()
        n_kernels_0 = 20
        kernel_size_0 = (5, 5)
        pool_size_0 = (2, 2)

        model.add(Convolution2D(n_kernels_0,
                                kernel_size_0[0], kernel_size_0[1],
                                border_mode='valid',
                                init=self.init,
                                W_constraint=maxnorm(3),
                                input_shape=input_shape))
        model.add(PReLU(init='zero', weights=None))
        model.add(MaxPooling2D(pool_size=pool_size_0))
        model.add(Dropout(0.2))

        n_kernels_1 = 50
        kernel_size_1 = (5, 5)
        pool_size_1 = (2, 2)

        model.add(Convolution2D(n_kernels_1,
                                kernel_size_1[0], kernel_size_1[1],
                                border_mode='same',
                                init=self.init,
                                W_constraint=maxnorm(3)))
        model.add(PReLU(init='zero', weights=None))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(4000, init=self.init, W_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(PReLU(init='zero', weights=None))
        model.add(Dense(self.out_size))
        model.add(Activation('softmax'))

        self.model = model


class CnnTriplanarMultisetCentroids(Network2DPatchDense):
    """
    - Multi-scale triplanar as first input
    - Distance to centroid (vector) as second input.
    """
    def __init__(self, patch_size, n_channels, n_regions, out_size):
        super(CnnTriplanarMultisetCentroids, self).__init__(patch_size,
                                                            n_channels,
                                                            n_regions,
                                                            out_size)
        self.network_type = 'CnnTriplanarMultisetCentroids'
        self.n_regions = n_regions

    def build(self):
        super(CnnTriplanarMultisetCentroids, self).build()
        if self.image_dim_ordering == 'tf':
            input_shape = (self.patch_size, self.patch_size, self.n_channels)
        else:
            input_shape = (self.n_channels, self.patch_size, self.patch_size)

        # model for patch feature
        model = Sequential()
        n_kernels_0 = 20
        kernel_size_0 = (5, 5)
        pool_size_0 = (2, 2)

        model.add(Convolution2D(n_kernels_0,
                                kernel_size_0[0], kernel_size_0[1],
                                border_mode='valid',
                                init=self.init,
                                W_constraint=maxnorm(3),
                                input_shape=input_shape))
        model.add(PReLU(init='zero', weights=None))
        model.add(MaxPooling2D(pool_size=pool_size_0))
        model.add(Dropout(0.2))

        n_kernels_1 = 50
        kernel_size_1 = (5, 5)
        pool_size_1 = (2, 2)

        model.add(Convolution2D(n_kernels_1,
                                kernel_size_1[0], kernel_size_1[1],
                                init=self.init,
                                W_constraint=maxnorm(3)))
        model.add(PReLU(init='zero', weights=None))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(Dropout(0.2))
        model.add(Flatten())

        # model for distance to centroid
        model_centroid = Sequential()

        def identity(x):
            return x + 0

        # same number of layers in patch-based network
        model_centroid.add(Lambda(identity, input_shape=(self.n_regions,)))
        model_centroid.add(Activation('linear'))
        model_centroid.add(Activation('linear'))
        model_centroid.add(Activation('linear'))

        # concatenate them and add a fully-connceted layer
        model_merged = Sequential()
        model_merged.add(Merge([model, model_centroid],
                               mode='concat',
                               concat_axis=1))
        model_merged.add(Dense(4000, init=self.init, W_constraint=maxnorm(3)))
        model_merged.add(PReLU(init='zero', weights=None))
        model_merged.add(Dropout(0.5))
        model_merged.add(Dense(self.out_size))
        model_merged.add(Activation('softmax'))
        model_merged.compile(optimizer=self.optimizer,
                             loss=self.loss,
                             metrics=self.metrics)
        self.model = model_merged
