import tensorflow as tf
import keras
from keras import Model
from keras import layers
import numpy as np
from keras import optimizers
from keras import initializers



#------------------------------------------------------------------------------------#
##  THIS IS A TRAINING MODEL, IMPLEMENTION DETAIL MAY DIFFER FROM PRODUCTION MODEL  ##
#------------------------------------------------------------------------------------#

# NOTE
# this is CNN_LSTM + GRU  model, derived from the one-hot model
#----------------------------------
# (CNN_LSTM + GRU) differences between one-hot model
#----------------------------------:
# removed pose firstever FC layer
# increased filtersize into 128
# replaced hand, pose CNN layer into 2 CNN_LSTMs
# CNN_LSTM used padding = "same"
# SLRmodel class uses GRU to combine CNN_LSTM results
#---------------------------------
# (one-hot) difference between legacy model
#----------------------------------: 
# one hot encoding
# ^ changed loss and activation function accordingly
# removed one dense layer from the final part
# increased learning rate
#--------------------

# global variables
model = None
checkpoint_manager = None
checkpoint = None
he_init = initializers.HeUniform()

# Hyperparamerters
# hand : conv3D
hand_filter_size = 128
hand_kernel_size = (3, 3)
hand_stride = 1
# pose: conv2D
pose_filter_size = 128
pose_kernel_size = 3
pose_stride = 1
# combined GRU
gru_units = 256
# combined_dense2_size = 128
combined_output_size = 3000
# optimizer
learning_rate = 0.0005
cce_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
# bin_acc_metric = keras.metrics.BinaryAccuracy()
cat_acc_metric = keras.metrics.CategoricalAccuracy()


# custom layer definition
class Conv2Plus1D(layers.Layer):
    def __init__(self, kernel_size, filters = 1, strides = (1,1,1), padding = 'valid'):
        """kernel_size is depth width height"""
        super().__init__()
        wh_stride = (1, strides[1], strides[2])
        t_stride = (strides[0], 1, 1)
        self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]), strides = wh_stride,
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1), strides = t_stride,
                      padding=padding)
        ])

    def call(self, x):
        return self.seq(x)


# hand model
class HandModel(Model):
    # run it every 2 frames in order to give it 2 temporal stride
    """input shape: (batch, timesteps, h, w, channels)\n
        output shape: (batch, timesteps, convolved_h * convolved_w * filters)"""
    def __init__(self, kernel_size, filters, strides):
        super().__init__()
        self.filters = filters
        self.clstm = layers.ConvLSTM2D(filters = filters, kernel_size = kernel_size, strides = strides,
                                       padding="same", dropout=0.1, recurrent_dropout=0.05, return_sequences=True)
        self.ln = layers.LayerNormalization()

    def call(self, x, training= False):
        x = self.clstm(x)
        conv_shape = x.shape
        # current shape: (batch, timesteps, convolved_h, convolved_w, filters)
        # x = tf.squeeze(x)
        x = layers.Reshape((conv_shape[1], conv_shape[2] * conv_shape[3] * self.filters))(x)
        return self.ln(x, training= training)



# pose model
class PoseModel(Model):
    """input shape: (batch, timesteps, channel, features)\n
        output shape: (batch, timesteps, filters * convolved_result)"""
    def __init__(self, kernel_size, filters):
        super().__init__()
        self.filters = filters
        self.clstm = layers.ConvLSTM1D(filters = filters, kernel_size = kernel_size, padding="same", data_format="channels_first",
                                        dropout=0.1, recurrent_dropout=0.05, return_sequences=True)
        self.ln = layers.LayerNormalization()

    def call(self, x, training= False):
        # input shape:  batch time channel features
        # output shape: (batch, timesteps, filters, convolved_result)
        x = self.clstm(x)
        conv_shape = x.shape
        x = layers.Reshape((conv_shape[1], self.filters * conv_shape[3]))(x)
        return self.ln(x, training = training)
        


# the main model class
class SLRModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.right_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.pose_model = PoseModel(kernel_size = pose_kernel_size, filters=pose_filter_size)
        self.gru = layers.GRU(gru_units)
        self.dense_out = layers.Dense(combined_output_size, activation='softmax')
        self.flat = layers.Flatten()

    def call(self, inputs, training= False):
        # inputs 0: L, 1: R, 2: Pose
        l_inputs, r_inputs, p_inputs = inputs
        l_res = self.left_hand_model(l_inputs, training=training)
        r_res = self.right_hand_model(r_inputs, training=training)
        p_res = self.pose_model(p_inputs, training=training)
        # l_res = self.flat(l_res)
        # r_res = self.flat(r_res)
        # p_res = self.flat(p_res)
        x = tf.concat([l_res, r_res, p_res], axis = 2)
        # current_shape: (batch, timesteps, hand_output_size * 2 + pose_output_size)
        x = self.gru(x, training=training)
        return self.dense_out(x)
        



model = SLRModel()
optimizer = optimizers.Adam(learning_rate = learning_rate)
# model.build((1,))
model.compile(optimizer = optimizer, loss=cce_loss, metrics=[cat_acc_metric])

def get_model():
    return model

# def model_summary():
#     return model.summary()

def reinit_model(run_eagerly = False):
    """reinitialize the model, reloads and resets the model"""
    model = SLRModel()
    optimizer = optimizers.Adam(learning_rate = learning_rate)
    # model.build((1,))
    model.compile(optimizer = optimizer, loss=cce_loss, metrics=[cat_acc_metric], run_eagerly = run_eagerly)
    return model


# utility functions
def encode_onehot2d(num_arr):
    "encode a number array into onehot 2d array"
    return tf.raw_ops.OneHot(indices = num_arr, depth = combined_output_size, on_value = 1.0, off_value = 0.0)

def decode_onehot2d(onehot_2d):
    "decode a onehot 2d array into a number array"
    return tf.argmax(onehot_2d, axis=-1)

# TODO change this to accomodate CNNLSTM + GRU
def serialize(vids, stride = 1, loss_weights_list = None):
    """input shape: (load_size, frames)\n
    ouput shape: (load_size, input_seq_size, 63 or 32, frames)"""
    each_size = []
    x_res = []
    weight_res = []
    for i, vid in enumerate(vids):
        window_count = 0
        start = 0
        while (start + 63) < len(vid):
            x_res.append(vid[start: start+63: stride])
            window_count += 1
            if loss_weights_list is not None:
                weight_res.append(loss_weights_list[i][start+63-1])
            start += 6
        each_size.append(window_count)
    if loss_weights_list is not None:
        return np.array(x_res), each_size, weight_res
    return np.array(x_res), each_size

def load_model(file_path):
    global model
    model = keras.models.load_model(file_path)
    return model

def convert_to_dataset(x_train, y_train, batch_size = 1, sample_weights=None):
    if sample_weights is not None:
        dataset_raw = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weights))
    else:
        dataset_raw = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset_raw.shuffle(buffer_size=len(x_train)).padded_batch(batch_size, drop_remainder=True)
    return dataset

# the part that doesnt get executed when imported
if __name__ == "__main__":
    pass

