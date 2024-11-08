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
# model detail
#
#

# global variables
model = None
checkpoint_manager = None
checkpoint = None
he_init = initializers.HeUniform()

# Hyperparamerters
# hand : conv3D
hand_filter_size = 1
hand_kernel_size = (33, 3, 3)
hand_stride = (2,1,1)
# pose: conv2D
pose_filter_size = 1
pose_dense_size = 3
pose_kernel_size = (17,3)
pose_stride = (1,1)
# combined FC
combined_dense1_size = 256
combined_dense2_size = 128
combined_output_size = 12
# optimizer
learning_rate = 0.0001
bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)
bin_acc_metric = keras.metrics.BinaryAccuracy()


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
    """input shape: (batch, time, h, w, channels)
        output shape: (batch, convolved_time, convolved_h * convolved_w * filter_size)"""
    def __init__(self, kernel_size = (33,3,3), filters= 1, strides = (2,1,1)):
        super().__init__()
        self.filters = filters
        self.conv21 = Conv2Plus1D(kernel_size = kernel_size, filters= filters, strides = strides)
        self.ln = layers.LayerNormalization()

    def call(self, x, training= False):
        x = self.conv21(x)
        conv_shape = x.shape
        # current shape: (batch, convolved_time, convolved_h, convolved_w, filter_size)
        x = tf.squeeze(x)
        x = layers.Reshape((conv_shape[1], conv_shape[2] * conv_shape[3] * self.filters))(x)
        return self.ln(x, training= training)



# pose model
class PoseModel(Model):
    """input shape: (batch, time, channel, features)\n
        output shape: (batch, convolved_time, xyz_channel * filter_size)"""
    def __init__(self, kernel_size = (17,3), filters = 1, dense_size = 3):
        super().__init__()
        self.dense_size = dense_size
        self.dense_td = layers.TimeDistributed(layers.Dense(dense_size, activation='relu', kernel_initializer= he_init))
        self.conv2_x = layers.Conv2D(filters=filters, kernel_size = kernel_size)
        self.conv2_y = layers.Conv2D(filters=filters, kernel_size = kernel_size)
        self.conv2_z = layers.Conv2D(filters=filters, kernel_size = kernel_size)
        self.ln = layers.LayerNormalization()

    def call(self, input, training= False):
        # input shape:  batch time channel features
        # output shape: batch channel conv_result
        temp = self.dense_td(input, training = training)
        # time channel FC_result -> channel time FC_result
        temp = layers.Permute((2, 1, 3))(temp)
        # below force adds required "channel" input for 2dCNN (not to confuse with xyz channel)
        temp = tf.expand_dims(temp, axis=4)
        # current shape: xyz_channel, time, FC_result, 1
        xyz = tf.split(temp, 3, axis = 1)
        x = self.conv2_x(tf.squeeze(xyz[0], axis= 1))
        y = self.conv2_y(tf.squeeze(xyz[1], axis= 1))
        z = self.conv2_z(tf.squeeze(xyz[2], axis= 1))
        conv_shape = x.shape
        # shape: (batch, conv_time, 1, filter_size) -> (batch, conv_time, filter_size) -> (batch, conv_time, xyz_ch, filter_size)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        z = tf.squeeze(z)
        temp = tf.stack([x, y, z], axis=2)
        temp = layers.Reshape((conv_shape[1], 3 * conv_shape[3]))(temp) # 3 from x y z 3 channels
        return self.ln(temp, training = training)
        


# the main model class
class SLRModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.right_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.pose_model = PoseModel(kernel_size = pose_kernel_size, filters=pose_filter_size, dense_size = pose_dense_size)
        self.dense1 = layers.Dense(combined_dense1_size, activation='gelu', kernel_initializer = he_init)
        self.dense2 = layers.Dense(combined_dense2_size, activation='gelu', kernel_initializer = he_init)
        # TODO 4096 max, check EXACT data size
        self.dense_out = layers.Dense(combined_output_size, activation='sigmoid')
        self.flat = layers.Flatten()

    def call(self, inputs):
        # inputs 0: L, 1: R, 2: Pose
        l_inputs, r_inputs, p_inputs = inputs
        l_res = self.left_hand_model(l_inputs)
        r_res = self.right_hand_model(r_inputs)
        p_res = self.pose_model(p_inputs)
        l_res = self.flat(l_res)
        r_res = self.flat(r_res)
        p_res = self.flat(p_res)
        x = tf.concat([l_res, r_res, p_res], axis = 1)
        # current_shape: (batch, hand_output_size * 2 + pose_output_size)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense_out(x)
        



model = SLRModel()
optimizer = optimizers.Adam(learning_rate = learning_rate)
# model.build((1,))
model.compile(optimizer = optimizer, loss=bce_loss, metrics=[bin_acc_metric])

def get_model():
    return model

# def model_summary():
#     return model.summary()

def reinit_model(run_eagerly = False):
    """reinitialize the model, reloads and resets the model"""
    model = SLRModel()
    optimizer = optimizers.Adam(learning_rate = learning_rate)
    # model.build((1,))
    model.compile(optimizer = optimizer, loss=bce_loss, metrics=[bin_acc_metric], run_eagerly = run_eagerly)
    return model


# utility functions

def bin2num(bin_arr):
    # big endian
    # lower index > higher exponent
    num = 0
    for val in bin_arr:
        num = (num << 1) | val
    return num

def bin2d_to_num(bin_arr_arr):
    # big endian
    # lower index > higher exponent
    num_arr = []
    for bin_arr in bin_arr_arr:
        num = 0
        for val in bin_arr:
            num = (num << 1) | val
        num_arr.append(num)
    return num_arr

# def num2bin(num, arr_len = -1):
#     # big endian
#     # lower index > higher exponent
#     bin_arr = [int(x) for x in bin(num)[2:]]
#     diff = arr_len - len(bin_arr)
#     if diff < 1:
#         return bin_arr
#     else:
#         pad = [0] * diff
#         return pad+bin_arr
    
def num_arr2bin(num_arr, out_len = -1):
    # big endian
    # lower index > higher exponent
    res = []
    for num in num_arr:
        str_list = list(np.binary_repr(num).zfill(out_len))
        bin_arr = np.array(str_list, dtype=np.float32)
        res.append(bin_arr)
    return np.array(res)

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

