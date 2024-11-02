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
# this is GRU over CNN model, derived from the one-hot model
#----------------------------------
# (GRU over CNN) differences between one-hot model
#----------------------------------:
# replaced the last hidden dense layer with GRU
# CNN temporal depth: fast 9, slow 5 frames
# CNN slidable window size: 9, 5 frames (no sliding)
# GRU overlap : 3, 2 frames each
# hop: 6 , 3 frames each
# serialize() output changed accordingly ^
# CNN depth in kernel_size changed accordingly ^
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
hand_filter_size = 1
hand_kernel_size = (9, 3, 3)
hand_stride = (2,1,1)
# pose: conv2D
pose_filter_size = 1
pose_dense_size = 3
pose_kernel_size = (5,3)
pose_stride = (1,1)
# combined FC
GRU_unit_size = 256
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
    """input shape: (batch, time, h, w, channels)
        output shape: (batch, convolved_time, convolved_h * convolved_w * filter_size)"""
    def __init__(self, kernel_size = (9,3,3), filters= 1, strides = (2,1,1)):
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
    def __init__(self, kernel_size = (9,3), filters = 1, dense_size = 3):
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
        # shape: (batch, conv_time(which is 1), 1, filter_size) -> (batch, filter_size) -> (batch, xyz_ch, filter_size)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        z = tf.squeeze(z)
        temp = tf.stack([x, y, z], axis=1)
        temp = layers.Reshape((conv_shape[1], 3 * conv_shape[3]))(temp) # 3 from x y z 3 channels
        return self.ln(temp, training = training)
        


# the main model class
class SLRModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.right_hand_model = HandModel(kernel_size = hand_kernel_size, filters = hand_filter_size, strides=hand_stride)
        self.pose_model = PoseModel(kernel_size = pose_kernel_size, filters=pose_filter_size, dense_size = pose_dense_size)
        self.gru = layers.GRU(GRU_unit_size, return_state= True)
        # self.dense1 = layers.Dense(combined_dense1_size, activation='gelu', kernel_initializer = he_init)
        self.dense_out = layers.Dense(combined_output_size, activation='softmax')
        self.flat = layers.Flatten()

    def call(self, inputs, states= None, return_states = False, training= False):
        # inputs 0: L, 1: R, 2: Pose
        l_inputs, r_inputs, p_inputs = inputs
        l_res = self.left_hand_model(l_inputs, training = training)
        r_res = self.right_hand_model(r_inputs, training = training)
        p_res = self.pose_model(p_inputs, training = training)
        l_res = self.flat(l_res)
        r_res = self.flat(r_res)
        p_res = self.flat(p_res)
        x = tf.concat([l_res, r_res, p_res], axis = 1)
        # current_shape: (batch, hand_output_size * 2 + pose_output_size)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        if return_states:
            return self.dense_out(x), states
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
    global model
    """reinitialize the model, reloads and resets the model"""
    keras.backend.clear_session()
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


def serialize(vids, stride = 1, loss_weights_list = None):
    """input shape: (load_size, frames)\n
    ouput shape: (load_size, input_seq_size, 9 or 5 depending on stride setting, frames)"""
    each_size = []
    x_res = []
    weight_res = []
    for i, vid in enumerate(vids):
        window_count = 0
        start = 0
        while (start + 9) < len(vid): # check if the end of this window goes over the end of the video
            x_res.append(vid[start: start+9: stride])
            window_count += 1
            if loss_weights_list is not None:
                weight_res.append(loss_weights_list[i][start+9-1])
            start += 6
            # ^ hop frames
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

