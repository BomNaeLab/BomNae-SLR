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
# difference between legacy model
#----------------------------------
# one hot encoding
# ^ changed loss and activation function accordingly
# removed one dense layer from the final part
# increased learning rate 

# global variables
model = None
checkpoint_manager = None
checkpoint = None
he_init = initializers.HeUniform()
glorot_init = initializers.GlorotUniform()

# Hyperparamerters
# # hand : conv3D
# hand_filter_size = 1
# hand_kernel_size = (33, 3, 3)
# hand_stride = (2,1,1)

# hand : conv3D : from CNNGRU
hand_filter_size = (64, 128, 96)
hand_kernel_size = ((13, 5, 5), (5,1,1), (5,3,3))
hand_activation = (None, "gelu", None)
hand_initializer = (glorot_init, he_init , glorot_init)
hand_padding = (('valid', 'same'), ('same', 'same'), ('same', 'same')) # temporal, spatial 
hand_stride = ((1,1,1), (2,1,1), (1,1,1))
# pose: conv2D : from CNNGRU
pose_filter_size = (64, 64)
pose_kernel_size = ((7,3), (3,3))
pose_activation = (None, None)
pose_initializer = (glorot_init , glorot_init)
pose_padding = ('valid', 'same') # for both temporal and spatial
pose_stride = (1 , 1)

# # pose: conv2D
# pose_filter_size = 1
# pose_dense_size = 3
# pose_kernel_size = (17,3)
# pose_stride = (1,1)
# combined FC
combined_dense1_size = 512
combined_output_size = 2363
# combined_output_size = 1880

# optimizer
learning_rate = 0.00005
cce_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
# bin_acc_metric = keras.metrics.BinaryAccuracy()
cat_acc_metric = keras.metrics.CategoricalAccuracy()


# custom layer definition
class Conv2Plus1D(layers.Layer):
    def __init__(self, kernel_size, filters, strides = 1, padding_list = ('valid', 'valid'), activation = None, kernel_initializer = 'glorot_uniform'):
        """kernel_size is depth width height"""
        super().__init__()
        wh_stride = (1, strides[1], strides[2])
        t_stride = (strides[0], 1, 1)
        self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]), strides = wh_stride,
                      padding=padding_list[1], activation= activation, kernel_initializer = kernel_initializer),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1), strides = t_stride,
                      padding=padding_list[0], activation= activation, kernel_initializer = kernel_initializer)
        ])

    def call(self, x):
        return self.seq(x)

# class Project(layers.Layer):
#     """
#     Project certain dimensions of the tensor as the data is passed through different 
#     sized filters and downsampled. 
#     """
#     def __init__(self, units):
#         super().__init__()
#         self.seq = keras.Sequential([
#             layers.Dense(units),
#             layers.LayerNormalization()
#         ])

#     def call(self, x):
#         return self.seq(x)

# hand model
class HandModel(Model):
    # run it every 2 frames in order to give it 2 temporal stride
    """takes lists of parameters for cnn layers, create list size number of cnn layers\n
    all parameters with the name list have to be the same sized list on the axis 0\n
        activation_list:
            string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
            on the layer but layer normalization is applied after the layer
    - call:
        input shape: (batch, time, h, w, channels)\n
        output shape: (batch, convolved_time, convolved_h , convolved_w , filter_size)"""
    def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding_list):
        super().__init__()
        self.conv0 =  Conv2Plus1D(kernel_size = kernel_size_list[0], filters= filters_list[0], strides = strides_list[0], 
                                   kernel_initializer= initializer_list[0], padding_list= padding_list[0])
        self.conv1 =  Conv2Plus1D(kernel_size = kernel_size_list[1], filters= filters_list[1], strides = strides_list[1], 
                                   kernel_initializer= initializer_list[1], padding_list= padding_list[1], activation=activation_list[1])
        self.conv2 =  Conv2Plus1D(kernel_size = kernel_size_list[2], filters= filters_list[2], strides = strides_list[2], 
                                   kernel_initializer= initializer_list[2], padding_list= padding_list[2])
        self.lnorm0 = layers.LayerNormalization(axis=(1,2,3))
        self.lnorm = layers.LayerNormalization(axis=(1,2,3))

    def call(self, x, training= False):
        x = self.conv0(x)
        x = self.lnorm0(x, training = training)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.lnorm(x, training = training)
        return x



# pose model
class PoseModel(Model):
    """takes lists of parameters for cnn layers, create list size number of cnn layers\n
    all parameters with the name list have to be the same sized list on the axis 0\n\n
        activation_list:
            string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
            on the layer but layer normalization is applied after the layer
    - call:
        input shape: (batch, time, features, channels)\n
        output shape: (batch, convolved_time, convolved_features , filter_size)"""
    def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding_list):
        super().__init__()

        self.conv0 = layers.Conv2D(kernel_size = kernel_size_list[0], filters= filters_list[0], strides = strides_list[0],
                                   kernel_initializer = initializer_list[0], padding= padding_list[0], activation=activation_list[0])
        self.conv1 = layers.Conv2D(kernel_size = kernel_size_list[1], filters= filters_list[1], strides = strides_list[1],
                                   kernel_initializer = initializer_list[1], padding= padding_list[1], activation=activation_list[1])
        self.lnorm = layers.LayerNormalization(axis=(1,2,3))

    def call(self, x, training= False):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.lnorm(x, training = training)
        return x
        


# the main model class
class SLRModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_hand_model = HandModel(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
                            padding_list= hand_padding)
        self.right_hand_model = HandModel(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
                            padding_list= hand_padding)
        self.pose_model = PoseModel(kernel_size_list= pose_kernel_size, filters_list= pose_filter_size,
                            activation_list= pose_activation, initializer_list= pose_initializer, strides_list= pose_stride,
                            padding_list= pose_padding)
        self.dense1 = layers.Dense(combined_dense1_size, activation='gelu', kernel_initializer = he_init)
        self.dense_out = layers.Dense(combined_output_size, activation='softmax')
        self.flat = layers.Flatten()
        # self.proj_hand = Project(6400)
        # self.proj_pose = Project(1920)

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


def serialize(vids, stride = 1, loss_weights_list = None, is_pose = False):
    """input shape: (load_size, frames)\n
    ouput shape: (load_size, input_seq_size, 63 or 32, frames)"""
    each_size = []
    x_res = []
    weight_res = []
    for i, vid in enumerate(vids):
        window_count = 0
        start = 0
        end = start + 63 # window_size
        while end < len(vid):
            window = vid[start: end: stride]
            if is_pose:
                window = np.swapaxes(window,-1, -2)
            x_res.append(window)
            window_count += 1
            if loss_weights_list is not None:
                weight_res.append(loss_weights_list[i][end-1])
            start += 6 # hop_length
            end += 6 # hop_length
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

