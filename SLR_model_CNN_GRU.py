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
# this is CNN GRU with Residual model, derived from the GRU over CNN legacy model
#----------------------------------
# (CNN GRU with Residual) differneces between GRU over CNN legacy
#----------------------------------:
# configurable CNN layer count (hand3, pose2)
# depthwise convolution
# CNN uses "same" padding
# removed pose first ever dense layer
# residual connection
#----------------------------------
# (GRU over CNN legacy) differences between one-hot model
#----------------------------------:
# replaced the last hidden dense layer with GRU
# changed CNN filter size into 128 
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
glorot_init = initializers.GlorotUniform()

# Hyperparamerters
# hand : conv3D
hand_filter_size = (48, 96, 64)
hand_kernel_size = ((9, 5, 5), (5,1,1), (5,3,3))
hand_activation = (None, "gelu", None)
hand_initializer = (glorot_init, he_init , glorot_init)
hand_padding = (('valid', 'same'), ('same', 'same'), ('same', 'same')) # temporal, spatial 
hand_stride = ((1,1,1), (2,1,1), (1,1,1))
# pose: conv2D
pose_filter_size = (48, 48)
pose_kernel_size = ((5,3), (3,3))
pose_activation = (None, None)
pose_initializer = (glorot_init , glorot_init)
pose_padding = ('valid', 'same') # for both temporal and spatial
pose_stride = (1 , 1)
# combined FC
GRU_unit_size = 256
# combined_dense2_size = 128
combined_output_size = 3000
# optimizer hyperparamerters
learning_rate = 0.005
cce_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
# bin_acc_metric = keras.metrics.BinaryAccuracy()
cat_acc_metric = keras.metrics.CategoricalAccuracy()

# serialize hyperparameters
# window_size = 17
# hop_dist = 10
# odd kernel size: 9, minimum=9
# if prev 'valid', 5 , stride=2,  minumum = 9+(5-1)*2
# if prev 'same', 5 , stride=2,  minumum = 0+(5-1)*2

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

class Separable_Conv2Plus1D(Model):
    def __init__(self, kernel_size, filters, strides = 1, padding_list = ('valid', 'valid'), activation = None, kernel_initializer = 'glorot_uniform'):
        """kernel_size is depth width height"""
        super().__init__()
        # wh_stride = (1, strides[1], strides[2])
        wh_stride = (strides[1], strides[2])
        t_stride = (strides[0],)
        # Spatial decomposition
        self.dc2d = layers.SeparableConv2D(filters=filters,
                      kernel_size=(kernel_size[1], kernel_size[2]), strides = wh_stride,
                      padding=padding_list[1], activation= activation, depthwise_initializer= kernel_initializer, pointwise_initializer=kernel_initializer)
        # Temporal decomposition
        self.c1d = layers.Conv1D(filters=filters, 
                      kernel_size=(kernel_size[0],), strides = t_stride,
                      padding=padding_list[0], activation= activation, kernel_initializer = kernel_initializer)

    def call(self, x):
        # input shape: (batch, time, h, w, channels)
        res_2d = []
        res_1d = []
        # print(x.shape)
        for i in range(x.shape[1]):
            # (batch, h, w, channels) per time
            # per_time = x[:,i]
            # print(i)
            # print(per_time.shape)
            res_2d.append(self.dc2d(x[:,i]))
        # res_2d: (time, batch, conv_h, conv_w, channels(same as the original input))
        x = tf.stack(res_2d, axis=1)
        # x = (batch, time,  conv_h, conv_w, channels)
        two_dim_shape = x.shape
        for h in range(two_dim_shape[2]):
            for w in range(two_dim_shape[3]):
                # (batch, time, channels) per features
                res_1d.append(self.c1d(x[:,:,h,w,:]))
        # res_1d: (h*w, batch, conv_time, filters)
        x = tf.stack(res_1d, axis=1)
        # x: (batch, h*w, conv_time, filters)
        x = layers.Permute((2,1,3))(x)
        # x: (batch, conv_time, h*w , filters)
        return layers.Reshape((x.shape[1], two_dim_shape[2], two_dim_shape[3], -1))(x)

class Project(layers.Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
    """
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

# # hand model
# class Conv2Plus1D_Network(Model):
#     # run it every 2 frames in order to give it 2 temporal stride
#     """takes lists of parameters for cnn layers, create list size number of cnn layers\n
#     all parameters with the name list have to be the same sized list on the axis 0\n
#         activation_list:
#             string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
#             on the layer but layer normalization is applied after the layer
#     - call:
#         input shape: (batch, time, h, w, channels)\n
#         output shape: (batch, convolved_time, convolved_h , convolved_w , filter_size)"""
#     def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding_list):
#         super().__init__()
#         self.layers_list = []
#         self.requires_training_arg = []
#         for i in range(len(kernel_size_list)):
#             if activation_list[i] == "layer_norm":
#                 conv = Conv2Plus1D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i], 
#                                    kernel_initializer= initializer_list[i], padding_list= padding_list[i])
#                 self.layers_list.append(conv)
#                 self.requires_training_arg.append(False)
#                 self.layers_list.append(layers.LayerNormalization(axis=(1,2,3)))
#                 self.requires_training_arg.append(True)
#             else:
#                 conv = Conv2Plus1D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i],
#                                    kernel_initializer = initializer_list[i], padding_list= padding_list[i], activation=activation_list[i])
#                 self.layers_list.append(conv)
#                 self.requires_training_arg.append(False)

#     def call(self, x, training= False):
#         for i,layer in enumerate(self.layers_list):
#             if self.requires_training_arg[i]:
#                 x = layer(x, training= training)
#             else:
#                 x = layer(x)
#         return x

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

# # pose model
# class Conv2D_Network(Model):
#     """takes lists of parameters for cnn layers, create list size number of cnn layers\n
#     all parameters with the name list have to be the same sized list on the axis 0\n\n
#         activation_list:
#             string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
#             on the layer but layer normalization is applied after the layer
#     - call:
#         input shape: (batch, time, features, channels)\n
#         output shape: (batch, convolved_time, convolved_features , filter_size)"""
#     def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding_list):
#         super().__init__()
#         self.layers_list = []
#         self.requires_training_arg = []
#         for i in range(len(kernel_size_list)):
#             if activation_list[i] == "layer_norm":
#                 conv = layers.Conv2D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i], 
#                                    kernel_initializer= initializer_list[i], padding= padding_list[i])
#                 self.layers_list.append(conv)
#                 self.requires_training_arg.append(False)
#                 self.layers_list.append(layers.LayerNormalization(axis=(1,2)))
#                 self.requires_training_arg.append(True)
#             else:
#                 conv = layers.Conv2D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i],
#                                    kernel_initializer = initializer_list[i], padding= padding_list[i], activation=activation_list[i])
#                 self.layers_list.append(conv)
#                 self.requires_training_arg.append(False)

#     def call(self, x, training= False):
#         for i,layer in enumerate(self.layers_list):
#             if self.requires_training_arg[i]:
#                 x = layer(x, training= training)
#             else:
#                 x = layer(x)
#         return x


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
        # self.left_hand_model = Conv2Plus1D_Network(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
        #                     activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
        #                     padding_list= hand_padding)
        # self.right_hand_model = Conv2Plus1D_Network(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
        #                     activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
        #                     padding_list= hand_padding)
        # self.pose_model = Conv2D_Network(kernel_size_list= pose_kernel_size, filters_list= pose_filter_size,
        #                     activation_list= pose_activation, initializer_list= pose_initializer, strides_list= pose_stride,
        #                     padding_list= pose_padding)
        self.left_hand_model = HandModel(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
                            padding_list= hand_padding)
        self.right_hand_model = HandModel(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride,
                            padding_list= hand_padding)
        self.pose_model = PoseModel(kernel_size_list= pose_kernel_size, filters_list= pose_filter_size,
                            activation_list= pose_activation, initializer_list= pose_initializer, strides_list= pose_stride,
                            padding_list= pose_padding)
        # self.gru = layers.GRU(GRU_unit_size, return_state= True)
        self.gru = layers.GRU(GRU_unit_size)
        # self.dense1 = layers.Dense(combined_dense1_size, activation='gelu', kernel_initializer = he_init)
        self.dense_out = layers.Dense(combined_output_size, activation='softmax')
        self.flat = layers.Flatten()
        self.proj_hand = Project(6400)
        self.proj_pose = Project(384)
        
    def flatten_residual_connection(self, original, result, residual_factor = 1.0):
        """ flattens the inputs and make residual connection
        """
        # print("orig orig:", original.shape)
        # print("result orig:", result.shape)
        original = self.flat(original)
        result = self.flat(result)
        # print("orig flat:", original.shape)
        # print("result flat:", result.shape)
        if result.shape[-1] == 6400:
            residual = self.proj_hand(original)
        elif result.shape[-1] == 384:
            residual = self.proj_pose(original)

        return layers.add([result, residual*residual_factor])
    
    def call(self, inputs, training= False):
        # inputs 0: L, 1: R, 2: Pose
        print(inputs)
        l_inputs, r_inputs, p_inputs = inputs
        # input shape: (batch(1) , window_count, window_size, features**)
        per_window = []
        for i in range(l_inputs.shape[1]):
            # ASSUMING LRP HAS THE SAME WINDOW COUNTS
            l_res = self.left_hand_model(l_inputs[:,i], training = training)
            # print("lh")
            l_res = self.flatten_residual_connection(l_inputs[:,i], l_res)
            # l_res = self.flat(l_res)
            
            r_res = self.right_hand_model(r_inputs[:,i], training = training)
            # print("rh")
            r_res = self.flatten_residual_connection(r_inputs[:,i], r_res)
            # r_res = self.flat(r_res)
            
            p_res = self.pose_model(p_inputs[:,i], training = training)
            # print("p")
            p_res = self.flatten_residual_connection(p_inputs[:,i], p_res)
            # p_res = self.flat(p_res)
            per_window.append(tf.concat([l_res, r_res, p_res], axis = 1))
            # concatenated shape: (batch, -1)
        x = tf.concat(per_window, axis=0)
        # x shape: (window_count, -1)
        x=tf.expand_dims(x,axis=0)
        # x shape: (batch(1), window_count, -1)
        # print(x.shape)
        
        x= self.gru(x, training=training)
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


# copied from CLGRU
def serialize(vid, stride = 1, window_size = 17, hop_length=10,  loss_weights = None, is_pose = False):
    """input vid shape: (frames, features**)\n
    ouput shape: (window_counts , ceil(window_size/stride) , features**)"""
    x_res = []
    weight_res = []
    window_count = 0
    # start is included
    # end is excluded
    start = 0
    end = start + window_size
    while end < len(vid):
        window = vid[start: end : stride]
        if is_pose:
            window = np.swapaxes(window,-1, -2)
        x_res.append(window)
        window_count += 1
        if loss_weights is not None:
            weight_res.append(loss_weights[end-1])
        start += hop_length
        end = start + window_size
    res = np.array(x_res)
    if loss_weights is not None:
        # temporary solution for batch size being none
        return np.expand_dims(res, axis=0), window_count, weight_res
    return np.expand_dims(res, axis=0), window_count

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

