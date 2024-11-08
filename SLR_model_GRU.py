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
# this is GRU over CNN "per video" model, derived from the GRU over CNN legacy model
#----------------------------------
# (GRU over CNN "per video") differneces between GRU over CNN legacy
#----------------------------------:
# configurable CNN layer count (hand3, pose2)
# CNN uses "same" padding
# removed pose first ever dense layer
# TODO consideration: residual connection?
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

# Hyperparamerters
# hand : conv3D
hand_filter_size = (128, 128, 128)
hand_kernel_size = ((9, 3, 3), (9,3,3), (9,3,3)) #TODO
hand_activation = ("layer_norm", "gelu", None)
hand_initializer = ('glorot_uniform', he_init , 'glorot_uniform')
hand_stride = ((2,1,1), (2,1,1), (2,1,1))
# pose: conv2D
pose_filter_size = (128, 128)
pose_kernel_size = ((5,3), (5,3)) #TODO
pose_activation = ("gelu", None)
pose_initializer = (he_init , 'glorot_uniform')
pose_stride = (1 , 1)
# combined FC
GRU_unit_size = 256
# combined_dense2_size = 128
combined_output_size = 3000
# optimizer hyperparamerters
learning_rate = 0.0005
cce_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
# bin_acc_metric = keras.metrics.BinaryAccuracy()
cat_acc_metric = keras.metrics.CategoricalAccuracy()


# custom layer definition
class Conv2Plus1D(layers.Layer):
    def __init__(self, kernel_size, filters, strides = 1, padding = 'valid', activation = None, kernel_initializer = 'glorot_uniform'):
        """kernel_size is depth width height"""
        super().__init__()
        wh_stride = (1, strides[1], strides[2])
        t_stride = (strides[0], 1, 1)
        self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]), strides = wh_stride,
                      padding=padding, activation= activation, kernel_initializer = kernel_initializer),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1), strides = t_stride,
                      padding=padding, activation= activation, kernel_initializer = kernel_initializer)
        ])

    def call(self, x):
        return self.seq(x)


# hand model
class Conv2Plus1D_Network(Model):
    # run it every 2 frames in order to give it 2 temporal stride
    """takes lists of parameters for cnn layers, create list size number of cnn layers\n
    all parameters with the name list have to be the same sized list on the axis 0\n
        activation_list:
            string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
            on the layer but layer normalization is applied after the layer
    - call:
        input shape: (batch, time, h, w, channels)\n
        output shape: (batch, convolved_time, convolved_h , convolved_w , filter_size)"""
    def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding = 'valid'):
        super().__init__()
        self.layers_list = []
        self.requires_training_arg = []
        for i in range(len(kernel_size_list)):
            if activation_list[i] is "layer_norm":
                conv = Conv2Plus1D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i], 
                                   kernel_initializer= initializer_list[i], padding=padding)
                self.layers_list.append(conv)
                self.requires_training_arg.append(False)
                self.layers_list.append(layers.LayerNormalization(axis=(1,2,3)))
                self.requires_training_arg.append(True)
            else:
                conv = Conv2Plus1D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i],
                                   kernel_initializer = initializer_list[i], padding=padding, activation=activation_list[i])
                self.layers_list.append(conv)
                self.requires_training_arg.append(False)

    def call(self, x, training= False):
        for i,layer in enumerate(self.layers_list):
            if self.requires_training_arg[i]:
                x = layer(x, training= training)
            else:
                x = layer(x)
        return x

# pose model
class Conv2D_Network(Model):
    """takes lists of parameters for cnn layers, create list size number of cnn layers\n
    all parameters with the name list have to be the same sized list on the axis 0\n\n
        activation_list:
            string, \"layer_norm\" or any activation function, if its \"layer_norm\", there is no activation
            on the layer but layer normalization is applied after the layer
    - call:
        input shape: (batch, time, features, channels)\n
        output shape: (batch, convolved_time, convolved_features , filter_size)"""
    def __init__(self, kernel_size_list, filters_list,  activation_list, initializer_list, strides_list, padding = 'valid'):
        super().__init__()
        self.layers_list = []
        self.requires_training_arg = []
        for i in range(len(kernel_size_list)):
            if activation_list[i] is "layer_norm":
                conv = layers.Conv2D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i], 
                                   kernel_initializer= initializer_list[i], padding=padding)
                self.layers_list.append(conv)
                self.requires_training_arg.append(False)
                self.layers_list.append(layers.LayerNormalization(axis=(1,2)))
                self.requires_training_arg.append(True)
            else:
                conv = layers.Conv2D(kernel_size = kernel_size_list[i], filters= filters_list[i], strides = strides_list[i],
                                   kernel_initializer = initializer_list[i], padding=padding, activation=activation_list[i])
                self.layers_list.append(conv)
                self.requires_training_arg.append(False)

    def call(self, x, training= False):
        for i,layer in enumerate(self.layers_list):
            if self.requires_training_arg[i]:
                x = layer(x, training= training)
            else:
                x = layer(x)
        return x
        


# the main model class
class SLRModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_hand_model = Conv2Plus1D_Network(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride)
        self.right_hand_model = Conv2Plus1D_Network(kernel_size_list= hand_kernel_size, filters_list= hand_filter_size,
                            activation_list= hand_activation, initializer_list= hand_initializer, strides_list= hand_stride)
        self.pose_model = Conv2D_Network(kernel_size_list= pose_kernel_size, filters_list= pose_filter_size,
                            activation_list= pose_activation, initializer_list= pose_initializer, strides_list= pose_stride)
        # self.gru = layers.GRU(GRU_unit_size, return_state= True)
        self.gru = layers.GRU(GRU_unit_size)
        # self.dense1 = layers.Dense(combined_dense1_size, activation='gelu', kernel_initializer = he_init)
        self.dense_out = layers.Dense(combined_output_size, activation='softmax')
        self.flat = layers.Flatten()
    
    def call(self, inputs, training= False):
        #TODO per video loop integration for CNNs
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
        x=tf.expand_dims(x,axis=1)
        
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


#TODO per video + pose channel transpose
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

