import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv2D
from keras.models import Model


#takes n_actions,memory_length,scope,height, width
def build_model(n_actions, memory_length, scope,h,w):

    # I only have a CPU to train on 
    # If you have better hardware, set to /gpu:0
    # Or if you have a cluster of gpus, set up accordingly

    with tf.device("/cpu:0"):
        with tf.name_scope(scope):

            #Model as specified in mnih et al., "Asynchronous Methods for Deep Reinforcement Learning"
            #Also specified in requirements 

            #s for state
            s = tf.placeholder(tf.float32, [None, memory_length, w, h], name="s")
            inputs = Input(shape=(memory_length, w, h,))
            model = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', padding='same', data_format='channels_first')(inputs)
            model = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same', data_format='channels_first')(model)
            model = Flatten()(model)
            model = Dense(256, activation='relu')(model)
            outputs = Dense(n_actions)(model)
     
            m = Model(inputs=inputs, outputs=outputs)
        
    return s, m