import tensorflow
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from custom_layers import CenterLossLayer, ModulatedAttention

def nonlinear_squashing(x):
    return (tensorflow.norm(x,2)/(1 + tensorflow.norm(x,2)))*(x/tensorflow.norm(x,2))


def get_model(num_outputs = 10, oltr=False, modulate = False):
    """
    Open-set Long-Tail Classifier
    """
    inputs = tensorflow.keras.Input(shape=(28,28,1), name = 'input_1')
    x = inputs
    num_feats = [32, 64]
    
    for num in num_feats:
        x = tensorflow.keras.layers.Conv2D(num, (3,3), padding = "same")(x)
        x = tensorflow.keras.layers.LeakyReLU(0.2)(x)
        x = tensorflow.keras.layers.MaxPool2D((2,2))(x)
    
    if modulate:
        x = ModulatedAttention(pool = True)(x)
    
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    
    feats = tensorflow.keras.layers.Dense(64,  name='feats')(x)
    feats = tensorflow.keras.layers.LeakyReLU(0.2)(feats)
    
    outs = []
    ins = []
    
    ins.append(inputs)
    
    if oltr:
        
        feats = tensorflow.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(feats)
        
        cls_ids = tensorflow.keras.Input(shape=(num_outputs), name = 'input_2')

        cl_func = CenterLossLayer(alpha = 0.5, embedding_shape = feats.shape[1], num_cls = num_outputs)

        center_loss, centers = cl_func([feats, cls_ids])
        
        # computing the reachability
        reachability = tensorflow.math.reduce_min(tensorflow.sqrt(tensorflow.math.reduce_sum(tensorflow.square(tensorflow.math.subtract(centers, tensorflow.expand_dims(feats, axis=1))), axis=-1)), axis=1, keepdims= True)
        
        # reachability (just to name it and access after)
        reachability = tensorflow.keras.layers.Lambda(lambda x: x, name = "reachability")(reachability)
        
        # hallucination
        o = tensorflow.keras.layers.Dense(centers.shape[0], activation = 'softmax',  name = 'hal')(feats)
        v_memory = K.dot(o,centers) #N * d
        
        # reachability (just to name it and access after)
        v_memory = tensorflow.keras.layers.Lambda(lambda x: x, name = "memory")(v_memory)
        
        # Concept Selector
        e = tensorflow.keras.layers.Dense(centers.shape[1], activation = 'tanh',  name = 'sel')(feats) # N x d
        
        # v_meta Equation
        v_meta = (e*v_memory + feats)*(1.0/reachability)
        
        # nonlinear squashing
        v_meta = tensorflow.keras.layers.Lambda(lambda x: nonlinear_squashing(x), name = "v_meta")(v_meta)
        
        output = tensorflow.keras.layers.Dense(num_outputs, activation = 'softmax', use_bias= False, name = 'cls', kernel_regularizer='l2')(v_meta)
        
        outs.append(output)
        ins.append(cls_ids)
        outs.append(center_loss)

    else:
        output = tensorflow.keras.layers.Dense(num_outputs, activation = 'softmax', name = 'cls')(feats)
        outs.append(output)
        
    
    return tensorflow.keras.Model(inputs=ins, outputs=outs, name = 'mnist_oltr')

