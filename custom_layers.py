from tensorflow.keras import layers
import tensorflow
import tensorflow.keras.backend as K


class CenterLossLayer(tensorflow.keras.layers.Layer):

    def __init__(self,alpha, embedding_shape, num_cls, **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.embedding = embedding_shape
        self.num_cls = num_cls

    def build(self,input_shape):
        self.centers=self.add_weight(name="centers",
                                    shape=(self.num_cls,self.embedding),
                                    initializer="uniform",
                                    trainable=False)
        super().build(input_shape)

    def call(self,x,mask=None):

        delta_centers=K.dot(K.transpose(x[1]),K.dot(x[1],self.centers)-x[0])
        centers_count=K.sum(K.transpose(x[1]),axis=-1,keepdims=True)+1
        delta_centers/=centers_count
        
        new_centers=self.centers-self.alpha*delta_centers
        self.add_update((self.centers,new_centers),x)

        self.result=x[0]-K.dot(x[1],self.centers)
        
        self.result=K.sum(self.result**2,axis=1,keepdims=True)
        
        return self.result, self.centers

    def compute_output_shape(self,input_shape):
        return K.int_shape(self.result)

class ModulatedAttention(tensorflow.keras.layers.Layer):
    def __init__(self, pool = False):
        super(ModulatedAttention, self).__init__()

        self.pool = pool
        
    def build(self, input_shape):
        
        self.conv_alpha = tensorflow.keras.layers.Conv2D(int(input_shape[-1]/2), (1,1))
        self.conv_theta = tensorflow.keras.layers.Conv2D(int(input_shape[-1]/2), (1,1))
        self.conv_phi = tensorflow.keras.layers.Conv2D(int(input_shape[-1]/2), (1,1))
        self.conv_mask = tensorflow.keras.layers.Conv2D(input_shape[-1], (1,1), use_bias= False)
        
        self.max_pool = tensorflow.keras.layers.MaxPool2D((2,2))
        self.flatten = tensorflow.keras.layers.Flatten()
        
        self.dense = tensorflow.keras.layers.Dense(input_shape[1]*input_shape[2], activation = "softmax")
        
    def call(self, inputs):
        
        # Modulated Attention
        theta = self.conv_theta(inputs)
        phi = self.conv_phi(inputs)
        alpha = self.conv_alpha(inputs)
        
        if self.pool:
            phi = self.max_pool(phi)
            alpha = self.max_pool(alpha)
        
        alpha = tensorflow.reshape(alpha, [-1, alpha.shape[-1]])
        theta = tensorflow.reshape(theta, [-1, theta.shape[-1]])
        phi = tensorflow.reshape(phi, [-1, phi.shape[-1]])
        
        out = tensorflow.reshape(tensorflow.matmul(tensorflow.nn.softmax(tensorflow.matmul(theta,tensorflow.transpose(phi))), alpha), [-1,inputs.shape[1],inputs.shape[2], int(inputs.shape[3]/2)])
        
        mod_attention = self.conv_mask(out)

        # Self Attention
        sp = self.flatten(inputs)
        sp = self.dense(sp)
        sp = tensorflow.reshape(sp, [-1, inputs.shape[1],inputs.shape[2],1])
    
        return  mod_attention*sp + inputs
