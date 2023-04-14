from tensorflow import keras
import numpy as np
from keras import layers
import tensorflow as tf
import random
import loaders

K = keras.backend

################################################################################
# Models and layers
################################################################################
def make_conv_classifier(codings_size):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''
    inputs = keras.layers.Input(shape=[28,28,2])
    z = keras.layers.SeparableConv2D(filters=32, kernel_size=3, activation='selu')(inputs)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.SeparableConv2D(filters=64, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.SeparableConv2D(filters=128, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(codings_size, activation='selu')(z)
    outputs = keras.layers.Dense(1, activation='sigmoid')(z)
    return keras.Model(inputs = [inputs], outputs=[outputs])

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs        
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    
def make_conv_base(codings_size):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    Encoder for the Triplet and Siamese network
    '''    
    inputs = keras.layers.Input(shape=[28,28])
    z = keras.layers.Reshape(target_shape=(28,28,1))(inputs)
    z = keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.Conv2D(filters=128, kernel_size=3, activation='selu')(z)
    z = keras.layers.Flatten()(z)
    codings = keras.layers.Dense(codings_size)(z)
    return keras.Model(inputs = [inputs], outputs=[codings])

def make_encoder_vae(codings_size):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    Encoder for the VAE, VAE-triplet, and VAE-siamese
    '''    
    inputs = keras.layers.Input(shape=[28,28])
    z = keras.layers.Reshape(target_shape=(28,28,1))(inputs)
    z = keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu')(z)
    z = keras.layers.MaxPooling2D(pool_size=2)(z)
    z = keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='selu')(z)
    z = keras.layers.Flatten()(z)
    codings_mean = keras.layers.Dense(codings_size)(z)
    codings_log_var = keras.layers.Dense(codings_size)(z)
    codings = Sampling()([codings_mean, codings_log_var])
    
    return keras.Model(inputs = [inputs], outputs=[codings_mean, codings_log_var, codings])


def make_decoder(codings_size):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    Decoder for the VAE models (VAE, VAE-triplet, & VAE-siamese)
    '''
    decoder_inputs = keras.layers.Input(shape=[codings_size])
    x = keras.layers.Dense(units=7*7*32, activation="selu")(decoder_inputs)
    x = keras.layers.Reshape(target_shape=(7, 7, 32))(x)
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='selu')(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='selu')(x)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    outputs = keras.layers.Reshape(target_shape=(28, 28))(x)
    return keras.Model(inputs=[decoder_inputs], outputs=[outputs]) 

class VAE(keras.Model):
    """ 
    A variational autoencoder
    """
    def __init__(self, an_encoder, a_decoder,  **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder
        
    def call(self, X):
        M, V, C = self.encoder(X)       
                        
        # Ran into an error when it came to the stacking of the decoder where the channel was 
        # placed right after the batch size instead of at the end
        X_hat = self.decoder(C)
        
        M_V   = tf.stack([M, V], axis=2)
        
        return {'recon':X_hat, 'mean-var':M_V}

class TripletNet(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Original code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder

    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0])        
        C_p = self.encoder(X[:, :, :, 1])
        C_n = self.encoder(X[:, :, :, 2])
            
        D_p = tf.norm(C_a-C_p, ord='euclidean', axis=1)
        D_n = tf.norm(C_a-C_n, ord='euclidean', axis=1)
        D   = tf.stack([D_p, D_n], axis = 1)
        
        return D
    
class TripletNetVAE(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Original code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, a_decoder, distance_func, **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder
        self.distance_func = distance_func

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_distance_func(self):
        return self.distance_func
        
    def call(self, X):
        '''
        X[:, 0]: is the anchor matrix
        X[:, 1]: is the positive matrix
        X[:, 2]: is the negative matrix
        '''                    

        M_a, V_a, C_a = self.encoder(X[:, :, :, 0])        
        M_p, V_p, C_p = self.encoder(X[:, :, :, 1])
        M_n, V_n, C_n = self.encoder(X[:, :, :, 2])
            
        X_hat = tf.stack([self.decoder(C_a), self.decoder(C_p), self.decoder(C_n)], axis=3) 

        M_V   = tf.stack([M_a, V_a, M_p, V_p, M_n, V_n], axis=2)
        
        D = self.distance_func(M_V)

        return {'recon':X_hat, 'mean-var':M_V, 'distance':D}

class SiameseNet(keras.Model):
    '''
    Authors: Anthony B Garza
             Rolando Garcia
             Hani Z Girgis
    A siamese network meant to be use with a contrastive loss
    '''
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder

    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0])
        C_b = self.encoder(X[:, :, :, 1])
            
        D = tf.norm(C_a-C_b, ord='euclidean', axis=1)
        
        return D
    
class SiameseNetVAE(keras.Model):
    '''
    Authors: Anthony B Garza
             Rolando Garcia
             Hani Z Girgis
    The siamese network combined with a variational auto encoder.
    '''
    def __init__(self, an_encoder, a_decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
           
    def call(self, X):
        M_1, V_1, C_1 = self.encoder(X[:, :, :, 0])        
        M_2, V_2, C_2 = self.encoder(X[:, :, :, 1])
            
        X_hat = tf.stack([self.decoder(C_1), self.decoder(C_2)], axis=3) 

        M_V   = tf.stack([M_1, V_1, M_2, V_2], axis=2)
        
        D = tf.norm(M_1-M_2, ord='euclidean', axis=1)
        
        return {'recon': X_hat, 'mean-var': M_V, 'distance': D}
    

################################################################################
# Distance Functions
################################################################################
def ecludean_distance(M_V):
    M_a = M_V[:, :, 0]
    M_p = M_V[:, :, 2]
    M_n = M_V[:, :, 4]
    D_p = tf.norm(M_a-M_p, ord='euclidean', axis=1)
    D_n = tf.norm(M_a-M_n, ord='euclidean', axis=1)
    D = tf.stack([D_p, D_n], axis = 1)
    
    return D


################################################################################
# Loss functions
# Each loss function returns a tensorflow compatible loss function with certain
# parameters set from arguments to the outer function.
################################################################################

def get_triplet_loss(alpha = 0.0):   
    def triplet_loss(_, D):

        basic = D[:, 0] - D[:, 1] + alpha
        return tf.reduce_mean(tf.maximum(basic, 0.))
    return triplet_loss

def get_kld_loss(image_count = 3):
    m_list = [x * 2 for x in range(image_count)]
    v_list = [x + 1 for x in m_list]
    def kld_loss(_, M_V):    
        M = tf.gather(M_V, indices=m_list, axis=2)
        V = tf.gather(M_V, indices=v_list, axis=2)

        latent_loss = -0.5 * K.sum(1 + V - K.exp(V) - K.square(M), axis = -1)
        return K.mean(latent_loss) / (image_count * 784)
    
    return kld_loss

def get_recon_loss():
    def recon_loss(y_true, y_hat):
        return tf.reduce_mean(tf.square(y_hat - y_true))
    return recon_loss

def get_contrastive_loss(alpha = 0.5):
    #
    # Credit: Based on code from https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/losses/contrastive.py#L72-L120
    #    
    def contrastive_loss(y, d):
        y = tf.cast(y, tf.float32)    
        return y * K.square(d) + (1.0 - y) * K.square(K.maximum(alpha - d, 0.0))
    return contrastive_loss

################################################################################
# Evaluation metrics
################################################################################
class TripletAccuracy(keras.metrics.Metric):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.alpha = alpha
        
    def update_state(self, _, D, sample_weight=None):
        correct = D[:, 0] + self.alpha < D[:, 1]
        self.total.assign_add( len(np.where(correct == True)[0]) )
        self.count.assign_add( tf.cast(len(D), tf.float32) )
        
        assert self.total <= self.count, f'{D.shape}'
    
    def result(self):
        assert self.total <= self.count
        return self.total / self.count
    
class PairAccuracy(keras.metrics.Metric):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.alpha = alpha
        
    def update_state(self, y, D, sample_weight=None):
        correct = len(np.where((y==1) & (D < self.alpha))[0]) + len(np.where((y==0) & (D >= self.alpha))[0])
        self.total.assign_add( correct )
        self.count.assign_add( tf.cast(len(D), tf.float32) )
    
    def result(self):
        return self.total / self.count
    
def evaluate_siamese_on_triplets(a_triplet_seq, a_siamese_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_siamese_net: A trained siamese network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the siamese net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        
        D_p = a_siamese_net.predict(x_batch[:, :, :, [0,1]], verbose = 0)
        D_n = a_siamese_net.predict(x_batch[:, :, :, [0,2]], verbose = 0)
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())
    
def evaluate_vae_on_triplets(a_triplet_seq, a_vae_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_vae_net: A trained vae network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the vae  net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        a = a_vae_net.encoder.predict(x_batch[:, :, :, [0]], verbose = 0)[2]
        p = a_vae_net.encoder.predict(x_batch[:, :, :, [1]], verbose = 0)[2]
        n = a_vae_net.encoder.predict(x_batch[:, :, :, [2]], verbose = 0)[2]
        D_p = tf.norm(a-p, ord='euclidean', axis=1)
        D_n = tf.norm(a-n, ord='euclidean', axis=1)
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())
    
def evaluate_siamese_vae_on_triplets(a_triplet_seq, a_siamese_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_siamese_net: A trained siamese network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the siamese net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        
        D_p = a_siamese_net.predict(x_batch[:, :, :, [0,1]], verbose = 0)['distance']
        D_n = a_siamese_net.predict(x_batch[:, :, :, [0,2]], verbose = 0)['distance']
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())
    
    

if __name__ == "__main__":
    sm = make_siamese_net(10)
    sm.get_encoder().summary()
