import tensorflow as tf
from keras import backend as K

def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_true = tf.one_hot(y_true,7)
        pt_1 = tf.compat.v1.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.compat.v1.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)
                
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis = 1) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0), axis = 1)
    return focal_loss_fixed
