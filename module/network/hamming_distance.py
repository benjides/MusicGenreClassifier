""" Hamming Distance metrics """
import keras.backend as K

def hamming_distance(y_true, y_pred):
    return K.mean(y_pred + y_true)
