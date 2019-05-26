from keras.models import Model, save_model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input, Dropout
from keras.optimizers import Adam
import numpy as np

CHAR_OFFSET = 32 # lower inclusive bound of ASCII codes to consider
CHAR_LIM = 126 # upper inclusive bound of ASCII codes to consider
NUM_CHARS=CHAR_LIM-CHAR_OFFSET
# take characters from 32 to 126 as acceptable inputs

def as_onehot(val, length):
    oneHot = np.zeros(length)
    oneHot[val] = 1
    return oneHot

def text_to_matrix(text, length=None, shift_right=False, mark_end=True):
    if length is None:
        length = len(text) # if no matrix output size is provided, it is simply the length of the input
    text_matrix = np.zeros((NUM_CHARS + 1, length + 1)) # extra row and column added for end of name marker

    offset = 0
    if shift_right:
        offset = length - len(text) - 1

    for i in range(len(text)):
        j = ord(text[i]) - CHAR_OFFSET
        text_matrix[j, i + offset] = 1

    if mark_end:
        text_matrix[NUM_CHARS, len(text) + offset] = 1 # end of string marker
    return text_matrix

def matrix_to_text(matrix):
    text = ""
    for i in range(matrix.shape[1]):
        if np.sum(matrix[:,i]) != 0 and matrix[-1, i] != 1:
            index = np.nonzero(matrix[:,i])[0][0]
            text += chr(index + CHAR_OFFSET)
    return text

def gen_intermediate_name_matrices(matrix):
    x = []
    y = []

    done = False
    for j in range(matrix.shape[1]-1): # ignore the final column, since for each column we also want to store the following character
        if not done and np.sum(matrix[:,j]) > 0:
            base = np.zeros(matrix.shape)
            base[:, :j+1] += matrix[:,:j+1]
            next_choice = matrix[:,j+1]
            x.append(base)
            y.append(next_choice)
            if np.nonzero(next_choice) == matrix.shape[0] - 1: # if end of string, stop
                done = True

    return x, y


# next step: generate dataset
def gen_dataset(names, length=None):
    if length is None:
        length = max([len(a) for a in names])

    pairs = []

    for name in names:
        matrix = text_to_matrix(name, length=length)
        x_int, y_int = gen_intermediate_name_matrices(matrix)
        for i in range(len(x_int)):
            pairs.append((x_int[i], y_int[i]))

    np.random.shuffle(pairs)

    x = np.asarray([np.reshape(a[0], (a[0].shape[0], a[0].shape[1], 1)) for a in pairs])
    y = np.asarray([a[1] for a in pairs])

    return x, y

def text_image_generator(data, batch_size=128, length=None):
    if length is None:
        length = max([len(a) for a in data])
    np.random.shuffle(data)
    while True:
        x = []
        y = []
        for name in data:
            matrix = text_to_matrix(name,length=length)
            x_group, y_group = gen_intermediate_name_matrices(
                matrix
            )
            x.extend(x_group)
            y.extend(y_group)
            if len(x) > batch_size:
                #print(np.asarray([np.reshape(a, (a.shape[0], a.shape[1], 1)) for a in x[:batch_size]]).shape)
                yield (np.asarray([np.reshape(a, (a.shape[0], a.shape[1], 1)) for a in x[:batch_size]])
                       , np.asarray(y[:batch_size]))
                x = []
                y = []




def generate_from_seed(seed, model, max_length=32):
    seed_len = len(seed)
    seed_matrix = text_to_matrix(seed, length=max_length, mark_end=False)
    for i in range(max_length):
        new = np.asarray([np.reshape(seed_matrix, (seed_matrix.shape[0], seed_matrix.shape[1], 1))])
        predicted = np.zeros((new.shape[1],))
        pred_index = np.argmax(model.predict(new))
        predicted[pred_index] = 1
        seed_matrix[:, i + seed_len] = predicted
        if pred_index == seed_matrix.shape[0] - 1: # if the end is reached, stop
            break
    out_text = matrix_to_text(seed_matrix)
    return out_text