from keras.models import Model, save_model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

CHAR_OFFSET = 32 # lower inclusive bound of ASCII codes to consider
CHAR_LIM = 126 # upper inclusive bound of ASCII codes to consider
NUM_CHARS=CHAR_LIM-CHAR_OFFSET
# take characters from 32 to 126 as acceptable inputs

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

def read_culture_names(filepath=None):
    readpath = "data/culture_ship_names.txt"
    if filepath is not None:
        readpath = filepath

    with open(readpath, 'r') as f:
        names = [a.strip().split(chr(9)) for a in f.readlines()] # lines are tab-delimited, ASCII tab code is 009

    names_list = []
    for line in names:
        for name in line:
            names_list.append(name)

    return names_list

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

def basic_conv_model(input_shape, output_size):
    if len(input_shape) < 3:
        input_shape = (input_shape[0], input_shape[1], 1)
    height = input_shape[0]
    print(height)
    channels=256
    in_val = Input(input_shape)

    x = Conv2D(channels, kernel_size=(19, 3), strides=(19,3), activation='relu')(in_val)
    #x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    #x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    x = Conv2D(channels, kernel_size=(5, 3), strides=(5, 3), activation='relu')(x)
    x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    #x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    #x = Conv2D(channels, kernel_size=(1, 3), strides=3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)

    """x = Conv2D(channels, kernel_size=3, padding='valid', activation='relu')(in_val)
    #x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    x = Conv2D(channels, kernel_size=1, activation='relu')(x)

    x = Conv2D(channels*2, kernel_size=3, strides=2, padding=
               'valid', activation='relu')(x)
    x = Conv2D(channels*2, kernel_size=1, activation='relu')(x)

    #x = Conv2D(channels * 3, kernel_size=3, strides=2, padding=
    #'valid', activation='relu')(x)
    #x = Conv2D(channels * 3, kernel_size=1, activation='relu')(x)"""

    #x = Flatten()(x)
    prediction = Dense(output_size, activation='softmax')(x)

    model = Model(in_val, prediction)
    return model

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


def train_model(model, data, epochs=3, batch_size=128, validation_split=0.1):
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    max_length = max([len(a) for a in data])
    validation_index = int(len(data) * validation_split)
    val_x, val_y = gen_dataset(data[:validation_index], length=max_length)
    data_generator = text_image_generator(data[validation_index:], batch_size, length=max_length)
    total_steps = np.mean([len(a) for a in data[validation_index:]]) * len(data[validation_index:])
    print("total steps:", total_steps)


    history=model.fit_generator(generator=data_generator, steps_per_epoch=total_steps//batch_size,
                                epochs=epochs,
                                validation_data=(val_x, val_y))
    #history = model.fit(x=data[0], y=data[1], batch_size=128, epochs=epochs, validation_split=0.1)


    return model

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

ship_names = read_culture_names()
max_name_length = max([len(a) for a in ship_names])
id = 5
print(ship_names[:25])
print(max_name_length)
test = text_to_matrix(ship_names[id], length=None)
ttext = matrix_to_text(test)
print("{}\n{}".format(ship_names[id], ttext))
#plt.imshow(test)
#plt.show()
"""x, y = gen_intermediate_name_matrices(test)
for i in range(len(x)):
    plt.subplot(1, 3, 1)
    plt.imshow(test)
    plt.subplot(1, 3, 2)
    plt.imshow(x[i])
    plt.subplot(1, 3, 3)
    yout = np.reshape(y[i], (y[i].shape[0], 1))
    plt.imshow(yout)
    plt.show()"""

ship_names = read_culture_names()
cutoff = 1000
max_name_length = max([len(a) for a in ship_names])
x, y = gen_dataset(ship_names[:2], length=max_name_length)
model = basic_conv_model(x[0].shape, y[0].shape[0])
#model = train_model(model, (x, y), epochs=50)
#model = train_model(model, ship_names, epochs=20)
#save_model(model, "models_test.h5")
#model = load_model("models_test.h5")

model = load_model("models/200_epoch_deep.h5")

max_name_length = max([len(a) for a in ship_names[:cutoff]])
test = text_to_matrix("ROU", length=max_name_length, mark_end=False)
print(generate_from_seed("ROU", model, max_name_length))
"""for i in range(32):
    new = np.asarray([np.reshape(test, (test.shape[0], test.shape[1], 1))])
    predicted = np.zeros((new.shape[1],))
    pred_index = np.argmax(model.predict(new))
    #print(model.predict(new))
    #print(pred_index)
    predicted[pred_index] = 1
    #print(new.shape, predicted.shape)
    test[:, i+3] = predicted
    print(matrix_to_text(test), len(matrix_to_text(test)))"""
#plt.imshow(test)
#plt.show()
run_demo = True
while run_demo:
    seed = input("Enter seed:\n")
    print('-----\n{}\n-----'.format(generate_from_seed(seed, model, max_name_length)))
    if 'n' in input('Try another? y/n\n'):
        run_demo=False