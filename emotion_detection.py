#datasets from here: https://stackoverflow.com/questions/30703485/data-sets-for-emotion-detection-in-text
#  and here: https://www.kaggle.com/c/sa-emotions/leaderboard
# (they're the same ones). Note that there were 427 occurrences of an unresolved symbol, which was simply replaced with a space
# unresolved symbols: , *Â¡, Â, ``,

import numpy as np
import pandas as pd
from basic_functions import *

MAX_CONTENT_LENGTH = 256
num_sentiments = 0
sentiment_to_id = {}
id_to_sentiment = {}

def read_dataset(filepath=None):
    #readpath = "data/text_emotion.csv"
    readpath = "data/sentiment_map_top5.csv"
    if filepath is not None:
        readpath = filepath

    data = pd.read_csv(readpath)
    sentiments = data['sentiment'].tolist()
    unique_sentiments = data['sentiment'].unique()
    num_sentiments = data['sentiment'].nunique()
    for i in range(num_sentiments): # keep track of which id means what...
        sent = unique_sentiments[i]
        sentiment_to_id[sent] = i
        id_to_sentiment[i] = sent
    content = data['content'].tolist()

    #max_content_length = max([len(a) for a in content])

    pairs = []

    #for i in range(len(content)):
    for i in range(10000):
        #print(i, max_content_length, content[i])
        matrix = text_to_matrix(content[i], length=MAX_CONTENT_LENGTH)
        sent = as_onehot(sentiment_to_id[sentiments[i]], num_sentiments)
        pairs.append((matrix, sent))
        print(i, '\r', end="")

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
    #x = Conv2D(channels, kernel_size=(1, 2), strides=(1, 2), activation='relu')(x)
    #x = Conv2D(channels, kernel_size=(1, 2), strides=(1, 2), activation='relu')(x)
    x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    #x = Conv2D(channels, kernel_size=1, activation='relu')(x)
    #x = Conv2D(channels, kernel_size=(1, 3), strides=3, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)
    x = Dense(output_size, activation='relu')(x)

    prediction = Dense(output_size, activation='softmax')(x)

    model = Model(in_val, prediction)
    return model

def train_model(model, data, epochs=3, batch_size=128, validation_split=0.1):
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=data[0], y=data[1], batch_size=128, epochs=epochs, validation_split=0.1)

    return model

if __name__ == '__main__':
    x, y = read_dataset()
    model = basic_conv_model(x[0].shape, y[0].shape[0])
    model = train_model(model, (x, y), epochs=100)
    save_model(model, "emotion_classifier.h5")

    #print(sentiment_to_id, id_to_sentiment, x[0].shape, y[0])