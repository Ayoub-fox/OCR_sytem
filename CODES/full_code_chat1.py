import tensorflow as tf

# GPU configuration must be set before any other TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Set memory growth to True
            tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally set a memory limit if necessary
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            # )
    except RuntimeError as e:
        print(e)

# Your existing imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
import os
import string
from shutil import copyfile, rmtree
import re
import cv2
from PIL import Image, ImageDraw
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, GRU, Dense, Lambda
import tensorflow.keras.backend as K

# 'set_b', 'set_c', 'set_d', 'set_e'
sets = ['set_a','set_b']

def get_Word(name):
    file_name = name.split("/")[-1].split(".")[0]
    load_profile = open('/'.join(name.split('/')[:len(name.split('/'))-2]) + "/tru/" + file_name + ".tru", "r",encoding="latin-1")
    label = load_profile.read().splitlines()[6]
    word = re.search(r"AW2:(.*?);", label).group(1).split('|')[:-1]
    return word

def evaluate_word(name):
    word = get_Word(name)
    for i, car in enumerate(word):
        if car[-1] == "1" or car[-1] == "2":
            word[i] = "-"
    return word

def get_lexicon_2(names):
    arabic_labels = []
    for name in names:
        arabic_labels = arabic_labels + evaluate_word(name)
    return list(dict.fromkeys(arabic_labels))

def get_lengths(names):
    d = {}
    for name in names:
        file_name = name.split("/")[-1].split(".")[0]
        word = get_Word(name)
        d[file_name] = len(word)
    return d

def open_image(name, img_size=[100, 300]):
    img = cv2.imread(name, 0)
    img = cv2.resize(img, (img_size[1], img_size[0]), Image.LANCZOS)
    img = cv2.threshold(img, 255 // 2, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(img)
    word = get_Word(name)
    return img, word

class Readf:
    def __init__(self, img_size=(100, 300), max_len=17, normed=False, batch_size=32, classes={}, mean=118.2423, std=36.72):
        self.batch_size = batch_size
        self.img_size = img_size
        self.normed = normed
        self.classes = classes
        self.max_len = max_len
        self.mean = mean
        self.std = std
        self.voc = list(self.classes.keys())
        if type(classes) == dict:
            self.blank = classes["-"]
   
    def make_target(self, text):
        return np.array([self.classes[char] if char in self.voc else self.classes['-'] for char in text])

    def get_labels(self, names):
        Y_data = np.full([len(names), self.max_len], self.blank)
        for i, name in enumerate(names):
            img, word = open_image(name, self.img_size)
            word = self.make_target(word)
            Y_data[i, 0:len(word)] = word
        return Y_data

    def get_blank_matrices(self):
        shape = (self.batch_size,) + self.img_size
        X_data = np.empty(shape)
        Y_data = np.full([self.batch_size, self.max_len], self.blank)
        input_length = np.ones((self.batch_size, 1))
        label_length = np.zeros((self.batch_size, 1))
        return X_data, Y_data, input_length, label_length

    def run_generator(self, names, downsample_factor=2):
        n_instances = len(names)
        N = n_instances // self.batch_size
        rem = n_instances % self.batch_size

        while True:
            X_data, Y_data, input_length, label_length = self.get_blank_matrices()
            i, n = 0, 0
            for name in names:
                img, word = open_image(name, self.img_size)
                word = self.make_target(word)
                if len(word) == 0:
                    continue
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = (self.img_size[0] + 4) // downsample_factor - 2
                X_data[i] = img[np.newaxis, :, :]
                i += 1
                if i == self.batch_size:
                    n += 1
                    inputs = {
                        'the_input': X_data,
                        'the_labels': Y_data,
                        'input_length': input_length,
                        'label_length': label_length,
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}
                    yield (inputs, outputs)
                    X_data, Y_data, input_length, label_length = self.get_blank_matrices()
                    i = 0
            if rem > 0:
                inputs = {
                    'the_input': X_data[:rem],
                    'the_labels': Y_data[:rem],
                    'input_length': input_length[:rem],
                    'label_length': label_length[:rem],
                }
                outputs = {'ctc': np.zeros([rem])}
                yield (inputs, outputs)

class CRNN:
    def __init__(self, img_w, img_h, output_size, max_len):
        self.img_w = img_w
        self.img_h = img_h
        self.output_size = output_size
        self.max_len = max_len
        self.conv_filters = 16  # Reduced the number of filters
        self.kernel_size = (3, 3)
        self.pool_size = 2
        self.time_dense_size = 32
        self.rnn_size = 64  # Reduced the RNN size
        self.model = self.build_model()

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(self):
        input_data = Input(name='the_input', shape=(self.img_h, self.img_w), dtype='float32')
        expanded_input = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_data)
        conv_1 = Conv2D(self.conv_filters, self.kernel_size, padding='same', activation='relu', name='conv1')(expanded_input)
        pool_1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='pool1')(conv_1)
        conv_2 = Conv2D(self.conv_filters, self.kernel_size, padding='same', activation='relu', name='conv2')(pool_1)
        pool_2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='pool2')(conv_2)
        conv_3 = Conv2D(self.conv_filters, self.kernel_size, padding='same', activation='relu', name='conv3')(pool_2)       ## change : we added this
        pool_3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='pool3')(conv_3)
        conv_to_rnn_dims = (self.img_w // (self.pool_size * 2), self.img_h // (self.pool_size * 2) * self.conv_filters)
        reshaped = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(pool_2)
        dense = Dense(self.time_dense_size, activation='relu', name='dense')(reshaped)
        rnn = Bidirectional(GRU(self.rnn_size, return_sequences=True), name='biGRU')(dense)                             ## change : we changed LSTM into GRU                 
        y_pred = Dense(self.output_size, activation='softmax', name='softmax')(rnn)
        labels = Input(name='the_labels', shape=[self.max_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        ctc_loss = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[ctc_loss, y_pred])
        model.summary()
        return model

# Training and evaluation
save_path = r"C:/Users/Spixer/Desktop/Classes/Deep learning/Project/Only_project/ifnenit_v2.0p1e_miniature"
path = r"C:/Users/Spixer/Desktop/Classes/Deep learning/Project/Only_project/ifnenit_v2.0p1e_miniature/data"
model_name = "OCR_IFNENIT_verl"

prng = RandomState(32)
batch_size = 32  # Adjusted batch size
imgh = 100  # Adjusted image height
imgw = 300  # Adjusted image width

try:
    rmtree(save_path + "/" + model_name)
except:
    pass

os.mkdir(save_path + "/" + model_name)

train = []
lexicon = []
for set in sets:
    PATH = path + '/' + set + '/' + 'tif'
    train.extend([dp + "/" + f for dp, dn, filenames in os.walk(PATH) for f in filenames if re.search('tif', f)])    # something shady happens here


prng.shuffle(train)
lexicon = get_lexicon_2(train)
classes = {j: i for i, j in enumerate(lexicon)}
inve_classes = {v: k for k, v in classes.items()}
length = len(train)
train, val = train[:int(length * 0.9)], train[int(length * 0.9):]
lenghts = get_lengths(train)
max_len = max(lenghts.values())

if '-' not in classes:
    classes['-'] = len(classes)  # Assign a new index to '-' if it's not already present

objet = Readf(img_size=(imgh, imgw), classes=classes)

output_size = len(classes) + 1
crnn = CRNN(imgw, imgh, output_size, max_len=17)
model = crnn.model
 
def ctc_loss_function(y_true, y_pred):
    return y_pred

model.compile(loss={'ctc': ctc_loss_function}, optimizer='adam')


train_generator = objet.run_generator(train)
val_generator = objet.run_generator(val)

train_steps = len(train) // batch_size
val_steps = len(val) // batch_size + 1

print(train_steps)

history = model.fit(train_generator,
                    steps_per_epoch=train_steps,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=20)

# Create empty arrays to store the evaluation results
losses = []

# Iterate over the validation generator and calculate loss for each batch
for i in range(val_steps):
    inputs, targets = next(val_generator)
    loss = model.evaluate(inputs, targets, verbose=0)
    losses.append(loss)

# Calculate the average validation loss
validation_loss = np.mean(losses)

# Print the evaluation result
print("Validation Loss:", validation_loss)

# Call the plot_training_history function
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_training_history(history)

def num_to_label(num, inv_classes):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            char = inv_classes.get(ch)
            if char is not None:  # Check if the character exists in the mapping
                ret += char
    return ret

# Create empty arrays to store the predictions
predictions_ctc_loss = []
predictions_probabilities = []

# Iterate over the validation generator and make predictions for each batch
for i in range(val_steps):
    inputs, _ = next(val_generator)

    # Reshape the input images to match the expected shape of the model
    inputs['the_input'] = inputs['the_input'].reshape((-1, imgh, imgw))

    # Make predictions with the model
    batch_ctc_loss, batch_predictions = model.predict(inputs)

    predictions_ctc_loss.append(batch_ctc_loss)
    predictions_probabilities.append(batch_predictions)

# Concatenate the predictions for all batches
predictions_ctc_loss = np.concatenate(predictions_ctc_loss, axis=0)
predictions_probabilities = np.concatenate(predictions_probabilities, axis=0)

# Best-path decoding
decoded = K.get_value(K.ctc_decode(predictions_probabilities,
                                   input_length=np.ones(predictions_probabilities.shape[0])*predictions_probabilities.shape[1],
                                   greedy=False)[0][0])

decoded_words = []
for d in decoded:
    decoded_word = num_to_label(d, inve_classes)
    decoded_words.append(decoded_word)

true_words = []  # List to store the true words

# Iterate over the validation set
for image_path in val:
    true_word = evaluate_word(image_path)  # Get the true word for the current image
    true_words.append(true_word)

# Compare the true words with the decoded words
for true_word, decoded_word in zip(true_words, decoded_words):
    if true_word == decoded_word:
        print("Correct: ", true_word)
    else:
        print("Incorrect: True word =", true_word, ", Decoded word =", decoded_word)

# Save the model
model.save(save_path + model_name + ".keras")
print(f"Model saved to {save_path + model_name + '.keras'}")
