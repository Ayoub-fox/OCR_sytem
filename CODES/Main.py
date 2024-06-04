import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import string
from shutil import copyfile, rmtree
import re
import cv2
from PIL import Image, ImageDraw
import glob
from keras.callbacks import EarlyStopping


save_path = "C:/Users/Spixer/Desktop/Classes/Deep learning/Project/Only_project"
path = "C:/Users/Spixer/Desktop/Classes/Deep learning/Project/Only_project/ifnenit_v2.0p1e_miniature/data/set_a/tif"
model_name = "OCR_IFNENIT_version_03-06"

prng = RandomState(32)


batch_size = 64
imgh = 100
imgw = 300

try:
    rmtree(save_path + "/" + model_name)
except:
    pass

os.mkdir(save_path + "/" + model_name)

train = [dp + "/" + f for dp, dn, filenames in os.walk(path)
         for f in filenames if re.search('tif', f)]

prng.shuffle(train)
lexicon = get_lexicon_2(train)
classes = {j: i for i, j in enumerate(lexicon)}
inve_classes = {v: k for k, v in classes.items()}

length = len(train)
train, val = train[:int(length * 0.9)], train[int(length * 0.9):]
lenghts = get_lengths(train)
max_len = max(lenghts.values())

objet = Readf(classes=classes)



img_w, img_h = 300, 100
output_size = len(classes) +1
crnn = CRNN(img_w, img_h, output_size, max_len)
model = crnn.model

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

train_generator = objet.run_generator(train)
val_generator = objet.run_generator(val)

train_steps = len(train) // batch_size
val_steps = len(val) // batch_size + 1

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_steps,
                              validation_data=val_generator,
                              validation_steps=val_steps,
                              epochs=150)
                              #callbacks=[early_stopping])




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
    import matplotlib.pyplot as plt
    
   
    plt.figure(figsize=(12, 6))
    # A dictionary containing the recorded values of different metrics during training
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

plot_training_history(history)

