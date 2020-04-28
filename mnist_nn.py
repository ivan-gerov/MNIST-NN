# %matplotlib inline # Jupyter

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configurating the default matplotlib parameters to display larger figures in Jupyter
matplotlib.rcParams['figure.figsize'] = [7, 7]

# Loading the data and spreading it
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scaling the values of each pixel between 0 and 1, in order to make it easier for the NN to learn
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Outputting the first digit from the dataset

#x_train[0] # A tensor of 28x28 pixels + value of 0 to 255 per pixel (lightness/darkness of pixel)

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# Building the model

model = tf.keras.models.Sequential() # Input Layer
model.add(tf.keras.layers.Flatten()) # Adding an inbuilt Keras "Flattening" layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Hidden Layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output Layer (softmax for probability distribution)

model.compile(optimizer='adam', # default go-to optimizer
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=3)

# Evaluating using the test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Loss: {val_loss}\nAccuracy: {val_acc}')

model.save('mnist_nn.model')

# Importing the model in order to make a prediction
model_2 = tf.keras.models.load_model('mnist_nn.model')

# Prediction only returns an array of a probability distribution,
# so np.argmax() is called on the prediction in order to get the most probable value 
predictions = model_2.predict(x_test)
print(np.argmax(predictions[0]))

# Outputting the actual image of the zeroeth digit in the x_text dataset
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()