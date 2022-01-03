import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tensorflow

mnist = tensorflow.keras.datasets.mnist  # Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Split dataset into two sets, one for training and one for
# testing

# Normalize the datasets to be all 0 or 1
x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
x_test = tensorflow.keras.utils.normalize(x_test, axis=1)


# # Setup the neural network model
# model = tensorflow.keras.models.Sequential()
# model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))  # picture size 28*28 pixels
# model.add(tensorflow.keras.layers.Dense(512, activation='relu'))
# model.add(tensorflow.keras.layers.Dense(512, activation='relu'))
# model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))
# # This is the output layer, 10 is the number of different outputs you want. In this case, 0 to 9
#
# # Compile the model, and display the process
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=70)  # train the model
#
# model.save('handwritten.model')
#
#
# # Compute and print the model accuracy and loss
# model = tensorflow.keras.models.load_model("handwritten.model")
#
# loss, accuracy = model.evaluate(x_test, y_test)
#
# print(loss)
# print(accuracy)


# Apply the model
model = tensorflow.keras.models.load_model('handwritten.model')

image_number = 1

while os.path.isfile(f"Digits/{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))  # Turn black to white, white too black
        prediction = model.predict(img)
        # print(f"All predictions for this digit are {prediction}")
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
