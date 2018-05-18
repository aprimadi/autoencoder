from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)    # (28, 28, 16)
x = MaxPooling2D((2, 2), padding='same')(x)                             # (14, 14, 16)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # (14, 14, 8)
x = MaxPooling2D((2, 2), padding='same')(x)                             # (7, 7, 8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # (7, 7, 8)
encoded = MaxPooling2D((2, 2), padding='same')(x)                       # (4, 4, 8)

# At this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)       # (4, 4, 8)
x = UpSampling2D((2, 2))(x)                                             # (8, 8, 8)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)            # (8, 8, 8)
x = UpSampling2D((2, 2))(x)                                             # (16, 16, 8)
x = Conv2D(16, (3, 3), activation='relu')(x)                            # (14, 14, 8)
x = UpSampling2D((2, 2))(x)                                             # (28, 28, 8)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)    # (28, 28, 1)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder = Model(input_img, encoded)

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    print(decoded_imgs[i].shape)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    print(encoded_imgs[i].shape)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
