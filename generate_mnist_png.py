import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist

# Load data (this downloads it automatically if not cached)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save a '4'
idx_4 = np.where(y_train == 4)[0][0]
img_4 = Image.fromarray(x_train[idx_4])
img_4.save('mnist_4.png')

# Save an '8'
idx_8 = np.where(y_train == 8)[0][0]
img_8 = Image.fromarray(x_train[idx_8])
img_8.save('mnist_8.png')

print("Successfully generated mnist_4.png and mnist_8.png in the project directory.")
