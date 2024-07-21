import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img # type: ignore 

def create_mask(filename):
    # Load the image
    # path = input("Enter the filename: ")
    img = load_img('images/' + filename, color_mode='grayscale') #, target_size=(512, 512)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array /= 255.0

    # Define the model with a single layer for the Sobel X filter
    model_sobel_x = Sequential([
        Conv2D(1, (3, 3), activation='linear', input_shape=(img_array.shape[1], img_array.shape[2], 1), padding='same')
    ])

    # Define the model with a single layer for the Sobel Y filter
    model_sobel_y = Sequential([
        Conv2D(1, (3, 3), activation='linear', input_shape=(img_array.shape[1], img_array.shape[2], 1), padding='same')
    ])

    # Define Sobel filters for edge detection
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32).reshape((3, 3, 1, 1))

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32).reshape((3, 3, 1, 1))

    # Set the weights of the Conv2D layers
    model_sobel_x.layers[0].set_weights([sobel_x, np.zeros(1)])
    model_sobel_y.layers[0].set_weights([sobel_y, np.zeros(1)])

    # Apply the Sobel filters to the image
    edges_x = model_sobel_x.predict(img_array)
    edges_y = model_sobel_y.predict(img_array)

    # Calculate the magnitude of the gradients
    edges_magnitude = np.sqrt(np.square(edges_x) + np.square(edges_y))

    # Normalize the edges
    edges_magnitude /= edges_magnitude.max()

    # Apply thresholding to get binary edges
    edges_binary = (edges_magnitude > 0.2).astype(np.float32)

    # Display the result
    edges_x = edges_x.squeeze()  # Squeeze the batch dimension
    edges_y = edges_y.squeeze()  # Squeeze the batch dimension
    edges_magnitude = edges_magnitude.squeeze()  # Squeeze the batch dimension
    edges_binary = edges_binary.squeeze()  # Squeeze the batch dimension
    img_array = img_array.squeeze()  # Squeeze the batch dimension

    combined = img_array - edges_binary
    clipped = np.clip(combined, a_min=0, a_max = 1)
    clipped[clipped < 0.95] = 0
    clipped[clipped >= 0.95] = 1

    display_images(img_array, edges_binary, combined, clipped)
    return (img_array, clipped)

def display_images(img_array, edges_binary, combined, clipped):
    plt.figure(figsize=(18, 5))
    plt.subplot(141), plt.imshow(img_array, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(142), plt.imshow(edges_binary, cmap='gray')
    plt.title('Binary Edges'), plt.xticks([]), plt.yticks([])

    plt.subplot(143), plt.imshow(combined, cmap='jet')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(144), plt.imshow(clipped, cmap='gray')
    plt.title('Masked Image'), plt.xticks([]), plt.yticks([])
    plt.show()

create_mask("trainline.png")
