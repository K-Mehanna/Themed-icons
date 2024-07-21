import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img # type: ignore 

def create_mask(path, filename):
    # Load the image
    combinedPath = os.path.join(path, filename)
    img = load_img(combinedPath, color_mode='grayscale')
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
    edges_binary = (edges_magnitude > 0.5).astype(np.float32)

    # Display the result
    edges_x = edges_x.squeeze()  # Squeeze the batch dimension
    edges_y = edges_y.squeeze()  # Squeeze the batch dimension
    edges_magnitude = edges_magnitude.squeeze()  # Squeeze the batch dimension
    edges_binary = edges_binary.squeeze()  # Squeeze the batch dimension
    img_array = img_array.squeeze()  # Squeeze the batch dimension

    combined = img_array - edges_binary
    clipped = np.clip(combined, a_min=0, a_max = 1)
    clipped[clipped < 0.9] = 0
    clipped[clipped >= 0.9] = 1

    if white_background_colour(clipped):
        clipped = 1 - clipped

    blurred = cv2.GaussianBlur(clipped.astype(np.float32), (3, 3), 0)
    
    # display_images(img_array, edges_binary, blurred, clipped)
    return (img_array, blurred) #clipped

def white_background_colour(array):
    # Extract the top, bottom, left, and right strips of the image, 5 pixels in
    top_strip = array[:5, :]
    bottom_strip = array[-5:, :]
    left_strip = array[:, :5]
    right_strip = array[:, -5:]

    # Concatenate all strips into a single array
    edge_strip = np.concatenate((top_strip.flatten(), bottom_strip.flatten(), left_strip.flatten(), right_strip.flatten()))
    new_edge_strip = edge_strip * 100
    avg = np.average(new_edge_strip)
    return avg > 50

def display_images(img_array, edges_binary, combined, clipped):
    plt.figure(figsize=(18, 5))
    plt.subplot(141), plt.imshow(img_array, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(142), plt.imshow(edges_binary, cmap='gray')
    plt.title('Binary Edges'), plt.xticks([]), plt.yticks([])

    plt.subplot(143), plt.imshow(combined, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(144), plt.imshow(clipped, cmap='gray')
    plt.title('Masked Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# create_mask("images", "trainline.png")
