import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img # type: ignore 
import matplotlib.pyplot as plt

from edge_detection import create_mask

def theme_icon(path, filename, bg_colour, fg_colour, icon_type, soft_edges_mask, harsh_edges_mask):
    bg = bg_colour.lstrip('#')
    (r, g, b) = tuple(int(bg[i:i+2], 16) for i in (0, 2, 4))

    img = load_img(os.path.join(path, filename))
    img_array = img_to_array(img)
    flattened_img = img_array.reshape(-1, img_array.shape[-1])
    print(flattened_img.shape)
    
    # test_mask = np.zeros((650, 866), dtype=int) # Remove this line
    flattened_mask = harsh_edges_mask.flatten() #soft_edges_mask.reshape(-1, soft_edges_mask.shape[-1])
    print(flattened_mask.shape)
    print(flattened_mask[50])
    flattened = np.zeros_like(flattened_img)
    print(flattened.shape)

    for i in range(flattened.shape[0]):
        new_val = (1 - flattened_mask[i]) * np.array([r, g, b])
        if icon_type == "bg_fg":
            fg = fg_colour.lstrip('#')
            (f_r, f_g, f_b) = tuple(int(fg[i:i+2], 16) for i in (0, 2, 4))
            new_val += flattened_mask[i] * np.array([f_r, f_g, f_b])
        else:
            new_val += flattened_mask[i] * flattened_img[i]
        flattened[i] = new_val
        
    img_array = flattened.reshape(img_array.shape)
    new_img = array_to_img(img_array)
    newName = 'new-' + filename
    new_img.save(os.path.join(path, newName))
    return newName

filename = "trainline.png"
(sm, hm) = create_mask("images", filename)

theme_icon("images", filename, "#ff0000", "#0000ff", "bg-only", sm, hm)