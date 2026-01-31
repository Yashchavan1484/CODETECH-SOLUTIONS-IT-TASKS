import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess image
def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    img = img.convert('RGB')

    long = max(img.size)
    scale = max_dim / long
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))

    img = img.resize(new_size)
    img = np.array(img) / 255.0
    img = img[np.newaxis, :]
    return img

# Display image
def show_image(image, title=None):
    image = np.squeeze(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

# Load pre-trained style transfer model
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Paths to images
content_path = "content.jpg"   # your photo
style_path = "style.jpg"       # artwork style image

# Load images
content_image = load_image(content_path)
style_image = load_image(style_path)

# Apply style transfer
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
show_image(content_image, "Content Image")

plt.subplot(1, 3, 2)
show_image(style_image, "Style Image")

plt.subplot(1, 3, 3)
show_image(stylized_image, "Stylized Image")

plt.show()
