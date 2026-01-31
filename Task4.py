import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, image_size=(256, 256)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img = np.array(img) / 255.0
    img = img[np.newaxis, :]
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Load content and style images
content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

# Load pre-trained Neural Style Transfer model
model = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Apply style transfer
stylized_image = model(content_image, style_image)[0]

# Post-process and save output image
stylized_image = tf.squeeze(stylized_image)
stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0)

plt.imsave("output.png", stylized_image.numpy())

print("✅ Style transfer complete. Output saved as output.png")
