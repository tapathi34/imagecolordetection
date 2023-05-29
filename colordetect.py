import cv2
import numpy as np
from collections import Counter

def get_top_colors(image_path, background_color):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 2D array
    pixels = image_rgb.reshape(-1, 3)

    # Remove background color from the pixels
    mask = np.all(pixels != background_color, axis=1)
    pixels = pixels[mask]

    # Count the occurrence of each color
    color_counts = Counter(map(tuple, pixels))

    # Get the top three most common colors
    top_colors = color_counts.most_common(3)

    return top_colors

# Example usage
image_path = 'C:\\Users\\naray\\PycharmProjects\\instancemodel\\test1.jpg'
background_color = (224, 224, 224)  # Background color in RGB format (white in this example)

top_colors = get_top_colors(image_path, background_color)

# Print the top three colors and their occurrence counts
for color, count in top_colors:
    print(f"Color: {color}, Count: {count}")
