import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter

# Sample dataset for CBR
X_train = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]])
y_train = np.array(['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta'])

# Function to perform Case-Based Reasoning (CBR) prediction
def cbr_prediction(new_instance, X_train, y_train):
    cosine_distances_arr = cosine_distances(new_instance.reshape(1, -1), X_train)[0]
    pairwise_distances_arr = pairwise_distances_argmin_min(new_instance.reshape(1, -1), X_train, metric='euclidean')[0]

    # Calculate the weighted average of cosine distances and pairwise distances
    distances = 0.6 * cosine_distances_arr + 0.4 * pairwise_distances_arr

    closest_indices = np.argsort(distances)
    closest_labels = y_train[closest_indices]
    color_counts = Counter(closest_labels)
    top_color = color_counts.most_common(1)[0][0]
    return top_color

# Example usage
X_test = np.array([[200, 50, 50], [50, 200, 50], [50, 50, 200]])

for instance in X_test:
    prediction = cbr_prediction(instance, X_train, y_train)
    print("CBR Prediction:", prediction)
