import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Extract y-values from the data
with open("raw_data.txt") as f:
    lines = f.readlines()
    data = [float(line.split(":")[-1]) for line in lines[3:]]


def compute_speed_changes(points):
    return [points[i+1] - points[i] for i in range(len(points)-1)]

scaler = MinMaxScaler(feature_range=(-1, 1))

# Create feature vectors for training
speed_changes = [(data[i:i+5]) for i in range(len(data)-4)]
print(speed_changes[:10])
for list_ in speed_changes:
    amp = abs(max(list_, key=abs) - min(list_, key=abs))
    for i in range(len(list_)):
        list_[i] *= amp

print(speed_changes[:10])
normalized_changes = scaler.fit_transform(np.array(speed_changes).reshape(-1, 5))

# # Reduce dimensions to 2 using PCA
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(normalized_changes)

# Train KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(normalized_changes)
#print(kmeans.cluster_centers_)

# Plot the reduced data
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
#
# # Plot the cluster centers
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
# plt.show()

def predict_states(points_list, model, scaler):
    changes = [compute_speed_changes(pts) for pts in points_list]
    normalized_changes = scaler.transform(np.array(changes))
    return model.predict(normalized_changes)

def plot_points_with_states(data, states):
    color_map = {
        0: 'red',  # accelerating
        1: 'green',  # constant
        2: 'blue'  # slowing down
    }

    for i, pts in enumerate(data):
        if max(pts) > 0:
            color = 'red'
        else:
            color = 'blue'
        plt.plot(range(i, i+6), pts, color=color, linewidth=2)

# Predict the state for each 5-point set in the training data
states = predict_states([data[i:i+6] for i in range(len(data)-5)], kmeans, scaler)
plot_points_with_states([data[i:i+6] for i in range(len(data)-5)][:100], states)
plt.show()

print(kmeans.predict([[0.1, 0.2, 0.3, 0.4, 0.5]]))
