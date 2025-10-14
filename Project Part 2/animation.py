from IPython.display import clear_output
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
X1 = pd.read_pickle("Xtrain2.pkl")
y = np.load("Ytrain2.npy")

skeleton_sequences = X1['Skeleton_Sequence'].values
sequence = skeleton_sequences[1]
sequence_array = np.array(sequence)
sequence_array = sequence_array.reshape(-1, 33, 2)

# Define skeleton edges
skeleton_edges = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32)
]

fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()  # Turn on interactive mode

for i, frame in enumerate(sequence_array):
    ax.clear()
    ax.set_title(f'Frame {i+1}/{sequence_array.shape[0]}')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')

    for kp in range(33):
        ax.plot(frame[kp, 0], frame[kp, 1], 'bo', markersize=5)
        ax.text(frame[kp, 0], frame[kp, 1], str(kp), fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.1'))

    for (j, k) in skeleton_edges:
        ax.plot([frame[j, 0], frame[k, 0]], [frame[j, 1], frame[k, 1]], 'k-', linewidth=2)

    plt.pause(0.01)

plt.ioff()
plt.show()