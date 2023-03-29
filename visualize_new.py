
import dataset_test

# Load the training data
training_data = dataset_test.training_data
data_point = training_data[8][0]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def animate_data_point(data_point):
    def update(frame, data, plot):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot = ax.scatter(data[frame, :, 0], data[frame, :, 1], data[frame, :, 2], c='r', marker='o')
        # Normalize the axis
        max_range = np.array([data_point[:, :, i].max() - data_point[:, :, i].min() for i in range(3)]).max() / 2.0
        mid = np.array([data_point[:, :, i].mean() for i in range(3)])
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_axis_off()
        ax.w_xaxis.line.set_visible(False)
        ax.w_yaxis.line.set_visible(False)
        ax.w_zaxis.line.set_visible(False)
        return plot,

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = ax.scatter(data_point[0, :, 0], data_point[0, :, 1], data_point[0, :, 2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, frames=len(data_point), fargs=(data_point, plot), interval=2, blit=False)
    
    plt.show()




animate_data_point(data_point)