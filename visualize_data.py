import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from dataset_test import training_data
import matplotlib.backends.backend_agg

zone = {'FH' : ['FH1', 'FH2', 'FH3'],
        'CH' : ['CH1', 'CH2', 'CH3'], 
        'RB' : ['RBM0', 'RBM1', 'RBM2', 'RBM3', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4' ],
        'LB' : ['LBM0', 'LBM1', 'LBM2', 'LBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4' ], 
        'N'  : ['MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL'],
        'MOU': ['Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8'], 
        'LC' : ['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8'],
        'RC' : ['RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8'], 
        'HD' : ['LHD', 'RHD'],
        'LD' : ['LLID', 'RLID'] }

points = ['CH1', 'CH2', 'CH3', 'FH1', 'FH2', 'FH3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8', 'LLID', 'RLID', 'MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL', 'LBM0', 'LBM1', 'LBM2', 'LBM3', 'RBM0', 'RBM1', 'RBM2', 'RBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4', 'Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8', 'LHD', 'RHD']
zone_indices = {subzone: zone_name for zone_name, subzones in zone.items() for subzone in subzones}
zone_indexes = [zone_indices[subzone] for subzone in points]
connections = [(i, j) for i in range(len(points)) for j in range(i+1, len(points)) if zone_indexes[i] == zone_indexes[j]]


def animate_facial_landmarks_3D(facial_landmarks):
    facial_landmarks = np.array(facial_landmarks)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(np.min(facial_landmarks[:,:,0]), np.max(facial_landmarks[:,:,0]))
    ax.set_ylim(np.min(facial_landmarks[:,:,1]), np.max(facial_landmarks[:,:,1]))
    ax.set_zlim(np.min(facial_landmarks[:,:,2]), np.max(facial_landmarks[:,:,2]))
    ax.set_box_aspect((np.ptp(facial_landmarks[:,:,0]), np.ptp(facial_landmarks[:,:,1]), np.ptp(facial_landmarks[:,:,2])))

    ax.axis('off')
    ax.grid(False)

    # Define the connections between facial landmarks
    zone_indices = {subzone: zone_name for zone_name, subzones in zone.items() for subzone in subzones}
    zone_indexes = [zone_indices[subzone] for subzone in points]
    connections = [(i, j) for i in range(len(points)) for j in range(i+1, len(points)) if zone_indexes[i] == zone_indexes[j]]

    lines = [ax.plot([], [], [], 'b-')[0] for _ in range(len(connections))]

    def update(frame):
        for i, (start, end) in enumerate(connections):
            x = facial_landmarks[frame, [start, end], 0]
            y = facial_landmarks[frame, [start, end], 1]
            z = facial_landmarks[frame, [start, end], 2]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)

        return lines

    anim = FuncAnimation(fig, update, frames=range(facial_landmarks.shape[0]), blit=False, interval=5)
    plt.show(block=True)
    return anim

# animate_facial_landmarks_3D(training_data[8][0])

print(points.index('Mou4'))

