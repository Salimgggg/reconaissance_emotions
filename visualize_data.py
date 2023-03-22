import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from dataset_test import training_data
import matplotlib.backends.backend_agg

zone = {'FH' : ['FH1', 'FH2', 'FH3'],
        'CH' : ['CH1', 'CH2', 'CH3'],
        'N'  : ['MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL'], 
        'LC' : ['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8'],
        'RC' : ['RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8'], 
        'HD' : ['LHD', 'RHD'],
        'LD' : ['LLID', 'RLID'] }

points = ['CH1', 'CH2', 'CH3', 'FH1', 'FH2', 'FH3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8', 'LLID', 'RLID', 'MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL', 'LBM0', 'LBM1', 'LBM2', 'LBM3', 'RBM0', 'RBM1', 'RBM2', 'RBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4', 'Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8', 'LHD', 'RHD']
for point in points:
    exec(f"{point} = '{point}'")
# Define the connections between facial landmarks

def generate_mesh(zone, points):
    # Create a dictionary to map each point to its corresponding zone
    zone_indices = {subzone: zone_name for zone_name, subzones in zone.items() for subzone in subzones if subzone in points}
    
    # Create a dictionary to map each zone to a list of its points
    zone_points = {zone_name: [point for point in points if zone_indices.get(point) == zone_name] for zone_name in zone.keys()}
    
    # Create a list of tuples representing the connections between points in the mesh
    connections = [(i, j) for i in range(len(points)) for j in range(i+1, len(points)) if zone_indices.get(points[i]) == zone_indices.get(points[j])]
    
    # Create a list of tuples representing the connections between points in the mesh, with the format (point_index_1, point_index_2)
    mesh = [(connection[0], connection[1]) for zone_name in zone.keys() for connection in connections if points[connection[0]] in zone_points[zone_name] and points[connection[1]] in zone_points[zone_name]]
    
    return mesh   


connections = generate_mesh(zone, points)
disconnections = [(['Mou2', 'Mou3', 'Mou4'], ['Mou6', 'Mou7', 'Mou8']), 
                  (['RC3', 'RC2'], ['RC4', 'RC6', 'RC8']), 
                  (['RC1'], ['RC3', 'RC7', 'RC8', 'RC6']),
                  (['RC4', 'RC2'], ['RC7', 'RC8']), 
                  (['RC5'], ['RC8']),
                  (['LC3', 'LC2'], ['LC4', 'LC6', 'LC8']), 
                  (['LC1'], ['LC3', 'LC7', 'LC8', 'LC6']), 
                  (['LC4', 'LC2'], ['LC7', 'LC8']), 
                  (['LC5'], ['LC8']), 
                  ([RLID], [LLID])]
add_connections = [(RNSTRL,[RC4, RC6, RC8, Mou1, Mou2]), 
                   (LNSTRL,[LC4, LC6, LC8, Mou4, Mou5]), 
                   (Mou5, [LC4, LC1, CH3]), 
                   (Mou1, [RC4, RC1, CH1]), 
                   (MH, [LC8, RC8, RBRO1, LBRO1, FH2, RLID, LLID]),
                   (RLID, [RC3, RC7, RC8, RBRO1, RBRO2, RBRO3, RBRO4 ]), (LLID, [LC3, LC7, LC8, LBRO1, LBRO2, LBRO3, LBRO4]),
                   (Mou4, [Mou5, Mou6, Mou7]), 
                   (Mou2, [Mou1, Mou8]), 
                   (Mou3, [Mou4, Mou2, Mou7]), 
                   (Mou7, [Mou8, Mou6, CH2, Mou3]), 
                   (RBM0, [RBRO2, RBRO1, RBM1]), (LBM0, [LBRO2, LBRO1, LBM1])
                   , (FH2, [RBM1, RBRO1, LBRO1, LBM1]), 
                   (RBM2, [RBM1, RBM3, RBRO4]), (RBM3, [RBRO3, RBM0]), 
                   (LBM2, [LBM1, LBM3, LBRO4]), (LBM3, [LBRO3, LBM0]), 
                   (LC1, CH3), (RC1, CH1), (LC3, LBRO4), (RC3, RBRO4), 
                   (FH3, LBM2), (FH1, RBM2), (Mou6, [CH3,Mou5]), (Mou8, [CH1, Mou1])
                   ]

def change_mesh(connections, disconnections, add_connections):
    # Add new connections to the list
    for add_pair in add_connections:
        from_point, to_point_or_list = add_pair
        if isinstance(to_point_or_list, list):
            for to_point in to_point_or_list:
                connections.append((points.index(from_point), points.index(to_point)))
        else:
            connections.append((points.index(from_point), points.index(to_point_or_list)))
            
    # Disconnect points according to disconnections list
    for disconnect_pair in disconnections:
        disconnect_from, disconnect_to = disconnect_pair
        for from_point in disconnect_from:
            for to_point in disconnect_to:
                connections = [connection for connection in connections if not (points[connection[0]] == from_point and points[connection[1]] == to_point) and not (points[connection[1]] == from_point and points[connection[0]] == to_point)]
                
    return connections

connections = change_mesh(connections, disconnections, add_connections)


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

animate_facial_landmarks_3D(training_data[8][0])

# print(points.index('Mou4'))

# print(RNSTRL)

