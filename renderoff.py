import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from tripy import delaunay

def read_off(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 'OFF' not in lines[0]:
            raise ValueError("Not a valid OFF file.")

        num_vertices, num_faces, _ = map(int, lines[1].split())

        vertices = []
        for i in range(2, 2 + num_vertices):
            vertex = list(map(float, lines[i].split()))
            vertices.append(vertex)

        faces = []
        for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
            face = list(map(int, lines[i].split()[1:]))
            faces.append(face)

        return np.array(vertices), faces


def render_off(file_path):
    vertices, faces = read_off(file_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')

    # Plot faces
    for face in faces:
        vertices_face = vertices[face, :]
        poly3d = [[tuple(vertices_face[i])] for i in range(len(face))]
        poly3d.append(poly3d[0])  # Connect the last point to the first to close the polygon
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
off_file_path = '/home/sse316/heng/cgn/datasets/shrec_13/point_clouds/piano/item_0/m642.off'
render_off(off_file_path)