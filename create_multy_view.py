import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 OFF 文件并提取顶点信息
def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError("Invalid OFF file format")

    num_vertices, _, _ = map(int, lines[1].split())
    vertices = []
    for line in lines[2:2+num_vertices]:
        vertex = list(map(float, line.split()))
        vertices.append(vertex)

    return np.array(vertices)

# 创建多视角图
def visualize_multiview(vertices, save_path=None):
    fig = plt.figure(figsize=(12, 4))

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=5, c='blue', alpha=0.6)
        ax.set_title(f'View {i+1}')
        ax.view_init(elev=20, azim=i * 120)  # 调整视角
        ax.axis('off')  # 关闭坐标轴

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

# OFF 文件路径
off_file_path = '..\datasets\shrec_13\SHREC13_SBR_TARGET_MODELS\models\m5.off'

# 从 OFF 文件读取顶点信息
vertices = read_off(off_file_path)

# 可视化多视角图并保存为文件
save_path = './output_image.png'
visualize_multiview(vertices, save_path)