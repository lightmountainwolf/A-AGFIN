from mayavi import mlab
import numpy as np
import os
def read_off(file_path):
    with open(file_path, 'r') as f:
        # 读取OFF文件头信息
        off_header = f.readline().strip()

        # 确保文件以"OFF"开头
        if off_header != 'OFF':
            raise ValueError('Invalid OFF file format')

        # 读取顶点数、面数和边数
        num_vertices, num_faces, _ = map(int, f.readline().strip().split())

        # 读取顶点坐标
        vertices = []
        for _ in range(num_vertices):
            vertex = list(map(float, f.readline().strip().split()))
            vertices.append(vertex)

        # 读取面信息
        faces = []
        for _ in range(num_faces):
            face = list(map(int, f.readline().strip().split()[1:]))
            faces.append(face)

    return vertices, faces

def processOff(file_path, out_put_path):
    # 指定本地OFF文件的路径

    # 读取OFF文件
    # 读取OFF文件
    vertices, faces = read_off(file_path)

    # 使用mlab.points3d创建3D点云

    # mlab.points3d(x, y, z, scale_factor=0.1)
    mlab.figure()
    x, y, z = np.array(vertices).T
    mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=0.1)
    k = 0
    # 创建一个循环，设置12个平均视角，并保存渲染的图像
    for i in range(4):
        # 计算当前视角的方位角和俯仰角
        azimuth = 360 * i / 4  # 平均分成12个角度
        for j in range(3):
            elevation = 360 * j / 3  # 你可以根据需要调整俯仰角度

            # 使用Mayavi设置视图
            mlab.view(azimuth=azimuth, elevation=elevation, distance=3)

            # 渲染并保存图像
            k += 1
            output_file_path = f'view_{k}.png'
            destination_path = os.path.join(out_put_path, output_file_path)
            print(destination_path)
            # mlab.savefig(destination_path)
            # print(f'Rendered and saved {output_file_path}')

    # 显示Mayavi窗口
    # mlab.show()
    mlab.close()
    return
off_file_path = '..\datasets\shrec_13\SHREC13_SBR_TARGET_MODELS\models\m0.off'
processOff(off_file_path,'')