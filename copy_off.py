# from mayavi import mlab
import numpy as np
import os
import shutil

def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"文件从 {source_path} 复制到 {destination_path}")
    except FileNotFoundError:
        print("找不到指定的文件")
    except PermissionError:
        print("没有足够的权限进行复制操作")
    except Exception as e:
        print(f"发生错误: {e}")
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
    copy_file(file_path, out_put_path)

    return
off_file_path = '.\datasets\shrec_13\SHREC13_SBR_TARGET_MODELS\models\m0.off'
processOff(off_file_path,'')