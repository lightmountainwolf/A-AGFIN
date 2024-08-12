import open3d as o3d

def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 读取 OFF 文件头信息
    if lines[0].strip() != 'OFF':
        raise ValueError("Invalid OFF file format")

    # 提取顶点和面信息
    num_vertices, num_faces, _ = map(int, lines[1].split())
    vertices = []
    for line in lines[2:2 + num_vertices]:
        vertex = list(map(float, line.split()))
        vertices.append(vertex)

    return vertices

def create_point_cloud(vertices):
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    return point_cloud

def visualize_point_cloud(point_cloud):
    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

# OFF 文件路径
off_file_path = '..\datasets\shrec_13\SHREC13_SBR_TARGET_MODELS\models\m10.off'

# 从 OFF 文件读取顶点信息
vertices = read_off(off_file_path)

# 创建点云对象
point_cloud = create_point_cloud(vertices)

# 可视化点云
visualize_point_cloud(point_cloud)