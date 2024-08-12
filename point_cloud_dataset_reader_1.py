from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def read_off_file(file_path):
    with open(file_path, 'r') as file:
        # 读取 OFF 文件内容
        lines = file.readlines()

        # 提取点云信息
        num_points = int(lines[1].split()[0])
        points = [list(map(float, line.split()))[0:3] for line in lines[2:2+num_points]]

        # 如果点数少于1024，将其复制以达到1024个点
        while len(points) < 1024:
            points += points[:1024 - len(points)]

        # 如果点数多于1024，使用最远点采样
        if len(points) > 1024:
             points = farthest_point_sample(np.array(points), 1024)

        points = pc_normalize(np.array(points))
    return np.array(points)

class PointCloudDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(classes,class_to_idx)
        return classes, class_to_idx

    def __init__(self, root, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root
        #print(self.root)
        self.classes, self.class_to_idx = self.find_classes(root)
        # print(self.classes, self.class_to_idx)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  /  <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label):
                point_cloud = ''
                for cloud in os.listdir(root + '/' + label  + '/' + item):
                    #print(view)
                    point_cloud = (root + '/' + label + '/' + item + '/' + cloud)   #path


                self.x.append(point_cloud)
                self.y.append(self.class_to_idx[label])
        # print(self.y)
                # x, max = read_off_file(point_cloud)
                # print(max)
                # if max> maxL:
                #     maxL = max
                #     print(maxL,'mm')

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_cloud = self.x[index]
        views = []
        # path = orginal_views[0].rsplit('/',1)[0]
        # for view in orginal_views:
        #     im = Image.open(view)
        #     im = im.convert('RGB')
        #     if self.transform is not None:
        #         im = self.transform(im)
        #     views.append(im)
        # return views, self.y[index],path
        path = orginal_cloud.split('/')[-2] + '/' + orginal_cloud.split('/')[-1]
        pcloud = read_off_file(orginal_cloud)

        return pcloud, self.y[index], path

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

# newloader = PointCloudDataSet('.\datasets\shrec_13\point_clouds')
# print(newloader)