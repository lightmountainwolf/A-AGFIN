import os
import glob
import sys
import argparse
import numpy as np
import scipy.io as sio
sys.path.append('/')
#sys.path.append('../mobilenet')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sketch_model import SketchModel
from view_model import MVCNN
from view_dataset_reader import MultiViewDataSet
from classifier_jz24096_test import Classifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from get_mu_logvar import Get_mu_logvar

from PIL import Image
import os
import shutil

correct_path='/home/sse316/heng/cgn/datasets/shrec_13/point_clouds/'
wrong_path='/home/sse316/heng/cgn/datasets/shrec_13/point_clouds/'

parser = argparse.ArgumentParser("feature extraction of sketch images")

# parser.add_argument('--sketch_path', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TESTING_SKETCHES/airplane/test/17.png')
parser.add_argument('--sketch_path', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TESTING_SKETCHES/airplane/test/test1.jpg')
parser.add_argument('--root_folder', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TESTING_SKETCHES')

parser.add_argument('--view-train-datadir', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TRAINING_SKETCHES')
parser.add_argument('--view-test-datadir', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13')
parser.add_argument('--workers', default=5, type=int,
                    help="number of data loading workers (default: 0)")

parser.add_argument('--batch-size', type=int, default=1)
# parser.add_argument('--num-classes', type=int, default=171)
# parser.add_argument('--num-train-samples', type=int, default=171*30)
# parser.add_argument('--num-test-samples', type=int, default=171*30)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--num-train-samples', type=int, default=90*30)
parser.add_argument('--num-test-samples', type=int, default=90*30)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str, default='/home/sse316/heng/cgn/trained/attentionpre')
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16','vgg19', 'resnet50'], default='resnet50')
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)

parser.add_argument('--cnn-feat-dim', type=int, default=2048)
parser.add_argument('--feat-dim', type=int, default=128)

parser.add_argument('--test-sketch-feat-file', type=str,
                    default='/home/sse316/heng/cgn/extract_features/alex_L2_test_sketch_feature.mat',
                    help="features flie of test sketches, .mat file")
parser.add_argument('--point_cloud-feat-flie', type=str,
                    default='/home/sse316/heng/cgn/point_feature_pointnet_connect_withweight_attention13pre.mat',
                    help="features flie of view images of 3d models, .mat file")

args = parser.parse_args()

def get_point_cloud_data(point_cloud_feat_flie):
    """" read the features and labels of sketches and 3D models
    Args:
        test_sketch_feat_file: features flie of test sketches, it is .mat file
        view_feat_flie: features flie of view images of 3d models
    """
    point_cloud_data_features1 = torch.load(point_cloud_feat_flie)
    # print(view_data_features1)

    """
    sketch_feature = sket_data_features['view_feature']
    print(sketch_feature.shape)
    sketch_label = sket_data_features['view_labels']
    """

    point_cloud_feature = point_cloud_data_features1['point_cloud_feature']
    point_cloud_label = point_cloud_data_features1['point_cloud_labels']
    point_cloud_paths=point_cloud_data_features1['point_cloud_paths']

    print(point_cloud_label, point_cloud_paths)

    return point_cloud_feature, point_cloud_label,point_cloud_paths

def cal_euc_distance(sketch_feat,view_feat):
    distance_matrix = pairwise_distances(sketch_feat,view_feat)

    return distance_matrix

def cal_cosine_distance(sketch_feat,view_feat):
    distance_matrix = cosine_similarity(sketch_feat,view_feat)

    return distance_matrix

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def main():
    point_cloud_feature, point_cloud_label,point_cloud_paths = get_point_cloud_data(args.point_cloud_feat_flie)
    np.set_printoptions(threshold=np.inf)

    # print('vl', point_cloud_label)
    classes,class_to_idx=find_classes("/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TESTING_SKETCHES")
    # print(classes,class_to_idx)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        # torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    sketch_model = SketchModel(args.model, args.num_classes, use_gpu=True)
    classifier = Classifier(12, args.cnn_feat_dim, args.num_classes)
    # classifier1 = torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(80) + '.pth')
    classifier1 = torch.load(args.model_dir + '/' + args.model + '_baseline_classifiernew' + '.pth')
    class_centroid = nn.functional.normalize(classifier1["module.fc5.weight"], dim=0).permute(1, 0)
    class_centroid = class_centroid.data.cpu().numpy()
    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    # Load model
    # sketch_model.load_state_dict(torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_sketch_model' + '_' + str(80) + '.pth'))
    # classifier.load_state_dict(torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(80) + '.pth'))

    sketch_model.load_state_dict(torch.load(args.model_dir + '/' + args.model + '_baseline_sketch_modelnew' + '.pth'))
    classifier.load_state_dict(torch.load(args.model_dir + '/' + args.model + '_baseline_classifiernew' + '.pth'))

    sketch_model.cuda()
    classifier.cuda()
    sketch_model.eval()
    classifier.eval()

    for type_folder in os.listdir(args.root_folder):
        type_folder_path = os.path.join(args.root_folder, type_folder)

        # 判断是否为文件夹
        if os.path.isdir(type_folder_path):

            # 获取类型文件夹下的所有测试文件夹
            test_folders = glob.glob(os.path.join(type_folder_path, 'test'))

            # 遍历每个测试文件夹
            for test_folder in test_folders:

                # 获取测试文件夹下的所有图像文件
                image_files = glob.glob(os.path.join(test_folder, '*.jpg')) + \
                              glob.glob(os.path.join(test_folder, '*.png'))  # 可根据实际情况添加其他图像格式

                # 遍历每个图像文件
                for image_file in image_files:
                    # print(image_file)  # 在这里你可以执行你需要的操作，比如读取图像等
                    # sketch_label=class_to_idx[args.sketch_path.split('/')[-2]]
                    sketch_label = class_to_idx[args.sketch_path.split('/')[-3]]
                    sketch_index = os.path.basename(image_file)
                    sketch_index = sketch_index.split(".")[0]
                    # print(sketch_index)

                    sketch_id = args.sketch_path.split('/')[-1].split('.')[0]
                    im = Image.open(image_file)
                    im = im.convert('RGB')
                    image_transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.5])])
                    data = image_transforms(im)
                    data = data.unsqueeze(dim=0)
                    if use_gpu:
                        data = data.cuda()
                    # print(batch_idx)
                    output = sketch_model.forward(data)
                    # print(output)
                    mu_embeddings, logits = classifier.forward(output,0)
                    # print(mu_embeddings, logits)
                    outputs = nn.functional.normalize(mu_embeddings, dim=1)
                    _, predicted = torch.max(logits.data, 1)

                    print("Predict Label: " + str(predicted.item()))

                    sketch_feat = outputs.detach().cpu().clone().numpy()
                    distance_matrix = cal_cosine_distance(sketch_feat, point_cloud_feature)

                    # print(distance_matrix.shape)
                    matrix = distance_matrix[0]
                    j = 0
                    res = []
                    dist_sort_index = np.argsort(-distance_matrix[0], axis=0)
                    file_name = sketch_index
                    file_path = os.path.join('retrieval_result_resnet50_attention', file_name)
                    with open(file_path, 'w') as file:
                        # 写入内容到文件
                        file.write(str(sketch_index))
                        file.write('\n')
                        while j < len(matrix):
                            temp = os.listdir(correct_path + point_cloud_paths[dist_sort_index[j]])[0]
                            temp = os.path.splitext(temp)[0]
                            file.write(temp)
                            file.write(' ')
                            file.write(str(matrix[dist_sort_index[j]]))
                            file.write('\n')
                            print(temp, matrix[dist_sort_index[j]])
                            j = j + 1


    # print(distance_matrix)
    # dist_sort_index = np.argsort(-distance_matrix[0], axis=0)
    # top_10_retrieval=dist_sort_index[:10]
    # print("Top 10 result: "+str(top_10_retrieval))
    # print("Top 10 result: " + point_cloud_paths[837])
    # print("Top 10 result label: " + str(point_cloud_label[top_10_retrieval].reshape(10,)))
    # print("Top 10 cosine similarity: "+str(distance_matrix[0][top_10_retrieval]))
    # print(os.path)
    # if not os.path.exists('/home/sse316/heng/cgn/retrieval_result/' + sketch_id):
    #     os.mkdir('/home/sse316/heng/cgn/retrieval_result/' + sketch_id)
    # for i in top_10_retrieval:
    #     if point_cloud_label[i][0]==sketch_label:
    #         file=os.path.join(correct_path,point_cloud_paths[i],os.listdir(correct_path+point_cloud_paths[i])[0])
    #         shutil.copy(file,'/home/sse316/heng/cgn/retrieval_result/' + sketch_id)
    #     else:
    #         file = os.path.join(wrong_path, point_cloud_paths[i], os.listdir(wrong_path + point_cloud_paths[i])[0])
    #         shutil.copy(file, '/home/sse316/heng/cgn/retrieval_result/' + sketch_id)


if __name__=='__main__':
    main()