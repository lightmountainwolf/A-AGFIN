# -*- coding: utf-8 -*-
import os
import argparse
from random import sample,randint

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from sketch_model import SketchModel
from view_model import MVCNN
from classifier import Classifier
from view_dataset_reader import MultiViewDataSet
from am_softmax import AMSoftMaxLoss
from focal_am_loss import FocalAMSoftMaxLoss

parser = argparse.ArgumentParser("Sketch_View Modality")
# dataset
# parser.add_argument('--sketch-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_sketch_train_picture')
# parser.add_argument('--val-sketch-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_sketch_test_picture')
# parser.add_argument('--view-datadir', type=str, default='E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_view_render_img')
parser.add_argument('--sketch-datadir', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TRAINING_SKETCHES')
parser.add_argument('--val-sketch-datadir', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/SHREC13_SBR_TESTING_SKETCHES')
parser.add_argument('--view-datadir', type=str, default='/home/sse316/heng/cgn/datasets/shrec_13/multy_view')
parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--sketch-batch-size', type=int, default=2)
parser.add_argument('--view-batch-size', type=int, default=2)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=81)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9, help="learning rate decay")
parser.add_argument('--feat-dim', type=int, default=4096, help="feature size")
parser.add_argument('--alph', type=float, default=12, help="L2 alph")
# model
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16', 'vgg19','resnet50'], default='alexnet')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=True)
# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-model-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0,1,2,3')
# parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str,default='/home/sse316/heng/cgn/trained')
parser.add_argument('--count', type=int, default=0)

args = parser.parse_args()
writer = SummaryWriter()


def get_data(sketch_datadir,view_datadir):
    """Image reading and image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),  # Randomly change the brightness, contrast, and saturation of the image
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomCrop(224),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])  # Imagenet standards

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])

    view_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])

    val_sketch_data=datasets.ImageFolder(root=sketch_datadir, transform=val_transform)
    val_sketch_dataloaders=DataLoader(val_sketch_data,batch_size=args.sketch_batch_size,num_workers=args.workers)

    sketch_data = datasets.ImageFolder(root=sketch_datadir, transform=image_transforms)
    sketch_dataloaders = DataLoader(sketch_data, batch_size=args.sketch_batch_size, shuffle=True, num_workers=args.workers)

    view_data = MultiViewDataSet(view_datadir, transform=view_transform)
    view_dataloaders = DataLoader(view_data, batch_size=args.view_batch_size, shuffle=True, num_workers=args.workers)

    return sketch_dataloaders,val_sketch_dataloaders,view_dataloaders

def val(sketch_model,classifier,val_sketch_dataloader,use_gpu):
    with torch.no_grad():
        sketch_model.eval()
        classifier.eval()
        sketch_size = len(val_sketch_dataloader)
        sketch_dataloader_iter = iter(val_sketch_dataloader)
        total = 0.0
        correct = 0.0
        for batch_idx in range(sketch_size):
            sketch = next(sketch_dataloader_iter)
            sketch_data, sketch_labels = sketch
            if use_gpu:
                sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()

            sketch_features = sketch_model.forward(sketch_data)
            _, logits = classifier.forward(sketch_features)
            _, predicted = torch.max(logits.data, 1)
            total += sketch_labels.size(0)
            correct += (predicted == sketch_labels).sum()

        val_acc = correct.item() / total
        return val_acc


def train(sketch_model,view_model, classifier,criterion_soft,criterion_am,
          optimizer_model, sketch_dataloader,view_dataloader,use_gpu):
    sketch_model.train()
    view_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    view_size = len(view_dataloader)
    print(view_size)
    sketch_size = len(sketch_dataloader)
    print(sketch_size)



    sketch_dataloader_iter = iter(sketch_dataloader)

    view_dataloader_iter = iter(view_dataloader)


    for batch_idx in range(max(view_size,sketch_size)):
        if sketch_size>view_size:
            sketch = next(sketch_dataloader_iter)

            try:
                view=next(view_dataloader_iter)

            except:
                del view_dataloader_iter
                view_dataloader_iter=iter(view_dataloader)
                view=next(view_dataloader_iter)
        else:
            view = next(view_dataloader_iter)
            try:
                sketch=next(sketch_dataloader_iter)
            except:
                del sketch_dataloader_iter
                sketch_dataloader_iter=iter(sketch_dataloader)
                sketch=next(sketch_dataloader_iter)

        sketch_data, sketch_labels = sketch
        # view_data, view_labels = view
        view_data = view[0]
        view_labels = view[1]
        view_data = np.stack(view_data, axis=1)
        view_data = torch.from_numpy(view_data)
        # print(view_data)
        if use_gpu:
            sketch_data, sketch_labels,view_data,view_labels = sketch_data.cuda(), sketch_labels.cuda(),\
                                                               view_data.cuda(),view_labels.cuda()

        sketch_features = sketch_model.forward(sketch_data)
        view_features = view_model.forward(view_data)

        concat_feature = torch.cat((sketch_features,view_features),dim=0)
        concat_labels = torch.cat((sketch_labels,view_labels), dim=0)
        print(concat_labels)
        print(concat_feature[4:6])
        print("___________________________________________")
        """
        if view_labels.shape[0] % 2 ==0:
            index = randint(0,30)
            concat_feature = torch.cat((concat_feature,concat_feature[index].view(1,-1)),dim=0)
            concat_labels = torch.cat((concat_labels,concat_labels[index].view(1,)),dim=0)
        if view_labels.shape[0] % 2 !=0:
            index = randint(0,25)
            concat_feature = torch.cat((concat_feature,concat_feature[index].view(1,-1)),dim=0)
            concat_labels = torch.cat((concat_labels,concat_labels[index].view(1,)),dim=0)
            concat_feature = concat_feature[0:24]
            concat_labels = concat_labels[0:24]"""

        #print(concat_labels.shape)
        #if args.model == 'alexnet':
#         concat_feature = concat_feature.view(-1,2,args.feat_dim).transpose(0, 1).contiguous().view(-1, args.feat_dim)
#         concat_labels = concat_labels.view(-1,2,).transpose(0, 1).contiguous().view(-1,)
        #elif args.model == 'resnet50':
            #concat_feature = concat_feature.view(-1,2,args.feat_dim).transpose(0, 1).contiguous().view(-1, args.feat_dim)
            #concat_labels = concat_labels.view(-1,2,).transpose(0, 1).contiguous().view(-1,)




        _,logits = classifier.forward(concat_feature)
        cls_loss = criterion_am(logits, concat_labels)
        loss = cls_loss


        _, predicted = torch.max(logits.data, 1)
        total += concat_labels.size(0)
        correct += (predicted == concat_labels).sum()
        avg_acc = correct.item() / total

        optimizer_model.zero_grad()
        loss.backward()

        optimizer_model.step()
        # return

        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, max(view_size, sketch_size), loss.item()))
            # print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, view_size, loss.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        writer.add_scalar("Loss", loss.item(), args.count)
        writer.add_scalar("average accuracy", avg_acc, args.count)
    
    return avg_acc


def train_sketch(sketch_model, classifier, criterion_soft, criterion_am,optimizer_model, sketch_dataloader, use_gpu):
    sketch_model.train()
    classifier.train()

    total = 0.0
    correct = 0.0

    sketch_size = len(sketch_dataloader)

    sketch_dataloader_iter = iter(sketch_dataloader)

    for batch_idx in range(sketch_size):
        sketch = next(sketch_dataloader_iter)
        sketch_data, sketch_labels = sketch

        if use_gpu:
            sketch_data, sketch_labels = sketch_data.cuda(), sketch_labels.cuda()

        sketch_features = sketch_model.forward(sketch_data)

        concat_feature = sketch_features
        concat_labels = sketch_labels

        _, logits = classifier.forward(concat_feature)
        cls_loss = criterion_am(logits, concat_labels)
        loss = cls_loss

        _, predicted = torch.max(logits.data, 1)
        total += concat_labels.size(0)
        correct += (predicted == concat_labels).sum()
        avg_acc = correct.item() / total

        optimizer_model.zero_grad()
        loss.backward()

        optimizer_model.step()

        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, sketch_size, loss.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))

        args.count += 1

        writer.add_scalar("Loss", loss.item(), args.count)
        writer.add_scalar("average accuracy", avg_acc, args.count)

    return avg_acc


def main():
    print(os.getcwd())
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    best_acc=0

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")



    print("Creating model: {}".format(args.model))

    sketch_model = SketchModel(args.model, args.num_classes)
    sketch_model.cuda()

    view_model = MVCNN(args.model,args.num_classes)
    view_model.cuda()

    #if args.model == 'alexnet':
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)
    classifier.cuda()
    #elif args.model == 'resnet50':
        #classifier = Classifier(args.alph,args.feat_dim, args.num_classes)
        #classifier.cuda()

    ignored_keys = ["L2Classifier.fc2","L2Classifier.fc4"]
    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        view_model = nn.DataParallel(view_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    # best_sketch_model_path = args.model_dir + '/' +args.model+'_best_baseline_sketch_model' + '.pth'
    # best_view_model_path = args.model_dir + '/' +args.model+'_best_baseline_view_model' + '.pth'
    # best_classifier_model_path = args.model_dir + '/' +args.model+'_best_baseline_classifier' + '.pth'

    # best_sketch_model_path = args.model_dir + '/' + args.model + '_baseline_sketch_model' + '_' + str(14) + '.pth'
    # best_view_model_path = args.model_dir + '/' + args.model+'_baseline_view_model' + '_'+str(14)  + '.pth'
    # best_classifier_model_path = args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(14) + '.pth'

    # sketch_model.load_state_dict(torch.load(best_sketch_model_path))
    # view_model.load_state_dict(torch.load(best_view_model_path))
    # classifier.load_state_dict(torch.load(best_classifier_model_path))

    #sketch_model.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_sketch_model' + '_' +str(70) +  '.pth'))
    #view_model.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_view_model' + '_' + str(70) + '.pth'))
    #classifier.load_state_dict(torch.load(args.model_dir + '/' + 'softmax_classifier' + '_' + str(70) + '.pth'))
    #classifier = torch.load(args.model_dir + '/' + 'softmax_classifier' + '_' + str(70) + '.pth')
    #state_dict = {k: v for k, v in classifier.items() if k in classifier.keys()}



    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    #print(pretrained_dict)
    # Cross Entropy Loss and Center Loss
    criterion_am = FocalAMSoftMaxLoss()
    criterion_soft = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD([{"params":sketch_model.parameters()},{"params":view_model.parameters()},
                                       {"params":classifier.parameters(),"lr":args.lr_model*10}],
                                      lr=args.lr_model, momentum=0.9, weight_decay=1e-4)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.stepsize, eta_min=0, last_epoch=-1)

    sketch_trainloader,val_sketch_dataloader, view_trainloader = get_data(args.sketch_datadir, args.view_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        train(sketch_model,view_model,classifier,criterion_soft,criterion_am,
              optimizer_model, sketch_trainloader, view_trainloader,use_gpu)
        # return
        val_acc=val(sketch_model,classifier,val_sketch_dataloader,use_gpu)
        print("\tAverage Val Accuracy: %.4f" % (val_acc))

        # if epoch > 60 and epoch % args.save_model_freq == 0:
        if epoch >= 0 and epoch % 10 == 0:

            torch.save(sketch_model.state_dict(),
                       args.model_dir + '/' + args.model+'_baseline_sketch_model' + '_' +str(epoch) + '.pth')
            torch.save(view_model.state_dict(),
                       args.model_dir + '/' + args.model+'_baseline_view_model' + '_'+str(epoch)  + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' + args.model+'_baseline_classifier' + '_'+str(epoch)  + '.pth')
        if epoch> 20 and val_acc > best_acc:
        # if epoch > 0 and val_acc > best_acc:
            best_acc=val_acc
            torch.save(sketch_model.state_dict(),
                       args.model_dir + '/' +args.model+'_best_baseline_sketch_model' + '.pth')
            torch.save(view_model.state_dict(),
                       args.model_dir + '/' +args.model+ '_best_baseline_view_model' + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' +args.model+ '_best_baseline_classifier' + '.pth')

        if args.stepsize > 0: scheduler.step()
    writer.close()

def main_sketch():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    best_acc = 0

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating model: {}".format(args.model))

    sketch_model = SketchModel(args.model, args.num_classes)
    sketch_model.cuda()
    classifier = Classifier(args.alph, args.feat_dim, args.num_classes)
    classifier.cuda()
    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    criterion_am = AMSoftMaxLoss()
    criterion_soft = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD([{"params": sketch_model.parameters()},
                                       {"params": classifier.parameters(), "lr": args.lr_model * 10}],
                                      lr=args.lr_model, momentum=0.9, weight_decay=1e-4)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    sketch_trainloader, view_trainloader = get_data(args.sketch_datadir, args.view_datadir)
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print("++++++++++++++++++++++++++")
        # save model

        avg_acc = train_sketch(sketch_model, classifier, criterion_soft, criterion_am,
                        optimizer_model, sketch_trainloader, use_gpu)


        if epoch > 60 and epoch % args.save_model_freq == 0:
            torch.save(sketch_model.state_dict(),
                       args.model_dir + '/' + args.model + '_baseline_sketch_model' + '_' + str(epoch) + '.pth')
            torch.save(classifier.state_dict(),
                       args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(epoch) + '.pth')

        if args.stepsize > 0: scheduler.step()
    writer.close()

if __name__ == '__main__':
    main()