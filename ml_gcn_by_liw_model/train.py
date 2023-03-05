#@Time      :2019/12/22 0:30
#@Author    :zhounan
#@FileName  :train.py
import sys
sys.path.append('../')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import torchvision.models as models
from ml_gcn_by_liw_model.util import *
import torch
import torch.nn as nn
import torch.optim as optim
from ml_gcn_by_liw_model.models_cat import resnet101_wildcat
from ml_gcn_by_liw_model.models import Inceptionv3Rank
from ml_gcn_by_liw_model.voc import Voc2007Classification, Voc2012Classification
from itertools import combinations
import torchvision.transforms as transforms
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='VOC')
parser.add_argument('--data', default='../data/voc2007', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')                                        
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--image_size', default=299, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--k', default=0.2, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=0.7, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--maps', default=8, type=int,
                    metavar='N', help='number of maps per class (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--model_num', default=3, type=int, metavar='N',
                    help='number of model')                    

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def instance_wise_loss(output, y):
    y_i = torch.eq(y, torch.ones_like(y))
    y_not_i = torch.eq(y, -torch.ones_like(y))

    column = torch.unsqueeze(y_i, 2)
    row = torch.unsqueeze(y_not_i, 1)
    truth_matrix = column * row
    column = torch.unsqueeze(output, 2)
    row = torch.unsqueeze(output, 1)
    sub_matrix = column - row
    exp_matrix = torch.exp(-sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, (1, 2))
    y_i_sizes = torch.sum(y_i, 1)
    y_i_bar_sizes = torch.sum(y_not_i, 1)
    normalizers = y_i_sizes * y_i_bar_sizes
    normalizers_zero = torch.logical_not(torch.eq(normalizers, torch.zeros_like(normalizers)))
    normalizers = normalizers[normalizers_zero]
    sums = sums[normalizers_zero]
    loss = sums / normalizers
    loss = torch.sum(loss)
    return loss

def label_wise_loss(output, y):
    output = torch.transpose(output, 0, 1)
    y = torch.transpose(y,0, 1)
    return instance_wise_loss(output, y)
'''
def criterion(output, y):
    loss = 0.5 * instance_wise_loss(output, y) + label_wise_loss(output, y)
    return loss
'''
def save_checkpoint(model, is_best, best_score, save_model_path, filename='checkpoint.pth.tar'):
    filename_ = filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)
    if is_best:
        filename_best = 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)

        filename_best = os.path.join(save_model_path, 'model_best.pth.tar'.format(score=best_score))
        shutil.copyfile(filename, filename_best)
def save_checkpoint_list(model, i, is_best, best_score, save_model_path, filename='checkpoint.pth.tar'):
    filename_ = str(i) + filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)
    if is_best:
        filename_best = str(i)+'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)

        #filename_best = os.path.join(save_model_path, 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        #shutil.copyfile(filename, filename_best)
def train(model, epoch, optimizer, train_loader, criterion):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = np.asarray(target)
        target[target== 0] = 1
        target[target==-1] = 0
        target = torch.from_numpy(target)
        #print(target)
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        optimizer.zero_grad()
        #data.requires_grad = True
        output = model.forward(data)
        #target.requires_grad = True
        #print(output)
        loss = criterion(output, target)
        #print(loss)
        #sys.exit()
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
def train_list(model_list, epoch, optimizer_list, train_loader, criterion, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = np.asarray(target)
        target[target== 0] = 1
        target[target==-1] = 0
        target = torch.from_numpy(target)    
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for model in model_list:
            output = model(data)
            loss += criterion(output, target)/model_num

        
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
def train_gal(model_list, epoch, optimizer_list, train_loader, criterion, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = np.asarray(target)
        target[target== 0] = 1
        target[target==-1] = 0
        target = torch.from_numpy(target)      
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        data_list = []
        #d_l = []
        for i in range(model_num):
            data_list.append(data.clone().detach())
            #d_l.append(data.clone().detach())
        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for i, model in enumerate(model_list):
            data_list[i].requires_grad = True
            #d_l[i].requires_grad = True
            output = model(data_list[i])
            loss += criterion(output, target)/model_num
        
        #print(loss.require_grad)
        #print(data.require_grad)
        '''
        x_grads = torch.autograd.grad(loss, data_list,create_graph=True)
        
        x_grads = list(x_grads)
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)
            proj_loss += torch.exp(cosine_similarity)

        proj_loss = torch.log(proj_loss).mean()
        loss = loss + 0.5*proj_loss
        total_loss += loss.item()
        total_size += data.size(0)
        print(loss)
        '''
        #loss.backward(retain_graph=True)
        #print(data_list[0].grad)
        #sys.exit()
        loss.backward(retain_graph=True)
        #print(data_list[0].grad)
        
        #sys.exit()
        #print(loss)
        
        #x_grads = torch.autograd.grad(loss, data_list, retain_graph=True, create_graph = True)
        
        x_grads = []
        for i,data in enumerate(data_list):
            x_grads.append(data.grad)
        x_grads = list(x_grads)
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)
            proj_loss += torch.exp(cosine_similarity)

        proj_loss = 0.5*torch.log(proj_loss).mean()
        proj_loss.requires_grad = True
        loss = loss + proj_loss
        total_loss += loss.item()
        total_size += data.size(0)
        #print(loss)

        proj_loss.backward()
        #loss.backward()
        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
def test_list(model_list, test_loader, criterion, model_num):
    from utils import evaluate_metrics
    for model in model_list:
        model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            target = np.asarray(target)
            target[target== 0] = 1
            target[target==-1] = 0
            target = torch.from_numpy(target)        
            output = 0
            loss = 0
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            for model in model_list:
                o = model(data)
                output += o
                loss +=  criterion(o, target)/model_num
            output /= model_num
            test_loss += loss.item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets==-1] = 0
    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    metrics = evaluate_metrics.evaluate(targets, outputs, pred)
    print(metrics)
    return test_loss                
def test(model, test_loader, criterion):
    from utils import evaluate_metrics
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            target = np.asarray(target)
            target[target== 0] = 1
            target[target==-1] = 0
            target = torch.from_numpy(target)
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets==-1] = 0
    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    metrics = evaluate_metrics.evaluate(targets, outputs, pred)
    print(metrics)
    return test_loss

def main_voc2007():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)
    train_dataset = Voc2007Classification(args.data, 'train')
    val_dataset = Voc2007Classification(args.data, 'val')
    test_dataset = Voc2007Classification(args.data, 'test')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms
    test_dataset.transform = data_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    # load model
    model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
    print('classifier', model.classifier)
    print('spatial pooling', model.spatial_pooling)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    print('ok')

    # define optimizer
  

    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)        

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader, criterion)
        val_loss = test(model, val_loader, criterion)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model, test_loader, criterion)
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../checkpoint/cat/voc2007/',
                        filename='voc2007_checkpoint.pth.tar')
def main_gal_voc2007():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)
    train_dataset = Voc2007Classification(args.data, 'train')
    val_dataset = Voc2007Classification(args.data, 'val')
    test_dataset = Voc2007Classification(args.data, 'test')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms
    test_dataset.transform = data_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    # load model
    #model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
    model_list = []
    for i in range(args.model_num):
        model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        model_list.append(model)
    #model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    #optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list: 
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)           
        optimizer_list.append(optimizer)
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    print('ok')

    # define optimizer
  

    '''if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)        
    '''
    
    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_gal(model_list, epoch, optimizer_list, train_loader, criterion, args.model_num)
        val_loss = test_list(model_list, val_loader, criterion, args.model_num)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test_list(model_list, test_loader, criterion, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint_list(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/cat/new_gal_change_voc2007/',
                            filename='voc2012_checkpoint.pth.tar')
            i = i+1                        
def main_no_voc2007():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)
    train_dataset = Voc2007Classification(args.data, 'train')
    val_dataset = Voc2007Classification(args.data, 'val')
    test_dataset = Voc2007Classification(args.data, 'test')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms
    test_dataset.transform = data_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    # load model
    #model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
    model_list = []
    for i in range(args.model_num):
        model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        model_list.append(model)
    #model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    #optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list: 
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)           
        optimizer_list.append(optimizer)
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    print('ok')

    # define optimizer
  

    '''if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)        
    '''
    
    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_list(model_list, epoch, optimizer_list, train_loader, criterion, args.model_num)
        val_loss = test_list(model_list, val_loader, criterion, args.model_num)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test_list(model_list, test_loader, criterion, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint_list(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/cat/no_voc2007/',
                            filename='voc2012_checkpoint.pth.tar')
            i = i+1
def main_voc2012():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)

    train_dataset = Voc2012Classification('../data/voc2012', 'train')
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    test_dataset = Voc2012Classification('../data/voc2012', 'val')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.dataset.transform = data_transforms
    val_dataset.dataset.transform = data_transforms
    test_dataset.transform = data_transforms

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model, test_loader)
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../checkpoint/mlliw/voc2012/',
                        filename='voc2012_checkpoint.pth.tar')
if __name__ == '__main__':
    #main_voc2007()
    #main_no_voc2007()
    main_gal_voc2007()
    
    #main_voc2012()