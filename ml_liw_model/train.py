# @Time      :2019/12/22 0:30
# @Author    :zhounan
# @FileName  :train.py
import sys

sys.path.append('../')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import torchvision.models as models
from ml_liw_model.util import *
import torch
import torch.optim as optim
from ml_liw_model.models import Inceptionv3Rank
from ml_liw_model.voc import Voc2007Classification, Voc2012Classification
from itertools import combinations
import torchvision.transforms as transforms
import os
import shutil

parser = argparse.ArgumentParser(description='VOC')
parser.add_argument('--data', default='../data/voc2007', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
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
parser.add_argument('--model_num', default=3, type=int, metavar='N',
                    help='number of model')
parser.add_argument('--loss_type', default='norm_cos', type=str,
                    help='ce, project_loss, GPMR, norm_cos, norm_cos1')
parser.add_argument('--para_norm', default=0.06, type=float,
                    help='para before norm_loss')
parser.add_argument('--para_cos', default=0.06, type=float,
                    help='para before cos_loss')
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
    y = torch.transpose(y, 0, 1)
    return instance_wise_loss(output, y)


def criterion(output, y):
    loss = 0.5 * instance_wise_loss(output, y) + label_wise_loss(output, y)
    return loss


def save_checkpoint(model, i, is_best, best_score, save_model_path, filename='checkpoint.pth.tar'):
    filename_ = str(i) + filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)
    if is_best:
        filename_best = str(i) + 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)

        # filename_best = os.path.join(save_model_path, 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        # shutil.copyfile(filename, filename_best)


def Ensemble_Entropy(y_true, y_pred, class_num, model_num):
    y_p = torch.split(y_pred, class_num, dim=-1)

    y_p_all = 0
    for i in range(model_num):
        y_p_all += y_p[i] / model_num
    log_offset = 1e-20

    sum = torch.sum(-torch.mul(y_p_all, torch.log(y_p_all + log_offset)), dim=-1)

    return sum


def log_det(y_true, y_pred, model_num, model_classes, batch_size_):
    '''
        zero = torch.tensor(0, dtype=torch.float32)
        det_offset = 1e-6
        bool_R_y_true = torch.not_equal(torch.ones_like(y_true) - y_true, zero)  # batch_size X (num_class X num_models), 2-D
        print(bool_R_y_true)
        mask_non_y_pred = torch.masked_select(y_pred, bool_R_y_true)  # batch_size X (num_class-x) X num_models, 1-D
        print('bool_R:')
        print(mask_non_y_pred.shape)
        mask_non_y_pred = torch.reshape(mask_non_y_pred,
                                     [batch_size, model_num, -1])  # batch_size X num_model X (num_class-1), 3-D
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p = 2, dim=2,
                                                    keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
        matrix = torch.matmul(mask_non_y_pred,
                           torch.transpose(mask_non_y_pred, 1, 2))  # batch_size X num_model X num_model, 3-D
        all_log_det = torch.logdet(
            matrix + det_offset * torch.unsqueeze(torch.eye(model_num), 0))  # batch_size X 1, 1-D
        return all_log_det
        '''

    y_t = torch.split(y_true, 1, dim=0)
    y_p = torch.split(y_pred, 1, dim=0)
    batch_size = len(y_t)

    # sys.exit()
    # print(y_t[0].shape)
    all_log_det = 0
    for i in range(batch_size):
        zero = torch.tensor(0, dtype=torch.float32)
        det_offset = 1e-6
        bool_R_y_true = torch.ne(torch.ones_like(y_t[i]) - y_t[i], zero)
        # print(bool_R_y_true)
        mask_non_y_pred = torch.masked_select(y_p[i], bool_R_y_true)
        # print('bool_R:')
        # print(mask_non_y_pred.shape)
        mask_non_y_pred = torch.reshape(mask_non_y_pred,
                                        [1, model_num, -1])
        # print('bool_R:')
        # print(mask_non_y_pred.shape)
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=2,
                                                       keepdim=True)
        # print('bool_R:')
        # print(mask_non_y_pred.shape)
        matrix = torch.matmul(mask_non_y_pred,
                              torch.transpose(mask_non_y_pred, 1, 2))
        # print('bool_R:')
        # print(matrix.shape)
        # print(matrix.device)
        # print(det_offset.device)
        # print(model_num.device)
        all_log_det += torch.logdet(
            matrix + det_offset * torch.unsqueeze(torch.eye(model_num), 0).cuda())
        # print('bool_R:')
        # print(all_log_det.shape)
    return all_log_det / batch_size


def train(model_list, epoch, optimizer_list, train_loader, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for model in model_list:
            output = model(data)
            loss += criterion(output, target) / model_num

        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def train_gpmr(model_list, epoch, optimizer_list, train_loader, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        data_list = []
        for i in range(model_num):
            data_list.append(data.clone().detach())

        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for i, model in enumerate(model_list):
            data_list[i].requires_grad = True
            output = model(data_list[i])
            loss += criterion(output, target) / model_num

        # print(loss.require_grad)
        # print(data.require_grad)
        x_grads = torch.autograd.grad(loss, data_list, create_graph=True)

        x_grads = list(x_grads)
        x_combinations = list(combinations(x_grads, 2))

        div_loss = 0
        eq_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)
            div = cosine_similarity + (1 / (model_num - 1))
            div = torch.mul(div, div)
            div_loss += div
        div_loss = (2 / (model_num * (model_num - 1))) * div_loss
        div_loss = div_loss.mean()
        for i in range(model_num):
            eq1 = torch.norm(x_grads[i], p=2)
            eq2 = 0
            for j in range(model_num):
                eq2 += torch.norm(x_grads[j], p=2)
            eq2 = eq2 / args.model_num
            eq = eq1 - eq2
            eq = torch.mul(eq, eq)
            eq_loss += eq
        eq_loss = eq_loss / args.model_num
        eq_loss = eq_loss.mean()
        loss = loss + 0.1 * div_loss + 10 * eq_loss

        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()

        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def get_project_loss_NC(x_grads, diversity_loss_type, para_norm, para_cos):
    if diversity_loss_type in ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5',
                               'norm_cos6', 'norm_cos7', 'norm_cos8', 'norm_cos9', 'norm_cos10', 'norm_cos11']:  # 除以g

        g = torch.zeros(x_grads[0].size(0), x_grads[0].size(1), x_grads[0].size(2), x_grads[0].size(3)).cuda()
        gn = torch.zeros(x_grads[0].size(0)).cuda()
        for i in range(args.model_num):
            g += x_grads[i] / args.model_num
            xi = x_grads[i].view(x_grads[i].size(0), -1)
            # print('111', torch.div(torch.norm(xi, dim=1), args.model_num))
            # print('222', gn)
            gn += torch.div(torch.norm(xi, dim=1), args.model_num)
            # print("=====", torch.norm(xi, dim=1).shape)   64

        norm = torch.zeros(x_grads[0].size(0)).cuda()
        for i in range(args.model_num):
            xi = x_grads[i].view(x_grads[i].size(0), -1)
            a = torch.norm(xi, dim=1) - gn
            a = torch.div(a, gn)  # ******************
            b = torch.zeros(x_grads[0].size(0)).cuda()
            for j in range(args.model_num):
                if j != i:
                    xj = x_grads[j].view(x_grads[j].size(0), -1)
                    tmp = torch.norm(xj, dim=1) - gn
                    b += torch.div(tmp, gn)  # ******************
                    # b += tmp / args.model_num
            proj = torch.mul(a, b)
            norm += torch.exp(proj)
        norm_loss = torch.log(norm).mean()

        cos = torch.zeros(x_grads[0].size(0)).cuda()
        qn = g.view(g.size(0), -1)
        for i in range(args.model_num):
            fi = x_grads[i].view(x_grads[i].size(0), -1)
            cos1 = torch.cosine_similarity(fi, qn, dim=1)
            cos1 = torch.exp(cos1)
            cos2 = torch.zeros(x_grads[0].size(0)).cuda()
            for j in range(args.model_num):
                if j != i:
                    fj = x_grads[j].view(x_grads[j].size(0), -1)
                    cos2 += torch.cosine_similarity(fj, qn, dim=1)
                    # cos2 += torch.exp(tmp)
            proj = torch.mul(cos1, cos2)
            cos += torch.exp(proj)
        cos_loss = torch.log(cos).mean()
        return para_norm * norm_loss + para_cos * cos_loss


def train_ncen(model_list, epoch, optimizer_list, train_loader, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        data_list = []
        for i in range(model_num):
            data_list.append(data.clone().detach())

        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for i, model in enumerate(model_list):
            data_list[i].requires_grad = True
            output = model(data_list[i])
            loss += criterion(output, target) / model_num

        # print(loss.require_grad)
        # print(data.require_grad)
        x_grads = torch.autograd.grad(loss, data_list, create_graph=True)

        diversity_loss = get_project_loss_NC(list(x_grads), args.loss_type, args.para_norm, args.para_cos)

        (loss + diversity_loss).backward()

        total_loss += (loss + diversity_loss).item()
        total_size += data.size(0)

        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def train_gal(model_list, epoch, optimizer_list, train_loader, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        data_list = []
        for i in range(model_num):
            data_list.append(data.clone().detach())

        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        for i, model in enumerate(model_list):
            data_list[i].requires_grad = True
            output = model(data_list[i])
            loss += criterion(output, target) / model_num

        # print(loss.require_grad)
        # print(data.require_grad)
        x_grads = torch.autograd.grad(loss, data_list, create_graph=True)

        x_grads = list(x_grads)
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)
            proj_loss += torch.exp(cosine_similarity)

        proj_loss = torch.log(proj_loss).mean()
        loss = loss + 0.5 * proj_loss
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()

        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def train_adp(model_list, epoch, optimizer_list, train_loader, model_num):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        data_list = []
        target_list = []
        for i in range(model_num):
            data_list.append(data.clone().detach())
            target_list.append(target.clone().detach())
        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss = 0
        y_pred_list = []
        for i, model in enumerate(model_list):
            # data_list[i].requires_grad = True

            output = model(data_list[i])
            y_pred_list.append(output)
            loss += criterion(output, target) / model_num
        y_pred = torch.cat((y_pred_list[0], y_pred_list[1], y_pred_list[2]), dim=1)
        y_true = torch.cat((target_list[0], target_list[1], target_list[2]), dim=1)
        EE = torch.mean(Ensemble_Entropy(y_true, y_pred, 20, model_num))
        log_dets = torch.mean(log_det(y_true, y_pred, model_num, 20, args.batch_size))

        loss = loss - 2 * EE - 0.5 * log_dets

        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()

        for optimizer in optimizer_list:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model_list, test_loader, model_num):
    from utils import evaluate_metrics
    for model in model_list:
        model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = 0
            loss = 0
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            for model in model_list:
                o = model(data)
                output += o
                loss += criterion(o, target) / model_num
            output /= model_num
            test_loss += loss.item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets == -1] = 0
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/none_voc2007/',
                            filename='voc2007_checkpoint.pth.tar')
            i += 1


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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/Rnone_voc2012/',
                            filename='voc2012_checkpoint.pth.tar')
            i += 1


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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_gal(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/gal_onlytest_voc2007/',
                            filename='gal_voc2007_checkpoint.pth.tar')
            i += 1


def main_gal_voc2012():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_gal(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/gal_voc2012/',
                            filename='voc2012_checkpoint.pth.tar')
            i += 1


def main_gpmr_voc2007():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []

    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_gpmr(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)

        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/gpmr_voc2007/',
                            filename='gpmr_voc2007_checkpoint.pth.tar')
            i += 1


def main_gpmr_voc2012():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_gpmr(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/new_gpmr_voc2012/',
                            filename='voc2012_checkpoint.pth.tar')
            i += 1


def main_adp_voc2007():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []

    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_adp(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/2_5_adp_voc2007/',
                            filename='voc2007_checkpoint.pth.tar')
            i = i + 1


def main_adp_voc2012():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_adp(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/2_5_adp_voc2012/',
                            filename='voc2012_checkpoint.pth.tar')
            i = i + 1


def main_ncen_voc2007():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []

    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_ncen(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)

        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/ncen_voc2007/',
                            filename='ncen_voc2007_checkpoint.pth.tar')
            i += 1


def main_ncen_voc2012():
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
    model_list = []
    for i in range(args.model_num):
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.aux_logit = False
        for param in model.parameters():
            param.requires_grad = False
        model = Inceptionv3Rank(model, num_classes)
        model_list.append(model)
    # model = models.inception_v3(pretrained=True)
    if use_cuda:
        for model in model_list:
            model = model.cuda()
    optimizer_list = []
    # optimizer = optim.Adam(model[0].model.fc.parameters())

    for model in model_list:
        optimizer = optim.Adam(model.model.fc.parameters())
        optimizer_list.append(optimizer)
    # Use exponential decay for fine-tuning optimizer
    print('optimizer set')
    scheduler_list = []
    for optimizer in optimizer_list:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        scheduler_list.append(scheduler)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train_ncen(model_list, epoch, optimizer_list, train_loader, args.model_num)
        val_loss = test(model_list, val_loader, args.model_num)
        for scheduler in scheduler_list:
            scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model_list, test_loader, args.model_num)
        i = 0
        for model in model_list:
            save_checkpoint(model, i, is_best, best_loss,
                            save_model_path='../checkpoint/mlliw/ncen_voc2012/',
                            filename='ncen_checkpoint.pth.tar')
            i += 1


if __name__ == '__main__':
    # main_voc2007()
    # main_voc2012()
    # main_gal_voc2012()
    main_gal_voc2007()
    # main_gpmr_voc2007()
    # main_gpmr_voc2012()
    # main_adp_voc2012()
