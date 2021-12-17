import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import argparse

from create_dataset import *
import torch.backends.cudnn as cudnn

import time as time
import numpy as np
import matplotlib.pyplot as plt

import sys
import copy
torch.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='deconv_3d', type=str)
parser.add_argument('--model_size', default='M', type=str)
parser.add_argument('--task', default='semantic', type=str, help='choose task: semantic, depth, normal')
parser.add_argument('--dataroot', default='/home/yiren/datasets/nyuv2', type=str, help='dataset root')
parser.add_argument('--data_size', default='S', type=str)
parser.add_argument('--small_set', action='store_true')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--class_nb', default=13, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--total_epoch', default=100, type=int)
parser.add_argument('--window_size', default=10, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--analysis', action='store_true')
parser.add_argument('--logfile', default='./nohup/log.txt', type=str)
args = parser.parse_args()

if args.model_size == 'L':
    num_channels = [64,128,256,512,512]
elif args.model_size == 'M':
    num_channels = [32,64,128,256,256]

if args.data_size == 'S':
    data_ratio = 10
elif args.data_size == 'M':
    data_ratio = 5
elif args.data_size == 'L':
    data_ratio = 3
elif args.data_size == 'XL':
    data_ratio = 2


if args.arch == 'deconv_1d':
    from models.model_deconv_modules_1d import ShareNet, BranchNet
    GradDim = num_channels[0]
elif args.arch == 'deconv_2d':
    from models.model_deconv_modules_2d import ShareNet, BranchNet
    GradDim = num_channels[1]
elif args.arch == 'deconv_3d':
    from models.model_deconv_modules_3d import ShareNet, BranchNet
    GradDim = num_channels[2]
elif args.arch == 'deconv_4d':
    from models.model_deconv_modules_4d import ShareNet, BranchNet
    GradDim = num_channels[3]
elif args.arch == 'deconv_5d':
    from models.model_deconv_modules_5d import ShareNet, BranchNet
    GradDim = num_channels[4]

elif args.arch == 'resnet_1d':
    from models.model_resnet_modules_1d import ShareNet, BranchNet
    GradDim = num_channels[0]
elif args.arch == 'resnet_2d':
    from models.model_resnet_modules_2d import ShareNet, BranchNet
    GradDim = num_channels[0]
elif args.arch == 'resnet_3d':
    from models.model_resnet_modules_3d import ShareNet, BranchNet
    GradDim = num_channels[1]
elif args.arch == 'resnet_4d':
    from models.model_resnet_modules_4d import ShareNet, BranchNet
    GradDim = num_channels[2]
elif args.arch == 'resnet_5d':
    from models.model_resnet_modules_5d import ShareNet, BranchNet
    GradDim = num_channels[3]

elif args.arch == 'conv_decoder':
    from models.model_conv_decoder import ShareNet, BranchNet
    num_channels = [32,64,128,256,512]
    GradDim = num_channels[-1]

elif args.arch == 'res_decoder':
    from models.model_res_decoder import ShareNet, BranchNet
    num_channels = [32,64,128,256,512]
    GradDim = num_channels[-1]

if os.path.isdir('./checkpoints/%s/%s'%(args.data_size, args.arch)) == False:
    os.makedirs('./checkpoints/%s/%s'%(args.data_size, args.arch))

save_path1 = './checkpoints/%s/%s/share-%s.pth' % (args.data_size, args.arch, args.task)
save_path2 = './checkpoints/%s/%s/branch-%s.pth' % (args.data_size, args.arch, args.task)

def main():
    if args.task == 'semantic':
        num_classes = args.class_nb
    elif args.task == 'depth' :
        num_classes = 1
    elif args.task == 'normal':
        num_classes = 3

    # define model, optimizer and scheduler
    device = torch.device('cuda')
    share_model = ShareNet(num_channels=num_channels).to(device)
    task_model = BranchNet(output_dim=num_classes, num_channels=num_channels).to(device)

    share_optimizer = optim.SGD(share_model.parameters(), lr=args.lr)
    task_optimizer = optim.SGD(task_model.parameters(), lr=args.lr)

    share_scheduler = optim.lr_scheduler.StepLR(share_optimizer, step_size=0.5*args.total_epoch, gamma=0.5)
    task_scheduler = optim.lr_scheduler.StepLR(task_optimizer, step_size=0.5*args.total_epoch, gamma=0.5)

    # compute parameter space
    print('  Total params in share model: %.2fM' % (sum(p.numel() for p in share_model.parameters())/1000000.0))
    print('  Total params in task model: %.2fM' % (sum(p.numel() for p in task_model.parameters())/1000000.0))

    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC\n'
          'DEPTH_LOSS ABS_ERR REL_ERR\n'
          'NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

    # define dataset path
    dataset_path = args.dataroot
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, return_idx=False)
    nyuv2_test_set = NYUv2(root=dataset_path, train=False, return_idx=False)

    nyuv2_train_loader_full = torch.utils.data.DataLoader(
                             dataset=nyuv2_train_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4)

    nyuv2_train_loader_part = torch.utils.data.DataLoader(
                            dataset=Subset(nyuv2_train_set, [i for i in range(len(nyuv2_train_set)) if i % data_ratio == 0 ]),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)

    if args.small_set:
        nyuv2_train_loader = nyuv2_train_loader_part
    else:
        nyuv2_train_loader = nyuv2_train_loader_full
    print('#%d of batch(step) in the training loader'%(len(nyuv2_train_loader)))

    nyuv2_val_loader = torch.utils.data.DataLoader(
        dataset=Subset(nyuv2_test_set, [i for i in range(len(nyuv2_test_set)) if i % 2 != 0 ]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=Subset(nyuv2_test_set, [i for i in range(len(nyuv2_test_set)) if i % 2 == 0 ]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True)

    start_epoch = 0
    cudnn.benchmark = True
    best_val_loss = float('inf')
    val_losses = []
    test_losses = []

    current_share_states = [0]*args.window_size
    current_task_states = [0]*args.window_size

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, share_model, task_model, nyuv2_train_loader, share_optimizer, task_optimizer, device)
        val_loss = test(epoch, share_model, task_model, nyuv2_val_loader, device, val=True)
        test_loss = test(epoch, share_model, task_model, nyuv2_test_loader, device, val=False)
        share_scheduler.step()
        task_scheduler.step()

        val_losses.append(val_loss)
        test_losses.append(test_loss)

        del current_share_states[0]
        del current_task_states[0]
        current_share_states.append(copy.deepcopy(share_model.state_dict()))
        current_task_states.append(copy.deepcopy(task_model.state_dict()))

        if epoch < args.window_size:
            print('')
        else:
            avg_val_loss = np.mean(val_losses[-args.window_size:])
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                avg_test_loss = np.mean(test_losses[-args.window_size:])
                print('***** best found at end of epoch %d with loss: %.4f *****' %(epoch, avg_test_loss))
                # save the best model (on validation) within the window_size T
                if args.save_model:
                    best_index = -(args.window_size - np.argmin(val_losses[-args.window_size:]))
                    share_state = {'epoch': epoch + 1 + best_index, 'share_model_state': current_share_states[best_index]}
                    torch.save(share_state, save_path1)
                    task_state = {'epoch': epoch + 1 + best_index, 'task_model_state': current_task_states[best_index]}
                    torch.save(task_state, save_path2)
            else:
                print('')

    print('minimun test loss during the whole training is %.4f' % (np.min(test_losses)))


def train(epoch, share_model, task_model, nyuv2_train_loader, share_optimizer, task_optimizer, device):
    for param_group in share_optimizer.param_groups:
        LR = param_group['lr']

    share_model.train()
    task_model.train()
    running_loss = 0.0
    total = 0
    end = time.time()

    for i, data in enumerate(nyuv2_train_loader, 0):
        train_data, train_label, train_depth, train_normal = data
        train_data, train_label = train_data.to(device), train_label.type(torch.LongTensor).to(device)
        train_depth, train_normal = train_depth.to(device), train_normal.to(device)

        pred = share_model(train_data)
        pred = task_model(pred)
        if args.task == 'semantic':
            # pred = F.log_softmax(pred, dim=1)
            # loss = F.nll_loss(pred, train_label, ignore_index=-1)
            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred, train_label)
        if args.task == 'depth':
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(train_depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            loss = torch.sum(torch.abs(pred - train_depth) * binary_mask) / torch.nonzero(binary_mask).size(0)
        if args.task == 'normal':
            pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(train_normal, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            loss = 1 - torch.sum((pred * train_normal) * binary_mask) / torch.nonzero(binary_mask).size(0)

        share_optimizer.zero_grad()
        task_optimizer.zero_grad()
        loss.backward()
        share_optimizer.step()
        task_optimizer.step()

        # save and print statistics
        running_loss += loss.item()
        total += 1

        # measure elapsed time
        batch_time = (time.time() - end)

    print('TRAIN epoch %4d, loss=%.4f, time=%.1f,'%(epoch, running_loss/total, batch_time), end=' ')


def test(epoch, share_model, task_model, nyuv2_test_loader, device, val=False):
    share_model.eval()
    task_model.eval()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    total = 0
    end = time.time()

    with torch.no_grad():
        for i, (test_data, test_label, test_depth, test_normal) in enumerate(nyuv2_test_loader):
            test_data, test_label = test_data.to(device),  test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            pred = share_model(test_data)
            pred = task_model(pred)

            if args.task == 'semantic':
                pred = F.log_softmax(pred, dim=1)
                loss = F.nll_loss(pred, test_label, ignore_index=-1)
                running_loss2 += compute_miou(pred, test_label, device).item()
                running_loss3 += compute_acc(pred, test_label, device).item()
            if args.task == 'depth':
                # binary mark to mask out undefined pixel space
                binary_mask = (torch.sum(test_depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
                loss = torch.sum(torch.abs(pred - test_depth) * binary_mask) / torch.nonzero(binary_mask).size(0)
                loss2, loss3 = depth_error(pred, test_depth, device)
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
            if args.task == 'normal':
                pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
                # binary mark to mask out undefined pixel space
                binary_mask = (torch.sum(test_normal, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
                loss = 1 - torch.sum((pred * test_normal) * binary_mask) / torch.nonzero(binary_mask).size(0)
                loss2, loss3 = normal_error(pred, test_normal, device)
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()

            running_loss1 += loss.item()
            total += 1

            # measure elapsed time
            batch_time = (time.time() - end)

    if val==True:
        if args.task == 'semantic':
            print(' VAL, loss=%.4f, iou=%.4f, acc=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = -running_loss2/total
        elif args.task == 'depth':
            print(' VAL, loss=%.4f, abs_err=%.4f, rel_err=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss1/total
        elif args.task == 'normal':
            print(' VAL, loss=%.4f, mean=%.4f, median=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss2/total

    elif val==False:
        if args.task == 'semantic':
            print(' TEST, loss=%.4f, iou=%.4f, acc=%.4f, time=%.1f'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = -running_loss2/total
        elif args.task == 'depth':
            print(' TEST, loss=%.4f, abs_err=%.4f, rel_err=%.4f, time=%.1f'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss1/total
        elif args.task == 'normal':
            print(' TEST, loss=%.4f, mean=%.4f, median=%.4f, time=%.1f'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss2/total

    return res


def compute_miou(x_pred, x_output, device):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(args.class_nb):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).to(device))
            true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).to(device))
            mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size


def compute_acc(x_pred, x_output, device):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                                  torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        else:
            pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                                              torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
    return pixel_acc / batch_size


def depth_error(x_pred, x_output, device):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)


def normal_error(x_pred, x_output, device):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error)


def analysis():
    train_loss = []
    test_loss1 = []
    test_loss2 = []
    test_loss3 = []
    with open(args.logfile) as f:
        for line in f:
            if line.startswith('TRAIN'):
                train_loss.append(float(line.split(',')[2].split('=')[-1].split(']')[0]))
            if line.startswith('TEST'):
                test_loss1.append(float(line.split(',')[2].split('=')[-1].split(']')[0]))
                test_loss2.append(float(line.split(',')[3].split('=')[-1].split(']')[0]))
                test_loss3.append(float(line.split(',')[4].split('=')[-1].split(']')[0]))

    if args.task == 'semantic':
        label1 = 'acc'
        label2 = 'iou'
        label3 = 'test loss'
    elif args.task == 'depth':
        label1 = 'abs err'
        label2 = 'test loss'
        label3 = 'rel err'
    elif args.task == 'normal':
        label1 = 'test loss'
        label2 = 'mean'
        label3 = 'median'
    t = np.arange(0,args.total_epoch,1)
    if args.task == 'semantic':
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, train_loss, 'black', label='train loss')
        ax2.plot(t, test_loss2, 'red', label=label2)
        ax2.set_ylim([0, 0.3])
    elif args.task == 'depth':
        plt.plot(t, train_loss, 'black', label='train loss')
        plt.plot(t, test_loss2, 'red', label=label2)
        plt.ylim([0, 1.2])
    elif args.task == 'normal':
        plt.plot(t, train_loss, 'black', label='train loss')
        plt.plot(t, test_loss1, 'blue', label=label1)
        plt.ylim([0, 0.4])

    plt.legend()
    plt.savefig('demo.png')


if __name__ == '__main__':
    for arg in vars(args):
        print(arg, getattr(args, arg))
    if args.analysis:
        try:
            analysis()
        except KeyboardInterrupt:
            pass
    else:
        try:
            main()
        except KeyboardInterrupt:
            pass
