from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import argparse

from create_dataset import *
import torch.backends.cudnn as cudnn

import time as time
import numpy as np
import copy
import sys

import matplotlib.pyplot as plt
torch.set_printoptions(threshold=sys.maxsize)

def loopy(dataloader):
    while True:
        for x in iter(dataloader): yield x

class Gradient(nn.Module):
    def __init__(self, dim=128):
        super(Gradient, self).__init__()
        self.coef = nn.Parameter(torch.ones(dim))
        self.dim = dim

        for param in self.parameters():
            with torch.no_grad():
                var = torch.empty(dim).normal_(0,0.02)
            param.data.add_(var)

    def forward(self, g, H=36, W=48, B=2):
        p = torch.zeros(self.dim, self.dim).cuda()
        torch.diagonal(p).copy_(self.coef)

        g = g.permute(0,2,3,1)
        y = F.linear(g, p)
        y = y.permute(0,3,1,2)
        return y

class DiagGradient(nn.Module):
    def __init__(self, dim=128):
        super(DiagGradient, self).__init__()
        self.coef = nn.Parameter(torch.ones(1,dim,1,1))

        for param in self.parameters():
            with torch.no_grad():
                var = torch.empty(1,dim,1,1).normal_(0,0.02)
            param.data.add_(var)

    def forward(self, g, H=36, W=48, B=2):
        y = torch.mul(self.coef, g)
        return y

class MatGradient(nn.Module):
    def __init__(self, dim=128):
        super(MatGradient, self).__init__()
        self.coef = nn.Parameter(torch.zeros(dim,dim))

        diag = torch.eye(dim)
        for param in self.parameters():
            with torch.no_grad():
                var = torch.empty(dim,dim).normal_(0,0.02)
            param.data.add_(var)
            param.data.add_(diag)

    def forward(self, g, H=36, W=48, B=2):
        g = g.permute(0,2,3,1)
        y = F.linear(g, self.coef)
        y = y.permute(0,3,1,2)
        return y

class RankDefGradient(nn.Module):
    def __init__(self, dim=128, rank=3):
        super(RankDefGradient, self).__init__()
        self.diag = nn.Parameter(torch.ones(1,1,1,dim))
        self.coef = nn.Parameter(torch.empty(dim,rank).normal_(0,0.05))
        self.dim = dim

        for (name, param) in self.named_parameters():
            if name == 'diag':
                with torch.no_grad():
                    var = torch.empty(1,1,1,dim).normal_(0,0.02)
                param.data.add_(var)

    def forward(self, g, H=36, W=48, B=2):
        p = torch.mm(self.coef, self.coef.T)

        g = g.permute(0,2,3,1)
        y = torch.mul(g, self.diag) + F.linear(g, p)
        y = y.permute(0,3,1,2)
        return y

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='deconv_3d', type=str)
parser.add_argument('--rank_projection', action='store_true')
parser.add_argument('--complete_projection', action='store_true')
parser.add_argument('--rank', default=5, type=int)
parser.add_argument('--model_size', default='M', type=str)
parser.add_argument('--data_size', default='S', type=str)
parser.add_argument('--dataroot', default='/home/yiren/datasets/nyuv2', type=str, help='dataset root')
parser.add_argument('--target_task', default='depth', type=str)
parser.add_argument('--auxi_task', default='normal', type=str)
parser.add_argument('--window_size', default=10, type=int)
parser.add_argument('--class_nb', default=13, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--gradient_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_controller', action='store_true')
parser.add_argument('--analysis', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--fine_share_lr', default=0.01, type=float)
parser.add_argument('--fine_target_lr', default=0.1, type=float)
parser.add_argument('--fine_load_pretrain_target', action='store_true')
args = parser.parse_args()

if args.rank_projection:
    controller_folder = 'rank-%s'%(args.rank)
elif args.complete_projection:
    controller_folder = 'rank-complete'
else:
    controller_folder = 'rank-0'

if args.model_size == 'L':
    num_channels = [64,128,256,512, 512]
elif args.model_size == 'M':
    num_channels = [32,64,128,256, 256]

if args.data_size == 'S':
    data_ratio = 10
    fine_epoch = 200
elif args.data_size == 'M':
    data_ratio = 5
    fine_epoch = 100
elif args.data_size == 'L':
    data_ratio = 3
    fine_epoch = 70
elif args.data_size == 'XL':
    data_ratio = 2
    fine_epoch = 40

total_step = 40000
lr_step = 20000
step_size = 200
print('********** The training has total steps #%d and reduce LR at step #%d **********'%(total_step, lr_step))


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

if args.target_task == 'semantic':
    target_output_dim = args.class_nb
elif args.target_task == 'depth':
    target_output_dim = 1
elif args.target_task == 'normal':
    target_output_dim = 3

if args.auxi_task == 'semantic':
    auxi_output_dim = args.class_nb
elif args.auxi_task == 'depth':
    auxi_output_dim = 1
elif args.auxi_task == 'normal':
    auxi_output_dim = 3

add_save_path = str(args.update_freq)
if os.path.isdir('./checkpoints/%s/%s/%s/%s/%s/auxi-%s'%(args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task)) == False:
    os.makedirs('./checkpoints/%s/%s/%s/%s/%s/auxi-%s'%(args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task))

save_path1 = './checkpoints/%s/%s/%s/%s/meta-share-%s-%s.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.auxi_task, args.target_task)
save_path2 = './checkpoints/%s/%s/%s/%s/meta-branch-%s-%s.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.auxi_task, args.target_task)
def controller_save_path(epoch):
    return './checkpoints/%s/%s/%s/%s/%s/auxi-%s/target-%s-auxi-%s-epoch-%d.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, args.target_task, args.auxi_task, epoch)

def main():
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

    print('#%d of batch(step) in the share training loader'%(len(nyuv2_train_loader_full)))
    print('#%d of batch(step) in the target training loader'%(len(nyuv2_train_loader_part)))

    train_iter_full = loopy(nyuv2_train_loader_full)
    train_iter_part = loopy(nyuv2_train_loader_part)

    device = torch.device('cuda')
    if not args.finetune:
        # Model definition
        share_model = ShareNet(num_channels=num_channels).to(device)
        target_model = BranchNet(output_dim=target_output_dim, num_channels=num_channels).to(device)
        auxi_model = BranchNet(output_dim=auxi_output_dim, num_channels=num_channels).to(device)

        # Optimizer definition
        share_optimizer = optim.SGD(share_model.parameters(), lr=args.lr)
        target_optimizer =  optim.SGD(target_model.parameters(), lr=args.lr)
        auxi_optimizer =  optim.SGD(auxi_model.parameters(), lr=args.lr)

        # Scheduler definition
        share_scheduler = optim.lr_scheduler.StepLR(share_optimizer, step_size=lr_step, gamma=0.5)
        target_scheduler = optim.lr_scheduler.StepLR(target_optimizer, step_size=lr_step, gamma=0.5)
        auxi_scheduler = optim.lr_scheduler.StepLR(auxi_optimizer, step_size=lr_step, gamma=0.5)

        # Gradient projection matrix
        if args.rank_projection:
            controller = RankDefGradient(dim=GradDim, rank=args.rank).to(device)
        elif args.complete_projection:
            controller = MatGradient(dim=GradDim).to(device)
        else:
            controller = DiagGradient(dim=GradDim).to(device)
        controller_optimizer = optim.Adam(controller.parameters(), lr=args.gradient_lr, weight_decay=1e-5)
        controller_scheduler = optim.lr_scheduler.StepLR(controller_optimizer, step_size=lr_step, gamma=0.5)

        # compute parameter space
        print('  Total params in share model: %.2fM' % (sum(p.numel() for p in share_model.parameters())/1000000.0))
        print('  Total params in task model: %.2fM' % (sum(p.numel() for p in target_model.parameters())/1000000.0))
        print('  Total params in auxi model: %.2fM' % (sum(p.numel() for p in auxi_model.parameters())/1000000.0))

        print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC\n'
              'DEPTH_LOSS ABS_ERR REL_ERR\n'
              'NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

        start_step = 0
        cudnn.benchmark = True

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        test_losses = []

        current_share_states = [0]*args.window_size
        current_target_states = [0]*args.window_size

        # for initial running loss on training stat
        running_loss = 0.0
        total = 0
        best_epoch = 0
        end_time = time.time()
        for step in range(start_step, total_step):
            epoch = int(step/step_size) - 1
            if epoch - best_epoch > 100:
                sys.exit('Early Stopping!')

            data_train_full = next(train_iter_full)
            current_state = train_share_model_with_auxi(step, data_train_full, share_model, auxi_model, share_optimizer, auxi_optimizer, controller, device)
            if step % args.update_freq == 0:
                data_train_part = next(train_iter_part)
                tr_loss = train_controller_and_target(step, data_train_part, share_model, target_model, auxi_model, share_optimizer, target_optimizer, auxi_optimizer, controller, controller_optimizer, current_state, device)
                running_loss += tr_loss
                total += 1

            # learning rate schedule
            share_scheduler.step()
            target_scheduler.step()
            auxi_scheduler.step()
            controller_scheduler.step()

            if step % step_size == 0 and step != 0:
                for param_group in share_optimizer.param_groups:
                    LR = param_group['lr']
                train_losses.append(running_loss/total)
                print('TRAIN %4d, loss=%.4f, LR %.3f, time %.1f sec' % (epoch, running_loss/total, LR,time.time()-end_time), end=' ')
                running_loss = 0.0
                total = 0
                end_time = time.time()

                val_loss = test(epoch, share_model, target_model, nyuv2_val_loader, device, val=True)
                test_loss = test(epoch, share_model, target_model, nyuv2_test_loader, device, val=False)
                val_losses.append(val_loss)
                test_losses.append(test_loss)

                del current_share_states[0]
                del current_target_states[0]
                current_share_states.append(copy.deepcopy(share_model.state_dict()))
                current_target_states.append(copy.deepcopy(target_model.state_dict()))

                if epoch < args.window_size:
                    print('')
                else:
                    avg_val_loss = np.mean(val_losses[-args.window_size:])
                    avg_train_loss = np.mean(train_losses[-args.window_size:])
                    if avg_val_loss < best_val_loss:
                        best_epoch = epoch
                        best_val_loss = avg_val_loss
                        avg_test_loss = np.mean(test_losses[-args.window_size:])
                        avg_train_loss = np.mean(train_losses[-args.window_size:])
                        print('***** best found at end of epoch %d with train loss: %.4f, val loss: %.4f and test loss: %.4f *****' %(epoch, avg_train_loss, best_val_loss, avg_test_loss))
                        # save the best model (on validation) within the window_size T
                        if args.save_model:
                            best_index = -(args.window_size - np.argmin(val_losses[-args.window_size:]))
                            share_state = {'epoch': epoch + 1 + best_index, 'share_model_state': current_share_states[best_index]}
                            torch.save(share_state, save_path1)
                            task_state = {'epoch': epoch + 1 + best_index, 'task_model_state': current_target_states[best_index]}
                            torch.save(task_state, save_path2)
                    else:
                        print('')

                # Save the projection matrix at each epoch
                controller_state = {'controller_state': controller.state_dict()}
                if args.save_controller:
                    torch.save(controller_state, controller_save_path(epoch))

        print('mean train loss at the end of training is %.4f' % (np.mean(train_losses[-args.window_size:])))
        print('minimun test loss during the whole training is %.4f' % (np.min(test_losses)))

    else:
        # Model definition
        share_model = ShareNet(num_channels=num_channels).to(device)
        target_model = BranchNet(output_dim=target_output_dim, num_channels=num_channels).to(device)

        # Load the saved model
        share_model.load_state_dict(torch.load(save_path1)['share_model_state'])
        print('  ==> model dir: %s'%(save_path1))
        print('  ==> share model re-loaded from meta step: %d' % (torch.load(save_path1)['epoch']) )
        if args.fine_load_pretrain_target:
            target_model.load_state_dict(torch.load(save_path2)['task_model_state'])
            print('  ==> target model re-loaded from meta step: %d' % (torch.load(save_path2)['epoch']) )


        # Optimizer definition
        share_optimizer = optim.SGD(share_model.parameters(), lr=args.fine_share_lr)
        target_optimizer =  optim.SGD(target_model.parameters(), lr=args.fine_target_lr)

        # Scheduler definition
        share_scheduler = optim.lr_scheduler.StepLR(share_optimizer, step_size=0.5*fine_epoch, gamma=0.5)
        target_scheduler = optim.lr_scheduler.StepLR(target_optimizer, step_size=0.5*fine_epoch, gamma=0.5)

        # compute parameter space
        print('  Total params in share model: %.2fM' % (sum(p.numel() for p in share_model.parameters())/1000000.0))
        print('  Total params in task model: %.2fM' % (sum(p.numel() for p in target_model.parameters())/1000000.0))

        best_val_loss = float('inf')
        best_epoch = 0
        val_losses = []
        test_losses = []
        for epoch in range(fine_epoch):
            if epoch - best_epoch > 50 and args.target_task=='semantic':
                sys.exit('Early Stopping!')

            train(epoch, share_model, target_model, nyuv2_train_loader_part, share_optimizer, target_optimizer, device)
            val_loss = test(epoch, share_model, target_model, nyuv2_val_loader, device, val=True)
            test_loss = test(epoch, share_model, target_model, nyuv2_test_loader, device, val=False)

            share_scheduler.step()
            target_scheduler.step()

            val_losses.append(val_loss)
            test_losses.append(test_loss)

            if epoch < args.window_size:
                print('')
            else:
                avg_val_loss = np.mean(val_losses[-args.window_size:])
                if avg_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = avg_val_loss
                    avg_test_loss = np.mean(test_losses[-args.window_size:])
                    print('***** best at epoch %d with val: %.4f and test: %.4f *****' %(epoch, best_val_loss, avg_test_loss))
                else:
                    print('')

        print('minimun test loss during the whole training is %.4f' % (np.min(test_losses)))


def train_share_model_with_auxi(step, data, share_model, auxi_model, share_optimizer, auxi_optimizer, controller, device):
    for param_group in share_optimizer.param_groups:
        share_lr = param_group['lr']

    # Set the mode for each part of model
    share_model.train()
    auxi_model.train()
    controller.eval()

    train_data, train_label, train_depth, train_normal = data
    train_data, train_label = train_data.to(device), train_label.type(torch.LongTensor).to(device)
    train_depth, train_normal = train_depth.to(device), train_normal.to(device)

    data = {}
    data['semantic'], data['depth'], data['normal'] = train_label, train_depth, train_normal

    auxi_label = data[args.auxi_task]

    # forward pass through shared network
    inter_output = share_model(train_data)

    # input for specific branch model for aux-task
    auxi_input = inter_output.clone().detach().requires_grad_()

    # forward pass through specific branch aux-task
    auxi_pred = auxi_model(auxi_input)

    # compute loss for aux-task
    if args.auxi_task == 'semantic':
        auxi_loss = nn.CrossEntropyLoss(ignore_index=-1)(auxi_pred, auxi_label)
    elif args.auxi_task == 'depth':
        binary_mask2 = (torch.sum(auxi_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        auxi_loss = torch.sum(torch.abs(auxi_pred - auxi_label) * binary_mask2) / torch.nonzero(binary_mask2).size(0)
    elif args.auxi_task == 'normal':
        auxi_pred = auxi_pred / torch.norm(auxi_pred, p=2, dim=1, keepdim=True)
        binary_mask3 = (torch.sum(auxi_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        auxi_loss = 1 - torch.sum((auxi_pred * auxi_label) * binary_mask3) / torch.nonzero(binary_mask3).size(0)

    # backward pass on aux-task model
    auxi_optimizer.zero_grad()
    auxi_loss.backward()
    ########## UPDATE THE AUXILIARY MODEL ##########
    auxi_optimizer.step()

    # the gradients used for backward pass in shared network
    g_auxi = auxi_input.grad.clone().detach()
    with torch.no_grad():
        inter_grads = controller(g_auxi, inter_output.size(2), inter_output.size(3), inter_output.size(0))

    # backward pass on the shared network with modified gradients
    share_model.zero_grad()
    inter_output.backward(inter_grads)
    ########## UPDATE THE SHARED MODEL ##########
    share_optimizer.step()

    # Save the state of share model, primay task model and auxiliary tasks model.
    current_state = {'share_model_state': copy.deepcopy(share_model.state_dict())}
    current_state['auxi_model_state'] = copy.deepcopy(auxi_model.state_dict())
    return current_state


def train_controller_and_target(epoch, data, share_model, target_model, auxi_model, share_optimizer, target_optimizer, auxi_optimizer, controller, controller_optimizer, current_state, device):
    for param_group in share_optimizer.param_groups:
        share_lr = param_group['lr']

    share_model.train()
    auxi_model.train()
    target_model.train()
    controller.train()

    # the input and labels for 3 tasks
    train_data, train_label, train_depth, train_normal = data
    train_data, train_label = train_data.to(device), train_label.type(torch.LongTensor).to(device)
    train_depth, train_normal = train_depth.to(device), train_normal.to(device)

    data = {}
    data['semantic'], data['depth'], data['normal'] = train_label, train_depth, train_normal

    target_label = data[args.target_task]
    auxi_label = data[args.auxi_task]

    # forward pass through shared network
    inter_output = share_model(train_data)

    # input for specific branch model for auxiliary task
    auxi_input = inter_output.clone().detach().requires_grad_()

    # forward pass through specific branch model for auxiliary task
    auxi_pred = auxi_model(auxi_input)

    # compute loss for aux-task
    if args.auxi_task == 'semantic':
        auxi_loss = nn.CrossEntropyLoss(ignore_index=-1)(auxi_pred, auxi_label)
    elif args.auxi_task == 'depth':
        binary_mask2 = (torch.sum(auxi_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        auxi_loss = torch.sum(torch.abs(auxi_pred - auxi_label) * binary_mask2) / torch.nonzero(binary_mask2).size(0)
    elif args.auxi_task == 'normal':
        auxi_pred = auxi_pred / torch.norm(auxi_pred, p=2, dim=1, keepdim=True)
        binary_mask3 = (torch.sum(auxi_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        auxi_loss = 1 - torch.sum((auxi_pred * auxi_label) * binary_mask3) / torch.nonzero(binary_mask3).size(0)

    # backward pass on auxiliary model
    auxi_optimizer.zero_grad()
    auxi_loss.backward()
    # Do NOT update the parameters in auxiliary model

    # the gradients used for backward pass in shared network
    g_auxi = auxi_input.grad.clone().detach()
    inter_grads = controller(g_auxi, inter_output.size(2), inter_output.size(3), inter_output.size(0))

    # current theta_1
    fast_weights = OrderedDict((name, param) for (name, param) in share_model.named_parameters())

    # create_graph flag for computing second-derivative
    share_optimizer.zero_grad()
    share_grads = torch.autograd.grad(outputs=inter_output, inputs=share_model.parameters(), grad_outputs=inter_grads, create_graph=True)
    share_params = [param.data for param in list(share_model.parameters())]

    # compute theta^+ by applying sgd
    fast_weights = OrderedDict((name, param - share_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), share_grads, share_params))

    # paste the running mean and var of BN here
    fast_bn_mean = OrderedDict((m, share_model.state_dict()[m]) for m in share_model.state_dict() if 'bn' in m and 'mean' in m)
    fast_bn_var = OrderedDict((m, share_model.state_dict()[m]) for m in share_model.state_dict() if 'bn' in m and 'var' in m)

    # second forward pass
    share_model.zero_grad()
    inter_output_new = share_model.forward(train_data, fast_weights, fast_bn_mean, fast_bn_var)
    target_input = inter_output_new.clone().detach().requires_grad_()

    target_pred = target_model(target_input)
    if args.target_task == 'semantic':
        target_loss = nn.CrossEntropyLoss(ignore_index=-1)(target_pred, target_label)
    elif args.target_task == 'depth':
        binary_mask2 = (torch.sum(target_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        target_loss = torch.sum(torch.abs(target_pred - target_label) * binary_mask2) / torch.nonzero(binary_mask2).size(0)
    elif args.target_task == 'normal':
        target_pred = target_pred / torch.norm(target_pred, p=2, dim=1, keepdim=True)
        binary_mask3 = (torch.sum(target_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        target_loss = 1 - torch.sum((target_pred * target_label) * binary_mask3) / torch.nonzero(binary_mask3).size(0)

    # This measures the training loss on the target task
    running_loss = target_loss.item()

    #################################
    target_model.zero_grad()
    target_loss.backward()
    inter_grads_new = target_input.grad.clone().detach()
    ########## UPDATE THE TARGET MODEL ##########
    target_optimizer.step()

    controller_optimizer.zero_grad()
    inter_output_new.backward(inter_grads_new)
    ########## UPDATE THE CONTROLLER ##########
    controller_optimizer.step()

    ########## UPDATE THE SHARE ALSO ##########
    with torch.no_grad():
        for (name, param) in share_model.named_parameters():
            param -= share_lr * param.grad
    ########## UPDATE THE SHARE ALSO ##########

    # share_model.load_state_dict(current_state['share_model_state'])
    auxi_model.load_state_dict(current_state['auxi_model_state'])
    return running_loss


def train_target(epoch, data, share_model, target_model, share_optimizer, target_optimizer, device):
    for param_group in share_optimizer.param_groups:
        share_lr = param_group['lr']

    share_model.train()
    target_model.train()

    # the input and labels for 3 tasks
    train_data, train_label, train_depth, train_normal = data
    train_data, train_label = train_data.to(device), train_label.type(torch.LongTensor).to(device)
    train_depth, train_normal = train_depth.to(device), train_normal.to(device)

    data = {}
    data['semantic'], data['depth'], data['normal'] = train_label, train_depth, train_normal

    target_label = data[args.target_task]

    # forward pass through shared network
    inter_output = share_model(train_data)
    target_input = inter_output.clone().detach().requires_grad_()

    target_pred = target_model(target_input)
    if args.target_task == 'semantic':
        target_loss = nn.CrossEntropyLoss(ignore_index=-1)(target_pred, target_label)
    elif args.target_task == 'depth':
        binary_mask2 = (torch.sum(target_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        target_loss = torch.sum(torch.abs(target_pred - target_label) * binary_mask2) / torch.nonzero(binary_mask2).size(0)
    elif args.target_task == 'normal':
        target_pred = target_pred / torch.norm(target_pred, p=2, dim=1, keepdim=True)
        binary_mask3 = (torch.sum(target_label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
        target_loss = 1 - torch.sum((target_pred * target_label) * binary_mask3) / torch.nonzero(binary_mask3).size(0)

    # This measures the training loss on the target task
    running_loss = target_loss.item()

    #################################
    target_model.zero_grad()
    target_loss.backward()
    inter_grads_new = target_input.grad.clone().detach()
    ########## UPDATE THE TARGET MODEL ##########
    target_optimizer.step()

    return running_loss


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
        if args.target_task == 'semantic':
            # pred = F.log_softmax(pred, dim=1)
            # loss = F.nll_loss(pred, train_label, ignore_index=-1)
            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred, train_label)
        if args.target_task == 'depth':
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(train_depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            loss = torch.sum(torch.abs(pred - train_depth) * binary_mask) / torch.nonzero(binary_mask).size(0)
        if args.target_task == 'normal':
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

    print('FINETUNE epoch %4d, loss=%.4f, time=%.1f,'%(epoch, running_loss/total, batch_time), end=' ')


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

            if args.target_task == 'semantic':
                pred = F.log_softmax(pred, dim=1)
                loss = F.nll_loss(pred, test_label, ignore_index=-1)
                running_loss2 += compute_miou(pred, test_label, device).item()
                running_loss3 += compute_acc(pred, test_label, device).item()
            if args.target_task == 'depth':
                # binary mark to mask out undefined pixel space
                binary_mask = (torch.sum(test_depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
                loss = torch.sum(torch.abs(pred - test_depth) * binary_mask) / torch.nonzero(binary_mask).size(0)
                loss2, loss3 = depth_error(pred, test_depth, device)
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
            if args.target_task == 'normal':
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
        if args.target_task == 'semantic':
            print(' VAL, loss=%.4f, iou=%.4f, acc=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = -running_loss2/total
        elif args.target_task == 'depth':
            print(' VAL, loss=%.4f, abs_err=%.4f, rel_err=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss1/total
        elif args.target_task == 'normal':
            print(' VAL, loss=%.4f, mean=%.4f, median=%.4f, time=%.1f,'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss2/total

    elif val==False:
        if args.target_task == 'semantic':
            print(' TEST, loss=%.4f, iou=%.4f, acc=%.4f, time=%.1f'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = -running_loss2/total
        elif args.target_task == 'depth':
            print(' TEST, loss=%.4f, abs_err=%.4f, rel_err=%.4f, time=%.1f'%(running_loss1/total, running_loss2/total, running_loss3/total, batch_time), end=' ')
            res = running_loss1/total
        elif args.target_task == 'normal':
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
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    if os.path.isdir('./out/%s/%s/%s/%s/%s/auxi-%s'% (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task)) == False:
        os.makedirs('./out/%s/%s/%s/%s/%s/auxi-%s'% (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task))

    if args.rank_projection:
        if args.rank == 3:
            controller = RankDefGradient(dim=GradDim, rank=args.rank)

            total_epoch = int(total_step/step_size) - 1
            for epoch in range(total_epoch):
                new_save_path = './checkpoints/%s/%s/%s/%s/%s/auxi-%s/target-%s-auxi-%s-epoch-%d.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, args.target_task, args.auxi_task, epoch)
                checkpoint = torch.load(new_save_path, map_location='cpu')
                diag = checkpoint['controller_state']['diag'].numpy()
                try:
                    m = checkpoint['controller_state']['m'].numpy()
                except KeyError:
                    m = checkpoint['controller_state']['coef'].numpy()

                x = np.arange(0,GradDim,1)
                plt.figure(1,figsize=(8, 6))
                for param in controller.parameters():
                    for i in range(4):
                        plt.subplot(4,1,i+1)
                        plt.grid()
                        if i == 0:
                            plt.title('epoch: %d'%epoch)
                        if i < 3:
                            plt.plot(x,  m[:,i])
                            plt.ylim([-1.5, 1.5])
                            plt.grid()
                        else:
                            plt.plot(x, diag[0,0,0,:] + m[:,0]**2 + m[:,1]**2 + m[:,2]**2)
                            plt.ylim([-1.5, 3.5])
                            plt.grid()
                plt.savefig('./out/%s/%s/%s/%s/%s/auxi-%s/%d.png'% (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, epoch))
                plt.close()
        else:
            pass
    elif args.complete_projection:
        controller = MatGradient(dim=GradDim)

        total_epoch = int(total_step/step_size) - 1
        for epoch in range(total_epoch):
            new_save_path = './checkpoints/%s/%s/%s/%s/%s/auxi-%s/target-%s-auxi-%s-epoch-%d.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, args.target_task, args.auxi_task, epoch)
            checkpoint = torch.load(new_save_path, map_location='cpu')

            try:
                m = checkpoint['controller_state']['m'].numpy()
            except KeyError:
                m = checkpoint['controller_state']['coef'].numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = y = np.arange(0, GradDim, 1)
            X, Y = np.meshgrid(x, y)

            ax.plot_surface(X, Y, m)

            ax.set_xlabel('X dim')
            ax.set_ylabel('Y dim')
            ax.set_zlabel('grad')
            ax.set_zlim([-0.5,0.5])

            plt.savefig('./out/%s/%s/%s/%s/%s/auxi-%s/%d.png'% (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, epoch))
            plt.close()
    else:
        # define model, optimizer and scheduler
        controller = DiagGradient(dim=GradDim)

        total_epoch = int(total_step/step_size) - 1
        for epoch in range(total_epoch):
            new_save_path = './checkpoints/%s/%s/%s/%s/%s/auxi-%s/target-%s-auxi-%s-epoch-%d.pth' % (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, args.target_task, args.auxi_task, epoch)
            checkpoint = torch.load(new_save_path, map_location='cpu')

            try:
                data = checkpoint['controller_state']['m']
            except KeyError:
                data = checkpoint['controller_state']['coef']
            x = np.arange(0,GradDim,1)
            plt.figure(1)
            plt.subplot(1,1,1)
            plt.title('epoch: %d'%epoch)
            plt.plot(x, data[0,:,0,0].numpy())
            plt.ylim([-2, 2])
            plt.grid()
            plt.savefig('./out/%s/%s/%s/%s/%s/auxi-%s/%d.png'% (args.data_size, args.arch, controller_folder, add_save_path, args.target_task, args.auxi_task, epoch))
            plt.close()


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
