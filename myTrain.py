import argparse
import logging
import os
import time
import sys

import pandas as pd
import torch
import torch_geometric
import numpy as np
import random
# from visualizer import get_local
# get_local.activate()

from torch.ao.nn.quantized.functional import threshold
from torchvision import transforms, models
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score # 平均精度
from sklearn.metrics import recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup

from optim.createOptimizer import create_optimizer
from utils.metric import LabelSmoothingLoss, SPLlabelSmoothing, compute_kl_loss, CompactnessLoss
from utils.AdversarialAttacks import PGD, AdversarialAttacks_Switch, FGM, AWP
import contextlib
from dataset.erDataset import ERDataset, GraphDataset, FixedOrderSampler

def setupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

def parseArgs():
    mParser = argparse.ArgumentParser()
    # mParser.add_argument(
    #     "--",
    #     type=,
    #     default=,
    #     help=
    # )
    # exp
    mParser.add_argument("--exp_name", type=str, default=None)
    # data
    mParser.add_argument("--data_path", type=str, default=None)
    mParser.add_argument("--graphPath", type=str, default=None)
    mParser.add_argument("--data_name", type=str, default=None)
    # mParser.add_argument("--train_path", type=str, default=None)
    # mParser.add_argument("--valid_path", type=str, default=None)
    # mParser.add_argument("--label_path", type=str, default=None)
    mParser.add_argument("--image_size", type=int, default=224)
    # data process
    mParser.add_argument("--mosaic", action='store_true')
    # pretrainModel load
    mParser.add_argument("--pretrainName", type=str, default="swinv2_base_window12_192")
    mParser.add_argument("--pretrainPartPath", type=str, default="swinv2_base_patch4_window12_192_22k")
    mParser.add_argument("--pretrainLoadPath", type=str, default=None)
    # model
    mParser.add_argument("--num_classes", type=int, default=None)
    mParser.add_argument("--epoch", type=int, default=15)
    mParser.add_argument("--batch_size", type=int, default=64)
    mParser.add_argument("--lr", type=float, default=1e-4)
    mParser.add_argument("--num_workers", type=int, default=4)
    mParser.add_argument("--warmup_ratio", type=float, default=0.06)
    mParser.add_argument('--accumulation_steps', type=int, default=1)
    mParser.add_argument('--max_grad_norm', type=float, default=1.0)
    #arch
    mParser.add_argument('--base_dim', type=int, default=1024)
    #gin
    mParser.add_argument('--hidden_dim', type=int, default=256)
    mParser.add_argument('--num_node_features', type=int, default=None)
    # r_drop
    mParser.add_argument("--R_drop",action='store_true')
    mParser.add_argument("--alpha", type=float, default=1.0)
    mParser.add_argument("--alphaKLAdj", type=float, default=1.0)
    # graphormer
    mParser.add_argument("--graphArgs", default=None)
    # mParser.add_argument("--pretrained_model_name", type=str, default="pcqm4mv1_graphormer_base")
    # mParser.add_argument("--max_nodes", type=int, default=5000)

    mParser.add_argument("--lambda_1", type=float, default=1e-4)
    mParser.add_argument("--k", type=int, default=9)
    mParser.add_argument("--feat_dim", type=int, default=128)
    # attack
    mParser.add_argument("--FGM", action='store_true')
    mParser.add_argument("--PGD", action='store_true')
    mParser.add_argument("--AWP", action='store_true')
    mParser.add_argument('--attack_start_epoch', type=int, default=0)
    mParser.add_argument('--attack_step', default=3)
    mParser.add_argument('--adv_param', type=str, default='shared')
    mParser.add_argument("--Switch", action='store_true')
    # ema
    mParser.add_argument("--amp", action='store_true')
    mParser.add_argument('--ema', action='store_true')
    mParser.add_argument('--ema_steps', type=int, default=32)
    mParser.add_argument('--ema_warmup_ratio', type=float, default=0.1)
    mParser.add_argument('--ema_decay', default=0.999)

    mParser.add_argument("--test_best", type=float, default=0)
    mParser.add_argument("--nomax", type=float, default=1e-3)
    mParser.add_argument("--nomin", type=float, default=0.0)

    # mParser.add_argument("--ablaExp", action='store_true')

    return mParser.parse_args()

import gc
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import timm
from model.downModel import DownModel
from utils.tools import Step, Logger
from utils.EMA import ExponentialMovingAverage

# if __name__ == "__main__":
def runTraining():
    args = parseArgs()
    now = time.localtime()
    timeStr = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday) + "-" + str(
        now.tm_hour) + "-" + str(
        now.tm_min)
    if args.exp_name is None:
        args.exp_name = "exp_" + timeStr
    else:
        args.exp_name = args.exp_name # + "_" + timeStr
    if(args.num_classes is None or args.data_path is None or args.data_name is None or args.num_node_features is None):
        print(f"args.num_classes is None or args.data_path is None or args.data_name is None or args.num_node_features is None")
        sys.exit(0)
    print(f"{args.exp_name}: ")
    print(f"detail: {args.data_name}, {args.num_classes} classes, bs:{args.batch_size}")

    log_name = args.exp_name # +"_"+timeStr
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    logger = Logger(f'./logs/{log_name}.txt', 'a')
    logger.log(args)
    # # 清理已有 handlers
    # logging.getLogger("numexpr").setLevel(logging.WARNING)
    # # for h in root_logger.handlers:
    # #     root_logger.removeHandler(h)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s %(levelname)s %(message)s",
    #     datefmt="%Y-%m-%d %H:%M",
    #     handlers=[logging.FileHandler("./logs/" + log_name + ".log", "w", "utf-8")],
    # )
    # # logging.getLogger("numexpr").setLevel(logging.WARNING)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"using {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    # # my own
    # model = ViT(image_size=args.image_size, num_classes=args.num_classes)
    # model.to(device)
    # for name, module in model._modules.items():
    #     print(name, " : ", module)
    # weights_dict = torch.load("D:\\Documents\\Python\\myDriveModel\\ViTModel\\imagenet21k+imagenet2012_ViT-B_16-224.pth", map_location=device)
    # weights_dict = weights_dict['state_dict']
    # for k in weights_dict.keys():
    #     print(k)
    # del_keys = ['classifier.weight', 'classifier.bias']
    # for k in del_keys:
    #     del weights_dict[k]
    # print(model.load_state_dict(weights_dict, strict=False))

    # timm
    # ViT
    # pretrainName = "vit_base_patch16_224"
    # pretrained_cfg = timm.models.create_model(pretrainName).default_cfg
    # pretrained_cfg['file'] = r'D:\Documents\Python\myDriveModel\timm\vit_base_patch16_224.augreg2_in21k_ft_in1k\pytorch_model.bin'
    # # checkpoint_path = r'D:\Documents\Python\myDriveModel\ViTModel\B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
    # arch = timm.models.create_model(pretrainName, pretrained=True, pretrained_cfg=pretrained_cfg, num_classes=0)
    # SwinT
    pretrainName = args.pretrainName
    pretrained_cfg = timm.models.create_model(pretrainName).default_cfg
    pretrained_cfg['file'] = f'pretrainModel\\{args.pretrainPartPath}.pth'
    arch = timm.create_model(pretrainName, pretrained=True, pretrained_cfg=pretrained_cfg, drop_path_rate=0.2, num_classes=0)
    # # vgg16
    # pretrainName = args.pretrainName
    # pretrained_cfg = timm.models.create_model(pretrainName).default_cfg
    # pretrained_cfg['file'] = args.pretrainLoadPath
    # arch = timm.create_model(pretrainName, pretrained_cfg=pretrained_cfg, pretrained=True, num_classes=0)

    # # resnet
    # arch = models.resnet50(weights='IMAGENET1K_V1')
    # arch = models.resnet34()
    # weights_dict = torch.load(r"D:\Documents\Python\myDriveModel\resnet\resnet34-b627a593.pth")
    # arch.load_state_dict(weights_dict, strict=False)
    # arch.load_state_dict(
    #     torch.load(r"D:\Documents\Python\myDriveModel\resnet\resnet18_msceleb.pth", map_location="cpu")[
    #         "state_dict"
    #     ],
    #     strict=True,
    # )
    # arch = nn.Sequential(*list(arch.children())[:-2])

    archName = "SwinT"
    model = DownModel(arch=arch, num_classes=args.num_classes, args=args, dim=args.base_dim, name=archName, k=args.k)
    model.to(device)
    # model_ft = timm.create_model('resnet18', pretrained=True,
    #                              pretrained_cfg_overlay=dict(file='/home/xiaoxin/Documents/hc/-/bin/pytorch_model.bin'))

    data_transforms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomHorizontalFlip(p=0.45),
            # transforms.RandomVerticalFlip(p=0.45),
            transforms.RandomApply(
                [transforms.RandomRotation(20), transforms.RandomCrop(args.image_size, padding=32)],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]
    )
    testData_transforms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]
    )

    # trainvalid_dataset = SRDataset(data_path=args.data_path,
    #                         data_name = args.data_name,
    #                         phase="train&valid",
    #                         corruptes=os.listdir(args.data_path),
    #                         transform=data_transforms)
    # setupSeed(42)
    #
    # train_size = int(0.8 * len(trainvalid_dataset))
    # val_size = len(trainvalid_dataset) - train_size
    # train_dataset, val_dataset = random_split(trainvalid_dataset, [train_size, val_size])

    ####
    train_dataset = ERDataset(data_path=args.data_path,
                            dataset=args.data_name,
                            phase="train",
                            transform=data_transforms,
                            mosaic=args.mosaic
                              )
    val_dataset = ERDataset(data_path=args.data_path,
                                   dataset=args.data_name,
                                   phase="val",
                                   transform=testData_transforms,
                                   )
    train_graDataset = GraphDataset(
                            root=os.path.join(args.data_path, "graph"),
                            dataset=args.data_name,
                            mode="train",
    )
    val_graDataset = GraphDataset(
                            root=os.path.join(args.data_path, "graph"),
                            dataset=args.data_name,
                            mode="val",
    )
    test_graDataset = GraphDataset(
        root=os.path.join(args.data_path, "graph"),
        dataset=args.data_name,
        mode="test",
    )
    setupSeed(42)
    # logging.info(f"detail: {args.data_name}, {args.num_classes} classes, bs:{args.batch_size}")
    logger.log(f"detail: {args.data_name}, {args.num_classes} classes, bs:{args.batch_size}")
    # #
    # train_size = int(0.5 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # # less for test code
    # train_size = int(0.1 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    # del _

    print("train dataset size:", train_dataset.__len__())
    print("valid dataset size:", val_dataset.__len__())

    indices = np.random.permutation(train_dataset.__len__())
    sampler = FixedOrderSampler(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # shuffle=True,
        sampler=sampler,
        drop_last=True,
    )
    train_graLoader = torch_geometric.loader.DataLoader(
        train_graDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        follow_batch=["x_a"],
        # shuffle=True,
        sampler=sampler,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_graLoader = torch_geometric.loader.DataLoader(
        val_graDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        follow_batch=["x_a"],
        shuffle=False,
        pin_memory=True,
    )
    trSize = train_dataset.__len__()
    vaSize = val_dataset.__len__()
    del train_dataset, val_dataset, val_graDataset, train_graDataset, test_graDataset
    gc.collect()

    # Define losses
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_c = CompactnessLoss(args.k, args.feat_dim)
    criterion_smooth = LabelSmoothingLoss(num_classes=args.num_classes, smoothing=0.1)
    # criterion_SPLsmooth = SPLlabelSmoothing(num_classes=args.num_classes, smoothing=0.1)

    # params = list(model.parameters())
    # optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[10, 18, 25, 32], gamma=0.1
    # )
    # learning rate warmup
    optimizer = create_optimizer(model, model_lr={'others':args.lr})#, layerwise_learning_rate_decay=LR_LAYER_DECAY)
    total_steps = args.epoch * trSize // args.batch_size
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps,
                                                num_warmup_steps=warmup_steps)
    # scaler = GradScaler()
    # tricks
    if args.PGD:
        pgd = PGD(model,adv_param=args.adv_param)
    if args.FGM:
        fgm = FGM(model,adv_param=args.adv_param)
    if args.AWP:
        awp = AWP(model,adv_param=args.adv_param)
    if args.ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.batch_size * args.accumulation_steps * args.ema_steps / args.epoch
        alpha = 1.0 - args.ema_decay
        alpha = min(1.0, alpha * adjust)
        logger.log('EMA decay:', 1 - alpha)
        print('EMA decay:', 1 - alpha)
        model_ema = ExponentialMovingAverage(model, device='cuda', decay=1.0 - alpha)
        step_ema = Step()
        # checkpoint_ema = Checkpoint(model=model_ema.module, step=step_ema)
        model_ema.eval()
    step = Step()
    model_swa = None
    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext
    if args.Switch:
        attack_switch = AdversarialAttacks_Switch(args, bool_name_list=['PGD', 'FGM', 'AWP'])
    # training loop
    best_acc = 0
    # thrSPL = 0.5
    # growing_factor = 1.12
    iterLoss1 = []
    iterLoss2 = []
    epochLoss1 = []
    epochLoss2 = []
    for epoch in tqdm(range(1, args.epoch + 1)):
        running_loss = 0.0
        running_adjLoss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        # for i, (imgs, targets) in enumerate(train_loader):
        for i, (graph, (imgs, targets)) in enumerate(zip(train_graLoader, train_loader)):
            if args.Switch:
                attack_switch.random_select()
            step.forward(imgs.shape[0])
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            graph = graph.to(device)
            targets = targets.to(device).long()
            # v = torch.zeros(args.batch_size).int()
            # with autocast():
            with amp_cm():
                # logits, fvsn_feat = model(imgs)
                logits, fvsn_feat, outs, adj = model(imgs, graph)
                # print(f"logits shape: {logits.shape}, targets shape: {targets.shape}")
                loss = (
                    # 1*criterion_cls(logits, targets)
                    1 * criterion_smooth(logits, targets)
                    # +args.lambda_1 * criterion_c(fvsn_feat)
                )
                if (args.R_drop):
                    # logits_r, fvsn_feat_r = model(imgs)
                    logits_r, fvsn_feat_r, outs_j, adj_j = model(imgs, graph)
                    loss_r = (
                        1 * criterion_smooth(logits_r, targets)
                        # + args.lambda_1 * criterion_c(fvsn_feat_r)
                    )
                    kl_loss = compute_kl_loss(logits, logits_r)
                    loss = 0.5*loss + 0.5*loss_r + args.alpha * kl_loss
                #
                out_adj = []
                for kI in range(args.k):
                    # print(f"kI: {kI}, {outs[kI].shape}")
                    _, predicts = torch.max(outs[kI], 1)
                    # print(f"kI: {kI}, predicts: {predicts.shape}, targets: {targets.shape}")
                    correct_num = torch.eq(predicts, targets).sum()
                    out_adj.append(float(correct_num/targets.shape[0]))
                    # out_adj.append(criterion_smooth(outs[kI], targets))
                    # print(f"kI: {kI}, {out_adj[kI]}")
                out_adj = torch.tensor(out_adj, dtype=torch.float)
                # print(f"out_adj: {out_adj.shape}, adj: {adj.shape}")
                out_adj = out_adj.to(device)
                kl_adj = compute_kl_loss(out_adj, adj)
                loss += args.alphaKLAdj * kl_adj
                iterLoss1.append(0.5 * loss.item() + 0.5 * loss_r.item())
                iterLoss2.append(kl_adj.item())

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            if scheduler is not None:
                scheduler.step()
            # SPL trick
            # thrSPL *= growing_factor

            running_loss += loss.item()
            running_adjLoss += kl_adj.item()
            _, predicts = torch.max(logits, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
        # cache = get_local.cache
        # print(f"cache keys: {list(cache.keys())}")
        # attMap = cache[list(cache.keys())[0]]
        # print(f"{list(cache.keys())[0]} len: {len(attMap)}, shape: {attMap[0].shape}")
        acc = correct_sum.float() / float(trSize)
        running_loss = running_loss / iter_cnt
        running_adjLoss = running_adjLoss / iter_cnt
        tqdm.write(
            "[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f, running_adjLoss:%.3f"
            % (epoch, acc, running_loss, optimizer.param_groups[0]["lr"], running_adjLoss)
        )
        logger.log(
            "[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f, running_adjLoss:%.3f"
            % (epoch, acc, running_loss, optimizer.param_groups[0]["lr"], running_adjLoss)
        )
        # 释放
        del logits, loss, _, predicts, running_loss
        gc.collect()
        torch.cuda.empty_cache()
        if epoch > args.attack_start_epoch - 1 and args.PGD:
            pgd.backup_grad()  # 保存正常的grad
            # 对抗训练
            for t in range(args.attack_step):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data

                if t != args.attack_step - 1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()  # 恢复正常的grad
                with amp_cm():
                    logits, fvsn_feat = model(imgs)
                    loss_pgd = (
                            1 * criterion_smooth(logits, targets)
                            + args.lambda_1 * criterion_c(fvsn_feat)
                    )
                if args.amp:
                    scaler.scale(loss_pgd).backward()
                else:
                    loss_pgd.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数
                del loss_pgd
                if epoch > args.attack_start_epoch - 1 and args.FGM:
                    fgm.attack()  # embedding被修改了
                    # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                    with amp_cm():
                        logits, fvsn_feat = model(imgs)
                        loss_fgm = (
                                1 * criterion_smooth(logits, targets)
                                + args.lambda_1 * criterion_c(fvsn_feat)
                        )
                    if args.amp:
                        scaler.scale(loss_fgm).backward()
                    else:
                        loss_fgm.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复Embedding的参数
                    del loss_fgm
                if epoch > args.attack_start_epoch - 1 and args.AWP:
                    awp.backup_grad()  # 保存正常的grad
                    awp.attack()  # embedding被修改了
                    # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                    with amp_cm():
                        logits, fvsn_feat = model(imgs)
                        loss_awp = (
                                1 * criterion_smooth(logits, targets)
                                + args.lambda_1 * criterion_c(fvsn_feat)
                        )
                    if args.amp:
                        loss_awp = scaler.scale(loss_awp).backward()
                    else:
                        loss_awp.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                    awp.restore()  # 恢复Embedding的参数
                    del loss_awp
        del imgs, targets
        gc.collect()
        # if (i + 1) % args.accumulation_steps == 0:
        #
        #     # grad clip
        #     if args.amp:
        #         scaler.unscale_(optimizer)
        #     if hasattr(optimizer, "clip_grad_norm"):
        #         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        #         optimizer.clip_grad_norm(args.max_grad_norm)
        #     else:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        #
        #     if args.amp:
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         optimizer.step()  # 优化一次
        #     scheduler.step()
        #     optimizer.zero_grad()  # 清空梯度
        #     if args.ema and (i + 1) % (args.accumulation_steps * args.ema_steps) == 0:
        #         model_ema.update_parameters(model)
        #         if epoch < int(args.ema_warmup_ratio * args.epoch):
        #             # Reset ema buffer to keep copying weights during warmup period
        #             model_ema.n_averaged.fill_(0)

        torch.cuda.empty_cache()

        with torch.no_grad():
            running_loss = 0.0
            running_adjLoss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            y_true = []
            y_pred = []

            model.eval()
            # for i, (imgs, targets) in enumerate(val_loader):
            for i, (graph, (imgs, targets)) in enumerate(zip(val_graLoader, val_loader)):
                imgs = imgs.to(device)
                graph = graph.to(device)
                targets = targets.to(device).long()

                # logits, fvsn_feat = model(imgs)
                logits, fvsn_feat, outs, adj = model(imgs, graph)
                loss = (
                    criterion_cls(logits, targets)
                )

                out_adj = []
                for kI in range(args.k):
                    # print(f"kI: {kI}, {outs[kI].shape}")
                    _, predicts = torch.max(outs[kI], 1)
                    # print(f"kI: {kI}, predicts: {predicts.shape}, targets: {targets.shape}")
                    correct_num = torch.eq(predicts, targets).sum()
                    out_adj.append(float(correct_num / targets.shape[0]))
                    # out_adj.append(criterion_smooth(outs[kI], targets))
                    # print(f"kI: {kI}, {out_adj[kI]}")
                out_adj = torch.tensor(out_adj, dtype=torch.float)
                # print(f"out_adj: {out_adj.shape}, adj: {adj.shape}")
                out_adj = out_adj.to(device)
                kl_adj = compute_kl_loss(out_adj, adj)

                running_loss += loss.item()
                running_adjLoss += kl_adj.item()
                iter_cnt += 1
                _, predicts = torch.max(logits, 1)
                # print("target: ", targets)
                # print("predict: ", predicts)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predicts.cpu().numpy().tolist())
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += logits.size(0)

            running_loss = running_loss / iter_cnt
            running_adjLoss = running_adjLoss/iter_cnt
            epochLoss1.append(running_loss)
            epochLoss2.append(running_adjLoss)
            # scheduler.step()

            # precision = precision_score(y_true, y_pred)
            # recall = recall_score(y_true, y_pred)
            # f1score = f1_score(y_true, y_pred)
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            bacc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
            UAR = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
            UF1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

            best_acc = max(acc, best_acc)
            logger.log(
                "[Epoch %d] Validation accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, running_adjLoss: %.3f, UAR:%.3f, UF1:%.3f"
                % (epoch, acc, bacc, running_loss, running_adjLoss, UAR, UF1)
            )
            logger.log("Best_acc:" + str(best_acc))
            tqdm.write(
                "[Epoch %d] Validation accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, running_adjLoss: %.3f, UAR:%.3f, UF1:%.3f"
                % (epoch, acc, bacc, running_loss, running_adjLoss, UAR, UF1)
            )
            tqdm.write("Best_bacc:" + str(best_acc))
            # 释放
            del imgs, targets, logits, loss, _, predicts, y_true, y_pred, running_loss
            gc.collect()
            torch.cuda.empty_cache()

            if acc >= best_acc:
                torch.save(
                    {
                        "iter": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join("checkpoints", "" + log_name + f"{archName}Epoch{args.epoch}_best.pth"),
                )
                tqdm.write("Best Model saved.")

            torch.save(
                {
                    "iter": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join("checkpoints", "" + log_name + f"{archName}Epoch{args.epoch}_latest.pth"),
            )
            tqdm.write("Latest Model saved.")
            # del model
            # gc.collect()
        evaluate(args, model, device, "image", logger, epoch)
    iterData = pd.DataFrame({'loss1':iterLoss1,
                             'loss2':iterLoss2})
    iterData.to_csv("./lossLog/iterData.csv", index=False)
    epochData = pd.DataFrame({'loss1': epochLoss1,
                             'loss2': epochLoss2})
    epochData.to_csv("./lossLog/epochData.csv", index=False)


def evaluate(args, model, device, mode, logger, epoch=None):
    testData_transforms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]
    )
    criterion_cls = torch.nn.CrossEntropyLoss()
    if(mode == "graph"):
        # teGraphDataset = GraphormerDataset(
        #     root=args.graphPath,
        #     dataset=args.data_name,
        #     mode="test",
        # )
        # teGraphLoader = CustomDataLoader2(
        #     teGraphDataset,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     follow_batch=["x"],
        #     pin_memory=True,
        #     collate_fn=myCollator,
        #     drop_last=True,
        # )
        teGraphDataset = GraphDataset(
            root=args.graphPath,
            dataset=args.data_name,
            mode="test",
        )
        teGraphLoader = torch_geometric.loader.DataLoader(
            teGraphDataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            follow_batch=["x"],
            shuffle=True,
            pin_memory=True,
            # drop_last=True,
        )
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            y_true = []
            y_pred = []

            model.eval()
            for step, graph in enumerate(teGraphLoader):
                for k in graph.keys():
                    # print(f"{k} type: {type(graph[k])}")
                    graph[k] = graph[k].to(device)
                targets = graph['y']

                logits, fvsn_feat = model(graph)
                loss = (
                    criterion_cls(logits, targets)
                )

                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(logits, 1)
                # print("target: ", targets)
                # print("predict: ", predicts)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predicts.cpu().numpy().tolist())
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += logits.size(0)

            running_loss = running_loss / iter_cnt
            # scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            bacc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
            UAR = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
            UF1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            # print(f"y_true: {y_true}")
            # print(f"y_pred: {y_pred}")
            logger.log(
                "Test accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, UAR:%.3f, UF1:%.3f"
                % (acc, bacc, running_loss, UAR, UF1)
            )
            tqdm.write(
                "Test accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, UAR:%.3f, UF1:%.3f"
                % (acc, bacc, running_loss, UAR, UF1)
            )
    elif(mode == "image"):
        test_dataset = ERDataset(data_path=args.data_path,
                                dataset=args.data_name,
                                phase="test",
                                transform=testData_transforms,
                                )
        test_graDataset = GraphDataset(
            root=os.path.join(args.data_path, "graph"),
            dataset=args.data_name,
            mode="test",
        )
        test_graLoader = torch_geometric.loader.DataLoader(
            test_graDataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            follow_batch=["x_a"],
            shuffle=False,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            y_true = []
            y_pred = []

            model.eval()
            for i, (graph, (imgs, targets)) in enumerate(zip(test_graLoader, test_loader)):
                imgs = imgs.to(device)
                graph = graph.to(device)
                targets = targets.to(device).long()

                # logits, fvsn_feat = model(imgs)
                logits, fvsn_feat, outs, adj = model(imgs, graph)
                # print(f"logits:{logits.shape}, targets:{targets.shape}")
                loss = (
                    criterion_cls(logits, targets)
                )

                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(logits, 1)
                # print("target: ", targets)
                # print("predict: ", predicts)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predicts.cpu().numpy().tolist())
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += logits.size(0)

            running_loss = running_loss / iter_cnt
            # scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            bacc = np.around(balanced_accuracy_score(y_true, y_pred), 4)
            UAR = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
            UF1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            # print(f"y_true: {y_true}")
            # print(f"y_pred: {y_pred}")
            logger.log(
                "Test accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, UAR:%.3f, UF1:%.3f"
                % (acc, bacc, running_loss, UAR, UF1)
            )
            tqdm.write(
                "Test accuracy: %.4f. Balanced Accuracy: %.4f. Loss: %.3f, UAR:%.3f, UF1:%.3f"
                % (acc, bacc, running_loss, UAR, UF1)
            )
            # 释放
            del imgs, targets, logits, loss, _, predicts, y_true, y_pred, running_loss
            gc.collect()
            torch.cuda.empty_cache()

    if(acc > args.test_best):
        args.test_best = acc
        logger.log(f"test best: {epoch}-{args.test_best}")
        print(f"test best: {epoch}-{args.test_best}")
    logger.log(f"test current: {epoch}-{acc}")
    print(f"test current: {epoch}-{acc}")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    runTraining()
    # trainGraphormer()
    # trainGraphNet()