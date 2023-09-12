
from __future__ import print_function
import warnings
from array import array
warnings.filterwarnings('ignore')

import argparse
import os,shutil,random,time,json
import numpy as np
from numpy import record
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timm.optim import AdamP,AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import config
from dataloader import YOLODataset
from utilities.loss import YoloLoss
from utilities.models import YOLOv3,YOLOv3_aspp
from utilities import Logger, AverageMeter
from utilities.augment import train_aug_od, val_aug_od
from utilities.metrics import check_class_accuracy
from utilities.metrics import batch_IOU
from utilities.metrics import mean_average_precision
from utilities.eval import get_evaluation_bboxes

# #===================================Arguments-start=======================================================================
parser = argparse.ArgumentParser(description = 'YOLO Training')
parser.add_argument('-j', '--workers', default=config.NUM_WORKERS, type=int, metavar='N',
                    help = 'number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default = config.NUM_EPOCHS, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
parser.add_argument('--size', type=int, default=config.IMAGE_SIZE, help='Input Image Size to Model.')

parser.add_argument('--train-batch', default = config.BATCH_SIZE, type = int, metavar = 'N',
                    help='train batch size')
# We keep train and test batch size as same
parser.add_argument('--lr', '--learning-rate', default = config.LEARNING_RATE, type=float,
                    metavar = 'LR', help = 'initial learning rate')

parser.add_argument('--weight-decay', '--wd', default = config.WEIGHT_DECAY, type=float,
                    metavar = 'W', help = 'weight decay (default: 5e-4)')
#Checkpoints
parser.add_argument('-c', '--checkpoint', default=config.CHECKPOINT_DIR, type=str, metavar='PATH',
                    help='checkpoint dir name (default: checkpoint)')
parser.add_argument('--resume', default = config.RESUME, type=str, metavar='PATH',
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int,  default = config.seed, help = 'seed for reproducibility')
parser.add_argument('--gpu', type = int,  default = config.GPU)                    
# parser.add_argument('--ignore_label', type=int,  default = config.ignore_label)                      
parser.add_argument('--clip', default = config.CLIP, type = float,
                    help = 'This method only clip the norm/magnitude only. Gradient descent will still be in the same direction') 
parser.add_argument('--backbone_freeze', default= config.BACKBONE_FREEZE, type=bool,help='')
parser.add_argument('--conf_threshold', default= config.CONF_THRESHOLD, type=float,help='')
parser.add_argument('--num_classes', type=int,  default = config.NUM_CLASSES)
parser.add_argument('--map_iou_th', default= config.MAP_IOU_THRESH, type=float,help='')
parser.add_argument('--nms_iou_th', default= config.NMS_IOU_THRESH, type=float,help='')
parser.add_argument('--scales', default= config.S, type=list,help='')
parser.add_argument('--img_dir', default= config.IMG_DIR, type=str,help='')
parser.add_argument('--label_dir', default= config.IMG_DIR, type=str,help='')
parser.add_argument('--anchors', default= config.ANCHORS, type=array,help='')
parser.add_argument('--step_size', default= config.STEP_SIZE, type=int,help='')
parser.add_argument('--accum_factor', default= config.ACCUM_FACTOR, type=int,help='')

#parser.add_argument('--accum_iter', default=config.accum_iter, type=int,
#                    help = 'if accumalation factor==1, that means no gradient accumalation') 

#parser.add_argument('--cls_wghts', default=False, type=bool,
#                    help='')
#parser.add_argument('--label_sm', default= config.label_sm, type=float,
#                    help='Label Smoothening to reduce the overconfidence level of prediction')
#parser.add_argument('--kl_div', type=bool, default=False)
#Device options

#==============================================================================================================================
args  = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# state = {}
#=====================================================================================================================
num_workers = config.NUM_WORKERS
epochs = config.NUM_EPOCHS
batch_size = config.BATCH_SIZE
lr = config.LEARNING_RATE
weight_decay = config.WEIGHT_DECAY
checkpoint_dir = config.CHECKPOINT_DIR
resume = config.RESUME
gpu = config.GPU
seed = config.seed
clip = config.CLIP
img_size = config.IMAGE_SIZE
root_dir =  config.DATASET
backbone_freeze = config.BACKBONE_FREEZE
num_classes  = config.NUM_CLASSES
scales = config.S
anchors = config.ANCHORS
image_path = config.IMG_DIR
label_path = config.LABEL_DIR
pin_memory = config.PIN_MEMORY
step_size = config.STEP_SIZE
conf_threshold=config.CONF_THRESHOLD
map_iou_th = config.MAP_IOU_THRESH
accum_factor = config.ACCUM_FACTOR
algo = 'yolov5'



# Use CUDA
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#torch.cuda.set_device(args.gpu)

#Random seed
if seed:
    random.seed(seed)
    torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


def train(train_loader, model, scaler, YoloLoss, scaled_anchors, optimizer, scheduler,device, epoch,checkpoint_dir=''):
    #switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avg = AverageMeter()
    class_acc = AverageMeter()
    object_acc = AverageMeter()
    object_iou_scale0 = AverageMeter()
    object_iou_scale1 = AverageMeter()
    object_iou_scale2 = AverageMeter()
    object_iou_scale_mean = AverageMeter()
    end = time.time()
    
    progress_bar = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )
        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                      YoloLoss(out[0], y0, scaled_anchors[0])
                    + YoloLoss(out[1], y1, scaled_anchors[1])
                    + YoloLoss(out[2], y2, scaled_anchors[2])
                      )




        #Gradient accumatlation if  accum_iter > 1
        # loss = loss / accum_factor
        # losses.append(loss.item())
        # #Compute gradient and do backpropagation step
        # scaler.scale(loss).backward()
        # #Weights update
        # if ((batch_idx + 1) % accum_factor == 0) or (batch_idx + 1 == len(train_loader)):
        #     #Gradient clipping prevtrain_diceenting grad exploding
        #     if clip is not None:
        #         torch.nn.utilities.clip_grad_norm_(model.parameters(),clip) # check it first or keep it less than 0.999
        #     #Updating the optimizer state after back propagation
        #     scaler.step(optimizer)
        #     # Updates the scale (grad-scale) for next iteration 
        #     scaler.update()
        #     #emptying optimizer
        #     optimizer.zero_grad()

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip) 
        scaler.step(optimizer)
        scaler.update()
        
        state['lr'] =   scheduler.optimizer.param_groups[0]['lr']
        
        class_accuracy, obj_accuracy, no_obj_accuracy = check_class_accuracy(y,out,conf_threshold,device)
        
        IOUs = []
        for idx in range(len(out)):
            IOUs.append(batch_IOU(out[idx], y[idx].to(device), scaled_anchors[idx]).mean().item())  
        mean_ious = np.mean(IOUs)
        
        
        losses_avg.update(loss.data, x.size(0))
        class_acc.update(class_accuracy.item(), x.size(0))
        object_acc.update(obj_accuracy.item(), x.size(0))
        object_iou_scale0.update(IOUs[0], x.size(0))
        object_iou_scale1.update(IOUs[1], x.size(0))
        object_iou_scale2.update(IOUs[2], x.size(0))
        object_iou_scale_mean.update(mean_ious, x.size(0))

        
        batch_time.update(time.time() - end)
        end = time.time()
        #plot progress
        progress_bar.set_description('(Epoch {epoch} | lr {lr}  | Batch: {bt:.3f}s | Loss: {loss:.4f} | obj_acc: {obj_acc: .4f} | clas_acc: {clas_acc: .4f} | mean_iou: {mean_iou: .4f}) '.format(
                    epoch=epoch + 1,
                    lr=state['lr'],
                    #batch=batch_idx + 1,
                    #size=len(train_loader),
                    #data=data_time.avg,
                    bt = batch_time.avg,
                    loss = losses_avg.avg,
                    obj_acc = object_acc.avg,
                    clas_acc = class_acc.avg,
                    mean_iou = object_iou_scale_mean.avg
                    ))
        
    accuracies = {'obj_acc': object_acc.avg, 'class_acc': class_acc.avg}
    ious = {'scale0':object_iou_scale0.avg,'scale1':object_iou_scale1.avg,'scale2':object_iou_scale2.avg}
    return losses_avg.avg, accuracies, ious
        
        

@torch.no_grad()
def val(val_loader, model, YoloLoss, scaled_anchors, map_iou_th, epoch, device, checkpoint_dir = '.'):
    #switch to train mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avg = AverageMeter()
    class_acc = AverageMeter()
    object_acc = AverageMeter()
    object_iou_scale0 = AverageMeter()
    object_iou_scale1 = AverageMeter()
    object_iou_scale2 = AverageMeter()
    object_iou_scale_mean = AverageMeter()
    
    
    
    all_pred_boxes = []
    all_true_boxes = []
    
    end = time.time()
    
    progress_bar = tqdm(val_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(progress_bar):
        
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )
        with torch.set_grad_enabled(False):
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                      YoloLoss(out[0], y0, scaled_anchors[0])
                    + YoloLoss(out[1], y1, scaled_anchors[1])
                    + YoloLoss(out[2], y2, scaled_anchors[2])
                      )

        class_accuracy, obj_accuracy, no_obj_accuracy = check_class_accuracy(y,out,conf_threshold,device)
        
        # iou_scale_0 = batch_IOU(out[0], y0, scaled_anchors[0]).mean()
        # iou_scale_1 = batch_IOU(out[1], y1, scaled_anchors[1]).mean()
        # iou_scale_2 = batch_IOU(out[2], y2, scaled_anchors[2]).mean()
        IOUs = []
        for idx in range(len(out)):
            IOUs.append(batch_IOU(out[idx], y[idx].to(device), scaled_anchors[idx]).mean().item())
            
        mean_ious = np.mean(IOUs)
        
        
        
        
        # pred_boxes, true_boxes = get_evaluation_bboxes(
        #                                 pred = out,
        #                                 labels = y,
        #                                 iou_threshold=config.NMS_IOU_THRESH,
        #                                 anchors=config.ANCHORS,
        #                                 threshold=config.CONF_THRESHOLD,
        #                                 device = device
        #                                 )
        #
        #
        # for idx in range(len(pred_boxes)):
        #     all_pred_boxes.append(pred_boxes[idx])
        #     all_true_boxes.append(true_boxes[idx])
      
        
        losses_avg.update(loss.data, x.size(0))
        class_acc.update(class_accuracy.item(), x.size(0))
        object_acc.update(obj_accuracy.item(), x.size(0))
        object_iou_scale0.update(IOUs[0], x.size(0))
        object_iou_scale1.update(IOUs[1], x.size(0))
        object_iou_scale2.update(IOUs[2], x.size(0))
        object_iou_scale_mean.update(mean_ious, x.size(0))
        
        
        batch_time.update(time.time() - end)
        end = time.time()
        #plot progress
        progress_bar.set_description('( Loss: {loss:.4f} | obj_acc: {obj_acc: .4f} | clas_acc: {clas_acc: .4f} | mean_iou: {mean_iou: 0.4f}) '.format(
                    loss = losses_avg.avg,
                    obj_acc = object_acc.avg,
                    clas_acc = class_acc.avg,
                    mean_iou = object_iou_scale_mean.avg,
                    ))
    map = 0
    #map = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=map_iou_th, box_format="midpoint", num_classes=num_classes)    
    accuracies = {'obj_acc': object_acc.avg, 'class_acc': class_acc.avg}
    ious = {'scale0':object_iou_scale0.avg,'scale1':object_iou_scale1.avg,'scale2':object_iou_scale2.avg}
    #print('mean average precision at th =0.5: ',map)
    return losses_avg.avg, accuracies, ious, object_iou_scale_mean.avg

def main():
    best_map = 0
    start_epoch = 0
    
    train_dataset = YOLODataset(root_dir + 'train.csv',
                    transform = train_aug_od(size = img_size),
                    S = scales,
                    C = num_classes,
                    img_dir=image_path,
                    label_dir=label_path,
                    anchors=anchors,
                    )
    
    val_dataset = YOLODataset(
                    root_dir + 'test.csv',
                    transform = val_aug_od(size = img_size),
                    S=scales,
                    C = num_classes,
                    img_dir=image_path,
                    label_dir=label_path,
                    anchors=anchors
                            )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
                        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
                        )
    
    print("")
    print("num_samples in train dataset: ", batch_size*len(train_loader))
    print("num_samples in val dataset: ", batch_size*len(val_loader))
    
    print("")
    
    loss_fn = YoloLoss()
    
    DEVICE = torch.device('cuda:'+ str(gpu))
    
    model = YOLOv3(num_classes=num_classes).to(DEVICE)
    #model = YOLOv3_aspp(num_classes=num_classes).to(DEVICE)
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    #for name, param in model.segformer.encoder
    if backbone_freeze:
        for name, param in model.named_parameters():
        #list the names of parameters/layers which you want to train
            if not 'ScalePrediction' in name:  #
                param.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr , weight_decay = weight_decay )
    else:
        #2.Optimizer 
        optimizer = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
    #Resume
    title = config.CHECKPOINT_DIR
    checkpoint_dir = os.path.join('checkpoints', title)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir,exist_ok=True)
    print('Saving training files and checkpoints to {}'.format(checkpoint_dir)) 
    if resume:
        #Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume,map_location=DEVICE)
        best_map = checkpoint['best_map']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict = True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_dir,'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), title=title)    
        logger.set_names(['Learning Rate', 'Train Loss', 'Val Loss', 
                          't_clas_acc.', 'v_clas_acc.',
                          't_obj_acc.', 'v_obj_acc.',
                          'iou_scale0', 'iou_scale1', 'iou_scale2', 
                          'mean_iou.'])
   
    scheduler = StepLR(optimizer, step_size=int(len(train_loader)*step_size), gamma=0.1)
       
    writer = SummaryWriter(log_dir=checkpoint_dir)
    
    with open(os.path.join(checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent = 2)

    scaler = torch.cuda.amp.GradScaler()
    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(scales).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(DEVICE)
    for epoch in range(start_epoch, epochs):
    # 
        train_loss, train_acc, train_iou = train(
                                    train_loader=train_loader,
                                    model = model,
                                    scaler = scaler,
                                    YoloLoss = loss_fn,
                                    scaled_anchors = scaled_anchors,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=DEVICE,
                                    epoch=epoch,
                                    checkpoint_dir=checkpoint_dir)
        
        val_loss, val_acc, val_iou, mean_iou = val(
                                        val_loader = val_loader,
                                        model=model,
                                        YoloLoss = loss_fn, 
                                        scaled_anchors = scaled_anchors, 
                                        device=DEVICE,
                                        map_iou_th = map_iou_th, 
                                        epoch=epoch,
                                        checkpoint_dir = checkpoint_dir
                                        )
        
        
        writer.add_scalar('train/loss',train_loss, (epoch + 1))
        writer.add_scalar('train/class_acc', train_acc['class_acc'], (epoch + 1))
        writer.add_scalar('train/obj_acc',train_acc['obj_acc'], (epoch + 1))
        writer.add_scalar('train/iou_scale0', train_iou['scale0'], (epoch + 1))
        writer.add_scalar('train/iou_scale1', train_iou['scale1'], (epoch + 1))
        writer.add_scalar('train/iou_scale2', train_iou['scale2'], (epoch + 1))
        
        writer.add_scalar('val/loss', val_loss, (epoch + 1))
        writer.add_scalar('val/class_acc', val_acc['class_acc'], (epoch + 1))
        writer.add_scalar('val/obj_acc', val_acc['obj_acc'], (epoch + 1))
        writer.add_scalar('val/iou_scale0', val_iou['scale0'], (epoch + 1))
        writer.add_scalar('val/iou_scale1', val_iou['scale1'], (epoch + 1))
        writer.add_scalar('val/iou_scale2', val_iou['scale2'], (epoch + 1))
        writer.add_scalar('val/mean_iou', mean_iou, (epoch + 1))

        #append logger file
        logger.append([state['lr'], train_loss, val_loss, 
               train_acc['class_acc'],  val_acc['class_acc'], 
               train_acc['obj_acc'], val_acc['obj_acc'],
               val_iou['scale0'], val_iou['scale1'], val_iou['scale1'],
               mean_iou,
               ])
        
        save_checkpoint(
                       { 'epoch': epoch + 1,
                        'state_dict':model.state_dict(),
                        #'acc': train_acc['class_acc'],
                        'best_map': best_map,
                        'optimizer' : optimizer.state_dict()
                        },
                    mean_iou > best_map, 
                    checkpoint=checkpoint_dir)
                
        
        if mean_iou > best_map:
            # savemat(checkpoint_dir+'/val_dice_score.mat', mdict_val)
            # savemat(checkpoint_dir+'/train_dice_score.mat', mdict_train)
            best_map = max(mean_iou, best_map)
            #print("checkpoint is saved")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
                
    logger.close()
    print('best mean_iou:', best_map)
        

def save_checkpoint(state, is_best,checkpoint=""):
    if is_best:
        print("=> Saving MODEL BEST")
        torch.save(state, checkpoint + '/model_best.pth.tar')
    else:
        print("=> Saving checkpoint")
        torch.save(state, checkpoint + '/checkpoint.pth.tar')
        
    
if __name__ == '__main__':
    main()

        
        
        
