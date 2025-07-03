from calendar import month
import sys
from data.pc_dataset import ModelNetDataLoader, ScanObjectNN
import argparse
import numpy as np
import os
from datetime import datetime
import torch
import logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from tqdm import tqdm
from models.pc.utils import provider
import yaml
from spikingjelly.clock_driven import functional
from timm.models import create_model

from models.pc import spt

#改成采用parser的形式
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# 新版中只需要compulsory输入一下config路径
# 相应的配置完全要求在yml中完成，不支持通过terminal配置
parser.add_argument('--config',default='configs/spt/modelnet40.yml',type=str, metavar='FILE',
                    help='YAML config file specifying model arguments')


logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


# Helper function for output directory and logger setup
def setup_logger_and_output_dir(model_name, base_output):
    # Generate timestamp for unique directory
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base_output, f"{model_name}_{time_str}")
    os.makedirs(out_dir, exist_ok=True)
    # Setup logger: clear existing handlers and add stream and file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    # Terminal handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    # File handler
    log_path = os.path.join(out_dir, "train.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    return out_dir, log_path


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def test(model, loader, num_class=40):
    classifier = model.eval()
    functional.reset_net(classifier)
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0] if num_class != 15 else target
        points, target = points.cuda(), target.cuda()

        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        functional.reset_net(classifier)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main():


    # omegaconf.OmegaConf.set_struct(args, False)
    # 获取yml内容
    args, args_text = _parse_args()

    # Setup unique output directory using model name and timestamp
    # base_output is from args.output or default "outputs"
    base_output = args.output if hasattr(args, "output") and args.output else "outputs"
    model_name = args.model if hasattr(args, "model") else "model"
    out_dir, log_path = setup_logger_and_output_dir(model_name, base_output)
    # Override args.output to point to this run's directory
    args.output = out_dir
    logger.info(f"Outputs will be saved to {out_dir}")

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(g) for g in args.device])

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    root = args.data_dir
    if 'ModelNet' in args.dataset:
        TRAIN_DATASET = ModelNetDataLoader(root=root, nclass=args.num_class, npoint=args.num_point, split='train', normal_channel=args.normal)
        TEST_DATASET = ModelNetDataLoader(root=root, nclass=args.num_class, npoint=args.num_point, split='test', normal_channel=args.normal)
    elif args.dataset == 'ScanObjectNN':
        TRAIN_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='training')
        TEST_DATASET = ScanObjectNN(root=root, num_points=args.num_point, split='test')
    else: raise NotImplementedError(f'{args.name} dataset is not found')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=6)

    '''MODEL LOADING'''
    args.input_dim = 6 if args.normal else 3
    # shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    # create model
    ## models/Hengshuang/model.py/PointTransfomerCls
    # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
    criterion = torch.nn.CrossEntropyLoss()

    all_args = vars(args)
    model_init_keys = ['num_point','nblocks','nneighbor','num_class','input_dim',
                      'spike_mode','step','use_encoder','num_samples','blocks',
                       'transformer_dim']
    model_init_cfg = {key: all_args[key] for key in model_init_keys if key in all_args}
    classifier = create_model(
        model_name=args.model,
        **model_init_cfg,
        )

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs, which is {args.device}")
        classifier = torch.nn.DataParallel(classifier)
        classifier.to(torch.device('cuda:0'))
    else:
        logger.info(f"Using {torch.cuda.device_count()} GPU, which is {args.device}")
        classifier.to(torch.device('cuda:0'))
    # logger.info(f"Model Structure: {classifier}")

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
        logger.info(f'Using AdamW as model optimizer, lr is {args.learning_rate}')
        logger.info(f'Using StepLR as model scheduler, learning rate decay {scheduler.gamma} for every {scheduler.step_size} epochs')
    else:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)
        logger.info(f'Using SGD as model optimizer, lr is {args.learning_rate}')
        logger.info(f'Using MultiStepLR as model scheduler, learning rate is dropped by 10x at epochs 120 and 160')

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier.train()
        functional.reset_net(classifier) #spikingjelly reset
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            target = target[:, 0] if 'ModelNet' in args.dataset else target
            points = provider.shuffle_points(points)
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points) #完成一次forward
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

            functional.reset_net(classifier)
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        logger.info(f"Train learning rate is {optimizer.param_groups[0]['lr']}")


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args.num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Current Epoch: %d, Test Instance Accuracy: %f, Class Accuracy: %f'% ((epoch+1), instance_acc, class_acc))
            logger.info('Best Epoch: %d, Best Instance Accuracy: %f, Class Accuracy: %f'% (best_epoch, best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                os.makedirs(args.output, exist_ok=True)
                savepath = os.path.join(args.output, 'best_model.pth')
                logger.info('Saving at %s'% savepath)
                state = {
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    main()
