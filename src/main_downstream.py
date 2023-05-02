import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import tqdm
from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config_downstream.yaml')
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    print('Load config ', os.path.join(config['ckpt_path'], 'config_downstream.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_downstream.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase' or key == 'gpu' or key == 'continue_train' or key == 'ckpt_name' or key == 'ckpt_path':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
    else:
        save_config_yaml(config['ckpt_path'], config, postfix='_downstream')
print(config)

# define dataset
if config['phase'] == 'test':
    config['shuffle'] = False
Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
            noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
            data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
            fold=config['fold'], shuffle=config['shuffle'])

trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

# define model
if config['model_name'] in ['CLS']:
    model = CLS(config).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# froze encoder
if config['froze_encoder']:
    for param in model.encoder.parameters():
        param.requires_grad = False

# define optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4, amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    start_epoch = -1

def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'all': 0, 'cls': 0.}
        global_iter0 = global_iter

        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1
            # if iter >= 5:
            #     break


            if 'classification' in config['task']:
                label = sample['label'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'age_regression':
                label = sample['age'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'adas_regression':
                label = sample['adas'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'mmse_regression':
                label = sample['mmse'].to(config['device'], dtype=torch.float)
            else:
                raise ValueError('Not support!')
            label = normalize_label(config, label)

            if config['use_feature'] == ['z']:
                img1 = sample['img'].to(config['device'], dtype=torch.float).unsqueeze(1)
                pred = model.forward_single(img1)
            else:
                img1 = sample['img1'].to(config['device'], dtype=torch.float).unsqueeze(1)
                img2 = sample['img2'].to(config['device'], dtype=torch.float).unsqueeze(1)
                pred = model.forward(img1, img2)

            # if img1.shape[0] <= config['batch_size'] // 2:
            #     break

            # run model
            loss_cls = model.compute_task_loss(pred, label)
            loss = config['lambda_cls'] * loss_cls
            loss_all_dict['cls'] += loss_cls.item()
            loss_all_dict['all'] += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            enc_old = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            cls_old = torch.cat([param.view(-1) for param in model.classifier.parameters()]).detach().cpu().numpy()
            optimizer.step()
            optimizer.zero_grad()

            enc_new = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            cls_new = torch.cat([param.view(-1) for param in model.classifier.parameters()]).detach().cpu().numpy()
            print(np.abs(enc_new-enc_old).sum(), np.abs(cls_new-cls_old).sum())

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f]' % (epoch, iter, loss.item(), loss_cls.item()))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        loss_all_dict['task'] = 0.
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        print('---------------------Evaluating Val Set-------------------- ')
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['all']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        print('---------------------Evaluating Train Set-------------------- ')
        stat = evaluate(phase='test', set='train', save_res=False)
        save_result_stat(stat, config, info='train')
        print(stat)

        print('---------------------Evaluating Test Set-------------------- ')
        stat = evaluate(phase='test', set='test', save_res=False)
        save_result_stat(stat, config, info='test')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, config['ckpt_path'])


def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'cls': 0.}

    label_list = []
    pred_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            # if iter >= 5:
            #     break
            if 'classification' in config['task']:
                label = sample['label'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'age_regression':
                label = sample['age'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'adas_regression':
                label = sample['adas'].to(config['device'], dtype=torch.float)
            elif config['task'] == 'mmse_regression':
                label = sample['mmse'].to(config['device'], dtype=torch.float)
            else:
                raise ValueError('Not support!')
            label = normalize_label(config, label)

            if config['use_feature'] == ['z']:
                img1 = sample['img'].to(config['device'], dtype=torch.float).unsqueeze(1)
                pred = model.forward_single(img1)
            else:
                img1 = sample['img1'].to(config['device'], dtype=torch.float).unsqueeze(1)
                img2 = sample['img2'].to(config['device'], dtype=torch.float).unsqueeze(1)
                pred = model.forward(img1, img2)

            # if img1.shape[0] <= config['batch_size'] // 2:
            #     break

            # run model
            loss_cls = model.compute_task_loss(pred, label)
            loss = config['lambda_cls'] * loss_cls
            loss_all_dict['cls'] += loss_cls.item()
            loss_all_dict['all'] += loss.item()

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)

        perf = compute_task_metrics(label_list, pred_list, config)
        loss_all_dict['task'] = perf

        if phase == 'test' and save_res:
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('pred', data=pred_list)

    return loss_all_dict

if config['phase'] == 'train':
    train()
    config['shuffle'] = False
    Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
                noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
                data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
                fold=config['fold'], shuffle=config['shuffle'])
    trainDataLoader = Data.trainLoader
    testDataLoader = Data.testLoader

    stat = evaluate(phase='test', set='test', save_res=True)
    print(stat)
    stat = evaluate(phase='test', set='train', save_res=True)
    print(stat)
else:
    stat = evaluate(phase='test', set='test', save_res=True)
    print(stat)
    stat = evaluate(phase='test', set='train', save_res=True)
    print(stat)
