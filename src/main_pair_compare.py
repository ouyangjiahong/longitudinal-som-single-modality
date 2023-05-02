import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config_pair_compare.yaml')
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)
print(time_label)

# ckpt folder, load yaml config
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    print('Load config ', os.path.join(config['ckpt_path'], 'config_pair_compare.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_pair_compare.yaml'))
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
        save_config_yaml(config['ckpt_path'], config, postfix='_pair')
print(config)

# define dataset
if config['phase'] == 'test':
    config['shuffle'] = False
Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
            noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
            data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
            fold=config['fold'], shuffle=config['shuffle'], train_all=config['train_all'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader
iter_max = len(trainDataLoader) * config['epochs']

# define model
if config['model_name'] == 'AE':
    model = AE().to(config['device'])
elif config['model_name'] == 'LSSL':
    model = LSSL(config=config).to(config['device'])
elif config['model_name'] == 'LNE':
    model = LNE(config=config).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    try:
        [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    except:
        [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
        # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    start_epoch = -1

def train():
    global_iter = len(trainDataLoader) * (start_epoch+1)
    monitor_metric_best = 100
    start_time = time.time()

    # stat = evaluate(phase='val', set='val', save_res=False)
    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0.}
        global_iter0 = global_iter

        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            # if iter > 10:
            #     break

            image1 = sample['img1'].to(config['device'], dtype=torch.float)
            image2 = sample['img2'].to(config['device'], dtype=torch.float)
            if len(image1.shape) == 4:
                image1 = image1.unsqueeze(1)
                image2 = image2.unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)
            interval = sample['interval'].to(config['device'], dtype=torch.float)

            # run model
            if config['model_name'] == 'AE' or config['model_name'] == 'LSSL':
                [z_e1, z_e2], [recon_ze1, recon_ze2] = model.forward(image1, image2, interval)
            elif config['model_name'] == 'LNE':
                [z_e1, z_e2], [recon_ze1, recon_ze2] = model.forward(image1, image2, interval)
                adj_mx = model.build_graph_batch([z_e1, z_e2])
                delta_z, delta_h = model.compute_social_pooling_delta_z_batch([z_e1, z_e2], interval, adj_mx)
            else:
                raise ValueError('Not support!')

            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(image1, recon_ze1) + model.compute_recon_loss(image2, recon_ze2))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_dir'] > 0:
                if config['model_name'] == 'LSSL':
                    loss_dir = model.compute_direction_loss([z_e1, z_e2])
                elif config['model_name'] == 'LNE':
                    loss_dir = model.compute_direction_loss(delta_z, delta_h)
                else:
                    raise ValueError('Not support!')
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir'] += loss_dir.item()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon=[%.4f], dir=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon.item(), loss_dir.item()))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter

        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False, epoch=epoch)
        # monitor_metric = stat['all']
        monitor_metric = stat['recon']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info='', epoch=0):
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

    loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0.}
    input1_list = []
    input2_list = []
    label_list = []
    age_list = []
    interval_list = []
    recon1_list = []
    recon2_list = []
    ze1_list = []
    ze2_list = []
    subj_id_list = []
    case_order_list = []

    # set_model_batchnorm(model)
    with torch.no_grad():
    # while True:

        for iter, sample in enumerate(loader, 0):
            # if iter > 10:
            #     break

            image1 = sample['img1'].to(config['device'], dtype=torch.float)
            image2 = sample['img2'].to(config['device'], dtype=torch.float)
            if len(image1.shape) == 4:
                image1 = image1.unsqueeze(1)
                image2 = image2.unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)
            interval = sample['interval'].to(config['device'], dtype=torch.float)

            # run model
            if config['model_name'] == 'AE' or config['model_name'] == 'LSSL':
                [z_e1, z_e2], [recon_ze1, recon_ze2] = model.forward(image1, image2, interval)
            elif config['model_name'] == 'LNE':
                [z_e1, z_e2], [recon_ze1, recon_ze2] = model.forward(image1, image2, interval)
                adj_mx = model.build_graph_batch([z_e1, z_e2])
                delta_z, delta_h = model.compute_social_pooling_delta_z_batch([z_e1, z_e2], interval, adj_mx)
            else:
                raise ValueError('Not support!')

            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(image1, recon_ze1) + model.compute_recon_loss(image2, recon_ze2))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_dir'] > 0:
                if config['model_name'] == 'LSSL':
                    loss_dir = model.compute_direction_loss([z_e1, z_e2])
                elif config['model_name'] == 'LNE':
                    loss_dir = model.compute_direction_loss(delta_z, delta_h)
                else:
                    raise ValueError('Not support!')
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir'] += loss_dir.item()


            if phase == 'test' and save_res:
                input1_list.append(image1.detach().cpu().numpy())
                input2_list.append(image2.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())
                recon1_list.append(recon_ze1.detach().cpu().numpy())
                recon2_list.append(recon_ze2.detach().cpu().numpy())
                ze1_list.append(z_e1.detach().cpu().numpy())
                ze2_list.append(z_e2.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())
                interval_list.append(interval.detach().cpu().numpy())
                subj_id_list.append(sample['subj_id'])
                case_order_list.append(np.stack([sample['case_order'][0].numpy(), sample['case_order'][1].numpy()], 1))


        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        if phase == 'test' and save_res:
            input1_list = np.concatenate(input1_list, axis=0)
            input2_list = np.concatenate(input2_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)
            interval_list = np.concatenate(interval_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            subj_id_list = np.concatenate(subj_id_list, axis=0)
            case_order_list = np.concatenate(case_order_list, axis=0)
            recon1_list = np.concatenate(recon1_list, axis=0)
            recon2_list = np.concatenate(recon2_list, axis=0)
            ze1_list = np.concatenate(ze1_list, axis=0)
            ze2_list = np.concatenate(ze2_list, axis=0)
            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('subj_id', data=subj_id_list)
            h5_file.create_dataset('case_order', data=case_order_list)
            h5_file.create_dataset('input1', data=input1_list)
            h5_file.create_dataset('input2', data=input2_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('interval', data=interval_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('recon1', data=recon1_list)
            h5_file.create_dataset('recon2', data=recon2_list)
            h5_file.create_dataset('ze1', data=ze1_list)
            h5_file.create_dataset('ze2', data=ze2_list)

    return loss_all_dict

if config['phase'] == 'train':
    train()
    config['shuffle'] = False
    Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
                noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
                data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
                fold=config['fold'], shuffle=config['shuffle'], train_all=config['train_all'])
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
