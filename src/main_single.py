import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil
import sklearn.cluster

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config_single.yaml')
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
    print('Load config ', os.path.join(config['ckpt_path'], 'config_single.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_single.yaml'))
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
        save_config_yaml(config['ckpt_path'], config)
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
if config['model_name'] in ['SOM']:
    model = SOM(config=config).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
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
        loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0.}
        global_iter0 = global_iter

        # init emb
        if config['warmup_epochs'] == epoch and (config['init_emb'] == 'pretrained-kmeans' or config['init_emb'] == 'pretrained-kmeans-switch'):
            if os.path.exists(os.path.join(config['ckpt_path'], 'init_emb_weights.npz')):
                data = np.load(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'))
                init = data['emb']
                print('Load pre-saved initialization!')
            else:
                with torch.no_grad():
                    z_e_list = []
                    for _, sample in enumerate(trainDataLoader, 0):
                        image = sample['img'].to(config['device'], dtype=torch.float)
                        if len(image.shape) == 4:
                            image = image.unsqueeze(1)

                        z_e, _ = model.encoder(image)
                        z_e_list.append(z_e.detach().cpu().numpy())
                    z_e_list = np.concatenate(z_e_list, 0)
                    kmeans = sklearn.cluster.KMeans(n_clusters=config['embedding_size'][0]*config['embedding_size'][1]).fit(z_e_list)
                    init = kmeans.cluster_centers_.reshape(config['embedding_size'][0], config['embedding_size'][1], -1)
                    np.savez(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'), emb=init)
            model.init_embeddings_ep_weight(init)
            model.train()


        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            # if iter > 10:
            #     break

            image = sample['img'].to(config['device'], dtype=torch.float)
            if len(image.shape) == 4:
                image = image.unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)

            # run model
            recons, zs = model(image)

            recon_ze = recons[0]
            recon_zq = recons[1]

            z_e = zs[0]
            z_q = zs[1]
            k = zs[2]
            sim = zs[3]

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = model.compute_recon_loss(image, recon_ze)
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                loss_recon_zq = model.compute_recon_loss(image, recon_zq)
                loss += config['lambda_recon_zq'] * loss_recon_zq
            else:
                loss_recon_zq = torch.tensor(0.)

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = model.compute_commit_loss(z_e, z_q)
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = model.compute_som_loss(z_e, k, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max)
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            # emb_old = model.embeddings.detach().cpu().numpy()
            # enc_old = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            # dec_old = torch.cat([param.view(-1) for param in model.decoder.parameters()]).detach().cpu().numpy()

            optimizer.step()
            optimizer.zero_grad()

            # emb_new = model.embeddings.detach().cpu().numpy()
            # enc_new = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            # dec_new = torch.cat([param.view(-1) for param in model.decoder.parameters()]).detach().cpu().numpy()
            # print(np.abs(emb_new-emb_old).mean(), np.abs(enc_new-enc_old).mean(), np.abs(dec_new-dec_old).mean())
            # pdb.set_trace()

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon=[%.4f], recon_zq=[%.4f], commit=[%.4f], som=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon.item(), loss_recon_zq.item(), loss_commit.item(), loss_som.item()))
                print('Num. of k:', torch.unique(k).shape[0])


        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        if 'k' not in loss_all_dict:
            loss_all_dict['k'] = 0

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
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0.}
    input_list = []
    label_list = []
    age_list = []
    recon_list = []
    recon_zq_list = []
    ze_list = []
    zq_list = []
    k_list = []
    sim_list = []
    subj_id_list = []
    case_order_list = []

    with torch.no_grad():
        recon_emb = model.recon_embeddings()

        for iter, sample in enumerate(loader, 0):
            # if iter > 10:
            #     break

            image = sample['img'].to(config['device'], dtype=torch.float)
            if len(image.shape) == 4:
                image = image.unsqueeze(1)
            label = sample['label'].to(config['device'], dtype=torch.float)

            recons, zs = model(image)

            recon_ze = recons[0]
            recon_zq = recons[1]
            z_e = zs[0]
            z_q = zs[1]
            k = zs[2]
            sim = zs[3]

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = model.compute_recon_loss(image, recon_ze)
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                loss_recon_zq = model.compute_recon_loss(image, recon_zq)
                loss += config['lambda_recon_zq'] * loss_recon_zq
            else:
                loss_recon_zq = torch.tensor(0.)

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = model.compute_commit_loss(z_e, z_q)
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = model.compute_som_loss(z_e, k)
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)


            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()

            if phase == 'test' and save_res:
                input_list.append(image.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())
                recon_list.append(recon_ze.detach().cpu().numpy())
                recon_zq_list.append(recon_zq.detach().cpu().numpy())
                ze_list.append(z_e.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())
                subj_id_list.append(sample['subj_id'])
                case_order_list.append(sample['case_order'].numpy())
                zq_list.append(z_q.detach().cpu().numpy())
                sim_list.append(sim.detach().cpu().numpy())
            k_list.append(k.detach().cpu().numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        if phase == 'test' and save_res:
            input_list = np.concatenate(input_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            subj_id_list = np.concatenate(subj_id_list, axis=0)
            case_order_list = np.concatenate(case_order_list, axis=0)
            recon_list = np.concatenate(recon_list, axis=0)
            recon_zq_list = np.concatenate(recon_zq_list, axis=0)
            ze_list = np.concatenate(ze_list, axis=0)
            zq_list = np.concatenate(zq_list, axis=0)
            k_list = np.concatenate(k_list, axis=0)
            sim_list = np.concatenate(sim_list, axis=0)
            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('subj_id', data=subj_id_list)
            h5_file.create_dataset('case_order', data=case_order_list)
            h5_file.create_dataset('input', data=input_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('recon', data=recon_list)
            h5_file.create_dataset('recon_zq', data=recon_zq_list)
            h5_file.create_dataset('ze', data=ze_list)
            h5_file.create_dataset('embeddings', data=model.embeddings.detach().cpu().numpy())
            h5_file.create_dataset('recon_emb', data=recon_emb.detach().cpu().numpy())
            h5_file.create_dataset('zq', data=zq_list)
            h5_file.create_dataset('k', data=k_list)
            h5_file.create_dataset('sim', data=sim_list)
        else:
            k_list = np.concatenate(k_list, axis=0)

        print('Number of used embeddings:', np.unique(k_list).shape)
        loss_all_dict['k'] = np.unique(k_list).shape[0]

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
