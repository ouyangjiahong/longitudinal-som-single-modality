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

_, config = load_config_yaml('config_pair.yaml')
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
    print('Load config ', os.path.join(config['ckpt_path'], 'config_pair.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_pair.yaml'))
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
if config['model_name'] in ['SOMPairVisit']:
    model = SOMPairVisit(config=config).to(config['device'])
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

    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0.}
        global_iter0 = global_iter

        # init emb
        if config['warmup_epochs'] == epoch and (config['init_emb'] == 'pretrained-kmeans' or config['init_emb'] == 'pretrained-kmeans-switch' or config['init_emb'] == 'pretrained-kmeans-lssl'):
            if os.path.exists(os.path.join(config['ckpt_path'], 'init_emb_weights.npz')):
                data = np.load(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'))
                init = data['emb']
                init_dz = data['emb_dz']
                print('Load pre-saved initialization!')
            else:
                with torch.no_grad():
                    z_e_list = []
                    delta_z_list = []
                    for _, sample in enumerate(trainDataLoader, 0):
                        image1 = sample['img1'].to(config['device'], dtype=torch.float)
                        image2 = sample['img2'].to(config['device'], dtype=torch.float)
                        interval = sample['interval'].to(config['device'], dtype=torch.float)
                        if len(image1.shape) == 4:
                            image1 = image1.unsqueeze(1)
                            image2 = image2.unsqueeze(1)
                        image = torch.cat([image1, image2], dim=0)

                        z_e, _ = model.encoder(image)
                        z_e_list.append(z_e.detach().cpu().numpy())

                        bs = image1.shape[0]
                        delta_z_list.append(((z_e[bs:]-z_e[:bs])/interval.unsqueeze(1)).detach().cpu().numpy())
                    z_e_list = np.concatenate(z_e_list, 0)
                    delta_z_list = np.concatenate(delta_z_list, 0)
                    kmeans = sklearn.cluster.KMeans(n_clusters=config['embedding_size'][0]*config['embedding_size'][1]).fit(z_e_list)
                    init = kmeans.cluster_centers_.reshape(config['embedding_size'][0], config['embedding_size'][1], -1)
                    idx = kmeans.predict(z_e_list[:z_e_list.shape[0]//2])
                    init_dz = np.concatenate([delta_z_list[idx==k].mean(0) for k in range(config['embedding_size'][0]*config['embedding_size'][1])], axis=0).reshape(config['embedding_size'][0], config['embedding_size'][1], -1)
                    np.savez(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'), emb=init, emb_dz=init_dz)
            model.init_embeddings_ep_weight(init)
            model.init_embeddings_dz_ema_weight(init_dz)
            model.train()


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
            recons, zs = model.forward_pair_z(image1, image2, interval)

            recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
            recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
            z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
            z_q1, z_q2 = zs[1][0], zs[1][1]
            k1, k2 = zs[2][0], zs[2][1]
            sim1, sim2 = zs[3][0], zs[3][1]

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(image1, recon_ze1) + model.compute_recon_loss(image2, recon_ze2))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                loss_recon_zq = 0.5 * (model.compute_recon_loss(image1, recon_zq1) + model.compute_recon_loss(image2, recon_zq2))
                loss += config['lambda_recon_zq'] * loss_recon_zq
            else:
                loss_recon_zq = torch.tensor(0.)

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = 0.5 * (model.compute_som_loss(z_e1, k1, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max) + \
                                model.compute_som_loss(z_e2, k2, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max))
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            if config['lambda_dir'] > 0 and (config['dir_reg'] == 'LSSL' or epoch >= config['warmup_epochs']):
                if config['dir_reg'] == 'LSSL':
                    if epoch < config['warmup_epochs']:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff)
                    else:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff) + config['emb_dir_ratio'] * model.compute_emb_lssl_direction_loss()
                elif config['dir_reg'] == 'LNE':
                    if config['is_grid_ema']:
                        loss_dir = model.compute_lne_grid_ema_direction_loss(z_e_diff, k1)
                        model.update_grid_dz_ema(z_e_diff, k1)
                    else:
                        loss_dir = model.compute_lne_direction_loss(z_e_diff, sim1)
                else:
                    raise ValueError('Do not support this direction regularization method!')
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()
            loss_all_dict['dir'] += loss_dir.item()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            emb_old = model.embeddings.detach().cpu().numpy()
            enc_old = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            dec_old = torch.cat([param.view(-1) for param in model.decoder.parameters()]).detach().cpu().numpy()

            optimizer.step()
            optimizer.zero_grad()

            emb_new = model.embeddings.detach().cpu().numpy()
            enc_new = torch.cat([param.view(-1) for param in model.encoder.parameters()]).detach().cpu().numpy()
            dec_new = torch.cat([param.view(-1) for param in model.decoder.parameters()]).detach().cpu().numpy()
            print(np.abs(emb_new-emb_old).mean(), np.abs(enc_new-enc_old).mean(), np.abs(dec_new-dec_old).mean())
            # pdb.set_trace()

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon=[%.4f], recon_zq=[%.4f], commit=[%.4f], som=[%.4f], dir=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon.item(), loss_recon_zq.item(), loss_commit.item(), loss_som.item(), loss_dir.item()))
                print('Num. of k:', torch.unique(k1).shape[0], torch.unique(k2).shape[0])

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        if 'k1' not in loss_all_dict:
            loss_all_dict['k1'] = 0
        if 'k2' not in loss_all_dict:
            loss_all_dict['k2'] = 0

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

    loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0}
    input1_list = []
    input2_list = []
    label_list = []
    age_list = []
    interval_list = []
    recon1_list = []
    recon2_list = []
    recon_zq_list = []
    ze1_list = []
    ze2_list = []
    ze_diff_list = []
    zq_list = []
    k1_list = []
    k2_list = []
    sim1_list = []
    sim2_list = []
    subj_id_list = []
    case_order_list = []

    with torch.no_grad():
        recon_emb = model.recon_embeddings()

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

            recons, zs = model.forward_pair_z(image1, image2, interval)

            recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
            recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
            z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
            z_q1, z_q2 = zs[1][0], zs[1][1]
            k1, k2 = zs[2][0], zs[2][1]
            sim1, sim2 = zs[3][0], zs[3][1]

                        # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(image1, recon_ze1) + model.compute_recon_loss(image2, recon_ze2))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                loss_recon_zq = 0.5 * (model.compute_recon_loss(image1, recon_zq1) + model.compute_recon_loss(image2, recon_zq2))
                loss += config['lambda_recon_zq'] * loss_recon_zq
            else:
                loss_recon_zq = torch.tensor(0.)

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = 0.5 * (model.compute_som_loss(z_e1, k1) + model.compute_som_loss(z_e2, k2))
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            if config['lambda_dir'] > 0 and (config['dir_reg'] == 'LSSL' or epoch >= config['warmup_epochs']):
                if config['dir_reg'] == 'LSSL':
                    if epoch < config['warmup_epochs']:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff)
                    else:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff) + config['emb_dir_ratio'] * model.compute_emb_lssl_direction_loss()
                elif config['dir_reg'] == 'LNE':
                    if config['is_grid_ema']:
                        loss_dir = model.compute_lne_grid_ema_direction_loss(z_e_diff, k1)
                        model.update_grid_dz_ema(z_e_diff, k1)
                    else:
                        loss_dir = model.compute_lne_direction_loss(z_e_diff, sim1)
                else:
                    raise ValueError('Do not support this direction regularization method!')
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)


            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()
            loss_all_dict['dir'] += loss_som.item()

            if phase == 'test' and save_res:
                input1_list.append(image1.detach().cpu().numpy())
                input2_list.append(image2.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())
                recon1_list.append(recon_ze1.detach().cpu().numpy())
                recon2_list.append(recon_ze2.detach().cpu().numpy())
                recon_zq_list.append(recon_zq1.detach().cpu().numpy())
                ze1_list.append(z_e1.detach().cpu().numpy())
                ze2_list.append(z_e2.detach().cpu().numpy())
                ze_diff_list.append(z_e_diff.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())
                interval_list.append(interval.detach().cpu().numpy())
                subj_id_list.append(sample['subj_id'])
                case_order_list.append(np.stack([sample['case_order'][0].numpy(), sample['case_order'][1].numpy()], 1))
                zq_list.append(z_q1.detach().cpu().numpy())
                sim1_list.append(sim1.detach().cpu().numpy())
                sim2_list.append(sim2.detach().cpu().numpy())
            k1_list.append(k1.detach().cpu().numpy())
            k2_list.append(k2.detach().cpu().numpy())

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
            recon_zq_list = np.concatenate(recon_zq_list, axis=0)
            ze1_list = np.concatenate(ze1_list, axis=0)
            ze2_list = np.concatenate(ze2_list, axis=0)
            ze_diff_list = np.concatenate(ze_diff_list, axis=0)
            zq_list = np.concatenate(zq_list, axis=0)
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)
            sim1_list = np.concatenate(sim1_list, axis=0)
            sim2_list = np.concatenate(sim2_list, axis=0)

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
            h5_file.create_dataset('recon_zq', data=recon_zq_list)
            h5_file.create_dataset('ze1', data=ze1_list)
            h5_file.create_dataset('ze2', data=ze2_list)
            h5_file.create_dataset('ze_diff', data=ze_diff_list)
            h5_file.create_dataset('zq', data=zq_list)
            h5_file.create_dataset('k1', data=k1_list)
            h5_file.create_dataset('k2', data=k2_list)
            h5_file.create_dataset('sim1', data=sim1_list)
            h5_file.create_dataset('sim2', data=sim2_list)
            h5_file.create_dataset('embeddings', data=model.embeddings.detach().cpu().numpy())
            h5_file.create_dataset('recon_emb', data=recon_emb.detach().cpu().numpy())
        else:
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)

        print('Number of used embeddings:', np.unique(k1_list).shape, np.unique(k2_list).shape)
        loss_all_dict['k1'] = np.unique(k1_list).shape[0]
        loss_all_dict['k2'] = np.unique(k2_list).shape[0]

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
