import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
import sklearn.cluster
import copy

import pdb


class EncoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(EncoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(DecoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, dropout=False):
        super(Encoder, self).__init__()

        if dropout:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.1, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.2, num_conv=num_conv)
        else:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

        self.conv4 = nn.Sequential(
                        EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv),
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1))
        # self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        # self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # (16,4,4,4)
        return conv4.view(x.shape[0], -1), [conv3, conv2, conv1]
        # fc = self.fc(conv4.view(x.shape[0], -1))
        # return fc, [conv3, conv2, conv1]


class Decoder(nn.Module):
    def __init__(self, out_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, shortcut=False):
        super(Decoder, self).__init__()
        self.shortcut = shortcut
        # self.fc = nn.Linear(1024, 1024)
        if self.shortcut:
            self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = DecoderBlock(8*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = DecoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv1 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)
        else:
            self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = DecoderBlock(4*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, x, feat_list=[]):
        # fc = self.fc(x)
        # conv4 = self.conv4(fc.view(x.shape[0], 16, 4, 4, 4))
        x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        conv4 = self.conv4(x_reshaped)
        if self.shortcut:
            conv3 = self.conv3(torch.cat([conv4, feat_list[0]], 1))
            conv2 = self.conv2(torch.cat([conv3, feat_list[1]], 1))
            conv1 = self.conv1(torch.cat([conv2, feat_list[2]], 1))
        else:
            conv3 = self.conv3(conv4)
            conv2 = self.conv2(conv3)
            conv1 = self.conv1(conv2)
        output = self.conv0(conv1)
        return output


class SOM(nn.Module):
    def __init__(self, config):
        super(SOM, self).__init__()

        self.config = config
        self.device = config['device']
        self.latent_size = config['latent_size']
        self.embedding_size = config['embedding_size']
        self.dataset_name = config['dataset_name']
        init_emb = config['init_emb']

        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.embeddings = nn.Parameter(torch.fmod(torch.randn(self.embedding_size[0],self.embedding_size[1],self.latent_size,requires_grad=True),2) * 0.05, requires_grad=True)

    def compute_node_edge_weight(self, emb, idx):
        edge_sum = 0
        edge_count = 0
        row = idx // self.embedding_size[1]
        col = idx % self.embedding_size[1]
        if row != 0:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx-self.embedding_size[1]])
            edge_count += 1
        if row != self.embedding_size[0] - 1:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx+self.embedding_size[1]])
            edge_count += 1
        if col != 0:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx-1])
            edge_count += 1
        if col != self.embedding_size[1] - 1:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx+1])
            edge_count += 1
        return edge_sum / edge_count

    def reorganize_som_embeddings(self, emb_init):
        num_run = 10000
        emb_rand = copy.copy(emb_init.reshape(self.embedding_size[0]*self.embedding_size[1], -1))
        for run_idx in range(num_run):
            idx_sel = np.random.choice(self.embedding_size[0]*self.embedding_size[1], (2,))
            emb_tpm = copy.copy(emb_rand)
            emb_tpm[idx_sel[0]] = emb_rand[idx_sel[1]]
            emb_tpm[idx_sel[1]] = emb_rand[idx_sel[0]]
            weight_old = self.compute_node_edge_weight(emb_rand, idx_sel[0]) + self.compute_node_edge_weight(emb_rand, idx_sel[1])
            weight_new = self.compute_node_edge_weight(emb_tpm, idx_sel[0]) + self.compute_node_edge_weight(emb_tpm, idx_sel[1])
            if weight_new < weight_old:
                emb_rand = emb_tpm
            del emb_tpm
        return emb_rand

    def init_embeddings_ep_weight(self, init):
        if self.config['init_emb']  == 'pretrained-kmeans':
            init_reorganized = init
        else:
            init_reorganized = self.reorganize_som_embeddings(init)
        init = torch.tensor(init_reorganized, requires_grad=True).to(self.device)
        print('Finish embedding initialization by k-means!')
        self.embeddings.data = init.view(self.embedding_size[0], self.embedding_size[1], -1)
        recon_emb = self.recon_embeddings()
        res_path = os.path.join(self.config['ckpt_path'], 'recon_emb_warmup.npy')
        np.save(res_path, recon_emb.detach().cpu().numpy())

    def compute_zq_distance(self, z_e):
        z_dist = torch.sum((z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2, dim=-1)
        return z_dist

    # for each z_e, find the nearest embedding z_q
    def compute_zq(self, z_e, global_iter=-1, iter_max=-1):
        z_dist = self.compute_zq_distance(z_e)
        k = torch.argmin(z_dist.view(z_e.shape[0], -1), dim=-1)
        k_1 = k // self.embedding_size[1]
        k_2 = k % self.embedding_size[1]
        k_stacked = torch.stack([k_1, k_2], dim=1)
        z_q = self._gather_nd(self.embeddings, k_stacked)
        return z_q, k

    # find the neighbours of z_q
    def compute_zq_neighbours(self, z_q, k):
        k_1 = k // self.embedding_size[1]
        k_2 = k % self.embedding_size[1]
        k1_down = torch.where(k_1 < self.embedding_size[0] - 1, k_1 + 1, k_1)
        k1_up = torch.where(k_1 > 0, k_1 - 1, k_1)
        k2_right = torch.where(k_2 < self.embedding_size[1] - 1, k_2 + 1, k_2)
        k2_left = torch.where(k_2 > 0, k_2 - 1, k_2)
        z_q_up = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_down = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_left = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_right = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_neighbours = torch.stack([z_q, z_q_up, z_q_down, z_q_left, z_q_right], dim=1)  # check whether gradient get back if no z_q
        # z_q_neighbours = torch.stack([z_q_up, z_q_down, z_q_left, z_q_right], dim=1)  # check whether gradient get back if no z_q
        return z_q_neighbours

    # compute manhattan distance for each embedding to given nearest embedding index k
    def compute_manhattan_distance(self, k):
        k_1 = (k // self.embedding_size[1]).unsqueeze(1).unsqueeze(2)
        k_2 = (k % self.embedding_size[1]).unsqueeze(1).unsqueeze(2)
        row = torch.arange(0, self.embedding_size[0]).long().repeat(self.embedding_size[1],1).transpose(1,0).to(self.device)
        col = torch.arange(0, self.embedding_size[1]).long().repeat(self.embedding_size[0],1).to(self.device)
        row_diff = torch.abs(k_1 - row)
        col_diff = torch.abs(k_2 - col)
        return (row_diff + col_diff).float()

    # gather the elements of given index
    def _gather_nd(self, params, idxes, dims=(0,1)):
        if dims == (0,1):
            outputs = params[idxes[:,0], idxes[:,1]]
        else:
            outputs = params[:, idxes[:,0], idxes[:,1]]
        return outputs

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)
    
    # standard commitment loss, make sure z_q close to z_e (fix z_e)
    def compute_commit_loss(self, z_e, z_q):
        return torch.mean((z_e.detach() - z_q) ** 2) + self.config['commit_ratio'] * torch.mean((z_e - z_q.detach()) ** 2)

    # compute similarity between z_e and embeddings
    def compute_similarity(self, z_e, sim_type='softmax'):
        z_dist_flatten = self.compute_zq_distance(z_e).view(z_e.shape[0], -1)
        if sim_type == 'softmax':
            sim_flatten = F.softmax(- z_dist_flatten / torch.std(z_dist_flatten, dim=1).unsqueeze(1))
        else:
            raise ValueError('Not supporting this similarity type')
        return sim_flatten

    # som loss based on grid distance to zq, from XADLiME paper
    def compute_som_loss(self, z_e, k, iter=-1, iter_max=-1, Tmax=1., Tmin=0.1):
        Tmin = self.config['Tmin']
        Tmax = self.config['Tmax']
        dis_ze_emb = self.compute_zq_distance(z_e.detach())
        dis_zq_manhattan = self.compute_manhattan_distance(k)               # (bs, 32, 32)
        if iter != -1 and iter_max != -1:
            self.T = Tmax * (Tmin / Tmax)**(iter / iter_max)
        else:
            self.T = Tmin
        weight = torch.exp(-0.5 * dis_zq_manhattan**2 / (self.embedding_size[0] * self.embedding_size[1] * self.T**2))
        weight_normalized = weight / weight.view(-1, self.embedding_size[0] * self.embedding_size[1]).sum(1).unsqueeze(1).unsqueeze(2)
        som = torch.mean(weight_normalized * dis_ze_emb)
        return som

    def recon_embeddings(self):
        emb_resize = self.embeddings.reshape(-1, self.latent_size)
        recon_emb_list = []
        i = 0
        while(1):
            if (i+1) * 64 < emb_resize.shape[0]:
                recon_emb = self.decoder(emb_resize[i*64:(i+1)*64, :])
                recon_emb_list.append(recon_emb)
            else:
                recon_emb = self.decoder(emb_resize[i*64:, :])
                recon_emb_list.append(recon_emb)
                break
            i += 1
        recon_emb_list = torch.cat(recon_emb_list, 0)
        return recon_emb_list

    def forward(self, x, global_iter=-1, iter_max=-1):
        z_e, feat_list = self.encoder(x)
        z_q, k = self.compute_zq(z_e, global_iter, iter_max)
        sim = self.compute_similarity(z_e, sim_type='softmax')
        recon_ze = self.decoder(z_e, feat_list)
        recon_zq = self.decoder(z_q, feat_list)
        return [recon_ze, recon_zq], [z_e, z_q, k, sim]


class SOMPairVisit(SOM):
    def __init__(self, config):
        super(SOM, self).__init__()

        self.config = config
        self.device = config['device']
        self.latent_size = config['latent_size']
        self.embedding_size = config['embedding_size']
        self.dataset_name = config['dataset_name']
        init_emb = config['init_emb']

        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.embeddings = nn.Parameter(torch.fmod(torch.randn(self.embedding_size[0],self.embedding_size[1],self.latent_size,requires_grad=True),2) * 0.05, requires_grad=True)

        if ('dir_reg' in config and config['dir_reg'] == 'LSSL') or config['init_emb'] == 'pretrained-kmeans-lssl':
            self.direction = nn.Linear(1, self.latent_size)

        if 'is_grid_ema' in config and config['is_grid_ema']:
            self.embeddings_dz_ema = nn.Parameter(torch.zeros(self.embedding_size[0],self.embedding_size[1],self.latent_size, requires_grad=False), requires_grad=False)

    def reorganize_lssl_som_embeddings(self, emb_init):
        emb_init = torch.tensor(emb_init, requires_grad=True).to(self.device).view(-1, emb_init.shape[-1])
        d_vec = self.direction(torch.ones(emb_init.shape[0], 1).to(self.device))
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        proj = torch.sum(emb_init * d_vec, 1) / d_vec_norm
        emb_sort = emb_init[torch.argsort(proj)]
        emb_sort = torch.transpose(emb_sort.reshape(self.embedding_size[1], self.embedding_size[0], -1), 0,1)
        return emb_sort

    def init_embeddings_ep_weight(self, init):
        if self.config['init_emb']  == 'pretrained-kmeans':
            init_reorganized = init
        elif self.config['init_emb'] == 'pretrained-kmeans-lssl':
            init_reorganized = self.reorganize_lssl_som_embeddings(init)
        else:
            init_reorganized = self.reorganize_som_embeddings(init)
        init = torch.tensor(init_reorganized, requires_grad=True).to(self.device)
        print('Finish embedding initialization by k-means!')
        self.embeddings.data = init.view(self.embedding_size[0], self.embedding_size[1], -1)
        recon_emb = self.recon_embeddings()
        res_path = os.path.join(self.config['ckpt_path'], 'recon_emb_warmup.npy')
        np.save(res_path, recon_emb.detach().cpu().numpy())

    def init_embeddings_dz_ema_weight(self, init):
        if self.config['is_grid_ema'] == True:
            self.embeddings_dz_ema.data = torch.tensor(init, requires_grad=False).to(self.device)
            print('Finished grid ema initialization!')
    
    def compute_lssl_direction_loss(self, z_e_diff):
        bs = z_e_diff.shape[0]
        delta_z = z_e_diff
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        d_vec = self.direction(torch.ones(bs, 1).to(self.device))
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        cos = torch.sum(delta_z * d_vec, 1) / (delta_z_norm * d_vec_norm)
        return (1. - cos).mean()

    def compute_emb_lssl_direction_loss(self, thres=0.5):
        emb_pair = []
        for i in range(1, self.embedding_size[1]):
            emb_pair.append(self.embeddings[:,i] - self.embeddings[:,i-1])
        emb_pair = torch.cat(emb_pair, dim=0)
        emb_pair_norm = torch.norm(emb_pair, dim=1) + 1e-12
        d_vec = self.direction(torch.ones(emb_pair.shape[0], 1).to(self.device))
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        cos = torch.sum(emb_pair * d_vec, 1) / (emb_pair_norm * d_vec_norm)
        loss = torch.max(thres - cos, torch.zeros_like(cos)).mean()
        return loss

    def compute_lne_direction_loss(self, z_e_diff, sim, thres=0.95):
        delta_z = z_e_diff
        sim_norm = torch.norm(sim, dim=1) + 1e-12
        sim_normalized = sim / sim_norm.view(-1,1)
        adj_mx = torch.matmul(sim_normalized, sim_normalized.T).detach()
        adj_mx[adj_mx < thres] = 0.
        delta_h = torch.matmul(adj_mx, delta_z.detach()) / adj_mx.sum(1, keepdim=True)
        print(adj_mx.sum() / delta_z.shape[0])

        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        delta_h_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (delta_z_norm * delta_h_norm)

        return (1. - cos).mean()

    def compute_lne_grid_ema_direction_loss(self, delta_z, k):
        delta_z_grid = self.embeddings_dz_ema[k//self.embedding_size[1], k%self.embedding_size[1]]
        delta_z_grid_norm = torch.norm(delta_z_grid, dim=1) + 1e-12
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_z_grid.detach(), 1) / (delta_z_norm * delta_z_grid_norm.detach())
        return (1. - cos).mean()

    def update_grid_dz_ema(self, delta_z, k):
        for r in range(self.embedding_size[0]):
            for c in range(self.embedding_size[1]):
                i = r * self.embedding_size[1] + c
                if (k == i).sum() == 0:
                    continue
                delta_z_new = delta_z[k == i].mean(0)
                self.embeddings_dz_ema[r, c] = self.config['grid_ema_rate'] * self.embeddings_dz_ema[r, c] + (1.-self.config['grid_ema_rate']) * delta_z_new.detach()

    def forward_pair_z(self, x1, x2, interval):
        bs = x1.shape[0]
        z_e, feat_list = self.encoder(torch.cat([x1, x2], dim=0))
        z_e1, z_e2 = z_e[:bs], z_e[bs:]

        z_e_diff = (z_e2 - z_e1) / interval.unsqueeze(1)
        recon_ze = self.decoder(z_e, feat_list)
        recon_ze1, recon_ze2 = recon_ze[:bs], recon_ze[bs:]

        z_q, k = self.compute_zq(z_e)
        sim = self.compute_similarity(z_e, sim_type='softmax')
        recon_zq = self.decoder(z_q, feat_list)

        z_q1, z_q2 = z_q[:bs], z_q[bs:]
        recon_zq1, recon_zq2 = recon_zq[:bs], recon_zq[bs:]
        k1, k2 = k[:bs], k[bs:]
        sim1, sim2 = sim[:bs], sim[bs:]

        return [[recon_ze1, recon_ze2], [recon_zq1, recon_zq2]], [[z_e1, z_e2, z_e_diff], [z_q1, z_q2], [k1, k2], [sim1, sim2]]


class Classifier(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
                    nn.BatchNorm1d(latent_size),
                    # nn.Dropout(0.2),
                    nn.Linear(latent_size, inter_num_ch),
                    nn.LeakyReLU(0.2),
                    nn.Linear(inter_num_ch, 1))

        self._init()

    def _init(self):
        for layer in self.fc.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.fc(x)


class CLS(nn.Module):
    def __init__(self, config):
        super(CLS, self).__init__()
        self.config = config
        self.device = config['device']
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)

        if self.config['is_som']:
            self.embedding_size = self.config['embedding_size']
            self.embeddings = nn.Parameter(torch.zeros(self.embedding_size[0],self.embedding_size[1],self.config['latent_size']), requires_grad=False)
            self.classifier = Classifier(latent_size=self.embedding_size[0]*self.embedding_size[1], inter_num_ch=self.embedding_size[0]*self.embedding_size[1])
        else:
            self.classifier = Classifier(latent_size=len(config['use_feature'])*config['latent_size'], inter_num_ch=64)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs, _ = self.encoder(torch.cat([img1, img2], 0))
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]

        if len(self.config['use_feature']) == 2:
            input = torch.cat([z1, delta_z], 1)
        elif 'z' in self.config['use_feature']:
            input = z1
        else:
            input = delta_z
        pred = self.classifier(input)
        return pred

    def forward_single(self, img1):
        z1, _ = self.encoder(img1)
        z1 = z1.view(img1.shape[0], -1)
        if self.config['is_som']:
            sim1 = self.compute_similarity(z1).view(img1.shape[0], -1)
            pred = self.classifier(sim1)
        else:
            pred = self.classifier(z1)
        return pred

    def compute_zq_distance(self, z_e):
        z_dist = torch.sum((z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2, dim=-1)
        return z_dist

    def compute_similarity(self, z_e):
        z_dist_flatten = self.compute_zq_distance(z_e).view(z_e.shape[0], -1)
        sim_flatten = F.softmax(- z_dist_flatten / torch.std(z_dist_flatten, dim=1).unsqueeze(1))
        return sim_flatten

    def compute_task_loss(self, pred, label):
        if 'classification' in self.config['task']:
            if self.config['task'] == 'NC_AD_classification':
                pos_weight=torch.tensor([1.7])
            elif self.config['task'] == 'pMCI_sMCI_classification':
                pos_weight=torch.tensor([1.4])
            else:
                raise ValueError('Not support!')
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device, dtype=torch.float))(pred.squeeze(1), label)
        elif 'regression' in self.config['task']:
            loss = nn.MSELoss()(pred.squeeze(1), label)
        else:
            raise ValueError('Not support!')
        return loss

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs, _ = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

class LSSL(nn.Module):
    def __init__(self, config):
        super(LSSL, self).__init__()

        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.direction = nn.Linear(1, 1024)
        self.gpu = config['device']

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs, _ = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss
    def compute_direction_loss(self, zs):
        z1, z2 = zs[0], zs[1]
        bs = z1.shape[0]
        delta_z = z2 - z1
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        d_vec = self.direction(torch.ones(bs, 1).to(self.gpu))
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        cos = torch.sum(delta_z * d_vec, 1) / (delta_z_norm * d_vec_norm)
        return (1. - cos).mean()

class LNE(nn.Module):
    def __init__(self, config, num_neighbours=5, agg_method='gaussian'):
        super(LNE, self).__init__()
        self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
        self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.num_neighbours = num_neighbours
        self.agg_method = agg_method
        self.gpu = config['device']

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs, _ = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    def forward_single(self, img):
        return self.encoder(img)

    def build_graph_batch(self, zs):
        z1 = zs[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, bs).to(self.gpu)
        for i in range(bs):
            for j in range(i+1, bs):
                dis_mx[i, j] = torch.sum((z1[i] - z1[j]) ** 2)
                dis_mx[j, i] = dis_mx[i, j]
        if self.agg_method == 'gaussian':
            adj_mx = torch.exp(-dis_mx/100)
        if self.num_neighbours < bs:
            adj_mx_filter = torch.zeros(bs, bs).to(self.gpu)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours+1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
                adj_mx_filter[i, i] = 0.
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.gpu))

    def build_graph_dataset(self, zs_all, zs):
        z1_all = zs_all[0]
        z1 = zs[0]
        ds = z1_all.shape[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, ds).to(self.gpu)
        for i in range(bs):
            for j in range(ds):
                dis_mx[i, j] = torch.sum((z1[i] - z1_all[j]) ** 2)
        if self.agg_method == 'gaussian':
            adj_mx = torch.exp(-dis_mx/100)
        if self.num_neighbours < bs:
            adj_mx_filter = torch.zeros(bs, ds).to(self.gpu)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours+1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.gpu))

    def compute_social_pooling_delta_z_batch(self, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z) / adj_mx.sum(1, keepdim=True)    # [bs, ls]
        return delta_z, delta_h

    def compute_social_pooling_delta_z_dataset(self, zs_all, interval_all, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]
        z1_all, z2_all = zs_all[0], zs_all[1]
        delta_z_all = (z2_all - z1_all) / interval_all.unsqueeze(1)      # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z_all) / adj_mx.sum(1, keepdim=True)    # [bs, ls]
        return delta_z, delta_h

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss, 1 - cos<delta_z, delta_h>
    def compute_direction_loss(self, delta_z, delta_h):
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        delta_h_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (delta_z_norm * delta_h_norm)
        return (1. - cos).mean()
