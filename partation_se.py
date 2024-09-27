import numpy as np
from matplotlib import pyplot as plt

import silearn
import skimage
import math
import torch
import matplotlib
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device( 'cpu')

class Partition:
    def __init__(self, t=1., se_t=2e-7):
        self.t, self.SE_t = t, se_t
        self.seg = None
        self.cur_data = None

    def construct_graph(self, img):
        '''
        For pictures of the same data set (same size),
        you can randomly select parts to determine a better radius and fix the radius,
        further reducing composition time.
        '''

        '''serch radius'''
        ds = range(3, 50, 2)
        imgH = img.shape[0]
        imgW = img.shape[1]
        entropies = []
        e_ent = []
        best_edges = None
        best_w = None
        for i, d in enumerate(ds):
            w, es, et = silearn.spatial_knn_graph(img, d * d + 1, d)
            w = torch.exp2(-w.double() / w.mean() / self.t)
            w = torch.clip(w, min=1e-256)
        
            edges = torch.cat([es.unsqueeze(1), et.unsqueeze(1)], dim=1)
            mx = torch.max(edges)
            v1 = silearn.scatter_sum(w, edges[:, 1], clip_length=mx + 1).reshape(-1)
            v_all = torch.sum(v1)
            probabilities = v1 / v_all
            non_zero_probabilities = probabilities[probabilities > 0]
            entropy = -torch.sum(non_zero_probabilities * torch.log2(non_zero_probabilities))
            entropy_value = entropy.item()
            entropies.append(entropy_value)
            e_ent.append(entropy_value/edges.size(0))
            if i >0 :
                if (e_ent[i-1] - e_ent[i] ) <= self.SE_t:
                    best_entropy = entropies[i-1]
                    best_d = ds[i-1]
                    print(f'\tGraph Construction: MAX_1D_SE: {best_entropy}, Best_layer_number: {best_d // 2}.')
        
                    return imgH * imgW, best_edges, best_w
        
            best_edges =edges
            best_w = w


    def get_adj_cover(self, idx, width):
        delta = torch.abs(idx[:, 0] - idx[:, 1])
        return (delta == 1) + (delta == width) + (delta == width + 1) + (delta == width - 1) > 0

    @staticmethod
    def reduction_edge(edges, edge_transform, *weights):
        cnt_e = edge_transform.max() + 1
        e1 = torch.zeros(size=(cnt_e, edges.shape[1]), dtype=edges.dtype, device=edges.device)
        edges = e1.scatter(0, edge_transform.reshape(-1, 1).repeat((1, 2)), edges)
        ret = [edges] + [silearn.scatter_sum(w, edge_transform) for w in weights]
        return tuple(ret)

    @staticmethod
    def get_edge_transform(edges, identical_flag=False):
        max_id = int(edges[:, 1].max() + 1)
        bd = 1
        shift = 0
        while bd <= max_id:
            bd = bd << 1
            shift += 1
        # todo: hash if shift is too big
        edge_hash = (edges[:, 0] << shift) + edges[:, 1]

        if identical_flag:
            _, transform, counts = torch.unique(edge_hash, return_inverse=True, return_counts=True)
            flag = counts[transform] != 1
            return transform, flag
        else:
            _, transform = torch.unique(edge_hash, return_inverse=True)
            return transform

    @staticmethod
    def sum_up_multi_edge(edges, *weights):

        trans = Partition.get_edge_transform(edges)
        r = Partition.reduction_edge(edges, trans, *weights)
        return r

    def single_perform(self, adj_cover, edge, w, tgt_size):
        it = 0
        seg = None
        op_num = None
        cluster_cur = None

        while True :
            transs = []
            if it == 0:
                it += 1
                adj_cover = adj_cover
                edges, trans_prob = edge, w
            else:
                transs.append(seg)
                edges, trans_prob, adj_cover = self.cur_data

            edge_s = edges[:, 0]
            edge_t = edges[:, 1]
            mx = torch.max(edges)

            v1 = silearn.scatter_sum(trans_prob, edge_t, clip_length=mx + 1).reshape(-1)
            vst = v1[edges]

            if it == 0:
                dH = (2*trans_prob * (torch.log2(trans_prob.sum()) - torch.log2(vst[:, 0] + vst[:, 1])))/trans_prob.sum()
            else:
                non_loop = edge_s != edge_t
                g1 = silearn.scatter_sum(trans_prob * (non_loop), edge_t, clip_length=mx + 1)
                gst = g1[edges]
                vx = vst.sum(dim=1)

                vin = vst - gst
                dH = ((vin[:, 0]) * torch.log2(vst[:, 0]) + \
                     (vin[:, 1]) * torch.log2(vst[:, 1]) - \
                     (vin[:, 0] + vin[:, 1]) * torch.log2(vx) + \
                      2 * trans_prob * (torch.log2(trans_prob.sum()) - torch.log2(vx)))/trans_prob.sum()


            mask_same = edge_s == edge_t
            dH[mask_same] = -1e10
            dH[adj_cover<=0] = -1e10
            values, dH_amax = silearn.scatter_max(dH, edge_s)
            mask = values > 0

            if op_num is None:
                # mask = values>0
                merge = dH_amax[mask]

                if merge.size(0) == 0:
                    op_num = cluster_cur - tgt_size
                    continue
            else:
                positive_values_size = values[mask].size(0)
                if positive_values_size == 0:
                    positive_values_size = op_num +1
                _, top_indices = torch.topk(values, min(op_num, positive_values_size))
                # _, top_indices = torch.topk(values, op_num)
                merge = dH_amax[top_indices]


            op_edges = edges[merge]
            sorted_edges = torch.sort(op_edges, dim=1)[0]
            op_edges = torch.unique(sorted_edges, dim=0, sorted=False)
            ids = op_edges[:,0]
            idt = op_edges[:,1]


            trans = torch.arange(edges.max() + 1, device=device)
            trans[ids] = trans[idt]
            lg_merge = math.log2(len(ids) + 2)
            for i in range(int(lg_merge)):
                trans[ids] = trans[trans[ids]]
            trans = torch.unique(trans, return_inverse=True)[1]
            transs.append(trans)

            trans_copy = trans.clone()


            if len(transs) != 0:
                trans = None
                for i in reversed(range(len(transs))):
                    if trans is None:
                        trans = transs[i]
                    else:
                        trans = trans[transs[i]]

            cluster_size = len(set(trans.tolist()))

            if cluster_size >= tgt_size :
                cluster_cur = cluster_size
                edges = trans_copy[edges]
                weights = [trans_prob, adj_cover.int()]
                ret = Partition.sum_up_multi_edge(edges, *weights)
                self.cur_data = [ret[0], ret[1], ret[2]]
                seg = trans
            else:
                op_num = cluster_cur - tgt_size
                continue

            if cluster_size == tgt_size:
                print(f"\tPartation：Target_size:{tgt_size}, Final_size: {cluster_size}.")
                break


        return seg

    def multi_perform(self, adj_cover, edge, w, tgt_size):
        tgt_size.sort(reverse=True)
        it = 0
        seg = None
        seg_set = []
        op_num = None
        cluster_cur = None
        while True:
            transs = []
            if it == 0:
                it += 1
                adj_cover = adj_cover
                edges, trans_prob = edge, w
            else:
                transs.append(seg)
                edges, trans_prob, adj_cover = self.cur_data

            edge_s = edges[:, 0]
            edge_t = edges[:, 1]
            mx = torch.max(edges)

            v1 = silearn.scatter_sum(trans_prob, edge_t, clip_length=mx + 1).reshape(-1)
            vst = v1[edges]

            if it == 0:
                dH = (2 * trans_prob * (
                            torch.log2(trans_prob.sum()) - torch.log2(vst[:, 0] + vst[:, 1]))) / trans_prob.sum()
            else:
                non_loop = edge_s != edge_t
                g1 = silearn.scatter_sum(trans_prob * (non_loop), edge_t, clip_length=mx + 1)
                gst = g1[edges]
                vx = vst.sum(dim=1)

                vin = vst - gst
                dH = ((vin[:, 0]) * torch.log2(vst[:, 0]) + \
                      (vin[:, 1]) * torch.log2(vst[:, 1]) - \
                      (vin[:, 0] + vin[:, 1]) * torch.log2(vx) + \
                      2 * trans_prob * (torch.log2(trans_prob.sum()) - torch.log2(vx))) / trans_prob.sum()

            mask_same = edge_s == edge_t
            dH[mask_same] = -1e10
            dH[adj_cover <= 0] = -1e10
            values, dH_amax = silearn.scatter_max(dH, edge_s)
            mask = values > 0

            if op_num is None:
                # mask = values>0
                merge = dH_amax[mask]

                if merge.size(0) == 0:
                    op_num = cluster_cur - tgt_size[0]
                    continue
            else:
                positive_values_size = values[mask].size(0)
                if positive_values_size == 0:
                    positive_values_size = op_num + 1
                _, top_indices = torch.topk(values, min(op_num, positive_values_size))
                # _, top_indices = torch.topk(values, op_num)
                merge = dH_amax[top_indices]

            op_edges = edges[merge]
            sorted_edges = torch.sort(op_edges, dim=1)[0]
            op_edges = torch.unique(sorted_edges, dim=0, sorted=False)
            ids = op_edges[:, 0]
            idt = op_edges[:, 1]

            trans = torch.arange(edges.max() + 1, device=device)
            trans[ids] = trans[idt]
            lg_merge = math.log2(len(ids) + 2)
            for i in range(int(lg_merge)):
                trans[ids] = trans[trans[ids]]
            trans = torch.unique(trans, return_inverse=True)[1]
            transs.append(trans)

            trans_copy = trans.clone()

            if len(transs) != 0:
                trans = None
                for i in reversed(range(len(transs))):
                    if trans is None:
                        trans = transs[i]
                    else:
                        trans = trans[transs[i]]

            cluster_size = len(set(trans.tolist()))

            if cluster_size >= tgt_size[0]:
                cluster_cur = cluster_size
                edges = trans_copy[edges]
                weights = [trans_prob, adj_cover.int()]
                ret = Partition.sum_up_multi_edge(edges, *weights)
                self.cur_data = [ret[0], ret[1], ret[2]]
                seg = trans
            else:
                op_num = cluster_cur - tgt_size[0]
                continue

            if cluster_size == tgt_size[0]:
                seg_set.append(seg.reshape(-1, 1))
                if  len(tgt_size)  == 1:
                    print(f"\tsize_set:{tgt_size}, Final_size: {cluster_size}.")
                    break
                elif len(tgt_size) > 1:
                    print(f"\tsize_set:{tgt_size},current size:{cluster_size}")
                    tgt_size = tgt_size[1:]
                    op_num = None

        return seg_set

    def fit(self, img, target_size, multi_scale):
        img_w = img.shape[1]

        img = torch.tensor(skimage.color.rgb2lab(img / 255).astype(np.float64)).to(device)

        self.cur_num_nodes, edges, w = self.construct_graph(img)

        adj_cover = self.get_adj_cover(edges, img_w)

        if multi_scale:
            if not isinstance(target_size, list):
                raise TypeError("Error: The 'target_size' variable must be a list when 'multi_scale' is True")
            seg_set = self.multi_perform(adj_cover=adj_cover, edge=edges, w=w, tgt_size=target_size)
            return seg_set

        else:
            if not isinstance(target_size, int):
                raise TypeError("Error: The 'target_size' variable must be a int value when 'multi_scale' is False")
            seg = self.single_perform(adj_cover=adj_cover, edge=edges, w=w, tgt_size=target_size)
            return seg.reshape(-1, 1)












