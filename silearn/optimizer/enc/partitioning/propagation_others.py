import math
import silearn.model.encoding_tree
from silearn.model.encoding_tree import GraphEncoding
from silearn.optimizer.enc.operator import Operator
import torch


class OperatorPropagation(Operator):

    def __init__(self, enc: GraphEncoding, objective="SE"):
        super().__init__(enc)
        self.edge_descriptors = []
        self.adj_cover = None
        self.adjacency_restriction = None
        self.objective = objective

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
    def sum_up_multi_edge(edges, *weights, operation_ptrs=None):
        if operation_ptrs is not None:
            em = edges[operation_ptrs]
            wm = [i[operation_ptrs] for i in weights]
            trans = OperatorPropagation.get_edge_transform(em)
            redu = OperatorPropagation.reduction_edge(em, trans, *wm)
            cnt = redu[0].shape[0]
            edges[operation_ptrs][:cnt] = redu[0]
            ret = [edges]
            for i in range(len(weights)):
                weights[i][operation_ptrs][:cnt] = redu[i + 1]
                weights[i][operation_ptrs][cnt:] = 0
                ret += [weights[i]]
            return ret
        else:
            trans = OperatorPropagation.get_edge_transform(edges)
            r = OperatorPropagation.reduction_edge(edges, trans, *weights)
            return r

    @staticmethod
    def sum_up_multi_edge_ts(edges, *weights, operation_ptrs=None):
        if operation_ptrs is not None:
            em = edges[operation_ptrs]
            wm = [i[operation_ptrs] for i in weights]
            trans = OperatorPropagation.get_edge_transform(em)
            redu = OperatorPropagation.reduction_edge(em, trans, *wm)
            cnt = redu[0].shape[0]
            operation_ptrs = operation_ptrs[:cnt]
            skip_ptrs = operation_ptrs[cnt:]
            edges[operation_ptrs] = redu[0]
            for i in range(len(weights)):
                weights[i][operation_ptrs] = redu[i + 1]
                weights[i][skip_ptrs] = 0
            return skip_ptrs
        else:
            trans = OperatorPropagation.get_edge_transform(edges)
            return OperatorPropagation.reduction_edge(edges, trans, *weights)

    def on_merged(self, edges, trans_prob, trans):
        """
        Listener on one Merge / Coarsening / Reduction step
        :param edges: Edge Idx Results after operation
        :param trans_prob: Edge Weight (Trans Prob) Results before operation
        :param trans: Node Id Mapping
        :return modified edges & trans_prob
        """
        return edges, trans_prob

    def on_operated(self, edges, trans_prob, trans):
        """
        Listener on overall Merge / Coarsening / Reduction operation
        :param edges: Edge Idx Results after operation
        :param trans_prob: Edge Weight (Trans Prob) Results before operation
        :param trans: Node Id Mapping
        :return modified edges & trans_prob
        """
        return edges, trans_prob

    def add_edge_descriptor(self, fea):
        self.edge_descriptors.append(fea)

    def perform(self, p=1.0,
                ter=0, n=200, contains_self_loops=None,
                adj_cover=None, min_com=1, max_com=math.inf, di_max=False, srt_M=False,
                m_scale=None, re_compute=True):
        assert 1 <= min_com <= max_com
        assert 0.0 <= p <= 1.0

        edges, trans_prob = self.enc.graph.edges

        self._log2m = torch.log2(trans_prob.sum())
        if m_scale != None:
            self._log2m += m_scale

        operated_cnt = math.inf
        transs = []
        merge_all = False
        if contains_self_loops is None:
            contains_self_loops = bool((edges[:, 0] == edges[:, 1]).any())

        vst = None
        dH0 = None

        cache = False

        current_num_vertices = self.enc.graph.num_vertices
        transs_start = 0
        if re_compute is not True and self.enc.node_id is not None:
            trans = self.enc.node_id

            cache = False
            if hasattr(self, "cur_data"):
                edges, trans_prob, adj_cover = self.cur_data
                current_num_vertices = edges.max() + 1
                contains_self_loops = True
            else:
                edges = trans[edges]
                if adj_cover is None:
                    edges, trans_prob = OperatorPropagation.sum_up_multi_edge(edges, trans_prob)
                else:
                    edges, trans_prob, adj_cover = OperatorPropagation.sum_up_multi_edge(edges, trans_prob, adj_cover)
                current_num_vertices = edges.max() + 1
                contains_self_loops = True
            transs.append(trans)
            transs_start = 1



        while operated_cnt > ter or not merge_all:

            max_operate_cnt = math.ceil((current_num_vertices - min_com) * p)  # math.ceil向上取最接近的整数

            edge_s = edges[:, 0]
            edge_t = edges[:, 1]
            mx = torch.max(edges)

            if operated_cnt <= ter + 1:
                merge_all = True

            if not cache:
                v1 = silearn.scatter_sum(trans_prob, edge_t, clip_length=mx + 1).reshape(-1)  # 得到每个点的体积
                vst = v1[edges]  # 得到每条边对应的，两侧体积

            if contains_self_loops:
                if not cache:
                    non_loop = edge_s != edge_t
                    g1 = silearn.scatter_sum(trans_prob * (non_loop), edge_t, clip_length=mx + 1)
                    gst = g1[edges]
                    vx = vst.sum(dim=1)

                    dH0 = self.get_dH(trans_prob, gst, vst, vx)
                    op = (dH0 > 0)
                    dH = dH0

                else:
                    op = (dH0 > 0)
                    dH = dH0

                if not torch.any(op):
                    break
                if not merge_all:
                    op = (dH >= torch.median(dH[op]))
                op = torch.logical_and(op, non_loop)
            else:
                dH = self.get_dH_v_eq_g(trans_prob, vst)  # 计算 wi / Vol(src) + Vol(drt)
                if not merge_all:
                    op = (dH >= torch.median(dH[dH > 0]))  # 判断dH在全部边的dH中，是否小于非负中值水平
                else:
                    op = dH > 0

            cache = True

            if adj_cover is not None:
                op = torch.logical_and(op, adj_cover > 0)

            merge = op

            # 挑出单边，避免重复合并
            same = (vst[merge, 0] == vst[merge, 1]).any()
            if same.any():
                hash_x = edge_s * 10007 % 1009
                hash_t = edge_t * 10007 % 1009
                merge = torch.logical_and(merge,
                                          torch.logical_or((vst[:, 0] < vst[:, 1]),
                                                           torch.logical_and((vst[:, 0] == vst[:, 1]),
                                                                             hash_x < hash_t)))
            else:
                merge = torch.logical_and(merge, vst[:, 0] < vst[:, 1])

            if not torch.any(merge):
                operated_cnt = 0
                continue

            id0 = edge_s
            id1 = edge_t

            id0 = id0[merge]
            id1 = id1[merge]
            dH = dH[merge]

            _, dH_amax = silearn.scatter_max(dH, id0)  # 选择每个点dH最高的边

            # dH_amax = dH_amax[dH_amax < dH.shape[0]]

            operated_cnt = int(dH_amax.shape[0])  # 操作数
            if operated_cnt == 0:
                continue


            # 双向max
            if operated_cnt > max_operate_cnt and di_max:
                id0 = id0[dH_amax]
                id1 = id1[dH_amax]
                dH = dH[dH_amax]
                _, dH_amax = silearn.scatter_max(dH, id1)
                dH_amax = dH_amax[dH_amax < dH.shape[0]]
                operated_cnt = int(dH_amax.shape[0])

            if operated_cnt > max_operate_cnt:
                _, idx = torch.sort(dH[dH_amax], descending=True)
                dH_amax = dH_amax[idx[:max_operate_cnt]]  # 选择当前dH最大的进行作为待处理的点
                operated_cnt = max_operate_cnt

            current_num_vertices -= operated_cnt

            ids = id0[dH_amax]
            idt = id1[dH_amax]
            trans = torch.arange(edges.max() + 1, device=self.enc.graph.device)

            trans[ids] = trans[idt] #把目标节点分区置为源节点的索引

            lg_merge = math.log2(operated_cnt + 2)


            for i in range(int(lg_merge)):
                trans[ids] = trans[trans[ids]]

            trans = torch.unique(trans, return_inverse=True)[1]

            transs.append(trans)

            if adj_cover is None:
                cache = False

                edges = trans[edges]
                weights = [trans_prob] + self.edge_descriptors
                ret = OperatorPropagation.sum_up_multi_edge(edges, *weights)
                edges = ret[0]
                trans_prob = ret[1]
                self.edge_descriptors = list(ret[2:])


            else:
                cache = False
                edges = trans[edges]


                weights = [trans_prob, adj_cover.int()] + self.edge_descriptors
                ret = OperatorPropagation.sum_up_multi_edge(edges, *weights)

                edges = ret[0]
                trans_prob = ret[1]
                adj_cover = ret[2]
                self.edge_descriptors = list(ret[3:])
            contains_self_loops = True
            self.adj_cover = adj_cover
            edges, trans_prob = self.on_merged(edges, trans_prob, trans)
            if current_num_vertices == int(min_com):
                break

        if len(transs) != 0:
            trans = None
            for i in reversed(range(len(transs))):
                if trans is None:
                    trans = transs[i]
                else:
                    trans = trans[transs[i]]

                # print(i, trans.shape)
                if i == transs_start:
                    edges, trans_prob = self.on_operated(edges, trans_prob, trans)
                # one extra trans: 0.01s

            self.enc.node_id = trans
        else:

            com0 = torch.arange(self.enc.graph.num_vertices, device=self.enc.graph.device)
            self.enc.node_id = com0
        self.cur_data = [edges, trans_prob, adj_cover]

    def perform_his(self, n, q=None, q_t=None, fs=False, re_his=False, high_dH=False, adj_cover=None):
        n_ = n
        if not hasattr(self, "cur_data"):
            edges, trans_prob = self.enc.graph.edges
            self.cur_data = [edges, trans_prob, adj_cover]
        while True:
            transs = []
            trans = self.enc.node_id
            edges, trans_prob, adj_cover = self.cur_data
            transs.append(trans)
            transs_start = 1
            current_num_vertices = edges.max() + 1
            graph_splits = [(s, min(s + n - 1, current_num_vertices - 1)) for s in range(0, current_num_vertices, n)]
            edge_s = edges[:, 0]
            edge_t = edges[:, 1]
            mx = torch.max(edges)
            v1 = silearn.scatter_sum(trans_prob, edge_t, clip_length=mx + 1).reshape(-1)
            vst = v1[edges]
            non_loop = edge_s != edge_t
            g1 = silearn.scatter_sum(trans_prob * (non_loop), edge_t, clip_length=mx + 1)
            gst = g1[edges]
            # vx = vst.sum(dim=1)
            ids = []
            idt = []
            for split in graph_splits:
                s = split[0]  # node indexing starts from 1
                e = split[1]
                ms = (s <= edge_s) & (edge_s <= e)
                mt = (s <= edge_t) & (edge_t <= e)
                rm = ms & mt
                edges_split = edges[rm]
                # print(len(edges_split))
                trans_prob_split = trans_prob[rm]
                adj_cover_split = adj_cover[rm]
                edges_split_s = edges_split[:, 0]
                edges_split_t = edges_split[:, 1]
                non_loop_split = edges_split_t != edges_split_s
                gst_split = gst[rm]
                vst_split = vst[rm]
                vx_split = vst_split.sum(dim=1)
                dH0 = self.get_dH(trans_prob_split, gst_split, vst_split, vx_split)
                dH = dH0
                op = (dH > 0)
                if not torch.any(op):
                    continue

                # if fs_:
                #     op = (dH > torch.median(dH[op]))
                # elif fs:
                #     op = (dH > torch.median(dH[op])) & (trans_prob_split > q)
                # else:
                #     op = (trans_prob_split > q)
                # print(q, torch.median(dH))
                # op = (trans_prob_split > q)
                # if fs:
                #     op = torch.logical_or((dH > torch.median(dH[op])), (trans_prob_split > q))
                # op = torch.logical_and(op, (edges_split_t != edges_split_s))
                # if not fs:
                #     op = torch.logical_and(op, adj_cover_split > 0)
                op = (dH > torch.median(dH[op]))
                if fs and high_dH:
                    op = (dH > torch.median(dH[op]))
                    # op = (dH > torch.median(dH[op]))
                    # op = (dH > torch.median(dH[~op]))
                    # op = dH > 0
                    # op = torch.logical_and(op, (trans_prob_split > q))
                op = torch.logical_and(op, non_loop_split)
                op = torch.logical_and(op, adj_cover_split > 0)
                merge = op
                same = (vst_split[merge, 0] == vst_split[merge, 1]).any()
                if same.any():
                    hash_x = edges_split_s * 10007 % 1009
                    hash_t = edges_split_t * 10007 % 1009
                    merge = torch.logical_and(merge,
                                              torch.logical_or((vst_split[:, 0] < vst_split[:, 1]),
                                                               torch.logical_and((vst_split[:, 0] == vst_split[:, 1]),
                                                                                 hash_x < hash_t)))
                else:
                    merge = torch.logical_and(merge, vst_split[:, 0] < vst_split[:, 1])

                id0 = edges_split_s[merge]
                id1 = edges_split_t[merge]
                dH = dH[merge]
                _, dH_amax = silearn.scatter_max(dH, id0)
                dH_amax = dH_amax[dH_amax < dH.shape[0]]
                # _, idx = torch.sort(dH[dH_amax], descending=True)
                # # idx = torch.randperm(dH_amax.shape[0])
                # dH_amax = dH_amax[idx[:max_operate_cnt]]  # 选择当前dH最大的进行作为待处理的点
                ids.extend(id0[dH_amax])
                idt.extend(id1[dH_amax])
            ids = torch.tensor(ids)
            idt = torch.tensor(idt)
            # if len(ids) == 0:
            #     if current_num_vertices < n:
            #         if fs_:
            #             print(current_num_vertices)
            #             print(n, q)
            #             break
            #         if fs:
            #             print(current_num_vertices)
            #             print(n, q)
            #             fs_ = True
            #             continue
            #         n = n_
            #         fs = True
            #         continue
            #     n = int(n * 1.2)
            #     if fs:
            #         q = q * 0.25  # max(0.25, 0.002 / q)
            #     continue
            # if len(ids) == 0:
            #     if current_num_vertices < n:
            #         if fs:
            #             print(current_num_vertices)
            #             print(n, q)
            #             # fs_ = True
            #             break
            #         n = max(n_, int(n * 0.05))
            #         fs = True
            #         print("23232323232323")
            #         continue
            #     n = int(n * 1.2)
            #     if fs:
            #         q = q * 0.25  # max(0.25, 0.002 / q)
            #     continue
            if len(ids) == 0:
                if current_num_vertices < n:
                    break
                n = int(n * 1.5)
                q = q * max(0.4, q_t / q)
                continue
            print("edges:", len(edges))
            print("operate_cnt:", len(ids))
            trans = torch.arange(edges.max() + 1, device=self.enc.graph.device)
            trans[ids] = trans[idt]
            lg_merge = math.log2(len(ids) + 2)
            for i in range(int(lg_merge)):
                trans[ids] = trans[trans[ids]]

            trans = torch.unique(trans, return_inverse=True)[1]
            transs.append(trans)
            edges = trans[edges]
            weights = [trans_prob, adj_cover.int()] + self.edge_descriptors
            ret = OperatorPropagation.sum_up_multi_edge(edges, *weights)
            edges = ret[0]
            trans_prob = ret[1]
            adj_cover = ret[2]
            self.edge_descriptors = list(ret[3:])
            self.adj_cover = adj_cover
            trans = None
            for i in reversed(range(len(transs))):
                if trans is None:
                    trans = transs[i]
                else:
                    trans = trans[transs[i]]
                if i == transs_start:
                    edges, trans_prob = self.on_operated(edges, trans_prob, trans)
            self.enc.node_id = trans
            self.cur_data = [edges, trans_prob, adj_cover]
            print(edges.max() + 1, current_num_vertices)
            if current_num_vertices < n:
                print(n, q)
                # print()
                # print(torch.median(trans_prob))
                # print(torch.median(trans_prob[non_loop]))
                # print(torch.median(trans_prob[trans_prob > torch.median(trans_prob)]))
                # print(torch.median(trans_prob[trans_prob < torch.median(trans_prob)]))
                q = torch.median(trans_prob[trans_prob > torch.median(trans_prob)])
                q_t = torch.median(trans_prob[trans_prob < torch.median(trans_prob)])
                if fs:
                    break
                if (edges.max() + 1) < int(n / 24):
                    break
                fs = re_his
                n = int(n / 48)
                if not re_his:
                    break




    def get_dH(self, trans_prob, gst, vst, vx):
        if self.objective == "SE":
            vin = vst - gst
            dH1 = (vin[:, 0]) * torch.log2(vst[:, 0]) + (vin[:, 1]) * torch.log2(vst[:, 1]) - (
                    vin[:, 0] + vin[:, 1]) * torch.log2(vx)
            dH2 = 2 * trans_prob * ((self._log2m) - torch.log2(vx))
            dH0 = dH1 + dH2
        elif self.objective == "SE_M":
            vin = vst - gst
            dH1 = (vin[:, 0]) * torch.log2(vst[:, 0]) + (vin[:, 1]) * torch.log2(vst[:, 1]) - (
                    vin[:, 0] + vin[:, 1]) * torch.log2(vx) - 2 * trans_prob * torch.log2(vx)
            # dH2 =  ((self._log2m) )
            dH0 = - trans_prob / dH1
        elif self.objective == "Modu":
            dH0 = - trans_prob + vst[:, 0] * vst[:, 1] / 2 ** self._log2m
            dH0 = - dH0

        elif self.objective == "Modu_gamma":
            dH0 = trans_prob / vst[:, 0] / vst[:, 1]

        elif self.objective == "NCut":
            dH0 = gst[:, 0] / vst[:, 0] + gst[:, 1] / vst[:, 1] - (gst.sum(dim=1) - 2 * trans_prob) / vx
        elif self.objective == "mapeq":
            ## todo : add G_ij
            all_prob = gst[:, 1].sum(keepdim=True) * 2 + vst[:, 1].sum(keepdim=True)
            inner_prob = (gst[:, 1] + vst[:, 1]).sum(keepdim=True)

            # Hq = gst[:, 1] / all_prob * torch.log2(gst[:, 1] / gst[:, 1].sum(keepdim = True))
            # Hp = gst[:, 1] / all_prob * torch.log2(gst[:, 1] / inner_prob) \
            #      + vst[:, 1] / all_prob * torch.log2(vst[:, 1] / inner_prob)

            # H0 = Hq.sum() + Hp.sum()
            # all_prob = all_prob - trans_prob * 2
            # inner_prob = inner_prob - trans_prob

            # Hq = gst[:, 1] / all_prob * torch.log2(gst[:, 1] / gst[:, 1].sum(keepdim = True))
            dH0 = 0

        else:
            raise NotImplementedError()
        return dH0

    def get_dH_v_eq_g(self, trans_prob, vst):
        if self.objective == "SE":
            return trans_prob * (self._log2m - torch.log2(vst[:, 0] + vst[:, 1]))
        elif self.objective == "SE_M":
            return trans_prob / torch.log2(vst[:, 0] + vst[:, 1])
        elif self.objective == "Modu":
            return trans_prob - vst[:, 0] * vst[:, 1] / 2 ** self._log2m

        elif self.objective == "Modu_gamma":
            return trans_prob / vst[:, 0] / vst[:, 1]

        elif self.objective == "NCut":
            vx = vst.sum(dim=1)
            return 2 - (vx - 2 * trans_prob) / vx
        else:
            raise NotImplementedError()

    def iterative_merge(self, verbose=False, min_com=1,
                        max_iteration=30, tau=0.1, sample_ratio=0.5, p=0.5, m_scale=-1):
        prob_e = torch.ones(self.enc.graph.num_edges, device=self.enc.graph.device)
        edges, _ = self.enc.graph.edges
        for i in range(max_iteration):
            rand = torch.rand(self.enc.graph.num_edges, device=self.enc.graph.device) * prob_e
            bound = torch.msort(rand)[int((prob_e.shape[0] - 1) * sample_ratio)]

            cover_adj = rand >= bound
            self.perform(adj_cover=cover_adj, min_com=min_com, p=p, m_scale=m_scale)
            self.perform(min_com=min_com, re_compute=False, p=p, m_scale=m_scale)
            c = torch.logical_not(cover_adj)
            operated = self.enc.node_id[edges[:, 0]] == self.enc.node_id[edges[:, 1]]
            # prob_e[torch.logical_and(cover_adj,torch.logical_not(operated))] *= 1 - tau
            prob_e[torch.logical_and(c, operated)] /= 1 - tau
            print(prob_e)

            print(self.enc.structural_entropy(reduction="sum"))

    # def swap(self, com_ids, k , ):
    #     pass

    # By Hujin

    # def iterative_merge_c(self, verbose = False, min_com = 1,
    #                     max_iteration = 30, tau = 0.1, sample_ratio = 0.5, p = 0.5, m_scale = 0):
    #     prob_e = torch.ones(self.enc.graph.num_edges, device=self.enc.graph.device)
    #     edges, _ = self.enc.graph.edges
    #     for i in range(max_iteration):
    #
    #         rand = torch.rand(self.enc.graph.num_edges, device=self.enc.graph.device) * prob_e
    #         bound = torch.msort(rand)[int((prob_e.shape[0] - 1) * sample_ratio)]
    #
    #         cover_adj = rand >= bound
    #         self.process_c(adj_cover = cover_adj)
    #         self.process_fast(min_com = min_com, re_compute=False, p = p, m_scale=m_scale)
    #         c = torch.logical_not(cover_adj)
    #         operated = self.enc.node_id[edges[:, 0]] == self.enc.node_id[edges[:, 1]]
    #         prob_e[torch.logical_and(c,torch.logical_not(operated))] *= 1 - tau
    #         prob_e[torch.logical_and(c,operated)] /= 1 - tau
    #         print(prob_e)
    #
    #         print(self.enc.structural_entropy(reduction="sum"))

    # def exchange_nodes(self, vin, vst, vx, trans_prob):
    #     dH1 = (vin[:, 0]) * torch.log2(vst[:, 0])
    #     dH2 = 2 * trans_prob * ((self._log2m) - torch.log2(vx))
    #
    #     dH3 = (vin[:, 0]) * torch.log2(vst[:, 0])
    #             vin[:, 0] + vin[:, 1]) * torch.log2(vx)
    #     dH4 = 2 * trans_prob * ((self._log2m) - torch.log2(vx))
    #
