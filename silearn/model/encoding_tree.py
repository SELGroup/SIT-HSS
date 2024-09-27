import networkx

import silearn
from silearn.graph import Graph
from silearn import *


class GraphEncoding:
    r"""
    The base Graph Encoding model
    """
    '''
        1. GraphEncoding:图编码的基类,定义了一些公共方法如stationary_dist、entropy_rate等。
        2. OneDim:一维图编码模型,继承自GraphEncoding,uncertainty方法计算一维模型的不确定性。
        3. Partitioning:划分图编码模型,node_id属性存储每个节点的模块标识,uncertainty方法计算模型的不确定性,structural_entropy计算结构熵。可以与hyper_partitioning组合。
        4. SoftPartitioning:软划分图编码模型,prob_clus属性存储每个节点属于每个模块的概率,uncertainty方法计算模型的不确定性。
        5. EncodingTree:树形图编码模型,parent_id属性存储每个层次的父节点标识,uncertainty方法沿树向上迭代计算不确定性,structural_entropy方法可以在不同层次上计算结构熵。as_partition可以获取任意层次的划分。
        主要功能与属性:
        - stationary_dist:获取图的平稳态分布
        - entropy_rate:计算图的熵率
        - uncertainty:计算模型的不确定性
        - structural_entropy:计算模型的结构熵
        - node_id:模块标识(Partitioning)
        - prob_clus:节点属于每个模块的概率(SoftPartitioning)
        - parent_id:树形模型每个层次的父节点标识(EncodingTree)
        - as_partition:获取EncodingTree任意层次的划分
        这段代码实现了图编码模型的框架及几种典型模型,通过属性如node_id、prob_clus和parent_id以及相关方法如uncertainty和structural_entropy定义每个模型的行为
    '''

    def __init__(self, g: Graph):
        self.graph = g

    def uncertainty(self, es, et, p):
        raise NotImplementedError("Not Implemented")

    def positioning_entropy(self):
        dist = self.graph.stationary_dist
        return silearn.entropy(dist, dist)

    def entropy_rate(self, reduction="vertex", norm=False):
        edges, p = self.graph.edges
        es, et = edges[:, 0], edges[:, 1]
        nw = self.graph.vertex_weight_es[es]
        entropy = silearn.entropy(p, p / nw)

        if norm:
            dist = self.graph.stationary_dist[es]
            entropy = entropy / self.positioning_entropy()

        if reduction == "none":
            return entropy
        elif reduction == "vertex":
            return silearn.scatter_sum(entropy, et)
        elif reduction == "sum":
            return entropy.sum()
        else:
            return entropy

    def structural_entropy(self, reduction="vertex", norm=False):
        edges, p = self.graph.edges
        es, et = edges[:, 0], edges[:, 1]
        # dist = self.graph.stationary_dist[es]
        dist = self.graph.stationary_dist[es]
        # tot = w.sum()
        entropy = p * self.uncertainty(es, et, p)

        if norm:
            entropy = entropy / silearn.entropy(dist, dist)
        if reduction == "none":
            return entropy
        elif reduction == "vertex":
            return silearn.scatter_sum(entropy, et)
        elif reduction == "sum":
            return entropy.sum()
        else:
            return entropy

    def to_networkx(self, create_using=networkx.DiGraph()):
        raise NotImplementedError()


class OneDim(GraphEncoding):

    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[es]
        return uncertainty(v1)


class Partitioning(GraphEncoding):
    node_id = None  # :torch.LongTensor

    def __init__(self, g: Graph, init_parition):
        super().__init__(g)
        self.node_id = init_parition

    def uncertainty(self, es, et, p):
        v1e = self.graph.stationary_dist[et]
        id_et = self.node_id[et]
        id_es = self.node_id[es]
        v2 = scatter_sum(self.graph.stationary_dist, self.node_id)
        v2e = v2[id_es]
        flag = id_es != id_et
        # print(v1e, v2, flag)
        return uncertainty(v1e / v2e) + flag * uncertainty(v2e / v2.sum())

    def structural_entropy(self, reduction="vertex", norm=False):
        entropy = super(Partitioning, self).structural_entropy(reduction, norm)
        if reduction == "module":
            et = self.graph.edges[2]
            return scatter_sum(entropy, self.node_id[et])
        return entropy

    def compound(self, hyper_partitioning):
        self.node_id = hyper_partitioning[self.node_id]



    def to_networkx(self,
                    create_using=networkx.DiGraph(),
                    label_name="partition"):
        nx_graph = self.graph.to_networkx(create_using=create_using)
        label_np = silearn.convert_backend(self.node_id, "numpy")
        for i in range(label_np.shape[0]):
            nx_graph._node[i][label_name] = label_np[i]
        return nx_graph


class SoftPartitioning(GraphEncoding):
    prob_clus = None  # :torch.Tensor V x C

    def __init__(self, g: Graph, init_parition):
        super().__init__(g)
        self.node_id = init_parition

    def uncertainty(self, es, et, p):
        vol = self.graph.vertex_weight_et
        vol_2 = (vol.unsqueeze(-1) * self.prob_clus).sum(dim = 0)
        # prob_t = self.prob_clus[et]
        #
        # return p * uncertainty(vol[et]) - (p * self.prob_clus[et]) uncertainty(vol_2[es])

    def structural_entropy2(self, adj):
        vol = adj.sum(dim = 0)
        vol_2 = (vol.unsqueeze(-1)) * self.prob_clus.sum(dim = 0)
        return adj * (uncertainty(vol))




class EncodingTree(GraphEncoding):
    parent_id: []

    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[et]
        cur_ids = es
        cur_idt = et
        ret = 0
        for i in range(len(self.parent_id)):
            id_es = self.parent_id[i][cur_ids]
            id_et = self.parent_id[i][cur_idt]
            vp = scatter_sum(
                v1,
                id_et)[id_et] if i != len(self.parent_id) - 1 else v1.sum()
            if i == 0:
                ret += uncertainty(v1 / vp)
            else:
                flag = cur_ids != cur_idt
                ret += flag * uncertainty(v1 / vp)
            v1 = vp
            cur_ids, cur_idt = id_es, id_et
        return ret

    def structural_entropy(self, reduction="vertex", norm=False):
        entropy = super(EncodingTree, self).structural_entropy(reduction, norm)
        if reduction.startswith("level"):
            level = int(reduction[5:])
            level = min(-len(self.parent_id), level)
            level = max(len(self.parent_id) - 1, level)
            et = self.graph.edges[0][:, 1]
            return scatter_sum(entropy, self.parent_id[level][et])
        return entropy

    """
    2-Dim Enc Tree: Level - -1, 0
    3-Dim Enc Tree: Level - -2, -1, 0, 1
    """

    def as_partition(self, level=-1):
        height = len(self.parent_id)
        assert -height <= level < height
        if level < 0:
            level = height + level
        if level != 0:
            trans = self.parent_id[level]
            for i in reversed(range(level)):
                trans = trans[self.parent_id[i]]
            return trans
        else:
            return self.parent_id
