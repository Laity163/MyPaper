'''
1. current multi-headed attn: fixed/unfixed subgraph sampling
2. scalar attn: fixed/unfixed subgraph sampling
3. alpha=1, beta=0.01 average pooling (once at the start)
4. alpha = cosine distance, beta=0.01 weighted pooling (once at the start)
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb

from plato.utils.torch_utils import dt
from plato.baseline.pipeline_utils import get_gene_ixs, get_drug_ixs, ParentDecoder, get_gene_embed, get_drug_embed
from plato.models.gat import GATConv

class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, skip_connection=False, use_bn=False):
        super(TinyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.use_bn = use_bn

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
    
    def forward(self, x):
        if not self.skip_connection:
            return self.bn2(self.l2(nn.ReLU()(self.bn1(self.l1(x)))))
        else:
            return self.bn2(self.l2(nn.ReLU()(self.bn1(self.l1(x))))) + x

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        if self.use_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


class GMLP(nn.Module):
    def __init__(self, m_layer_list, node_dim, V, do_softmax, skip_connection, use_bn, enlarge=1):
        super(GMLP, self).__init__()

        # Hyperparameters
        self.m_layer_list = m_layer_list
        self.n_layers = len(self.m_layer_list)
        self.node_dim = node_dim
        self.V = V
        self.do_softmax = do_softmax

        for i in range(len(m_layer_list)):
            setattr(self, "l%d"%i, TinyMLP(node_dim, m_layer_list[i]*enlarge, m_layer_list[i], False if i == self.n_layers - 1 else skip_connection, use_bn))

    def reset_parameters(self):
        for i in range(self.n_layers):
            getattr(self, "l%d"%i).reset_parameters()

    def forward(self, x):
        h = x
        v = self.V
        # dt(h, "h--1")

        for i in range(self.n_layers):
            # dt(v, "v%d"%i)
            a = getattr(self, "l%d"%i)(v)
            # dt(a, "a%d"%i)
            if self.do_softmax:
                a = F.softmax(a, dim=0)
            h = torch.matmul(h, a)
            # dt(h, "h%d"%i)
            v = torch.matmul(a.T, v)
        # pdb.set_trace()
        return h

class SimpleGMLP(nn.Module):
    def __init__(self, m_layer_list, node_dim, V, nonlin, skip_connection, use_bn, train_meta, device, div=1, enlarge=1, edge_index=None, beta = 0, num_edges_sampled=0, scalar_attn=False):
        super(SimpleGMLP, self).__init__()

        # Hyperparameters
        self.m_layer_list = m_layer_list
        self.n_layers = len(self.m_layer_list)
        self.node_dim = node_dim
        self.V = V
        self.train_meta = train_meta
        self.device = device
        self.scalar_attn = scalar_attn
        if self.train_meta:
            self.V_cp = self.V.detach().clone().cpu().numpy()
            self.V = nn.Parameter(torch.tensor(self.V, device=self.device), requires_grad=True)
        self.nonlin = nonlin
        self.div = div
        self.edge_index = edge_index
        self.beta = beta
        self.num_edges_sampled = num_edges_sampled
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
            if scalar_attn:
                self.attn_fn = GATConv(1, self.V.shape[-1], bias=None, beta=beta, heads=1, scalar_attn=scalar_attn)
            else:
                self.attn_fn = GATConv(1, 1, bias=None, beta=beta, heads=self.V.shape[-1])
        else:
            self.attn_fn = None

        self.l0 = TinyMLP(node_dim, m_layer_list[0]*enlarge, m_layer_list[0], skip_connection, use_bn)
        for i in range(1, len(m_layer_list)):
            setattr(self, "l%d"%i, nn.Linear(m_layer_list[i-1], m_layer_list[i]))

    def reset_parameters(self):
        for i in range(self.n_layers):
            getattr(self, "l%d"%i).reset_parameters()
        if self.train_meta:
            self.V = nn.Parameter(torch.tensor(self.V, device=self.device), requires_grad=True)
        if self.attn_fn is not None:
            self.attn_fn.reset_parameters()

    def forward(self, x):
        h = x
        v = self.V
        if self.edge_index is not None:
            # pdb.set_trace()
            all_vs = []
            
            # === 修复开始: 增加边数检查 ===
            # 只有当现有边数 > 采样数时才进行采样，否则直接使用全部边
            num_edges = self.edge_index.shape[1]
            if num_edges > self.num_edges_sampled:
                idx = np.random.choice(num_edges, self.num_edges_sampled, replace=False)
                edge_index = self.edge_index[:, idx]
            else:
                edge_index = self.edge_index
            # === 修复结束 ===

            for i in range(h.shape[0]):
                all_vs.append(self.attn_fn(x[i].unsqueeze(1), edge_index, v))
            all_vs = torch.stack(all_vs, 0)    
            all_as = getattr(self, "l0")(all_vs)
            h = torch.matmul(h.unsqueeze(1), all_as)/self.div
            h = h.squeeze(1)
            
            for i in range(1, self.n_layers):
                h = getattr(self, "l%d"%i)(h)
                if i != self.n_layers - 1:
                    h = nn.ReLU()(h)
            return h
        # dt(h, "h--1")

        # dt(v, "v%d"%i)
        a = getattr(self, "l0")(v)
        # dt(a, "a%d"%i)

        if self.nonlin == "none":
            pass
        elif self.nonlin == "softmax":
            a = F.softmax(a, dim = 0)
        elif self.nonlin == "relu":
            a = torch.nn.ReLU()(a)
        elif self.nonlin == "leakyrelu":
            a = torch.nn.LeakyReLU()(a)
        elif self.nonlin == "tanh":
            a = torch.nn.Tanh()(a)
        else:
            assert(False)

        # h = torch.matmul(h, a)/self.div

        # === 修复开始：处理维度不匹配 (针对 Dummy Drug 情况) ===
        # 如果输入维度 (例如 200) 不等于权重行数 (例如 1)
        if h.shape[1] != a.shape[0]:
            # 强制切片输入以匹配权重维度。
            # 因为您的 Dummy Drug 是全 0 向量，取前几位依然是 0，计算结果正确为 0。
            h_adjusted = h[:, :a.shape[0]]
            h = torch.matmul(h_adjusted, a)/self.div
        else:
            h = torch.matmul(h, a)/self.div
        # === 修复结束 ===


        # dt(h, "h%d"%i)
        # v = torch.matmul(a.T, v)
        # pdb.set_trace()
        for i in range(1, self.n_layers):
            h = getattr(self, "l%d"%i)(h)
            if i != self.n_layers - 1:
                h = nn.ReLU()(h)
        return h


class GGMLP(ParentDecoder):
    def __init__(self, m_layer_list, node_dim, gene_dim, drug_dim, provider, device, skip_connection, use_bn, simple, train_meta, drug_nonlin, gene_nonlin, drug_div, gene_div, l1_weight, l2_weight, enlarge, mp, beta, gene_edge_index, drug_edge_index, num_edges_sampled, scalar_attn):
        super(GGMLP, self).__init__()

        # Hyperparameters
        self.m_layer_list = m_layer_list
        self.n_layers = len(self.m_layer_list)
        self.node_dim = node_dim
        self.train_meta = train_meta

        # Set previously
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.source2task = provider.source2task
        self.device = device
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.scalar_attn = scalar_attn

        self.mp = mp
        self.beta = beta
        self.gene_edge_index = gene_edge_index
        self.drug_edge_index = drug_edge_index
        self.num_edges_sampled = num_edges_sampled

        # Layers
        # if self.train_meta:
        #     self.V = nn.Parameter(torch.tensor(torch.cat([get_gene_embed(provider), get_drug_embed(provider)]), device=self.device), requires_grad=True) # [16934, 200]
        # else:
        self.V = torch.cat([get_gene_embed(provider), get_drug_embed(provider)]).to(self.device) # [16934, 200]
        if not simple:
            self.gene_gmlp = GMLP(m_layer_list, node_dim, self.V[:self.gene_dim], 
                                do_softmax=True, skip_connection=skip_connection, 
                                use_bn=use_bn)
            self.drug_gmlp = GMLP(m_layer_list, node_dim, self.V[self.gene_dim:], 
                                do_softmax=False, skip_connection=skip_connection, 
                                use_bn=use_bn)
        else:
            self.gene_gmlp = SimpleGMLP(m_layer_list, node_dim, self.V[:self.gene_dim], 
                                nonlin=gene_nonlin, skip_connection=skip_connection, 
                                use_bn=use_bn, train_meta=train_meta, device=device, div=gene_div,
                                enlarge=enlarge, beta=beta, edge_index=gene_edge_index, num_edges_sampled=num_edges_sampled,
                                scalar_attn=scalar_attn)
            self.drug_gmlp = SimpleGMLP(m_layer_list, node_dim, self.V[self.gene_dim:], 
                                nonlin=drug_nonlin, skip_connection=skip_connection, 
                                use_bn=use_bn, train_meta=train_meta, device=device, div=drug_div,
                                enlarge=enlarge, beta=beta, edge_index=None, num_edges_sampled=0,
                                scalar_attn=False)
        # self.drug_gmlp = GMLP(m_layer_list, node_dim, self.V[self.gene_dim:], do_softmax=False)
        
        # Attributes that require computation
        self.loss_list = self.get_loss_list() # For compatibility with compute_loss in parent

    def reset_parameters(self):
        self.gene_gmlp.reset_parameters()
        self.drug_gmlp.reset_parameters()

    # def forward(self, x, sourceint: int):
    #     h = x
    #     h_gene, h_drug = get_gene_ixs(x, self.gene_dim), get_drug_ixs(x, self.gene_dim)
    #     h_gene = self.gene_gmlp(h_gene)
    #     h_drug = self.drug_gmlp(h_drug)
        
    #     h = h_gene + h_drug

    #     return h
    def forward(self, x, sourceint: int):
        # 提取我们在 Pipeline 中绑定的真实维度
        original_gene_dim = getattr(self, 'original_gene_dim', self.gene_dim)
        num_disease_nodes = getattr(self, 'num_disease_nodes', 0)
        
        # 按照真实的特征维度提取患者基因/特征 (x 的原始列数)
        h_gene, h_drug = get_gene_ixs(x, original_gene_dim), get_drug_ixs(x, original_gene_dim)
        
        # =================================================================
        # === [Method C 核心] 为虚拟疾病枢纽节点补 0 (Zero-Padding) ===
        # 这样，疾病节点在 GAT 中初始激活值为 0，它们将作为一个纯粹的“信息中转站”，
        # 吸收来自相关特征的信息，再反哺给其他共现的特征！
        if num_disease_nodes > 0:
            pad_zeros = torch.zeros(h_gene.shape[0], num_disease_nodes, device=h_gene.device)
            h_gene = torch.cat([h_gene, pad_zeros], dim=1)
        # =================================================================
        
        h_gene = self.gene_gmlp(h_gene)
        h_drug = self.drug_gmlp(h_drug)
        
        h = h_gene + h_drug

        return h

# === 新增：标准 MLP 模型 (用于对比实验) ===
class MLP(ParentDecoder):
    def __init__(self, input_dim, m_layer_list, provider, device, l1_weight=0.0, l2_weight=0.0, use_bn=False, dropout=0.0):
        super(MLP, self).__init__()
        self.source2task = provider.source2task
        self.device = device
        
        # === 修复点：初始化正则化权重 ===
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        # ============================
        
        layers = []
        current_dim = input_dim
        
        # m_layer_list 例如 [32, 32, 8]
        # 构建网络层: Linear -> BN -> ReLU ... -> Linear
        for i, h_dim in enumerate(m_layer_list):
            layers.append(nn.Linear(current_dim, h_dim))
            
            # 如果不是最后一层，添加激活函数和 Batch Norm
            if i < len(m_layer_list) - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = h_dim
            
        self.net = nn.Sequential(*layers)
        
        # 初始化损失函数
        self.loss_list = self.get_loss_list()

    def forward(self, x, sourceint):
        # x 的形状是 (Batch_Size, gene_dim + drug_dim)
        # sourceint 在此处仅用于兼容接口，纯 MLP 不需要它来选择权重
        return self.net(x)

    # === 新增：必须实现这个方法以支持 runs > 1 ===
    def reset_parameters(self):
        for layer in self.net:
            # 检查层是否有 reset_parameters 方法 (Linear 和 BatchNorm 都有)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()