import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import (JumpingKnowledge, MessagePassing,
                                global_add_pool, global_max_pool,
                                global_mean_pool)
class FourierEncoder(torch.nn.Module):
    """Node encoder using Fourier features.
    """
    def __init__(self, level, include_self=True):
        super(FourierEncoder, self).__init__()
        self.level = level
        self.include_self = include_self

    def multiscale(self, x, scales):
        return torch.hstack([x / i for i in scales])

    def forward(self, x):
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(-self.level / 2, self.level / 2, device=device, dtype=dtype)
        lifted_feature = torch.cat((torch.sin(self.multiscale(x, scales)), torch.cos(self.multiscale(x, scales))), 1)
        return lifted_feature

class LinearEncoder(nn.Module):
    """Node encoder using linear layers
    """
    def __init__(self, input_dim, hidden_dim, layer = 1, BN = False):
        super(LinearEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layer):
            self.layer.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer.append(nn.ReLU())
        if BN:
            self.layer.append(nn.BatchNorm1d(hidden_dim))
        

    def forward(self, x):
        x = self.layer(x)
        return x
    
class PreNormException(Exception):
    pass

class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super(PreNormLayer, self).__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException
        if self.shift is not None:
            input_ = input_ + self.shift
        if self.scale is not None:
            input_ = input_ * self.scale
        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units
        delta = sample_avg - self.avg
        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
    

import torch
from torch_geometric.nn import MessagePassing

class TripartiteGCNConv(MessagePassing):
    def __init__(self, emb_size, use_residual=True):
        super().__init__(aggr='add')
        self.emb_size = emb_size

        self.combined_linear = torch.nn.Linear(emb_size, emb_size)

        self.feature_module_src1 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size))
        
        self.feature_module_src2 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size))

        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False))

        self.feature_module_target = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size))

        self.cross_att = torch.nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)

        self.fusion_gate = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.Sigmoid())

        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(emb_size), 
            torch.nn.GELU(),
            torch.nn.Linear(emb_size, emb_size))

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(emb_size), 
            # torch.nn.Dropout(0.3)
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(3*emb_size, emb_size),
            torch.nn.GELU(),
            # torch.nn.Dropout(0.3),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LayerNorm(emb_size))

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.use_residual = use_residual

    def forward(self, src1_features, src2_features, edge_indices, edge_features, target_features):
        if self.use_residual:
            residual = target_features

        out_src1 = self.propagate(
            edge_indices[0],
            node_features=(src1_features, target_features),
            edge_features=edge_features[0],
            mode='src1'
        )
        
        out_src2 = self.propagate(
            edge_indices[1],
            node_features=(src2_features, target_features),
            edge_features=edge_features[1],
            mode='src2'
        )

        combined, _ = self.cross_att(
            query=out_src1.unsqueeze(1),
            key=out_src2.unsqueeze(1),
            value=out_src2.unsqueeze(1)
        )
        combined = combined.squeeze(1)

        gate = self.fusion_gate(torch.cat([out_src1, out_src2], dim=-1))
        fused_output = gate * out_src1 + (1 - gate) * out_src2

        output = self.output_module(
            torch.cat([
                self.post_conv_module(fused_output),
                combined,
                target_features
            ], dim=-1)
        )
        if self.use_residual:
            return output + residual
        else:
            return output

    def message(self, node_features_i, node_features_j, edge_features, mode):
        src_module = self.feature_module_src1 if mode == 'src1' else self.feature_module_src2

        trans_src = src_module(node_features_i)
        trans_edge = self.feature_module_edge(edge_features)
        trans_target = self.feature_module_target(node_features_j)

        combined = trans_target + trans_src + trans_edge
        combined = self.combined_linear(combined)  # 使用预定义的线性层
 
        def create_message(combined_input):
            return self.feature_module_final(combined_input)
    
        # Apply checkpoint with explicit reentrant parameter
        if self.training:
            output = checkpoint(
                create_message, 
                combined,
                use_reentrant=True  # Recommended setting
            )
        else:
            output = create_message(combined)
        return output
    
class BipartiteGCNConv(MessagePassing):
    def __init__(self, emb_size, use_residual=True):
        super().__init__(aggr='add')
        self.emb_size = emb_size

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False),
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )

        self.att_weight = torch.nn.Parameter(torch.Tensor(emb_size, 1))
        torch.nn.init.xavier_uniform_(self.att_weight)

        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(emb_size),
            torch.nn.GELU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(emb_size)
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.GELU(),
            # torch.nn.Dropout(0.3),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LayerNorm(emb_size)
        )
        self.combined_linear = torch.nn.Linear(emb_size, emb_size)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
        self.use_residual = use_residual

    def forward(self, left_features, edge_indices, edge_features, right_features):
        if self.use_residual:
            residual = right_features
        output = self.propagate(
            edge_indices, 
            size=(left_features.size(0), right_features.size(0)),
            node_features=(left_features, right_features),
            edge_features=edge_features
        )
        output = self.output_module(
            self.post_conv_module(output) + right_features
        )
        if self.use_residual:
            return output + residual
        else:
            return output

    def message(self, node_features_i, node_features_j, edge_features):
        att_i = node_features_i @ self.att_weight
        att_j = node_features_j @ self.att_weight
        attention = torch.sigmoid(att_i + att_j)

        combined = self.feature_module_left(node_features_i) + self.feature_module_edge(edge_features) + self.feature_module_right(node_features_j)
        combined = self.combined_linear(combined)

        def create_message(combined_input):
            return self.feature_module_final(combined_input)
    
        # Apply checkpoint with explicit reentrant parameter
        if self.training:
            output = checkpoint(
                create_message, 
                combined,
                use_reentrant=True  # Recommended setting
            )
        else:
            output = create_message(combined)
        return output * attention
    
# class GNNLayer(torch.nn.Module):
#     def __init__(self, emb_size):
#         super(GNNLayer, self).__init__()
#         self.emb_size = emb_size
#         self.learn_norm = True
#         self.track_norm = False
#         self.feature_module_left = nn.Linear(emb_size, emb_size, bias=True)
#         self.feature_module_right = nn.Linear(emb_size, emb_size, bias=True)
#         self.
        