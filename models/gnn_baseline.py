import numpy as np
import torch
import math
from dataset import GraphMILP
import torch.nn as nn
from models.layers import FourierEncoder, LinearEncoder, PreNormLayer, BipartiteGCNConv, TripartiteGCNConv
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
class MILPGCNNonRes(nn.Module):
    def __init__(
                self, 
                input_dim_x_Cvars,
                input_dim_x_Ivars, 
                input_dim_x_cons, 
                hidden_dim, 
                num_layers,
                max_integer_num:int, 
                learn_norm=True,
                only_discrete = False,
                 ):
        super(MILPGCNNonRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        time_embed_dim = hidden_dim//2
        self.only_discrete = only_discrete
        
        self.x_Cvars_linear = nn.Sequential(
            linear(input_dim_x_Cvars, hidden_dim),
            nn.ReLU(),
            linear(hidden_dim, hidden_dim),
        )
        
        self.x_Ivars_linear = nn.Sequential(
            linear(input_dim_x_Ivars, hidden_dim),
            nn.ReLU(),
            linear(hidden_dim, hidden_dim),
        )
        
        self.x_cons_linear = nn.Sequential(
            linear(input_dim_x_cons, hidden_dim),
            nn.ReLU(),
            linear(hidden_dim, hidden_dim),
        )
        # Convolution layers
        self.conv_vars2cons = nn.ModuleList([TripartiteGCNConv(hidden_dim, False) for _ in range(num_layers)]) \
            if not self.only_discrete else \
            nn.ModuleList([BipartiteGCNConv(hidden_dim, False) for _ in range(num_layers)])
        self.conv_cons2Ivars = nn.ModuleList([BipartiteGCNConv(hidden_dim, False) for _ in range(num_layers)])

        if not self.only_discrete:
            self.conv_cons2Cvars = nn.ModuleList([BipartiteGCNConv(hidden_dim, False) for _ in range(num_layers)])

        
        self.time_embed = nn.Sequential(
            linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                linear(
                    time_embed_dim,
                    hidden_dim,
                ),
            ) for _ in range(num_layers)
        ])
        self.init_params()
        self.per_layer_out_cons = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
        
        self.per_layer_out_Ivars = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
        
        self.per_layer_out_Cvars = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])

        self.Iout_layer = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, max_integer_num, kernel_size=1, bias=True)
        )
        
        self.Cout_layer = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        )
        
        self.num_integer_vars = max_integer_num
        
    def _forward_multimodal(
        self, 
        x_Cvars: torch.Tensor, 
        x_Ivars: torch.Tensor,
        x_cons: torch.Tensor,
        C2cons_edge_attr: torch.Tensor, 
        I2cons_edge_attr: torch.Tensor,
        C2cons_edge_index: torch.Tensor,
        I2cons_edge_index: torch.Tensor,
        timesteps
        ):
        x_Cvars_shape = x_Cvars.shape
        x_Ivars_shape = x_Ivars.shape
        x_cons_shape = x_cons.shape
        
        x_Cvars = self.x_Cvars_linear(x_Cvars)
        x_Ivars = self.x_Ivars_linear(x_Ivars)
        x_cons = self.x_cons_linear(x_cons)
        
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        broadcast_flag = True if time_emb.shape[0] == 1 else False
        broadcast_dim_Ivars = x_Ivars_shape[0]
        broadcast_dim_Cvars = x_Cvars_shape[0]
        broadcast_dim = broadcast_dim_Ivars + broadcast_dim_Cvars 
        
        inverse_C2cons_edge_index = C2cons_edge_index[[1, 0]]
        inverse_I2cons_edge_index = I2cons_edge_index[[1, 0]]

        
        for conv_vars2cons,  \
            conv_cons2Cvars, conv_cons2Ivars, \
            time_layer, out_layer_cons, out_layer_Ivars, out_layer_Cvars\
                                        in zip(
                                            self.conv_vars2cons,
                                            self.conv_cons2Cvars, self.conv_cons2Ivars,
                                            self.time_embed_layers, self.per_layer_out_cons,
                                            self.per_layer_out_Ivars, self.per_layer_out_Cvars
                                            ):
            x_cons = conv_vars2cons(x_Ivars, x_Cvars, 
                                     [I2cons_edge_index, C2cons_edge_index], [I2cons_edge_attr, C2cons_edge_attr], x_cons)
            x_Ivars = conv_cons2Ivars(x_cons, inverse_I2cons_edge_index, I2cons_edge_attr, x_Ivars)
            x_Cvars = conv_cons2Cvars(x_cons, inverse_C2cons_edge_index, C2cons_edge_attr, x_Cvars)
            torch.cuda.empty_cache()
            
            time_emb_tmp = time_layer(time_emb)
            if broadcast_flag:
                time_emb_tmp = time_emb_tmp.expand(broadcast_dim, -1)
            
            x_Ivars = time_emb_tmp[:broadcast_dim_Ivars] + out_layer_Ivars(x_Ivars)
            x_Cvars = time_emb_tmp[broadcast_dim_Ivars: ] + out_layer_Cvars(x_Cvars)
            x_cons = out_layer_cons(x_cons)

        x_Ivars = x_Ivars.reshape(1, x_Ivars_shape[0], -1, x_Ivars.shape[-1]).permute(0, 3, 1, 2)
        x_Cvars = x_Cvars.reshape(1, x_Cvars_shape[0], -1, x_Cvars.shape[-1]).permute(0, 3, 1, 2)
        x_cons = x_cons.reshape(1, x_cons_shape[0], -1, x_cons.shape[-1]).permute(0, 3, 1, 2)

        x_Ivars_out = self.Iout_layer(x_Ivars).reshape(-1, x_Ivars_shape[0]).permute(1, 0)
        x_Cvars_out = self.Cout_layer(x_Cvars).reshape(-1, x_Cvars_shape[0]).permute(1, 0).reshape(-1)
        return x_Ivars_out, x_Cvars_out
    
    def _forward_discrete(
        self, 
        x_Ivars: torch.Tensor,
        x_cons: torch.Tensor,
        I2cons_edge_attr: torch.Tensor,
        I2cons_edge_index: torch.Tensor,
        timesteps
        ):
        x_Ivars_shape = x_Ivars.shape
        x_cons_shape = x_cons.shape
        
        # x_Cvars = self.x_Cvars_linear(x_Cvars)
        x_Ivars = self.x_Ivars_linear(x_Ivars)
        x_cons = self.x_cons_linear(x_cons)
        
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        broadcast_flag = True if time_emb.shape[0] == 1 else False
        broadcast_dim_Ivars = x_Ivars_shape[0]
        broadcast_dim = broadcast_dim_Ivars 
        
        inverse_I2cons_edge_index = I2cons_edge_index[[1, 0]]

        
        for conv_vars2cons,  conv_cons2Ivars, \
            time_layer, out_layer_cons, out_layer_Ivars\
                                        in zip(
                                            self.conv_vars2cons,
                                            self.conv_cons2Ivars,
                                            self.time_embed_layers, self.per_layer_out_cons,
                                            self.per_layer_out_Ivars
                                            ):
            x_Ivars_in = x_Ivars
            x_cons_in = x_cons
            x_cons = conv_vars2cons(x_Ivars, I2cons_edge_index, I2cons_edge_attr, x_cons)
            x_Ivars = conv_cons2Ivars(x_cons, inverse_I2cons_edge_index, I2cons_edge_attr, x_Ivars)
            
            time_emb_tmp = time_layer(time_emb)
            if broadcast_flag:
                time_emb_tmp = time_emb_tmp.expand(broadcast_dim, -1)
            
            x_Ivars = x_Ivars_in + time_emb_tmp[:broadcast_dim_Ivars] + out_layer_Ivars(x_Ivars)
            x_cons = x_cons_in  + out_layer_cons(x_cons)
            # torch.cuda.empty_cache()

        x_Ivars = x_Ivars.reshape(1, x_Ivars_shape[0], -1, x_Ivars.shape[-1]).permute(0, 3, 1, 2)
        x_cons = x_cons.reshape(1, x_cons_shape[0], -1, x_cons.shape[-1]).permute(0, 3, 1, 2)

        x_Ivars_out = self.Iout_layer(x_Ivars).reshape(-1, x_Ivars_shape[0]).permute(1, 0)
        return x_Ivars_out, None
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        
    def forward(
        self,
        graph: GraphMILP,
        timesteps,
    ):
        return self._forward_multimodal(
            x_Cvars=graph['Cvars'].x,
            x_Ivars=graph['Ivars'].x,
            x_cons=graph['cons'].x,
            C2cons_edge_attr=graph['Cvars', 'to', 'cons'].edge_attr,
            I2cons_edge_attr=graph['Ivars', 'to', 'cons'].edge_attr,
            C2cons_edge_index=graph['Cvars', 'to', 'cons'].edge_index,
            I2cons_edge_index=graph['Ivars', 'to', 'cons'].edge_index,
            timesteps=timesteps 
        ) if not self.only_discrete else \
            self._forward_discrete(
                x_cons=graph['cons'].x,
                x_Ivars=graph['Ivars'].x,
                I2cons_edge_attr=graph['Ivars', 'to', 'cons'].edge_attr,
                I2cons_edge_index=graph['Ivars', 'to', 'cons'].edge_index,
                timesteps=timesteps
            )