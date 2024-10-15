from typing import Tuple, List
import math
from einops import rearrange
import torch
from torch import nn
import numpy as np
torch.set_printoptions(precision=15)
class TorchQuantizer(torch.nn.Module):
    def __init__(self, 
                 bitwidth=18, 
                 int_bitwidth=8, 
                 signed=True,
                 rounding='CONVERGENT',
                 saturation='WRAP',
                 calibration=False,
                 quantize=True,
                 dtype=torch.float64):
        super(TorchQuantizer, self).__init__()
        self.bitwidth = bitwidth
        self.int_bitwidth = int_bitwidth
        self.signed = signed
        self.m = pow(2, self.bitwidth) if quantize else 1 #in calibration mode, no need to calculate m
        self.m_i = pow(2, self.int_bitwidth) if quantize else 1
        self.q = self.m / self.m_i
        self.q = float(self.q)
        self.lower_bound = -self.m/2 if self.signed else 0
        self.upper_bound = self.m/2-1 if self.signed else self.m-1
        self.rounding = rounding
        self.saturation = saturation
        self.calibration = calibration
        self.quantize = quantize
        self.max_int_bits = torch.tensor(-torch.inf)
        self.max_value = torch.tensor(-torch.inf)
        self.min_frac_bits = torch.tensor(torch.inf)
    def forward(self, x):
        if self.quantize == False:
            return x
        if self.calibration:
            x_flat = x.flatten()
            x_flat = x_flat[x_flat != 0]
            #check if x_flat is not empty
            if x_flat.nelement() > 0:
                max_int_bits = torch.max(torch.ceil(torch.log2(torch.abs(x_flat))).max())
                max_int_bits += 1 if self.signed else 0
                min_frac_bits = torch.min(torch.ceil(torch.log2(torch.abs(x_flat))).min())
                min_frac_bits += 1 if self.signed else 0
                self.max_int_bits = torch.max(max_int_bits, self.max_int_bits).int()
                self.min_frac_bits = torch.min(min_frac_bits, self.min_frac_bits).int()
            return x
        if self.rounding == 'CONVERGENT':
            if self.saturation == 'WRAP':
                qx = ((torch.round(x * self.q) - self.lower_bound) % (self.upper_bound - self.lower_bound + 1) + self.lower_bound) / self.q
            else:
                qx = torch.clamp(torch.round(x * self.q), self.lower_bound, self.upper_bound)/self.q
        else:
            if self.saturation == 'WRAP':
                qx = ((torch.trunc(x * self.q) - self.lower_bound) % (self.upper_bound - self.lower_bound + 1) + self.lower_bound) / self.q
            else:
                qx = torch.clamp(torch.trunc(x * self.q), self.lower_bound, self.upper_bound)/self.q
        # if qx == nan, raise expcetion
        if torch.isnan(qx).any():
            print("x:",x)
            raise Exception("Quantized value is NaN")
        return qx
    def forward_inplace(self, x):
        if self.quantize == False:
            return x
        if self.calibration:
            x_flat = x.flatten()
            x_flat = x_flat[x_flat != 0]
            if x_flat.nelement() > 0:
                max_int_bits = torch.max(torch.ceil(torch.log2(torch.abs(x_flat))).max())
                max_int_bits += 1 if self.signed else 0
                min_frac_bits = torch.min(torch.ceil(torch.log2(torch.abs(x_flat))).min())
                min_frac_bits += 1 if self.signed else 0
                self.max_int_bits = torch.max(max_int_bits, self.max_int_bits).int()
                self.min_frac_bits = torch.min(min_frac_bits, self.min_frac_bits).int()
                #self.max_int_bits += 1 if self.signed else 0
                #self.min_frac_bits += 1 if self.signed else 0
            return x
        if self.rounding == 'CONVERGENT':
            if self.saturation == 'WRAP':
                x.mul_(self.q).round_().sub_(self.lower_bound).remainder_(self.upper_bound - self.lower_bound + 1).add_(self.lower_bound).div_(self.q)
            else:
                x.mul_(self.q).round_().clamp_(self.lower_bound, self.upper_bound).div_(self.q)
        else:
            if self.saturation == 'WRAP':
                x.mul_(self.q).trunc_().sub_(self.lower_bound).remainder_(self.upper_bound - self.lower_bound + 1).add_(self.lower_bound).div_(self.q)
            else:
                x.mul_(self.q).trunc_().clamp_(self.lower_bound, self.upper_bound).div_(self.q)
        #x.mul_(self.q).round_()
        #x.clamp_(self.lower_bound, self.upper_bound)
        #x.round_()
        #x.div_(self.q)
        return x

class QLinear(torch.nn.Linear):
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 bias:bool=True, 
                 device=None,
                 dtype=torch.float64,
                 quant_config:dict=None,
                 calibration=False):
        super(QLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_config = quant_config
        self.calibration = calibration
        self.weight_qtzr = TorchQuantizer(**quant_config['weight'], calibration=calibration)
        self.bias_qtzr = TorchQuantizer(**quant_config['bias'], calibration=calibration)
        self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        self.output_qtzr = TorchQuantizer(**quant_config['output'], calibration=calibration)
        self.dtpye = dtype
        #self.reset_parameters()
        
    #def reset_parameters(self):
    #    #reset to zero
    #    torch.nn.init.zeros_(self.weight, dtype=self.dtype)
    #    if self.bias is not None:
    #        torch.nn.init.zeros_(self.bias, dtype=self.dtype)
            
    def forward(self, x):
        qw = self.weight_qtzr(self.weight)
        qx = self.input_qtzr(x)
        qy = torch.matmul(qx, qw.t())
        if self.bias is not None:
            qy += self.bias_qtzr(self.bias)
        qy = self.output_qtzr(qy)
        return qy
    
class QFlashMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, 
                 embed_dim:int, 
                 num_heads:int, 
                 bias:bool=True, 
                 batch_first:bool=False, 
                 device=None, 
                 dtype=torch.float64,
                 quant_config:dict=None,
                 token_tile_size:int=1,
                 embed_tile_size:int=1,
                 head_tile_size:int=1,
                 max_neg_value:float=-80.0,
                 calibration=False):
        super(QFlashMultiheadAttention, self).__init__(embed_dim, 
                                                  num_heads,  
                                                  bias=bias, 
                                                  add_bias_kv=False, 
                                                  add_zero_attn=False,
                                                  kdim=None, 
                                                  vdim=None, 
                                                  batch_first=batch_first, 
                                                  device=device, 
                                                  dtype=dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scale = torch.tensor(1.0 / math.sqrt(self.head_dim))
        self.in_proj = QLinear(embed_dim, 
                               3*embed_dim, 
                               bias=bias, 
                               device=device, 
                               dtype=dtype,
                               quant_config=quant_config['in_proj'], calibration=calibration)
        self.scale_qtzr = TorchQuantizer(**quant_config['scale'], calibration=calibration)
        self.token_tile_size = token_tile_size
        self.embed_tile_size = embed_tile_size
        self.head_tile_size = head_tile_size
        self.max_neg_value = max_neg_value
        self.row_sum_qtzr = TorchQuantizer(**quant_config['row_sum'], calibration=calibration)
        self.exp_input_qtzr = TorchQuantizer(**quant_config['exp_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.exp_output_qtzr = TorchQuantizer(**quant_config['exp_output'], saturation='SAT', calibration=calibration)
        self.inv_input_qtzr = TorchQuantizer(**quant_config['inv_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.inv_output_qtzr = TorchQuantizer(**quant_config['inv_output'], saturation='SAT', calibration=calibration)
        self.attn_out_qtzr = TorchQuantizer(**quant_config['out_proj']['input'], calibration=calibration)
        self.out_proj = QLinear(embed_dim, 
                                embed_dim, 
                                bias=bias, 
                                device=device, 
                                dtype=dtype,
                                quant_config=quant_config['out_proj'], calibration=calibration)
        self.device = device
        self.dtype = dtype
        
    def forward(self, query, attn_mask=None):
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        query = self.in_proj.input_qtzr(query)
        tgt_len, bsz, embed_dim = query.shape
        head_dim = embed_dim // self.num_heads
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        
        # with open("Q.txt", 'a') as f:
        #     np.savetxt(f, q.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        # with open("K.txt", 'a') as f:
        #     np.savetxt(f, k.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        # with open("V.txt", 'a') as f:
        #     np.savetxt(f, v.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((bsz * self.num_heads, tgt_len, 1), dtype = self.dtype, device = self.device)
        all_row_maxes = torch.full((bsz * self.num_heads, tgt_len, 1), self.max_neg_value, dtype = self.dtype, device = self.device)

        num_tiles = math.ceil(tgt_len / self.token_tile_size)
        if attn_mask is not None and attn_mask.ndim == 2:
            mask = attn_mask.bool()

        if attn_mask is None:
            col_masks = (None,) * num_tiles
            mask = (col_masks,) * num_tiles 
        else:
            mask = ((mask,) * num_tiles) if mask.shape[-2] == 1 else mask.split(self.token_tile_size, dim = -2)
            mask = tuple(((row_mask,) * num_tiles) if row_mask.shape[-1] == 1 else row_mask.split(self.token_tile_size, dim = -1) for row_mask in mask)

        B, Nt, E = q.shape
        scale = self.scale_qtzr(self.scale)
        row_splits = zip(
            q.split(self.token_tile_size, dim = -2),
            o.split(self.token_tile_size, dim = -2),
            mask,
            all_row_sums.split(self.token_tile_size, dim = -2),
            all_row_maxes.split(self.token_tile_size, dim = -2),
        )
        for i, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            col_splits = zip(
                k.split(self.token_tile_size, dim = -2),
                v.split(self.token_tile_size, dim = -2),
                row_mask
            )
            for j, (kc, vc, col_mask) in enumerate(col_splits):
                attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc) * scale
                if col_mask is not None:
                    attn_weights.masked_fill_(col_mask, -1000000)
                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(row_maxes, block_row_maxes)
                
                # 将attn_weights的值存储到文件
                # with open("attn_weights.txt", 'a') as f:
                #     np.savetxt(f, attn_weights.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                # 将new_row_maxes的值存储到文件
                # with open("new_row_maxes.txt", 'a') as f:
                #     np.savetxt(f, new_row_maxes.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                att_weights = attn_weights - new_row_maxes
                quant_att_weights = self.exp_input_qtzr(att_weights)
                exp_weights = torch.exp(quant_att_weights)
                exp_weights = self.exp_output_qtzr(exp_weights)
                
                # 将exp_weights的值存储到文件
                # with open("exp_weights.txt", 'a') as f:
                #     np.savetxt(f, exp_weights.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                if col_mask is not None:
                    exp_weights.masked_fill_(col_mask, 0.0)
                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = 1e-10)
                exp_values = torch.einsum('... i j, ... j d -> ... i d', exp_weights, vc)
                exp_row_max_diff = self.exp_input_qtzr(row_maxes - new_row_maxes)
                exp_row_max_diff = self.exp_output_qtzr(torch.exp(exp_row_max_diff))
                
                # 将exp_row_max_diff的值存储到文件
                # with open("exp_row_max_diff.txt", 'a') as f:
                #     np.savetxt(f, exp_row_max_diff.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                new_row_sums = self.row_sum_qtzr(exp_row_max_diff * row_sums + block_row_sums)
                oc.mul_(exp_row_max_diff)
                oc.add_(exp_values)
                self.out_proj.input_qtzr.forward_inplace(oc)
                
                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)
                
                # 每次更新new_row_sums时将其存储到文件
                # with open("new_row_sums.txt", 'a') as f:
                #     np.savetxt(f, new_row_sums.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
            
            new_row_sums = self.inv_input_qtzr(new_row_sums)
            row_sum_inv = self.inv_output_qtzr(torch.reciprocal(new_row_sums + 1e-10))
            
            # 存储未乘上inv_row_sum的O
            # with open("O.txt", 'a') as f:
            #     np.savetxt(f, oc.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
            
            oc.mul_(row_sum_inv)
            self.out_proj.input_qtzr.forward_inplace(oc)
            
            # 将row_sum_inv的值存储到文件
            # with open("row_sum_inv.txt", 'a') as f:
            #     np.savetxt(f, row_sum_inv.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        attn_output = o.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        # with open("attn_output.txt", 'a') as f:
        #     np.savetxt(f, attn_output.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        return attn_output
        

    
class QLayerNorm(torch.nn.LayerNorm):
    def __init__(self, 
                 normalized_shape:Tuple[int, ...],
                 quant_config:dict=None,
                 calibration=False,
                 device='cpu',
                 dtype=torch.float64):
        super(QLayerNorm, self).__init__(normalized_shape, device=device, dtype=dtype)
        self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        self.scale_qtzr = TorchQuantizer(**quant_config['scale'], calibration=calibration)
        self.bias_qtzr = TorchQuantizer(**quant_config['bias'], calibration=calibration)
        self.output_qtzr = TorchQuantizer(**quant_config['output'], calibration=calibration)
        self.mean_qtzr = TorchQuantizer(**quant_config['mean'], calibration=calibration)
        self.var_input_qtzr = TorchQuantizer(**quant_config['var_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.var_output_qtzr = TorchQuantizer(**quant_config['var_output'], saturation='SAT', calibration=calibration)
        self.inv_embed_dim = torch.tensor(1.0 / self.normalized_shape[-1])
        dim_int = np.ceil(np.log2(1.0/self.normalized_shape[-1]))
        self.dim_qtzr = TorchQuantizer(bitwidth=18, int_bitwidth=dim_int, signed=False, calibration=calibration)
        self.inv_embed_dim = self.dim_qtzr(self.inv_embed_dim)
    def forward(self, x):
        x = self.input_qtzr(x)
        # with open("layernorm_data.txt", 'a') as f:
        #     np.savetxt(f, x.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        xsum = x.sum(dim=-1)
        xmean = xsum * self.inv_embed_dim
        xmean = self.mean_qtzr(xmean)
        xsqr = x**2
        xsqrsum = xsqr.sum(dim=-1)
        xsqrsum = xsqrsum * self.inv_embed_dim
        xvar = xsqrsum - xmean**2
        xvar = self.var_input_qtzr(xvar)
        xvar = torch.sqrt(xvar+1e-15)
        xvar = torch.reciprocal(xvar)
        xvar = self.var_output_qtzr(xvar)
        xnorm = (x - xmean.unsqueeze(-1)) * xvar.unsqueeze(-1)
        weight = self.scale_qtzr(self.weight)
        xnorm.mul_(weight)
        bias = self.bias_qtzr(self.bias)
        xnorm.add_(bias)
        xnorm = self.output_qtzr(xnorm)
        
        # 儲存xsum、xsqrsum、xmean和xvar至相應的txt文件
        # with open("sum.txt", 'a') as f:
        #     np.savetxt(f, xsum.detach().cpu().numpy(), fmt='%.6f')
        # with open("sqrsum.txt", 'a') as f:
        #     np.savetxt(f, xsqrsum.detach().cpu().numpy(), fmt='%.6f')
        # with open("mean.txt", 'a') as f:
        #     np.savetxt(f, xmean.detach().cpu().numpy(), fmt='%.6f')
        # with open("var.txt", 'a') as f:
        #     np.savetxt(f, xvar.detach().cpu().numpy(), fmt='%.6f')
        
        # 儲存xnorm至layernorm.txt
        # with open("layernorm.txt", 'a') as f:
        #     np.savetxt(f, xnorm.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        return xnorm
    
class QFeedForward(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 hidden_dim: int,
                 bias: bool = True, 
                 activation: str = 'relu',
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float64,
                 quant_config: dict = None,
                 calibration: bool = False):
        super(QFeedForward, self).__init__()
        self.in_proj = QLinear(embed_dim, 
                               hidden_dim, 
                               bias=bias, 
                               device=device, 
                               dtype=dtype,
                               quant_config=quant_config['in_proj'], calibration=calibration)
        self.activation = activation
        self.cdf_input_qtzr = TorchQuantizer(bitwidth=12, int_bitwidth=3, rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.cdf_output_qtzr = TorchQuantizer(bitwidth=18, int_bitwidth=0, signed=False, saturation='SAT', calibration=calibration)

        self.out_proj = QLinear(hidden_dim, 
                                embed_dim, 
                                bias=bias, 
                                device=device, 
                                dtype=dtype,
                                quant_config=quant_config['out_proj'], calibration=calibration)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        # with open("linear.txt", 'a') as f:
        #     np.savetxt(f, x.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        if self.activation == 'relu':
            x = nn.ReLU()(x)
        elif self.activation == 'gelu':
            cdf_input = self.cdf_input_qtzr(x)
            cdf_values = 0.5 * (1 + torch.erf(cdf_input / math.sqrt(2)))
            cdf_values = self.cdf_output_qtzr(cdf_values)
            x = x * cdf_values
        x = self.out_proj(x)
        return x
    
class QTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.0, 
                 activation: str = 'relu', 
                 norm_first: bool = True, 
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float64,
                 quant_config: dict = None,
                 calibration: bool = False,
                 src_mask: torch.Tensor = None):
        super(QTransformerEncoderLayer, self).__init__(embed_dim, 
                                                       num_heads, 
                                                       hidden_dim, 
                                                       dropout, 
                                                       activation, 
                                                       norm_first)
        self.self_attn = QFlashMultiheadAttention(embed_dim,
                                                    num_heads,
                                                    device=device,
                                                    dtype=dtype,
                                                    quant_config=quant_config['self_attn'], calibration=calibration)
        self.feedforward = QFeedForward(embed_dim,
                                    hidden_dim,
                                    activation=activation,
                                    device=device,
                                    dtype=dtype,
                                    quant_config=quant_config['ffn'], calibration=calibration)
        self.norm1 = QLayerNorm(embed_dim,
                                quant_config=quant_config['norm1'], calibration=calibration)
        self.norm2 = QLayerNorm(embed_dim,
                                quant_config=quant_config['norm2'], calibration=calibration)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        #self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        self.src_mask = src_mask

    def forward(self, 
                src: torch.Tensor, 
                src_mask: torch.Tensor = None) -> torch.Tensor:
        if self.norm_first:
            src = self.norm1.input_qtzr(src) #add input quantizer
            src_norm = self.norm1(src)
            src2 = self.self_attn(src_norm, attn_mask=src_mask)
            src = src + self.dropout(src2)
            src = self.norm2.input_qtzr(src) #add input quantizer
            src_norm = self.norm2(src)
            src2 = self.feedforward(src_norm)
            src = src + self.dropout(src2)
        else:
            src2 = self.self_attn(src, attn_mask=src_mask)
            src = src + self.dropout(src2)
            src = self.norm1(src)
            src2 = self.feedforward(src)
            src = src + self.dropout(src2)
            src = self.norm2(src)
        return src

class QTransformerEncoder(nn.TransformerEncoder):
    def __init__(self,
                 encoder_layer: List[QTransformerEncoderLayer],
                 num_layers: int,
                 norm: QLayerNorm,
                 input_qtzr: TorchQuantizer,
                 dtype: torch.dtype = torch.float64):
        super(QTransformerEncoder, self).__init__(encoder_layer[0], num_layers, norm)
        self.layer_list = encoder_layer
        self.norm = norm
        self.input_qtzr = input_qtzr
        self.dtype = dtype

    def forward(self, 
                src: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        src = self.input_qtzr(src)
        output = src
        for mod in self.layer_list:
            output = mod(output, src_mask=mask)
        output = self.norm(output)
        return output
    
    def transfer_weights(self, 
                         model: nn.Module):
        for i, layer in enumerate(self.layer_list):
            layer.norm1.weight.data = model.transformer_encoder.layers[i].norm1.weight.type(self.dtype)
            layer.norm1.bias.data = model.transformer_encoder.layers[i].norm1.bias.type(self.dtype)
            layer.norm2.weight.data = model.transformer_encoder.layers[i].norm2.weight.type(self.dtype)
            layer.norm2.bias.data = model.transformer_encoder.layers[i].norm2.bias.type(self.dtype)
            layer.self_attn.in_proj.weight.data = model.transformer_encoder.layers[i].self_attn.in_proj_weight.type(self.dtype)
            layer.self_attn.in_proj.bias.data = model.transformer_encoder.layers[i].self_attn.in_proj_bias.type(self.dtype)
            layer.self_attn.out_proj.weight.data = model.transformer_encoder.layers[i].self_attn.out_proj.weight.type(self.dtype)
            layer.self_attn.out_proj.bias.data = model.transformer_encoder.layers[i].self_attn.out_proj.bias.type(self.dtype)
            layer.feedforward.in_proj.weight.data = model.transformer_encoder.layers[i].linear1.weight.type(self.dtype)
            layer.feedforward.in_proj.bias.data = model.transformer_encoder.layers[i].linear1.bias.type(self.dtype)
            layer.feedforward.out_proj.weight.data = model.transformer_encoder.layers[i].linear2.weight.type(self.dtype)
            layer.feedforward.out_proj.bias.data = model.transformer_encoder.layers[i].linear2.bias.type(self.dtype)
        self.norm.weight.data = model.transformer_encoder.norm.weight.type(self.dtype)
        self.norm.bias.data = model.transformer_encoder.norm.bias.type(self.dtype)
        #print("debug: ", self.layer_list[0].self_attn.in_proj.weight.data)


def calibrate_transformer(qmodel: QTransformerEncoder, 
                          quant_config: dict, 
                          calibration_data: torch.Tensor,
                          calibration_mask: torch.Tensor = None
                          ) -> dict:
    with torch.no_grad():
        qmodel.eval()
        qy = qmodel(calibration_data, mask=calibration_mask)
        for i, layer in enumerate(qmodel.layer_list):
            #print("Calibrating layer:", id(layer.norm1.input_qtzr.max_int_bits))
            #print("Calibrating:", layer.norm1.input_qtzr.max_int_bits)
            quant_config[i]['norm1']['input']['int_bitwidth'] = layer.norm1.input_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['mean']['int_bitwidth'] = layer.norm1.mean_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['scale']['int_bitwidth'] = layer.norm1.scale_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['bias']['int_bitwidth'] = layer.norm1.bias_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['output']['int_bitwidth'] = layer.norm1.output_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['var_input']['int_bitwidth'] = layer.norm1.var_input_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['var_output']['int_bitwidth'] = layer.norm1.var_output_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['input']['int_bitwidth'] = layer.norm2.input_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['mean']['int_bitwidth'] = layer.norm2.mean_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['scale']['int_bitwidth'] = layer.norm2.scale_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['bias']['int_bitwidth'] = layer.norm2.bias_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['output']['int_bitwidth'] = layer.norm2.output_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['var_input']['int_bitwidth'] = layer.norm2.var_input_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['var_output']['int_bitwidth'] = layer.norm2.var_output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['input']['int_bitwidth'] = layer.self_attn.in_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['weight']['int_bitwidth'] = layer.self_attn.in_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['bias']['int_bitwidth'] = layer.self_attn.in_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['output']['int_bitwidth'] = layer.self_attn.in_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['input']['int_bitwidth'] = layer.self_attn.out_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['weight']['int_bitwidth'] = layer.self_attn.out_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['bias']['int_bitwidth'] = layer.self_attn.out_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['output']['int_bitwidth'] = layer.self_attn.out_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['row_sum']['int_bitwidth'] = layer.self_attn.row_sum_qtzr.max_int_bits.item()
            #quant_config[i]['self_attn']['exp_input']['int_bitwidth'] = layer.self_attn.exp_input_qtzr.max_int_bits.item()
            #quant_config[i]['self_attn']['exp_output']['int_bitwidth'] = layer.self_attn.exp_output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['inv_input']['int_bitwidth'] = layer.self_attn.inv_input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['inv_output']['int_bitwidth'] = layer.self_attn.inv_output_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['input']['int_bitwidth'] = layer.feedforward.in_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['weight']['int_bitwidth'] = layer.feedforward.in_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['bias']['int_bitwidth'] = layer.feedforward.in_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['output']['int_bitwidth'] = layer.feedforward.in_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['input']['int_bitwidth'] = layer.feedforward.out_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['weight']['int_bitwidth'] = layer.feedforward.out_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['bias']['int_bitwidth'] = layer.feedforward.out_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['output']['int_bitwidth'] = layer.feedforward.out_proj.output_qtzr.max_int_bits.item()
        quant_config['norm']['input']['int_bitwidth'] = qmodel.norm.input_qtzr.max_int_bits.item()
        quant_config['norm']['mean']['int_bitwidth'] = qmodel.norm.mean_qtzr.max_int_bits.item()
        quant_config['norm']['scale']['int_bitwidth'] = qmodel.norm.scale_qtzr.max_int_bits.item()
        quant_config['norm']['bias']['int_bitwidth'] = qmodel.norm.bias_qtzr.max_int_bits.item()
        quant_config['norm']['output']['int_bitwidth'] = qmodel.norm.output_qtzr.max_int_bits.item()
        quant_config['norm']['var_input']['int_bitwidth'] = qmodel.norm.var_input_qtzr.max_int_bits.item()
        quant_config['norm']['var_output']['int_bitwidth'] = qmodel.norm.var_output_qtzr.max_int_bits.item()
    return quant_config