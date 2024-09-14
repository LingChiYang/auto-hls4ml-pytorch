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
                 max_neg_value:float=-8.0,
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
        #save query and q k v to txt file
        #print("query", query)
        query = self.in_proj.input_qtzr(query)
        #print("query after", query)
        #with open("query.txt", 'a') as f:
        #    np.savetxt(f, query.reshape(-1, self.embed_dim).detach().numpy(), fmt='%.8f')
        #with open("q.txt", 'a') as f:
        #    np.savetxt(f, q.reshape(-1, self.embed_dim).detach().numpy(), fmt='%.8f')
        #with open("k.txt", 'a') as f:
        #    np.savetxt(f, k.reshape(-1, self.embed_dim).detach().numpy(), fmt='%.8f')
        #with open("v.txt", 'a') as f:
        #    np.savetxt(f, v.reshape(-1, self.embed_dim).detach().numpy(), fmt='%.8f')
        tgt_len, bsz, embed_dim = query.shape
        head_dim = embed_dim // self.num_heads
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        
        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((bsz * self.num_heads, tgt_len, 1), dtype = self.dtype, device = self.device)
        all_row_maxes = torch.full((bsz * self.num_heads, tgt_len, 1), self.max_neg_value, dtype = self.dtype, device = self.device)

        num_tiles = math.ceil(tgt_len / self.token_tile_size)
        if attn_mask is not None and attn_mask.ndim == 2:
            #attn_mask = rearrange(attn_mask, 'b n -> 1 1 b n')
            mask = attn_mask.bool()
            #print("attn_mask shape:", attn_mask.shape)

        if attn_mask is None:
            col_masks = (None,) * num_tiles
            mask = (col_masks,) * num_tiles 
        else:
            mask = ((mask,) * num_tiles) if mask.shape[-2] == 1 else mask.split(self.token_tile_size, dim = -2)
            #print("attn_mask shape1:", attn_mask.shape)
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
        #attn_weight_debug = torch.zeros((self.num_heads, tgt_len, tgt_len), dtype = self.dtype, device = self.device)
        exp_weight_debug = torch.zeros((self.num_heads*bsz, tgt_len, tgt_len), dtype = self.dtype, device = self.device)
        #row_max_debug = torch.zeros((self.num_heads, tgt_len, tgt_len), dtype = self.dtype, device = self.device)
        row_sum_debug = torch.zeros((self.num_heads*bsz, tgt_len, 1), dtype = self.dtype, device = self.device)
        #with open("K.txt", 'a') as f:
        #    np.savetxt(f, k.reshape(-1, tgt_len*head_dim).detach().numpy(), fmt='%.6f')
        #with open("Q.txt", 'a') as f:
        #    np.savetxt(f, q.reshape(-1, tgt_len*head_dim).detach().numpy(), fmt='%.6f')
        #with open("V.txt", 'a') as f:
        #    np.savetxt(f, v.reshape(-1, tgt_len*head_dim).detach().numpy(), fmt='%.6f')
        for i, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            col_splits = zip(
                k.split(self.token_tile_size, dim = -2),
                v.split(self.token_tile_size, dim = -2),
                row_mask
            )
            for j, (kc, vc, col_mask) in enumerate(col_splits):
                attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc) * scale
                if col_mask is not None:
                #    #print("col_mask:", ~col_mask)
                    attn_weights.masked_fill_(col_mask, -1000000)
                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(row_maxes, block_row_maxes)
                #row_max_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = new_row_maxes
                att_weights = attn_weights - new_row_maxes
                #attn_weight_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = attn_weights
                att_weights = self.exp_input_qtzr(att_weights)
                exp_weights = torch.exp(att_weights)
                exp_weights = self.exp_output_qtzr(exp_weights)
                if col_mask is not None:
                    exp_weights.masked_fill_(col_mask, 0.0)
                exp_weight_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = exp_weights
                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = 1e-10)
                exp_values = torch.einsum('... i j, ... j d -> ... i d', exp_weights, vc)
                exp_row_max_diff = self.exp_input_qtzr(row_maxes - new_row_maxes)
                exp_row_max_diff = self.exp_output_qtzr(torch.exp(exp_row_max_diff))
                new_row_sums = self.row_sum_qtzr(exp_row_max_diff * row_sums + block_row_sums)
                #exp_weight_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = exp_row_max_diff
                #row_sum_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = new_row_sums
                oc.mul_(exp_row_max_diff)
                oc.add_(exp_values)
                #if i == 0:
                #    print(f"exp weights: {exp_weights[0,0,0].item()}, vc {vc[0,0,0].item()}")
                #    print(f"oc {oc[0,0,0].item()}, exp_values {exp_values[0,0,0].item()}, exp_row_max_diff {exp_row_max_diff[0,0,0].item()}")
                #exp_weight_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, j*self.token_tile_size:(j+1)*self.token_tile_size] = exp_values
                #print("max oc:", oc.abs().max())
                self.out_proj.input_qtzr.forward_inplace(oc)
                #print(f"max oc after:{oc.abs().max()}, max bits:{self.attn_out_qtzr.max_int_bits}")
                
                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)
            new_row_sums = self.inv_input_qtzr(new_row_sums)
            row_sum_inv = self.inv_output_qtzr(torch.reciprocal(new_row_sums + 1e-10))
            row_sum_debug[:, i*self.token_tile_size:(i+1)*self.token_tile_size, :] = row_sum_inv
            #row_sum_inv = row_sum_inv.view(bsz*self.num_heads, tgt_len, 1)
            oc.mul_(row_sum_inv)
            self.out_proj.input_qtzr.forward_inplace(oc)
        #with open("O.txt", 'a') as f:
        #    np.savetxt(f, o.reshape(-1, tgt_len*head_dim).detach().numpy(), fmt='%.6f')
        attn_output = o.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #print("attn_output", attn_output)
        attn_output = self.out_proj(attn_output)
        #print("exp_weight_debug", exp_weight_debug)
        #print("row_sum_debug", row_sum_debug)
        #with open("attn_weights.txt", 'a') as f:
        #    np.savetxt(f, attn_weight_debug.reshape(-1, tgt_len*tgt_len).detach().numpy(), fmt='%.6f')
        #with open("exp_weights.txt", 'a') as f:
        #    np.savetxt(f, exp_weight_debug.reshape(-1, tgt_len*tgt_len).detach().numpy(), fmt='%.6f')
        #with open("row_max.txt", 'a') as f:
        #    np.savetxt(f, row_max_debug.reshape(-1, tgt_len*tgt_len).detach().numpy(), fmt='%.6f')
        #with open("row_sum.txt", 'a') as f:
        #    np.savetxt(f, row_sum_debug.reshape(-1, tgt_len*tgt_len).detach().numpy(), fmt='%.6f')
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
        #save x to file
        #with open("norm_in.txt", 'a') as f:
        #    np.savetxt(f, x[:,0,:].detach().numpy(), fmt='%.6f')
        #xsum = 0
        #for i in range(self.normalized_shape[-1]):
        #    #accumulate the 132th row
        #    xsum += x[132,0,i]
        #    print(f"xsum[{i}] = {xsum}")
        xmean = x.sum(dim=-1, keepdim=True)
        #print("x_sum: ",xmean[132,0,0].item())
        #with open("xsum.txt", 'a') as f:
        #    np.savetxt(f, xmean[:,0,:].detach().numpy(), fmt='%.6f')
        xmean.mul_(self.inv_embed_dim)
        xmean = self.mean_qtzr(xmean)
        #import numpy as np
        #with open("xmean.txt", 'a') as f:
        #    np.savetxt(f, xmean[:,0,:].detach().numpy(), fmt='%.6f')
        xsqr = x**2
        xsqrsum = xsqr.sum(dim=-1, keepdim=True)
        xsqrsum.mul_(self.inv_embed_dim)
    
        #with open("xvar.txt", 'a') as f:
        #    np.savetxt(f, xsqrsum[:,0,:].detach().numpy(), fmt='%.6f')
        xvar = xsqrsum - xmean**2
        #print('xvar', xvar)
        #with open("xvar_in.txt", 'a') as f:
        #    np.savetxt(f, xvar[:,0,:].detach().numpy(), fmt='%.6f')
        xvar = self.var_input_qtzr(xvar)
        xvar = torch.sqrt(xvar+1e-15)
        xvar = torch.reciprocal(xvar)
        xvar = self.var_output_qtzr(xvar)
        #print('xvar', xvar)
        #with open("xvar_out.txt", 'a') as f:
        #    np.savetxt(f, xvar[:,0,:].detach().numpy(), fmt='%.10f')
        #with open("xmean.txt", 'a') as f:
        #    np.savetxt(f, xmean[:,0,:].detach().numpy(), fmt='%.10f')
        xnorm = (x - xmean) * xvar
        #print('xnorm', xnorm)
        x_debug = x - xmean
        #with open("x_debug.txt", 'a') as f: #only save the 133 row
        #    np.savetxt(f, x_debug[132,0,:].detach().numpy(), fmt='%.10f')
        #with open("xnorm_133_before.txt", 'a') as f: #only save the 133 row
        #    np.savetxt(f, xnorm[132,0,:].detach().numpy(), fmt='%.10f')
        weight = self.scale_qtzr(self.weight)
        xnorm.mul_(weight)
        #with open("xnorm_133.txt", 'a') as f: #only save the 133 row
        #    np.savetxt(f, xnorm[132,0,:].detach().numpy(), fmt='%.6f')

        bias = self.bias_qtzr(self.bias)
        xnorm.add_(bias)
        xnorm = self.output_qtzr(xnorm)
        #with open("norm_out.txt", 'a') as f:
        #    np.savetxt(f, xnorm[:,0,:].detach().numpy(), fmt='%.6f')
        #print("xnorm", xnorm)
        return xnorm
    
class QFeedForward(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 hidden_dim: int,
                 bias: bool = True, 
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
        self.activation = nn.ReLU()
        self.out_proj = QLinear(hidden_dim, 
                                embed_dim, 
                                bias=bias, 
                                device=device, 
                                dtype=dtype,
                                quant_config=quant_config['out_proj'], calibration=calibration)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print('ffn x', x)
        #with open("./hls/ndt_calibrate3/tb_data/tb_input_features.dat", 'a') as f:
        #    np.savetxt(f, self.in_proj.input_qtzr(x[:,0,:]).reshape(1,-1).detach().numpy(), fmt='%.6f')
        x = self.in_proj(x)
        #save x to file
        #with open("ffn_hidden.txt", 'a') as f:
        #    np.savetxt(f, x[:,0,:].flatten().detach().numpy(), fmt='%.6f')
        #print('hidden x', x)
        x = self.activation(x)
        #print("relu x", x)
        x = self.out_proj(x)
        #print('ffn out', x)
        #with open("./hls/ndt_calibrate3/tb_data/tb_output_predictions.dat", 'a') as f:
        #    np.savetxt(f, x[:,0,:].reshape(1,-1).detach().numpy(), fmt='%.6f')
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
            #print("src:", src)
            #with open("src_norm1_in.txt", 'a') as f:
            #    np.savetxt(f, src[:,0,:].detach().numpy(), fmt='%.6f')
            src = self.norm1.input_qtzr(src) #add input quantizer
            #print('src', src)
            src_norm = self.norm1(src)
            #print('src_norm', src_norm)
            #import numpy as np
            #with open("src_norm1.txt", 'a') as f:
            #    np.savetxt(f, src_norm[:,0,:].detach().numpy(), fmt='%.6f')
            src2 = self.self_attn(src_norm, attn_mask=src_mask)
            #print('src2', src2)
            src = src + self.dropout(src2)
            #print("src:", src)
            src = self.norm2.input_qtzr(src) #add input quantizer
            #print('src2', src)
            src_norm = self.norm2(src)
            #print('src_norm', src_norm)
            src2 = self.feedforward(src_norm)
            #print("src before add:", src)
            #print("src2", src2)
            src = src + self.dropout(src2)
            #print("src after add:", src[0,0,:])
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
                          calibration_mask: torch.Tensor
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