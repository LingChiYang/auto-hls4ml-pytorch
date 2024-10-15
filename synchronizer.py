from typing import Dict

def gen_init_BRAMaware_state(num_layers:int, 
                             weight_bits:int=18,
                             table_input_bits:int=10,
                             table_output_bits:int=18,
                             intermediate_bits:int=18,
                             result_bits:int=18) -> Dict[str, int]:
    state = {}
    for i in range(num_layers):
        state.update({'layers_'+str(i)+'_norm1.Precision.var_table': table_output_bits,
                      'layers_'+str(i)+'_norm1.VarTableSize': table_input_bits,
                      'layers_'+str(i)+'_norm1.Precision.result': result_bits,
                      'layers_'+str(i)+'_self_attn.Precision.exp_table': table_output_bits,
                      'layers_'+str(i)+'_self_attn.ExpTableSize': table_input_bits,
                      'layers_'+str(i)+'_self_attn.Precision.inv_table': table_output_bits,
                      'layers_'+str(i)+'_self_attn.InvTableSize': table_input_bits,
                      'layers_'+str(i)+'_self_attn.Precision.in_proj_out': intermediate_bits,
                      'layers_'+str(i)+'_self_attn.Precision.out_proj_in': intermediate_bits,
                      'layers_'+str(i)+'_self_attn.Precision.in_proj_weight': weight_bits,
                      'layers_'+str(i)+'_self_attn.Precision.out_proj_weight': weight_bits,
                      'layers_'+str(i)+'_self_attn.Precision.result': result_bits,
                      'layers_'+str(i)+'_add1.Precision.result': result_bits,
                      'layers_'+str(i)+'_norm2.Precision.var_table': table_output_bits,
                      'layers_'+str(i)+'_norm2.VarTableSize': table_input_bits,
                      'layers_'+str(i)+'_norm2.Precision.result': result_bits,
                      'layers_'+str(i)+'_ffn.Precision.in_proj_weight': weight_bits,
                      'layers_'+str(i)+'_ffn.Precision.out_proj_weight': weight_bits,
                      'layers_'+str(i)+'_ffn.Precision.hidden': intermediate_bits,
                      'layers_'+str(i)+'_ffn.Precision.result': result_bits,
                      'layers_'+str(i)+'_add2.Precision.result': result_bits})
    state.update({'norm.Precision.var_table': table_output_bits,
                  'norm.VarTableSize': table_input_bits,
                  'norm.Precision.result': result_bits})
    return state

def gen_init_nonBRAMaware_state(num_layers:int) -> Dict[str, int]:
    state = {}
    for i in range(num_layers):
        state.update({'layers_'+str(i)+'_norm1.Precision.bias': 18,
                      'layers_'+str(i)+'_norm1.Precision.scale': 18,
                      'layers_'+str(i)+'_norm1.Precision.mean': 18,
                      'layers_'+str(i)+'_self_attn.Precision.in_proj_bias': 18,
                      'layers_'+str(i)+'_self_attn.Precision.out_proj_bias': 18,
                      'layers_'+str(i)+'_self_attn.Precision.row_sum': 18,
                      'layers_'+str(i)+'_self_attn.Precision.result': 18,
                      'layers_'+str(i)+'_ffn.Precision.in_proj_bias': 18,
                      'layers_'+str(i)+'_ffn.Precision.out_proj_bias': 18,
                      'layers_'+str(i)+'_norm2.Precision.bias': 18,
                      'layers_'+str(i)+'_norm2.Precision.scale': 18,
                      'layers_'+str(i)+'_norm2.Precision.mean': 18,})
    state.update({'norm.Precision.bias': 18,
                  'norm.Precision.scale': 18,
                  'norm.Precision.mean': 18})
    return state

import re
def sync_hls_config(hls_config:dict, 
                    state:dict) -> dict:
    for key in state.keys():
        subkey = key.split('.')
        if len(subkey) == 3:
            match = re.match(r'(ap_ufixed|ap_fixed|fixed|ufixed)<(\d+),(-?\d+)(?:,(\w+)(?:,(\w+)(?:,(\d+))?)?)?>', hls_config['LayerName'][subkey[0]][subkey[1]][subkey[2]])
            base_type, total_bits, integer_bits, rounding, saturation, sat_bits = match.groups()
            if 'table' in subkey[2]:
                hls_config['LayerName'][subkey[0]][subkey[1]][subkey[2]] = f'{base_type}<{state[key]},{integer_bits},{rounding},{saturation},{sat_bits}>'
            else:
                hls_config['LayerName'][subkey[0]][subkey[1]][subkey[2]] = f'{base_type}<{state[key]},{integer_bits},{rounding}>'
        elif len(subkey) == 2:
            hls_config['LayerName'][subkey[0]][subkey[1]] = int(2**state[key])
    return hls_config

def sync_quant_config(quant_config:dict, 
                      hls_config:dict,
                      state:dict) -> bool:
    valid = True
    for key in state.keys():

        subkey = key.split('.')
        signed = True
        if subkey[1] == 'Precision':
            varname = subkey[2]
        else:
            varname = subkey[1]
        layername = subkey[0]
        if layername != 'norm':
            layeridx = int(layername.split('_')[1])
            if layername.endswith('self_attn'):
                layername = 'self_attn'
            else:
                layername = layername.split('_')[2]
        else:
            if varname == 'VarTableSize':
                quant_config[layername]['var_input']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['var_input']['int_bitwidth']
                hls_config['LayerName'][layername]['VarTableRange'] = 2**hls_int
            elif varname == 'var_table':
                quant_config[layername]['var_output']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['var_output']['int_bitwidth']
            elif varname == 'result':
                quant_config[layername]['output']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['output']['int_bitwidth']
            elif varname == 'scale':
                quant_config[layername]['scale']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['scale']['int_bitwidth']
            elif varname == 'bias':
                quant_config[layername]['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['bias']['int_bitwidth']
            elif varname == 'mean':
                quant_config[layername]['mean']['bitwidth'] = state[key]
                hls_int = quant_config[layername]['mean']['int_bitwidth']
            hls_subkey = key.split('.')
            if len(hls_subkey) == 3:
                match = re.match(r'(ap_ufixed|ap_fixed|fixed|ufixed)<(\d+),(-?\d+)(?:,(\w+)(?:,(\w+)(?:,(\d+))?)?)?>', hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]])
                base_type, total_bits, integer_bits, rounding, saturation, sat_bits = match.groups()
                if 'table' in hls_subkey[2]:
                    hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]] = f'{base_type}<{state[key]},{hls_int},{rounding},{saturation},{sat_bits}>'
                    if state[key] < hls_int:
                        valid = False
                else:
                    hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]] = f'{base_type}<{state[key]},{hls_int},{rounding}>'
                    if state[key] < hls_int:
                        valid = False
            elif len(hls_subkey) == 2:
                hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]] = int(2**state[key])
                if state[key] < 9: #table size cannot be less than 2^9
                    valid = False
            continue

        if 'norm' in layername:
            if varname == 'VarTableSize':
                quant_config[layeridx][layername]['var_input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['var_input']['int_bitwidth']
                hls_config['LayerName'][subkey[0]]['VarTableRange'] = 2**hls_int
            elif varname == 'var_table':
                quant_config[layeridx][layername]['var_output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['var_output']['int_bitwidth']
            elif varname == 'result':
                quant_config[layeridx][layername]['output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['output']['int_bitwidth']
            elif varname == 'scale':
                quant_config[layeridx][layername]['scale']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['scale']['int_bitwidth']
            elif varname == 'bias':
                quant_config[layeridx][layername]['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['bias']['int_bitwidth']
            elif varname == 'mean':
                quant_config[layeridx][layername]['mean']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['mean']['int_bitwidth']
                #if layername == 'norm1':
                #    quant_config[layeridx]['self_attn']['in_proj']['input']['bitwidth'] = state[key]
                #elif layername == 'norm2':
                #    quant_config[layeridx]['ffn']['in_proj']['input']['bitwidth'] = state[key]
        elif 'self_attn' in layername:
            if varname == 'ExpTableSize':
                quant_config[layeridx][layername]['exp_input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['exp_input']['int_bitwidth']
                hls_config['LayerName'][subkey[0]]['ExpTableRange'] = 8
            elif varname == 'exp_table':
                quant_config[layeridx][layername]['exp_output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['exp_output']['int_bitwidth']
            elif varname == 'InvTableSize':
                quant_config[layeridx][layername]['inv_input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['inv_input']['int_bitwidth']
                hls_config['LayerName'][subkey[0]]['InvTableRange'] = 2**hls_int
            elif varname == 'inv_table':
                quant_config[layeridx][layername]['inv_output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['inv_output']['int_bitwidth']
            elif varname == 'in_proj_out':
                quant_config[layeridx][layername]['in_proj']['output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['in_proj']['output']['int_bitwidth']
            elif varname == 'out_proj_in':
                quant_config[layeridx][layername]['out_proj']['input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['input']['int_bitwidth']
            elif varname == 'in_proj_weight':
                quant_config[layeridx][layername]['in_proj']['weight']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['in_proj']['weight']['int_bitwidth']
            elif varname == 'in_proj_bias':
                quant_config[layeridx][layername]['in_proj']['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['in_proj']['bias']['int_bitwidth']
            elif varname == 'out_proj_weight':
                quant_config[layeridx][layername]['out_proj']['weight']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['weight']['int_bitwidth']
            elif varname == 'out_proj_bias':
                quant_config[layeridx][layername]['out_proj']['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['bias']['int_bitwidth']
            elif varname == 'result':
                quant_config[layeridx][layername]['out_proj']['output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['output']['int_bitwidth']
            elif varname == 'row_sum':
                quant_config[layeridx][layername]['row_sum']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['row_sum']['int_bitwidth']
                signed = False
        elif 'ffn' in layername:
            if varname == 'in_proj_weight':
                quant_config[layeridx][layername]['in_proj']['weight']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['in_proj']['weight']['int_bitwidth']
            elif varname == 'in_proj_bias':
                quant_config[layeridx][layername]['in_proj']['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['in_proj']['bias']['int_bitwidth']
            elif varname == 'out_proj_weight':
                quant_config[layeridx][layername]['out_proj']['weight']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['weight']['int_bitwidth']
            elif varname == 'out_proj_bias':
                quant_config[layeridx][layername]['out_proj']['bias']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['bias']['int_bitwidth']
            elif varname == 'hidden':
                quant_config[layeridx][layername]['in_proj']['output']['bitwidth'] = state[key]
                quant_config[layeridx][layername]['out_proj']['input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['input']['int_bitwidth']
            elif varname == 'result':
                quant_config[layeridx][layername]['out_proj']['output']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx][layername]['out_proj']['output']['int_bitwidth']
                #if quant_config.get(layeridx+1) is not None:
                #    quant_config[layeridx+1]['norm1']['input']['bitwidth'] = state[key]
                #if quant_config.get('norm') is not None:
                #    quant_config['norm']['input']['bitwidth'] = state[key]
        elif 'add1' in layername:
            if varname == 'result':
                quant_config[layeridx]['norm2']['input']['bitwidth'] = state[key]
                hls_int = quant_config[layeridx]['norm2']['input']['int_bitwidth']
        elif 'add2' in layername:
            if varname == 'result':
                if quant_config.get(layeridx+1) is not None:
                    quant_config[layeridx+1]['norm1']['input']['bitwidth'] = state[key]
                    hls_int = quant_config[layeridx+1]['norm1']['input']['int_bitwidth']
                elif quant_config.get('norm') is not None:
                    quant_config['norm']['input']['bitwidth'] = state[key]
                    hls_int = quant_config['norm']['input']['int_bitwidth']
        hls_subkey = key.split('.')
        if len(hls_subkey) == 3:
            match = re.match(r'(ap_ufixed|ap_fixed|fixed|ufixed)<(\d+),(-?\d+)(?:,(\w+)(?:,(\w+)(?:,(\d+))?)?)?>', hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]])
            base_type, total_bits, integer_bits, rounding, saturation, sat_bits = match.groups()
            if not signed:
                base_type = 'ap_ufixed'
            if 'table' in hls_subkey[2]:
                hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]] = f'{base_type}<{state[key]},{hls_int},{rounding},{saturation},{sat_bits}>'
                if state[key] < hls_int:
                    valid = False
            else:
                hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]][hls_subkey[2]] = f'{base_type}<{state[key]},{hls_int},{rounding}>'
                if state[key] < hls_int:
                    valid = False
        elif len(hls_subkey) == 2:
            hls_config['LayerName'][hls_subkey[0]][hls_subkey[1]] = int(2**state[key])
            if state[key] < 9: #table size cannot be less than 2^9
                valid = False
        #print(f'layer: {hls_subkey}, {state[key]}, {hls_int}, {valid}')
    return valid

def sync_hls_integer_config(hls_config:dict, 
                          quant_config:dict,
                          state:dict) -> dict:
    for key in state.keys():
        subkey = key.split('.')
        print(subkey)
    return hls_config