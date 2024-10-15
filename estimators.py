import numpy as np
import hls4ml
from typing import Dict, Union
class FixedPointQuantizer:
    def __init__(self, bitwidth=18, int_bitwidth=8, signed=True, rounding='RND_CONV'):
        self.int_bitwidth = int_bitwidth
        self.bits = bitwidth
        self.signed = signed
        self.rounding = rounding

    def __call__(self, data):
        m = 2**self.bits
        m_i = 2**self.int_bitwidth
        qdata = data * m / m_i
        qdata = np.round(qdata) if self.rounding == 'RND_CONV' else np.trunc(qdata)
        quant_data = (m_i / m) * np.clip(qdata, -m/2, m/2-1)
        return quant_data 
    
class VivadoVariableBRAMEstimator:
    """
    Args:
        name(str): name of the variable
        var_type(str): 'weight', 'table', 'PIPO' or 'FIFO'
        ram_type(str): 'bram', 'uram' or 'lutram'
        signed(bool): signed or unsigned
        bitwidth(int): total bitwidth of the variable
        int_bitwidth(int): integer bitwidth of the variable
        block_factor(int): block factor of the layer (reuse factor or prod(tiling factor))
        is_partition(bool): True if the variable is partitioned
        array_pragma(str): 'block' or 'cyclic'
    """
    def __init__(self, 
                 name: str, 
                 var_type: str = 'weight', 
                 data: np.array = None,
                 ram_type: str = 'bram',
                 signed : bool = True, 
                 bitwidth : int = 18, 
                 int_bitwidth : int = 8,
                 rounding : str = 'RND_CONV',
                 width : int = 1,
                 depth : int = 1,
                 partition : int = 1,
                 pipo_depth : int = 2):
        self.name = name
        self.var_type = var_type
        self.data = data
        self.ram_type = ram_type
        self.signed = signed
        self.bitwidth = bitwidth
        self.int_bitwidth = int_bitwidth
        self.rounding = rounding
        self.width = width
        self.depth = depth
        self.partition = partition
        self.pipo_depth = pipo_depth
        self.quantizer = FixedPointQuantizer(bitwidth, int_bitwidth, signed, rounding)
        if data is not None:
            assert len(data.shape) == 3, 'Data should be 3D'

    def to_binary(self):
        idata = self.quantizer(self.data) * (1 << (self.bitwidth - self.int_bitwidth))

        def to_fixed_bit_binary(x, bits):
            if x < 0:
                # Convert negative number to its two's complement representation
                x = (1 << bits) + x
            return format(x, f'0{bits}b')
        
        return np.vectorize(to_fixed_bit_binary)(idata.astype(int), self.bitwidth)
    
    def get_ram_data(self):
        binary_data = self.to_binary()
        max_in_single_ram = np.max(np.abs(self.data), axis=2)
        max_int_bitwidth = np.ceil(np.log2(max_in_single_ram)).astype(int)
        calibr_bit = self.int_bitwidth - max_int_bitwidth
        calibr_bit[calibr_bit > self.bitwidth] = self.bitwidth
        ram_data = np.array([[[s[calibr_bit[i,j]-1:] for s in u] for j, u in enumerate(v)] for i, v in enumerate(binary_data)])
        return ram_data
    
    def get_num_ram(self):
        def extend_sign_bits(ram_data):
            for i in range(ram_data.shape[0]):
                for j in range(ram_data.shape[1]):
                    max_len = max([len(s) for s in ram_data[i,j]])
                    for k in range(ram_data.shape[2]):
                        #for each data, extend the length of the data to the maximum length
                        ram_data[i, j, k] = ram_data[i, j, k].zfill(max_len) if ram_data[i, j, k][0] == '0' else ram_data[i, j, k].rjust(max_len, '1')
            return ram_data
        
        
        def find_ram_min_area(total_width, depth):
            bram_spec = {72: 512, 36: 1024, 18: 2048, 9: 4096, 4: 8192, 2: 16384, 1: 32768}
            remainder_bits = total_width
            num_ram = 0
            for i in bram_spec.keys():
                width_ratio = remainder_bits // i
                remainder_bits = remainder_bits % i
                ram_depth = bram_spec[i]
                depth_ratio = np.ceil(depth / ram_depth)
                #print(f'BRAM width: {i}, depth: {ram_depth}, width_ratio: {width_ratio}, depth_ratio: {depth_ratio}')
                num_ram += depth_ratio * width_ratio
            return num_ram
        #def round_to_point_five(num):
        #    return np.ceil(num*2)/2
        
        if self.var_type in ['weight', 'table']:
            assert self.data is not None, 'Data should be provided for weight and table'
            assert self.data.shape[0] == self.partition, 'Data partition should be equal to the partition'
            assert self.data.shape[1] == self.width, 'Data width should be equal to the width'
            assert self.data.shape[2] == self.depth, 'Data depth should be equal to the depth'
            if self.depth <= 128: #the ram efficiency is low for small depth
                return 0
            ram_data = self.get_ram_data()
            addr_range = np.ceil(np.log2(ram_data.shape[2]))
            depth_ratio = 1
            if addr_range < 9:
                addr_range = 9
            if addr_range > 15:
                depth_ratio = np.power(2, addr_range - 15)
                addr_range = 15
            ram_data = extend_sign_bits(ram_data)
            bram_spec = {9: 36, 10: 18, 11: 9, 12: 4, 13: 2, 14: 1, 15: 0.5}
            num_ram = 0
            for i_par in range(self.partition):
                aggr_ram_data = []
                for i_dep in range(self.depth):
                    ram_bits = ''
                    for i_wid in range(self.width):
                        ram_bits += ram_data[i_par, i_wid, i_dep]
                    aggr_ram_data.append(ram_bits)
                data_range = bram_spec[addr_range]
                width_ratio = np.ceil(max([len(s) for s in aggr_ram_data]) / data_range)
                num_ram += depth_ratio*width_ratio
            #print(f'addr_range: {addr_range}, ram_data.shape[2]: {ram_data.shape[2]}, depth_ratio: {depth_ratio}, width_ratio: {width_ratio}, num_ram: {num_ram}, width_max: {max([len(s) for s in aggr_ram_data])}')
            num_ram = num_ram/2.0 #18kBRAM count as 0.5
        elif self.var_type == 'PIPO':
            num_ram = 0
            for i in range(self.partition):
                num_ram += find_ram_min_area(self.width*self.bitwidth, self.depth*self.pipo_depth)
            #num_ram = num_ram*self.pipo_depth
        elif self.var_type == 'FIFO':
            addr_range = np.ceil(np.log2(self.depth))
            depth_ratio = 1
            if addr_range < 9:
                addr_range = 9
            if addr_range > 15:
                depth_ratio = np.power(2, addr_range - 15)
                addr_range = 15
            bram_spec = {9: 72, 10: 36, 11: 18, 12: 9, 13: 4, 14: 2, 15: 1}
            data_range = bram_spec[addr_range]
            width_ratio = np.ceil(self.width*self.bitwidth / data_range)
            num_ram = depth_ratio*width_ratio*self.partition
        #print(f'Number of BRAMs for {self.name}: {num_ram}')
        return num_ram
    
def parse_precision(precision: hls4ml.backends.fpga.fpga_types.APFixedPrecisionDefinition) -> Dict[str, Union[dict, str, int]]:
    return {
        'signed': precision.signed,
        'bitwidth': precision.width,
        'int_bitwidth': precision.integer,
        'rounding': str(precision.rounding_mode),
        #'saturation': str(precision.saturation_mode),
        #'sat_bits': precision.saturation_bits
    }
def parse_graph_edge(edge: hls4ml.backends.fpga.fpga_types.VivadoStreamVariableDefinition,
                     copys: int) -> Dict[str, Union[dict, str, int]]:
    precision = parse_precision(edge.type.precision)
    return {
        'var_type': 'FIFO',
        **precision,
        'partition': copys,
        'width': edge.type.n_pack/edge.type.n_elem if edge.type.unpack else edge.type.n_pack*edge.type.n_elem,
        'depth': edge.pragma[1]
    }
def parse_weight(weight: hls4ml.backends.fpga.fpga_types.StaticWeightVariableDefinition,
                 width: int,
                 partition: int) -> Dict[str, Union[dict, str, int]]:
    precision = parse_precision(weight.type.precision)
    return {
        'var_type' : 'weight',
        **precision,
        'partition': partition,
        'width': width,
        'depth': np.prod(weight.shape) // (width*partition),
        'data': weight.data_unquantized.reshape(partition, width, -1)
    }

def parse_table(node: hls4ml.model.layers.Layer,
                func: str) -> Dict[str, Union[dict, str, int]]:
    assert func in ['exp_table', 'inv_table', 'var_table'], 'Invalid table function'
    precision = parse_precision(get_hls_type(node.get_layer_precision(), func).precision)
    tiling_factor = node.get_attr('tiling_factor')
    if func == 'exp_table':
        table_range = node.get_attr('exp_table_range')
        table_size = node.get_attr('exp_table_size')
        num_cpy = tiling_factor[0]*tiling_factor[0]*node.get_attr('num_heads')
        num_cpy = int(np.ceil(num_cpy/2)) if num_cpy > 1 else num_cpy #because of the simple dual port BRAM
        data = generate_exp_table(table_range, table_size, num_cpy)
    elif func == 'inv_table':
        table_range = node.get_attr('inv_table_range')
        table_size = node.get_attr('inv_table_size')
        num_cpy = tiling_factor[0]*node.get_attr('num_heads')
        num_cpy = int(np.ceil(num_cpy/2)) if num_cpy > 1 else num_cpy #because of the simple dual port BRAM
        data = generate_inv_table(table_range, table_size, num_cpy)
    elif func == 'var_table':
        table_range = node.get_attr('var_table_range')
        table_size = node.get_attr('var_table_size')
        num_cpy = tiling_factor[0]
        num_cpy = int(np.ceil(num_cpy/2)) if num_cpy > 1 else num_cpy #because of the simple dual port BRAM
        data = generate_var_table(table_range, table_size, num_cpy)
    return {
        'var_type' : 'table',
        **precision,
        'partition': num_cpy,
        'width': 1,
        'depth': table_size,
        'data': data
    }

def get_hls_type(layer_precision: Dict[str, hls4ml.backends.fpga.fpga_types.TypeDefinition],
                 layer_name: str) -> hls4ml.backends.fpga.fpga_types.TypeDefinition:
    key = [k for k in layer_precision.keys() if layer_name in k][0]
    return layer_precision[key]

def get_depth(node: hls4ml.model.layers.Layer,
              class_name: str, 
              var: str) -> int:
    tiling_factor = node.get_attr('tiling_factor')
    if class_name == 'MultiheadAttention':
        if var == 'in_proj_out':
            return node.get_attr('seq_len')*node.get_attr('head_dim')//tiling_factor[0]*tiling_factor[2]
        elif var == 'out_proj_in':
            return node.get_attr('seq_len')*node.get_attr('head_dim')//tiling_factor[0]*tiling_factor[2]
        elif var == 'exp_table':
            return node.get_attr('exp_table_size')
        elif var == 'inv_table':
            return node.get_attr('inv_table_size')
    elif class_name == 'LayerNorm':
        #if var == 'row_buffer':
        #    return node.get_attr('seq_len')//tiling_factor[0]
        if var == 'var_table':
            return node.get_attr('table_size')
    elif class_name == 'FeedForwardNetwork':
        if var == 'accum':
            return node.get_attr('hidden_dim')//tiling_factor[2]
    return None

def get_width(node: hls4ml.model.layers.Layer,
              class_name: str, 
              var: str) -> int:
    tiling_factor = node.get_attr('tiling_factor')
    if class_name == 'MultiheadAttention':
        if var == 'in_proj_out':
            return tiling_factor[0]*tiling_factor[2]*node.get_attr('num_heads')
        elif var == 'out_proj_in':
            return tiling_factor[0]*tiling_factor[2]*node.get_attr('num_heads')
        elif var == 'exp_table':
            return tiling_factor[0]*tiling_factor[0]
        elif var == 'inv_table':
            return tiling_factor[0]
    elif class_name == 'LayerNorm':
        #if var == 'row_buffer':
        #    return node.get_attr('seq_len')//tiling_factor[0]
        if var == 'var_table':
            return 1
    elif class_name == 'FeedForwardNetwork':
        if var == 'accum':
            return tiling_factor[2]
    return None

def parse_pipo(node: hls4ml.model.layers.Layer,
               var: str,
               width: int,
               depth: int) -> Dict[str, Union[dict, str, int]]:
    #TODO: add pipo as variable to HLSLayer
    layer_precision = node.get_layer_precision()
    hls_type = get_hls_type(layer_precision, var)
    return {
        'var_type': 'PIPO',
        **parse_precision(hls_type.precision),
        'width': width,
        'depth': depth
    }

def generate_exp_table(table_range, table_size, num_cpy):
    table = np.zeros(table_size)
    for i in range(table_size):
        table[i] = np.exp(2*table_range*(i - table_size//2)/table_size)
    return np.tile(table, (num_cpy, 1, 1))
    
def generate_inv_table(table_range, table_size, num_cpy):
    table = np.zeros(table_size)
    for i in range(table_size):
        table[i] = 1.0/(table_range*i/table_size) if i != 0 else 0
    return np.tile(table, (num_cpy, 1, 1))

def generate_var_table(table_range, table_size, num_cpy):
    table = np.zeros(table_size)
    for i in range(table_size):
        table[i] = 1.0/np.sqrt(table_range*i/table_size) if i != 0 else 0
    return np.tile(table, (num_cpy, 1, 1))

class BaseBramConfig:
    def __init__(self, precision, width, depth):
        self.precision = precision
        self.width = width
        self.depth = depth

def parse_hls_model(hls_model):
    var_dict = {'MultiheadAttention': 
                {'pipo' : [],
                 'table' : [('exp_table', 'exp_diff_max'), ('exp_table', 'exp_attn'), ('inv_table', 'row_sum')]},
                'LayerNorm':
                {'pipo': [],
                 'table': [('var_table', 'inv_sqrt_var')]},
                'FeedForwardNetwork': 
                {'pipo': [],
                 'table': []}
               }

    quant_config = {}
    import re
    for node in hls_model.get_layers():
        if node.class_name == 'Input':
            continue
        quant_config[node.name] = {}
        tiling_factor = node.get_attr('tiling_factor')
        for w_var in node.get_weights():
            if 'in_proj_weight' in w_var.name and node.class_name == 'MultiheadAttention':
                print(tiling_factor[1]*tiling_factor[2]*node.get_attr('num_heads')*3)
                quant_config[node.name][re.sub(r'\d+$', '', w_var.name)] = parse_weight(w_var, tiling_factor[1]*tiling_factor[2]*node.get_attr('num_heads')*3,1)#type('BramConfig', (BaseBramConfig,), parse_weight(w_var, tiling_factor[1]*tiling_factor[2]))
            elif 'out_proj_weight' in w_var.name and node.class_name == 'MultiheadAttention':
                quant_config[node.name][re.sub(r'\d+$', '', w_var.name)] = parse_weight(w_var, tiling_factor[1]*tiling_factor[2]*node.get_attr('num_heads')*2,1)
            elif 'bias' in w_var.name or 'mask' in w_var.name:
                continue
            else:
                quant_config[node.name][re.sub(r'\d+$', '', w_var.name)] = parse_weight(w_var, tiling_factor[1]*tiling_factor[2],1)
        #strm_var = node.get_output_variable()
        #output_nodes = node.get_output_nodes()
        #if output_nodes is not []:
        #    quant_config[node.name]['layer_out'] = parse_graph_edge(strm_var, copys=len(output_nodes))#type('BramConfig', (BaseBramConfig,), parse_graph_edge(strm_var))
        #if node.class_name in var_dict:
        #    for pipo_var in var_dict[node.class_name]['pipo']:
        #        depth = get_depth(node, node.class_name, pipo_var[0])
        #        width = get_width(node, node.class_name, pipo_var[0])#tiling_factor[1]*tiling_factor[2]
        #        quant_config[node.name][pipo_var[1]] = parse_pipo(node, pipo_var[0], width, depth)#type('BramConfig', (BaseBramConfig,), parse_pipo(node, pipo_var, width, depth))
        #    for table_var in var_dict[node.class_name]['table']:
        #        #print(table_var)
        #        quant_config[node.name][table_var[1]] = parse_table(node, table_var[0])#type('BramConfig', (BaseBramConfig,), parse_table(precision))
    return quant_config