import random
import torch
import torch.nn as nn
import pandas as pd
from synchronizer import *
import sys
import logging

class Transformer4HLS(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, norm_first, device):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.device = device
        self._init_transformer()

    def _init_transformer(self):
        norm = nn.LayerNorm(self.d_model)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, 
                                       nhead=self.nhead,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=self.dropout,
                                       activation=self.activation,
                                       norm_first=self.norm_first,
                                       device=self.device),
            self.num_encoder_layers,
            norm=norm
        )

    def forward(self, src):  
        output = self.transformer_encoder(src)
        return output


model4hls = Transformer4HLS(d_model=192, 
                            nhead=3, 
                            num_encoder_layers=12, 
                            dim_feedforward=768, 
                            dropout=0, 
                            activation='gelu', 
                            norm_first=True, 
                            device='cpu')

from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from PIL import Image
import requests
import torch
import os

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
model.to(torch.device('cuda'))
img_folder_path = '/home/docker/deit_hls4ml/img_folder'

# 读取标签映射文件
label_mapping = {}
with open(os.path.join(img_folder_path, 'file_label_mapping.txt'), 'r') as f:
    for line in f:
        filename, label = line.strip().split(",")
        label_mapping[filename] = int(label)

# 读取图片并处理
image_tensors = []
true_labels = []
for filename in os.listdir(img_folder_path):
    if filename.endswith('.JPEG'):
        image_path = os.path.join(img_folder_path, filename)
        image = Image.open(image_path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt")
        image_tensors.append(inputs['pixel_values'])
        true_labels.append(label_mapping[filename])

# 将所有图片张量堆叠成一个大的张量
images_tensor = torch.cat(image_tensors, dim=0)
true_labels = torch.tensor(true_labels)
images_tensor = images_tensor.to(torch.device('cuda'))
true_labels = true_labels.to(torch.device('cuda'))

for i in range(12):
    wq = model.deit.encoder.layer[i].attention.attention.query.weight
    wk = model.deit.encoder.layer[i].attention.attention.key.weight
    wv = model.deit.encoder.layer[i].attention.attention.value.weight
    bq = model.deit.encoder.layer[i].attention.attention.query.bias
    bk = model.deit.encoder.layer[i].attention.attention.key.bias
    bv = model.deit.encoder.layer[i].attention.attention.value.bias
    #concatenate wq, wk, wv
    w_in_proj = torch.cat([wq, wk, wv], dim=0)
    model4hls.transformer_encoder.layers[i].self_attn.in_proj_weight.data = w_in_proj
    model4hls.transformer_encoder.layers[i].self_attn.in_proj_bias.data = torch.cat([bq, bk, bv], dim=0)
    model4hls.transformer_encoder.layers[i].self_attn.out_proj.weight.data = model.deit.encoder.layer[i].attention.output.dense.weight
    model4hls.transformer_encoder.layers[i].self_attn.out_proj.bias.data = model.deit.encoder.layer[i].attention.output.dense.bias
    model4hls.transformer_encoder.layers[i].linear1.weight.data = model.deit.encoder.layer[i].intermediate.dense.weight
    model4hls.transformer_encoder.layers[i].linear1.bias.data = model.deit.encoder.layer[i].intermediate.dense.bias
    model4hls.transformer_encoder.layers[i].linear2.weight.data = model.deit.encoder.layer[i].output.dense.weight
    model4hls.transformer_encoder.layers[i].linear2.bias.data = model.deit.encoder.layer[i].output.dense.bias
    model4hls.transformer_encoder.layers[i].norm1.weight.data = model.deit.encoder.layer[i].layernorm_before.weight
    model4hls.transformer_encoder.layers[i].norm1.bias.data = model.deit.encoder.layer[i].layernorm_before.bias
    model4hls.transformer_encoder.layers[i].norm2.weight.data = model.deit.encoder.layer[i].layernorm_after.weight
    model4hls.transformer_encoder.layers[i].norm2.bias.data = model.deit.encoder.layer[i].layernorm_after.bias
model4hls.transformer_encoder.norm.weight.data = model.deit.layernorm.weight
model4hls.transformer_encoder.norm.bias.data = model.deit.layernorm.bias
model4hls.to(torch.device('cpu'))
transformer_quant_config = torch.load('/home/docker/deit_hls4ml/deit_tiny_distilled_patch16_224_quant_config_calibrated_2000.pth')

from simanneal import Annealer
from quantizers import *
from synchronizer import *
from estimators import *
import hls4ml
import torch
from pprint import pprint
import copy

# 設置日誌
logging.basicConfig(filename='sa_output.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# 定義一個類來捕獲stdout和stderr
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = logging.getLogger()

    def write(self, message):
        self.terminal.write(message)
        if message != '\n':
            self.log.info(message)

    def flush(self):
        pass

# 重定向stdout和stderr
sys.stdout = Logger()
sys.stderr = Logger()

def layer_estimater(quant_config):
    bram_dict = {}  
    for layer_name in quant_config.keys():
        bram_dict[layer_name] = {}
        for var_name in quant_config[layer_name].keys():
            #pprint(quant_config[layer_name][var_name])
            bram_dict[layer_name][var_name] = VivadoVariableBRAMEstimator(name=var_name,**quant_config[layer_name][var_name])

    num_ram = 0
    for layer_name in bram_dict.keys():
        for var_name in bram_dict[layer_name].keys():
            ram_est = bram_dict[layer_name][var_name]
            num_ram += ram_est.get_num_ram()
    return num_ram


import os
path = '/home/docker/deit_hls4ml/sa_data/batch100_1e-1_1e-5_2000_100_alpha2'
os.makedirs(path, exist_ok=True)

class OptimizeAckley(Annealer):
    def __init__(self, initial_state, best_acc, model4hls, ref_model, src, transformer_quant_config, alpha):
        super(OptimizeAckley, self).__init__(initial_state)
        self.state_history = []
        self.energy_history = []
        self.acc1_history = []
        self.acc5_history = []
        self.num_ram_history = []
        self.acceptance_history = []
        self.improvement_history = []
        self.temperature_history = []
        self.prob_history = []
        self.best_acc = best_acc
        self.ref_model = ref_model
        self.src = src
        self.alpha = alpha
        self.penalty = 1
        self.transformer_quant_config = transformer_quant_config
        self.model4hls = model4hls
        self.hls_config = hls4ml.utils.config_from_pytorch_model(model4hls, 
                                                                granularity='name',
                                                                backend='Vitis',
                                                                input_shapes=[[1, 198, 192]], 
                                                                default_precision='ap_fixed<18,5,AP_RND_CONV>', 
                                                                inputs_channel_last=True, 
                                                                transpose_outputs=False)
        valid = sync_quant_config(self.transformer_quant_config, self.hls_config, self.state)

        for layer_config in self.hls_config['LayerName'].keys():
            if layer_config.endswith('self_attn'):
                self.hls_config['LayerName'][layer_config]['TilingFactor'] = [1,1,8]
            elif layer_config.endswith('ffn'):
                self.hls_config['LayerName'][layer_config]['TilingFactor'] = [1,1,96]
        self.hls_model = hls4ml.converters.convert_from_pytorch_model(
                                                                self.model4hls,
                                                                [[1, 198, 192]],
                                                                output_dir='./experiments/hls/deit_tiny_alpha1',
                                                                project_name='myproject',
                                                                backend='Vitis',
                                                                #part='xcu250-figd2104-2L-e',
                                                                part='xcu55c-fsvh2892-2L-e',
                                                                #board='alveo-u55c',
                                                                hls_config=self.hls_config,
                                                                io_type='io_tile_stream',
                                                            )
        self.qmodel = QTransformerEncoder(
                                            encoder_layer=[
                                                QTransformerEncoderLayer(
                                                    embed_dim=192,
                                                    num_heads=3,
                                                    hidden_dim=768,
                                                    activation='gelu',
                                                    norm_first=True,
                                                    quant_config=transformer_quant_config[i],
                                                    calibration=False,
                                                    device='cuda',
                                                    dtype=torch.float64
                                                ) for i in range(12)
                                            ],
                                            num_layers=12,
                                            norm=QLayerNorm(
                                                normalized_shape=192,
                                                quant_config=transformer_quant_config['norm'],
                                                calibration=False,
                                                device='cuda',
                                                dtype=torch.float64
                                            ),
                                            input_qtzr=TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=False, device='cuda', dtype=torch.float64),
                                            dtype=torch.float64
                                        )
        self.qmodel.transfer_weights(self.model4hls)
        self.qmodel.to(torch.device('cuda'))
    def move(self):
        valid = False
        while not valid:
            #deep copy
            new_state = copy.deepcopy(self.state)
            while True:
                rnd_key = random.choice(list(new_state.keys()))
                if 'weight' in rnd_key:
                    break
            rnd_bits = 1
            rnd_plus = random.choice([True, False])
            new_state[rnd_key] = new_state[rnd_key] + rnd_bits if rnd_plus else new_state[rnd_key] - rnd_bits
            valid = sync_quant_config(self.transformer_quant_config, self.hls_config, new_state)
        self.state = new_state
        self.state_history.append(self.state.copy())

    def energy(self):
        valid = sync_quant_config(self.transformer_quant_config, self.hls_config, self.state)
        self.model4hls.to(torch.device('cpu'))
        for layer_config in self.hls_config['LayerName'].keys():
            if layer_config.endswith('self_attn'):
                self.hls_config['LayerName'][layer_config]['TilingFactor'] = [1,1,8]
            elif layer_config.endswith('ffn'):
                self.hls_config['LayerName'][layer_config]['TilingFactor'] = [1,1,96]
        self.hls_model = hls4ml.converters.convert_from_pytorch_model(
                                                                self.model4hls,
                                                                [[1, 198, 192]],
                                                                output_dir='./hls/deit_tiny_w8_alpha1',
                                                                project_name='myproject',
                                                                backend='Vitis',
                                                                #part='xcu250-figd2104-2L-e',
                                                                part='xcu55c-fsvh2892-2L-e',
                                                                #board='alveo-u55c',
                                                                hls_config=self.hls_config,
                                                                io_type='io_tile_stream',
                                                            )
        self.qmodel = QTransformerEncoder(
                                            encoder_layer=[
                                                QTransformerEncoderLayer(
                                                    embed_dim=192,
                                                    num_heads=3,
                                                    hidden_dim=768,
                                                    activation='gelu',
                                                    norm_first=True,
                                                    quant_config=transformer_quant_config[i],
                                                    calibration=False,
                                                    device='cuda',
                                                    dtype=torch.float64
                                                ) for i in range(12)
                                            ],
                                            num_layers=12,
                                            norm=QLayerNorm(
                                                normalized_shape=192,
                                                quant_config=transformer_quant_config['norm'],
                                                calibration=False,
                                                device='cuda',
                                                dtype=torch.float64
                                            ),
                                            input_qtzr=TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=False, device='cuda', dtype=torch.float64),
                                            dtype=torch.float64
                                        )
        self.model4hls.to(torch.device('cuda'))
        self.qmodel.transfer_weights(self.model4hls)
        self.num_ram = layer_estimater(parse_hls_model(self.hls_model))
        self.ram_percentage = self.num_ram/2016.0
        print(f'Number of BRAMs: {self.num_ram}')
        self.ref_model.eval()
        self.qmodel.eval()
        self.qmodel.to(torch.device('cuda'))
        batch_size = 2000
        with torch.no_grad():
            embbed_out = self.ref_model.deit.embeddings(images_tensor)
            encoder_out2 = self.qmodel(embbed_out[0:batch_size,:,:].permute(1, 0, 2).type(torch.float64).to(torch.device('cuda')))
            encoder_out2 = encoder_out2.permute(1, 0, 2).type(torch.float32).to(torch.device('cuda'))
            print(encoder_out2.shape)
            cls_logits = self.ref_model.cls_classifier(encoder_out2[:, 0, :])
            distillation_logits = self.ref_model.distillation_classifier(encoder_out2[:, 1, :])
            # during inference, return the average of both classifier predictions
            logits = (cls_logits + distillation_logits) / 2
            #evaluate the accuracy
            _, predicted = logits.topk(5, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(true_labels[0:batch_size].view(1, -1).expand_as(predicted))
            self.top1_accuracy = correct[:1].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()
            self.top5_accuracy = correct[:5].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()
        print(f'top5_accuracy: {self.top5_accuracy}')
        print(f'top1_accuracy: {self.top1_accuracy}')
        if self.ram_percentage > 0.8:
            self.penalty = 2
        else:
            self.penalty = 1
        self.energy_value = self.penalty * self.ram_percentage - self.alpha * (self.top5_accuracy + self.top1_accuracy)/self.best_acc
        print(f'energy_value: {self.energy_value}')
        #save energy to energy_history
        self.energy_history.append(self.energy_value.copy())
        self.acc1_history.append(self.top1_accuracy)
        self.acc5_history.append(self.top5_accuracy)
        self.num_ram_history.append(self.num_ram)
        if len(self.num_ram_history) > 1:
            #if there is a gap between current number of ram and previous number of ram, then save the state
            #if self.num_ram_history[-1] - self.num_ram_history[-2] > 100:
            torch.save(self.state, f'{path}/state_{len(self.num_ram_history)}.pth')
        #if there is a gap between current top5_accuracy and previous top5_accuracy, then save the state
        #if len(self.acc5_history) > 1:
            #if self.acc5_history[-1] - self.acc5_history[-2] > 0.1:
            #torch.save(self.state, f'/home/docker/deit_hls4ml/sa_data/batch100_1e-1_1e-8_2000_100_alpha1/state_{len(self.acc5_history)}.pth')
        return self.energy_value
        
from quantizers import *
from synchronizer import *

BRAMstate = gen_init_BRAMaware_state(num_layers=12, weight_bits=8, table_input_bits=12, table_output_bits=18, intermediate_bits=24, result_bits=18)
DSPstate = gen_init_nonBRAMaware_state(num_layers=12)

#merge BRAMstate and DSPstate
state = {**BRAMstate, **DSPstate}
opt = OptimizeAckley(initial_state=state, 
                     best_acc=91.5 + 74.5, 
                     model4hls=model4hls, 
                     ref_model=model, 
                     src=images_tensor, 
                     transformer_quant_config=transformer_quant_config,
                     alpha=2)
schedule = {'tmax': 1e-1, 'tmin': 1e-05, 'steps': 2000, 'updates': 100}
opt.set_schedule(schedule)
best_solution, best_energy = opt.anneal()
logging.info("模擬退火優化完成")
torch.save(opt.hls_config, f'{path}/hls_config.pth')
torch.save(opt.transformer_quant_config, f'{path}/transformer_quant_config.pth')
torch.save(opt.state, f'{path}/state.pth')
torch.save(opt.energy_history, f'{path}/energy_history.pth')
torch.save(opt.num_ram_history, f'{path}/num_ram_history.pth')
torch.save(opt.acc1_history, f'{path}/acc1_history.pth')
torch.save(opt.acc5_history, f'{path}/acc5_history.pth')
torch.save(opt.state_history, f'{path}/state_history.pth')
torch.save(opt.acceptance_history, f'{path}/acceptance_history.pth')
torch.save(opt.improvement_history, f'{path}/improvement_history.pth')
