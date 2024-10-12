## 下載官方Distilled DeiT-T並處理數據
  ```python
  from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
  from PIL import Image
  import requests
  import torch
  import os
  feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
  model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
  img_folder_path = 'to/your/image/path'

  label_mapping = {}
  with open(os.path.join(img_folder_path, 'file_label_mapping.txt'), 'r') as f:
      for line in f:
          filename, label = line.strip().split(",")
          label_mapping[filename] = int(label)

  image_tensors = []
  true_labels = []
  for filename in os.listdir(img_folder_path):
      if filename.endswith('.JPEG'):
          image_path = os.path.join(img_folder_path, filename)
          image = Image.open(image_path).convert('RGB')
          inputs = feature_extractor(images=image, return_tensors="pt")
          image_tensors.append(inputs['pixel_values'])
          true_labels.append(label_mapping[filename])

  images_tensor = torch.cat(image_tensors, dim=0)
  true_labels = torch.tensor(true_labels)
  ```
## 建立`model4hls`使得HLS4ML可以識別，並轉移DeiT-T `model`權重至`model4hls`
  ```python
    from torch import nn
    import torch
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

torch.manual_seed(0)
model4hls = Transformer4HLS(d_model=192, 
                            nhead=3, 
                            num_encoder_layers=12, 
                            dim_feedforward=768, 
                            dropout=0, 
                            activation='gelu', 
                            norm_first=True, 
                            device='cpu')

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
```
## 生成`transformer_quant_config`量化transformer encoder，並calibrate quantized model觀察quantizer的數值範圍並重新生成calibated `transformer_quant_config`
#### Tips : 由於calibration可能會很久(取決於使用多大的calibation dataset)，建議將calibrated `transformer_quant_config`存檔
  ```python
  from quantizers import *
  from synchronizer import *
  import hls4ml
  import json
  import copy
  def load_transformer_quant_config(quant_config_path: str = "./quant_config.json",
                                    norm_quant_config_path: str = "./norm_quant_config.json",
                                    num_layers: int = 1) -> dict:
      with open(quant_config_path, 'r') as f:
          quant_config = json.load(f)
      with open(norm_quant_config_path, 'r') as f:
          norm_quant_config = json.load(f)
      transformer_quant_config = {}
      for i in range(num_layers):
          transformer_quant_config[i] = copy.deepcopy(quant_config)
      transformer_quant_config['norm'] = copy.deepcopy(norm_quant_config)
      return transformer_quant_config


  transformer_quant_config = load_transformer_quant_config(num_layers=12)
  qmodel = QTransformerEncoder([QTransformerEncoderLayer(192, 
                                                         3, 
                                                         768, 
                                                         activation='gelu', 
                                                         quant_config=transformer_quant_config[i], 
                                                         calibration=True, 
                                                         device='cpu') for i in range(12)], 
                               12, 
                               QLayerNorm(192, quant_config=transformer_quant_config['norm'], calibration=True, device='cpu'),
                               TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=True),
                               dtype=torch.float64)
  qmodel.transfer_weights(model4hls)
  qmodel.to(torch.device('cpu'))
  qmodel.eval()

  with torch.no_grad():
      embbed_out = model.deit.embeddings(images_tensor)
      transformer_quant_config = calibrate_transformer(qmodel, transformer_quant_config, embbed_out[0:1,:,:].permute(1, 0, 2).type(torch.float64).to(torch.device('cpu')))
  ```
## 生成`state` for Simulated Annealing(若沒有要透過Simulated Annealing優化，這邊只是作為同步`quant_config`和`hls_config`的方法)並測試sync_quant_config
- `state`包含影響BRAM數目的變數`BRAMstate`以及不影響BRAM數目的變數`DSPstate`(或者說影響DSP數目的變數，但目前並沒有 ***TODO : 將DSP相關變數加入Design Search Space***)
- `num_layers`為Transformer Block的數量
- `weight_bits`主要包含MHSA的兩個linear的weight(或者是Q、K、V的weight以及O的weight)、FFN的兩個linear layer的weight的bit-wdith
- `table_input_bits`和`table_output_bits`包含，MHSA的exponential、倒數查表、LayerNorm的variance查表、FFN的GeLU(CDF)查表。
  - 2的`table_input_bits`次方即為Look-up table的Entry數量，因此這個數值只會設置約12上下
  - `table_output_bits`即為Look-up table的width。由於BRAM的配置18 bits或9 bits的使用效率最高，因此這邊通常只會是這兩個數值或其倍數
- `intermediate_bits`包含MHSA中的QKV cache，由於對Deit-tiny來說，QKV所需緩存很大，因此使用UltraRAM實現，而UltraRAM使用72 bits = 24 bits* 3 heads最有效率，並不將此列入BRAM計算(***TODO : KV cache存至HBM***)
- `result_bits`包含所有layer的output，使用FIFO實現，由於選取適當的Tile size可減小FIFO深度，所以使用LUTRAM實現並不列入BRAM計算(***TODO : Formulize FIFO深度與Tile size的關係以估計LUTRAM數量***)
  ```python
  BRAMstate = gen_init_BRAMaware_state(num_layers=12, 
                                     weight_bits=32, 
                                     table_input_bits=32, 
                                     table_output_bits=32, 
                                     intermediate_bits=32, 
                                     result_bits=32)
  DSPstate = gen_init_nonBRAMaware_state(num_layers=12)
  state = {**BRAMstate, **DSPstate}

  config = hls4ml.utils.config_from_pytorch_model(model4hls, 
                                                granularity='name',
                                                backend='Vitis',
                                                input_shapes=[[1, 198, 192]], 
                                                default_precision='ap_fixed<18,5,AP_RND_CONV,AP_SAT>', 
                                                inputs_channel_last=True, 
                                                transpose_outputs=False)
  
  valid = sync_quant_config(transformer_quant_config, config, state)
  ```
## 建立quantize model `qmodel` 並載入calibared和sync up後的 `transformer_quant_config`。配置HLS config中的Tile size以最大化BRAM以及硬體使用效率並產生 `hls_model` 和HLS project
  ```python
  for layer_config in config['LayerName'].keys():
      if layer_config.endswith('self_attn'):
          config['LayerName'][layer_config]['TilingFactor'] = [1,1,1]
      elif layer_config.endswith('ffn'):
          config['LayerName'][layer_config]['TilingFactor'] = [1,1,12]
  hls_model = hls4ml.converters.convert_from_pytorch_model(
                                                              model4hls,
                                                              [[1, 198, 192]],
                                                              output_dir='./hls/deit_tiny_w8_Bdk-1_Bffn-12',
                                                              project_name='myproject',
                                                              backend='Vitis',
                                                              part='xcu55c-fsvh2892-2L-e',
                                                              #board='alveo-u55c',
                                                              hls_config=config,
                                                              io_type='io_tile_stream',
                                                          )
  hls_model.compile()
  ```
## 比較`qmodel` 、 `hls_model`和`model4hls`的輸出。理論上，`qmodel` 和 `hls_model`的輸入要一致
  ```python
  with torch.no_grad():
      embbed_out = model.deit.embeddings(images_tensor[0:1])
      encoder_out2 = model4hls(embbed_out)
      output = qmodel(embbed_out.permute(1, 0, 2).type(torch.float64))
      hls_output = hls_model.predict(embbed_out.numpy())
      print(output)
      print(encoder_out2)
      print(hls_output)
  ```

## 預測`hls_model`以及產生的HLS project的BRAM數目，這會`state`的配置與tile size的大小有關
  ```python
    from estimators import *
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
  num_ram = layer_estimater(parse_hls_model(hls_model))
  print(num_ram)
  ```
