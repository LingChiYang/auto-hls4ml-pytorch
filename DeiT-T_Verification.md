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

