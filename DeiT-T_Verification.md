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
