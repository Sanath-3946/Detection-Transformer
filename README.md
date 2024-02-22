# Detection-Transformer:
A DETR (DEtection Transfomer) model for object detection using PyTorch. DETR is a transformer-based model that directly predicts object bounding boxes and class labels in a single forward pass.  
# Architecture:
The architecture of this project is centered around the DETR (DEtection Transfomer) model, which combines a convolutional backbone (ResNet-50) with a transformer-based object detection framework. The ResNet-50 backbone extracts hierarchical features from input images, while the transformer processes these features to capture contextual relationships and dependencies. The model predicts class probabilities and bounding box coordinates directly in a single forward pass, eliminating the need for anchor boxes and non-maximum suppression. This end-to-end architecture enables efficient and accurate object detection, and the model is pretrained on the COCO dataset, consisting of 91 classes, including a background class.
# Dependencies:  
### Libraries to be installed:
!pip install torch 
!pip install Pillow  
!pip install matplotlib  
!pip install requests  
!pip install numpy  
!pip install torchvision  
# Dataset:COCO
The code uses the COCO (Common Objects in Context) dataset, a widely-used benchmark for object detection tasks. COCO contains a diverse set of images with 80 object categories, making it suitable for training and evaluating models like DETR. The dataset includes images of varying complexity, with multiple instances and object annotations, allowing the model to learn to detect and classify objects in different contexts. The class labels and colors specified in the code correspond to the COCO dataset categories, providing context for visualizing and interpreting the model's predictions.
For more information visit COCO Github Repository:https://github.com/ultralytics/ultralytics.
# BackBone:ResNet50
In this project, the backbone model refers to the ResNet-50 architecture, which serves as the feature extractor for the DETR (DEtection Transfomer) model. The ResNet-50 backbone plays a crucial role in capturing hierarchical features from input images, enabling the transformer to focus on relevant spatial and semantic information for object detection.
# Layers:
The neural network architecture used in this project, called DETR (DEtection Transfomer), comprises several key components:
### 1.ResNet-50 Backbone:
   The ResNet-50 architecture serves as the backbone of the model. It consists of deep residual blocks that allow the efficient training of very deep networks. In this implementation, the last fully connected layer of ResNet-50 is removed, as it is not needed for the object detection task.
### 2.Convolutional Layer (self.backbone.conv1):
   The first convolutional layer of the ResNet-50 backbone (`self.backbone.conv1`) processes the input image to extract low-level features. This layer is responsible for the initial transformation of the input image.
### 3.Batch Normalization (self.backbone.bn1) and ReLU Activation (self.backbone.relu):
   Batch normalization and rectified linear unit (ReLU) activation functions are applied after the initial convolution to normalize and introduce non-linearity to the features.
### 4.Residual Blocks (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4):
   Residual blocks form the bulk of the ResNet-50 architecture. These blocks contain skip connections (identity shortcuts) that help mitigate the vanishing gradient problem during training. The model uses four sets of residual blocks with increasing spatial resolutions in `layer1`, `layer2`, `layer3`, and `layer4`.
### 6.1x1 Convolutional Layer (self.conv):
   A 1x1 convolutional layer is applied to the output of the ResNet-50 backbone (`self.conv`). This layer reduces the channel dimensionality from 2048 to the specified `hidden_dim` (256 in this case).
### 7.Transformer Layer (self.transformer):
   The transformer layer applies the self-attention mechanism to capture long-range dependencies in the feature map. It consists of multiple encoder and decoder layers. The number of encoder and decoder layers is controlled by the `num_encoder_layers` and `num_decoder_layers` parameters, respectively.
### 8.Linear Layers for Prediction:
   The model uses two linear layers for prediction:
      `self.linear_class`: Predicts class probabilities. The output size is `num_classes + 1` to include a background class.
      `self.linear_bbox`: Predicts bounding box coordinates. The output size is 4, representing (x, y, width, height).
### 9.Positional Embeddings (self.query_pos, self.row_embed, self.col_embed):
   Positional embeddings are introduced to provide the transformer with spatial information. These embeddings include query positional embeddings (`self.query_pos`), row embeddings (`self.row_embed`), and column embeddings (`self.col_embed`).
