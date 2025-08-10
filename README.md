# Rural Road Semantic Segmentation 🚗🛣️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Computer%20Vision-green.svg)]()

Advanced semantic segmentation system for autonomous vehicle navigation in rural environments using DeepLabV3 and custom optimization techniques.

> Demonstrated expertise in designing, optimizing, and evaluating state-of-the-art deep learning models for computer vision with strong problem-solving and software engineering skills.

## 🎯 Overview

This project develops a robust **semantic segmentation pipeline** for classifying terrain elements in rural road scenarios. The system processes vehicle-mounted camera images to identify 8 distinct terrain classes essential for autonomous navigation:

- 🌅 **Sky** - Clear overhead areas
- 🛤️ **Smooth Trail** - Paved or well-maintained paths  
- 🪨 **Rough Trail** - Unpaved, rocky terrain
- 🌱 **Traversable Grass** - Safe vegetation areas
- 🌳 **High Vegetation** - Trees and tall bushes
- 🚫 **Non-traversable Low Vegetation** - Obstacle vegetation
- 💧 **Puddle** - Water hazards
- ⚠️ **Obstacle** - Physical barriers

## 🏆 Key Results

- **mIoU: 0.60+** (improved from baseline 0.51)
- **Memory Efficient**: <2.7GB VRAM training, <0.52GB inference
- **Class Imbalance Solved**: Rare classes (puddles, obstacles) improved from 0% to 56%+ IoU
- **Fast Convergence**: 8-10 epochs to optimal performance

🌍 Language Note
----------------

The **main report is in English**, but all **code comments and internal documentation** are written in **Italian**, as the project was developed during a group exam at the **University of Salerno (Italy)**.

Despite this, the **codebase follows international best practices**, with clear method names and class structures that make it easily understandable for global developers and recruiters.

## 🔧 Technical Architecture

### Model Pipeline
```
Input Image (RGB) → DeepLabV3 + ResNet101 → Custom ASPP → Classifier Head → 8-Class Segmentation
```

### Key Innovations
- **Custom ASPP Module**: Multi-scale context with dilations [3, 7, 3]
- **Weighted Loss Function**: Addresses severe class imbalance  
- **Selective Data Augmentation**: Targeted enhancement for rare classes
- **Auxiliary Head**: Improved gradient flow and stability
- **Smart Dropout**: 0.05 probability for noise robustness

## 📊 Performance Analysis

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| **Overall mIoU** | 0.5117 | 0.6026 | +17.8% |
| **Puddle IoU** | 0.0000 | 0.5617 | +561.7% |
| **Obstacle IoU** | ~0.20 | 0.5661 | +183% |
| **Training Time** | - | 8-10 epochs | Fast convergence |
| **Memory Usage** | - | 2.7GB train / 0.52GB inference | Efficient |

### Class-wise Performance
```
Sky:                     0.89 IoU
Smooth Trail:            0.78 IoU  
Traversable Grass:       0.72 IoU
Rough Trail:             0.65 IoU
High Vegetation:         0.61 IoU
Puddle:                  0.56 IoU ⭐ (was 0.00)
Obstacle:                0.57 IoU ⭐ (major improvement)
Non-traversable Veg:     0.19 IoU (challenging class)
```

## 🗃️ Dataset: Yamaha-CMU Off-Road (YCOR) Dataset
This project utilizes the **Yamaha-CMU Off-Road Dataset**, originally introduced by Maturana et al. (2017) in their seminal work *"Real-time Semantic Mapping for Autonomous Off-Road Navigation"*. The dataset consists of 1,076 images collected across four locations in Western Pennsylvania and Ohio, spanning three seasons with 8 semantic classes identical to our classification scheme.

### Comparative Results vs. Original Research

| Model | Architecture | mIoU | Key Strengths | Year |
|-------|-------------|------|---------------|------|
| **Our Model** | **DeepLabV3 + ResNet101** | **0.6026** | **Advanced ASPP, class balancing** | **2025** |
| from the paper | dark-fcn-448 | 0.4982 | Real-time performance | 2017 |
| from the paper | dark-fcn | 0.4854 | Lightweight architecture | 2017 |
| from the paper | cnns-fcn | 0.4666 | VGG-based baseline | 2017 |

### Breakthrough Achievement: Rare Class Detection
**Our major improvement over the original work lies in handling severely underrepresented classes:**

| Class | Original (dark-fcn-448) | Our Model | Improvement |
|-------|------------------------|-----------|-------------|
| **Puddle** | 0.00 IoU | 0.56 IoU | **+56% (∞% improvement)** |
| **Obstacle** | 0.36 IoU | 0.57 IoU | **+58% improvement** |
| Smooth Trail | 0.52 IoU | 0.78 IoU | +50% improvement |
| Rough Trail | 0.40 IoU | 0.65 IoU | +63% improvement |

The original authors noted: *"puddles achieve 0.0 IoU as the network tends to ignore it due to severe class imbalance"*. Our weighted loss and selective augmentation strategies successfully solved this critical limitation.

### Technical Evolution (2017 → 2025)
- **Architecture**: Custom FCN → State-of-the-art DeepLabV3
- **Backbone**: VGG-based → ResNet101 with skip connections
- **Context**: Basic convolutions → Advanced ASPP with dilated convolutions
- **Class Imbalance**: Ignored problem → Sophisticated weighted loss + oversampling
- **Training**: From scratch → Transfer learning with domain adaptation

### Performance Trade-off: Accuracy vs Speed
While our model achieves superior segmentation accuracy, there is a computational trade-off compared to the original lightweight architectures:

| Model | Inference Time | Hardware | Accuracy (mIoU) |
|-------|---------------|----------|-----------------|
| **Our DeepLabV3+ResNet101** | **45 ms** | **Tesla T4** | **0.6026** |
| Original dark-fcn | 21 ms | GT980M | 0.4854 |
| Original cnns-fcn | 37 ms | GT980M | 0.4666 |

Our **49ms inference time on Tesla T4** reflects the complexity of modern segmentation architectures. While the Tesla T4 significantly outperforms the GT980M used in the original work (7.5 TFLOPS vs 2.5 TFLOPS), our DeepLabV3+ResNet101 is computationally more demanding than the lightweight FCN architectures of 2017. This represents the classic **accuracy-speed trade-off**: we achieve **+21% higher mIoU** at the cost of **~2.1x slower inference**, which is acceptable for applications prioritizing segmentation quality over real-time constraints.

### Our Model Performance
| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| **Overall mIoU** | 0.5117 | 0.6026 | +17.8% |
| **Puddle IoU** | 0.0000 | 0.5617 | +561.7% |
| **Obstacle IoU** | ~0.20 | 0.5661 | +183% |
| **Training Time** | - | 8-10 epochs | Fast convergence |
| **Memory Usage** | - | 2.7GB train / 0.52GB inference | Efficient |

### Class-wise Performance
```
Sky:                     0.89 IoU ⭐ (+1% vs original 0.93)
Smooth Trail:            0.78 IoU ⭐ (+49% vs original 0.52)  
Traversable Grass:       0.72 IoU ⭐ (+0% vs original 0.72)
Rough Trail:             0.65 IoU ⭐ (+64% vs original 0.40)
High Vegetation:         0.61 IoU ⚠️ (-26% vs original 0.83)
Puddle:                  0.56 IoU ⭐ (+∞% vs original 0.00)
Obstacle:                0.57 IoU ⭐ (+58% vs original 0.36)
Non-traversable Veg:     0.19 IoU ⚠️ (-23% vs original 0.25)
```

Final note: the authors of the original paper tried to optimize efficiency and speed, while we tried to optimized for accuracy (and in that we succeded). It is considered interesting for future research to compare the inference time of our model with that of the original paper using the same hardware.

## 📁 Project Structure

```
📦 semantic-segmentation-deeplabv3-pytorch/  
│  
├── 📄 docs/ #  Documentation and reports  
│ ├── 1_main_report_ENGLISH.pdf  
│ ├── 2_experiments_list_ITALIAN.pdf  
│ ├── 3_experiments_tree_ITALIAN.png  
│ └── 4_presentation_ITALIAN.pdf  
│  
├── 📓 notebooks/
│ ├── 1_training_validation_split_protocol.ipynb  
│ ├── 2_training_script.ipynb  
│ └── 3_testing_script.ipynb  
│  
├── 🎯 pretrained models/  
│ └── best_model_pretrained_weights_deeplabv3.pth  
│  
├── LICENSE  
├── README.md 
└── requirements.txt
```

## 🎯 Problem-Solving Approach

### Challenge 1: Severe Class Imbalance
- **Problem**: Puddles (0% IoU), obstacles poorly detected
- **Solution**: Weighted CrossEntropy + selective oversampling  
- **Result**: Puddles improved to 56% IoU

### Challenge 2: Class Confusion (Rough vs Smooth Trail)
- **Problem**: Visually similar terrain types misclassified
- **Solution**: Texture-focused data augmentation
- **Result**: Better discrimination between trail types

### Challenge 3: Small Object Detection
- **Problem**: Missing small obstacles and puddles
- **Solution**: Custom ASPP with optimized dilation rates
- **Result**: Multi-scale context capture improved

### Challenge 4: Dataset Quality
- **Problem**: Incorrect labels in training data
- **Solution**: Manual filtering + dropout regularization
- **Result**: More robust learning despite noisy labels

## 🔬 Experimental Methodology

### Architecture Comparison
Tested multiple state-of-the-art segmentation networks:

| Architecture | Backbone | Dataset Pretraining | mIoU | Memory |
|-------------|-----------|-------------------|------|--------|
| **DeepLabV3** | **ResNet101** | **COCO** | **0.6026** | **2.7GB** |
| DeepLabV3 | ResNet50 | COCO | 0.5070 | 1.54GB |
| DeepLabV3 | MobileNetV3 | COCO | 0.5000 | <1GB |
| DeepLabV3+ | ResNet101 | Cityscapes | 0.5518 | 3.2GB |
| BiSeNetV2 | - | RUGD | 0.49 | 1.8GB |
| BiSeNetV2 | - | Rellis3D | 0.41 | 1.8GB |

### Validation Strategy
- **Stratified Split**: Ensures rare classes in both train/val
- **Memory Constraints**: <5GB training, <4GB inference
- **Metric Focus**: IoU over accuracy (more meaningful for segmentation)

## 📈 Future Improvements

- [ ] **Label Quality Enhancement**: Manual/automatic label correction
- [ ] **Domain-Specific Pretraining**: Train on Cityscapes/RUGD datasets  
- [ ] **Pseudo-Labeling**: Leverage unlabeled rural road data
- [ ] **Real-Time Optimization**: Model quantization and pruning
- [ ] **Multi-Modal Input**: Incorporate LiDAR/depth information

## 🛡️ Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: Compatible GPU recommended
- **RAM**: 8GB+ recommended
- **Storage**: 5GB for dataset + models

## 📋 Dependencies

Key libraries used:
```
torch>=1.9.0
torchvision>=0.10.0  
albumentations>=1.0.0
iterstrat>=0.1.2
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.21.0
Pillow>=8.0.0
tqdm>=4.60.0
```

👥 Team 6 – University of Salerno
---------------------------------

* [@francescopiocirillo](https://github.com/francescopiocirillo)
    
* [@alefaso-lucky](https://github.com/alefaso-lucky)    

* * *

📬 Contacts
-----------

✉️ Got feedback or want to contribute? Feel free to open an Issue or submit a Pull Request!

* * *

📈 SEO Tags
-----------

```
deep-learning, computer-vision, semantic-segmentation, pytorch, machine-learning, artificial-intelligence, deeplabv3, resnet, autonomous-vehicles, image-segmentation, neural-networks, convolutional-neural-networks, cnn, transfer-learning, data-augmentation, class-imbalance, model-optimization, terrain-classification, road-segmentation, vehicle-navigation, resnet101, resnet50, mobilenetv3, aspp, atrous-convolution, backbone-networks, encoder-decoder, auxiliary-head, dropout-regularization, weighted-loss
```

* * *

📄 License
----------

This project is licensed under the **MIT License**, a permissive open-source license that allows anyone to use, modify, and distribute the software freely, as long as credit is given and the original license is included.

> In plain terms: **use it, build on it, just don’t blame us if something breaks**.

> ⭐ Like what you see? Consider giving the project a star!

* * *
