# NeRF TensorFlow v2 Keras Re-Implementation
[Work in Progress]

Author: Naufal Suryanto

## Quickstart
### 1. Prepare Dataset
- Download the dataset from NeRF Official [[Here]](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Results
[TBD]

## Implementation and Features

### NeRF Model
- [x] Positional Encoding
- [x] Coarse and Fine Model with Hierarchical Sampling

### Training Supports
- [x] Single or multiple GPU training
- [x] Split the MLP prediction into chunks for fitting GPU memory
- [x] [Default] Use **graph** execution with tf.function for **better performance** (but may take longer initialization and larger memory usage)
- [x] [Optional] Use **eager** execution for **faster initialization and lower memory usage** (include --eagerly option when run the code)
- [x] Black or White Background (include --white_bg option when run the code for better result)
- [x] Log the training history in CSV, plot image, and sample image
- [x] Continue model training from last logging step

## References
### Code Implementation Inspired by
- Official Keras Code Example for NeRF: [[Link]](https://keras.io/examples/vision/nerf)
- Awesome NeRF PyTorch Implementation: [[Link]](https://github.com/kakaobrain/NeRF-Factory)
- NeRF Implementation Tutorial by PyImageSearch: [[Part-1]](https://pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/) [[Part-2]](https://pyimagesearch.com/2021/11/17/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-2/) [[Part-3]](https://pyimagesearch.com/2021/11/24/*computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-3/)


### Original Project Page
- NeRF: [[Project Page]](https://www.matthewtancik.com/nerf) [[Paper]](https://arxiv.org/abs/2003.08934) [[Code]](https://github.com/bmild/nerf)