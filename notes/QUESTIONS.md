- Some details of functions in [utils\utils.py](..\utils\utils.py) are not fully understood, especially ones pre-processing dataset.
They involve multiple coordinates projection and transformation.

- The maximum number of non-voxels (K) is different across samples, so it is not strightforward to use multiple GPUs in PyTorch.
Because it is hard to pack a batch of samples into a single tensor. (TensorFlow can assign certain samples to a certain GPU.)