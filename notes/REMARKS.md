# Sparse Tensor Representation
- Use `torch.sparse.FloatTensor` to replace `tf.scatter_nd` in [model\group_pointcloud.py](..\model\group_pointcloud.py)
- A toy example for illustration:
```
size = torch.Size([2, 3, 3, 3, 5])    # similary to '[batch_size, DEPTH, HEIGHT, WIDTH, 128]'
src = torch.Tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]) # similary to 'voxelwise'
index = torch.Tensor([[0, 0, 1, 2], [0, 1, 1, 1], [1, 2, 1, 0]])    # similary to 'coordinate'
dst = torch.sparse.FloatTensor(index.t(), src, size)    # similary to 'outputs'
```

# Deconv2D
- According to the formula of [`ConvTranspose2d`](https://pytorch.org/docs/0.4.1/nn.html?highlight=convtranspose2d#torch.nn.ConvTranspose2d) in PyTorch,
H_out=(H_in−1)×stride\[0\]−2×padding\[0\]+kernel_size\[0\], if the first Deconv2D is set to (128, 256, 3, 1, 0), we cannot keep the spatial size unchanged.
- In [model\rpn.py](..\model\rpn.py), change the first Deconv2D to (128, 256, 3, 1, 1).

# Gradient Clip
- In the TensorFlow implementation, it contains many lines to clip gradients in the file [model/model.py](https://github.com/qianguih/voxelnet/blob/b74823daa328fc2fa99452bf79793e1f3c32c72a/model/model.py#L98).
In PyTorch implementation [train.py](../train.py), only use a single function `'clip_grad_norm_'`.

# Non-maximum Suppression
- In the TensorFlow implementation, it uses Tensorflow API `'tf.image.non_max_suppression'` to conduct non-maximum suppression [model/model.py](https://github.com/qianguih/voxelnet/blob/b74823daa328fc2fa99452bf79793e1f3c32c72a/model/model.py#L136).
PyTorch does not provide such API.
- In PyTorch implementation, an third-party open source [NMS](https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185) code is adpoted. A potential problem is the number of candidate boxes is zero.
- An alternative option is a C language based code which can be found [here](https://github.com/multimodallearning/pytorch-mask-rcnn/tree/master/nms). It needs compiling but may be more faster.
