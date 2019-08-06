# Group-Depthwise-Conv3D-for-Keras
Group Conv3D in Keras, also can be used as Depthwise Conv3D.

group_multiplier: The number of convolution output channels for each group.
            The total number of output channels will be equal to `group_num * group_multiplier`.

group_size: default 1, means depthwise Conv; bigger than 1, means group Conv; input channel num should be
    integral multiple of group_size.
backend only suport tensorflow.

*Base code for the implementation is used from: https://github.com/alexandrosstergiou/keras-DepthwiseConv3D



## Usage
```python
from .Group_Depthwise_Conv3D import DepthwiseConv3D, GroupConv3D

x = DepthwiseConv3D(kernel_size=(3, 3, 3), name='depthwiseConv3d_1')(x) # DepthwiseConv3D
x = GroupConv3D(kernel_size=(1, 1, 1), group_size=256, group_multiplier=32, name='grou_conv3d_1')(x) # GroupConv3D
...
```
