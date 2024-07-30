from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

get_custom_objects().update({'CustomDepthwiseConv2D': CustomDepthwiseConv2D})
