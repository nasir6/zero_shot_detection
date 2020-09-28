from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .cls_model import ClsModel2
__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead', 'DoubleConvFCBBoxHead', 'ClsModel2'
]
