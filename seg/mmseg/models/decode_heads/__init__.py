# Copyright (c) OpenMMLab. All rights reserved.

from .fpn_head import *
from .fcn_head import *

__all__ = [
    'FPNHead', 'FPNHead_SNN', 'QFPNHead',
    'FCNHead_SNN'
]
