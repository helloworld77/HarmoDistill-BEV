from .cwd import ChannelWiseDivergence
from .l2loss import L2Loss 
from .fgd import FeatureLoss
from .bkl import BalanceKLDivergence

__all__ = [
    'L2Loss',
    'ChannelWiseDivergence',
    'FeatureLoss',
    'BalanceKLDivergence'
]
