from .clevr import CLEVR
from .clevrtex import  CLEVRTEX, collate_fn
from .multi_dsprites import MultiDSprites
from .tetrominoes import Tetrominoes

__all__ = ['CLEVR', 'CLEVRTEX', 'collate_fn', 'Tetrominoes', 'MultiDSprites']
