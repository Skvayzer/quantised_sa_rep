from .clevr import CLEVR
from .clevrtex import  CLEVRTEX, CLEVRTEX_Evaluator, collate_fn
from .multi_dsprites import MultiDSprites
from .tetrominoes import Tetrominoes

__all__ = ['CLEVR', 'CLEVRTEX', 'CLEVRTEX_Evaluator', 'collate_fn', 'Tetrominoes', 'MultiDSprites']
