from .clevr import CLEVR
from .clevrtex import  CLEVRTEX, collate_fn
from .multi_dsprites import MultiDSprites
from .tetrominoes import Tetrominoes
from .clevrmirror import CLEVR_Mirror

__all__ = ['CLEVR', 'CLEVRTEX', 'CLEVR_Mirror', 'collate_fn', 'Tetrominoes', 'MultiDSprites']
