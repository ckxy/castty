from .bambusa import Bambusa

from .misc import *

from .colander import *
from .color import *
from .convert import *
from .crop import *
from .erase import *
from .filp import *
from .image import *
from .label import *
from .pad import *
from .register import *
from .resize import *
from .tag import *
from .union import *
from .warp import *


class Bamboo(Bambusa):
    def __init__(self, internodes):
        self.internodes = []
        for k, v in internodes:
            self.internodes.append(eval(k)(**v))

    def __repr__(self):
        return 'Bamboo' + super(Bamboo, self).__repr__()

    def rper(self):
        return 'Oobmab' + super(Bamboo, self).rper()
