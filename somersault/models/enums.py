from enum import Enum
from pathlib import Path


class Models(Enum):
    ACROBAT = Path(__file__).parent.__str__() + "/acrobat.bioMod"
