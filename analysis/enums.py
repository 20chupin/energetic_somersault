from enum import Enum
from pathlib import Path


class ResultFolders(Enum):
    ACROBAT = Path(__file__).parent.parent.parent.__str__() + "/msd-somersaults-results/" + "ACROBAT_21-11-22"
    ACROBAT_1_TWIST = Path(__file__).parent.parent.parent.__str__() + "/msd-somersaults-results/" + "ACROBAT_1_TWIST"
    ACROBAT_2_TWIST = Path(__file__).parent.parent.parent.__str__() + "/msd-somersaults-results/" + "ACROBAT_2_TWIST"
    ACROBAT_3_TWIST = Path(__file__).parent.parent.parent.__str__() + "/msd-somersaults-results/" + "ACROBAT_3_TWIST"


