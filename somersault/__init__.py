"""
This package contains the code to run the optimal control problem of the somersault.
"""
VERSION = "0.1.0"

from .ocp.miller_ocp import MillerOcp as MillerOCP
from .ocp.miller_ocp_one_phase import MillerOcpOnePhase
from .models.enums import Models
