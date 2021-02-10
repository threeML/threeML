from dataclasses import dataclass, field
from enum import Enum, Flag, IntEnum
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf

class IntegrateMethod(IntEnum):
    trapz = 0
    quad = 1



@dataclass
class PointSourceDefaults:
    integrate_flux_method: IntegrateMethod = IntegrateMethod.trapz
    max_number_samples: int = 5000
