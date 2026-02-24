from dataclasses import dataclass
from enum import IntEnum


class IntegrateMethod(IntEnum):
    trapz = 0
    quad = 1


@dataclass
class PointSourceDefaults:
    integrate_flux_method: IntegrateMethod = IntegrateMethod.trapz
    max_number_samples: int = 5000
