"""Array types and result models for planning algorithms."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type ValueArray = npt.NDArray[np.float64]
type PolicyArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Result returned by a planning algorithm.

    Attributes
    ----------
    values : ValueArray
        Optimal or policy-specific value for every state.
    policy : PolicyArray
        Deterministic action selected for every state.
    iterations : int
        Number of outer iterations completed before convergence.
    """

    values: ValueArray
    policy: PolicyArray
    iterations: int
