"""Core types for finite Markov decision processes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

type State = int
type Action = int


@dataclass(frozen=True, slots=True)
class Transition:
    """Describe one possible outcome of taking an action.

    Attributes
    ----------
    probability : float
        Probability of this transition occurring.
    next_state : State
        State reached after the transition.
    reward : float
        Immediate reward received for the transition.
    terminated : bool
        Whether the transition ends the episode.
    """

    probability: float
    next_state: State
    reward: float
    terminated: bool


type TransitionModel = Mapping[
    State,
    Mapping[Action, Sequence[Transition]],
]
