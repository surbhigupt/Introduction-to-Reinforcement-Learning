"""Dynamic-programming algorithms for finite Markov decision processes."""

from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from numbers import Integral, Real

import numpy as np

from .models import PlanningResult, PolicyArray, ValueArray
from .types import Transition, TransitionModel

_DEFAULT_TOLERANCE = 1e-10
_DEFAULT_MAX_ITERATIONS = 100_000
_PROBABILITY_ABS_TOLERANCE = 1e-12
_PROBABILITY_REL_TOLERANCE = 1e-9


def policy_evaluation(
    policy: PolicyArray,
    transition_model: TransitionModel,
    gamma: float = 1.0,
    theta: float = _DEFAULT_TOLERANCE,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> ValueArray:
    """Evaluate a deterministic policy with iterative Bellman updates.

    Parameters
    ----------
    policy : PolicyArray
        One-dimensional array mapping each state to an action.
    transition_model : TransitionModel
        Finite transition model indexed by contiguous states and actions.
    gamma : float, default=1.0
        Discount factor in the closed interval ``[0, 1]``.
    theta : float, default=1e-10
        Positive convergence tolerance for the maximum value change.
    max_iterations : int, default=100000
        Maximum number of Bellman updates before raising ``RuntimeError``.

    Returns
    -------
    ValueArray
        State values for ``policy``.

    Raises
    ------
    TypeError
        If the transition model or an array has an invalid type.
    ValueError
        If an argument has an invalid shape, range, or value.
    RuntimeError
        If evaluation does not converge within ``max_iterations``.
    """

    num_states, num_actions = _validate_transition_model(transition_model)
    validated_policy = _validate_policy(policy, num_states, num_actions)
    _validate_algorithm_parameters(gamma, theta, max_iterations)

    values, _ = _evaluate_policy(
        validated_policy,
        transition_model,
        gamma,
        theta,
        max_iterations,
    )
    return values


def policy_improvement(
    values: ValueArray,
    transition_model: TransitionModel,
    gamma: float = 1.0,
) -> PolicyArray:
    """Construct a greedy deterministic policy from state values.

    Parameters
    ----------
    values : ValueArray
        One-dimensional array containing one value per state.
    transition_model : TransitionModel
        Finite transition model indexed by contiguous states and actions.
    gamma : float, default=1.0
        Discount factor in the closed interval ``[0, 1]``.

    Returns
    -------
    PolicyArray
        Greedy action for every state. Ties select the lowest action index.

    Raises
    ------
    TypeError
        If the transition model or an array has an invalid type.
    ValueError
        If an argument has an invalid shape, range, or value.
    """

    num_states, num_actions = _validate_transition_model(transition_model)
    validated_values = _validate_values(values, num_states)
    _validate_gamma(gamma)

    action_values = _calculate_action_values(
        validated_values,
        transition_model,
        gamma,
        num_states,
        num_actions,
    )
    return np.argmax(action_values, axis=1).astype(np.int64, copy=False)


def policy_iteration(
    transition_model: TransitionModel,
    gamma: float = 1.0,
    theta: float = _DEFAULT_TOLERANCE,
    *,
    initial_policy: PolicyArray | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    evaluation_max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> PlanningResult:
    """Find an optimal policy by alternating evaluation and improvement.

    Parameters
    ----------
    transition_model : TransitionModel
        Finite transition model indexed by contiguous states and actions.
    gamma : float, default=1.0
        Discount factor in the closed interval ``[0, 1]``.
    theta : float, default=1e-10
        Positive convergence tolerance used during policy evaluation.
    initial_policy : PolicyArray | None, default=None
        Optional deterministic starting policy. Defaults to action zero in
        every state.
    max_iterations : int, default=100000
        Maximum number of policy-improvement iterations.
    evaluation_max_iterations : int, default=100000
        Maximum Bellman updates allowed for each policy evaluation.

    Returns
    -------
    PlanningResult
        Converged values, optimal deterministic policy, and iteration count.

    Raises
    ------
    TypeError
        If the transition model or an array has an invalid type.
    ValueError
        If an argument has an invalid shape, range, or value.
    RuntimeError
        If evaluation or policy iteration exceeds its iteration limit.
    """

    num_states, num_actions = _validate_transition_model(transition_model)
    _validate_algorithm_parameters(gamma, theta, max_iterations)
    _validate_max_iterations(
        evaluation_max_iterations,
        "evaluation_max_iterations",
    )

    if initial_policy is None:
        policy = np.zeros(num_states, dtype=np.int64)
    else:
        policy = _validate_policy(
            initial_policy,
            num_states,
            num_actions,
        ).copy()

    for iteration in range(1, max_iterations + 1):
        values, _ = _evaluate_policy(
            policy,
            transition_model,
            gamma,
            theta,
            evaluation_max_iterations,
        )
        action_values = _calculate_action_values(
            values,
            transition_model,
            gamma,
            num_states,
            num_actions,
        )
        improved_policy = np.argmax(action_values, axis=1).astype(
            np.int64,
            copy=False,
        )

        if np.array_equal(policy, improved_policy):
            return PlanningResult(values, improved_policy, iteration)

        policy = improved_policy

    raise RuntimeError(
        "policy iteration did not converge within "
        f"{max_iterations} iterations"
    )


def value_iteration(
    transition_model: TransitionModel,
    gamma: float = 1.0,
    theta: float = _DEFAULT_TOLERANCE,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> PlanningResult:
    """Find optimal state values and a greedy deterministic policy.

    Parameters
    ----------
    transition_model : TransitionModel
        Finite transition model indexed by contiguous states and actions.
    gamma : float, default=1.0
        Discount factor in the closed interval ``[0, 1]``.
    theta : float, default=1e-10
        Positive convergence tolerance for the maximum value change.
    max_iterations : int, default=100000
        Maximum number of Bellman optimality updates.

    Returns
    -------
    PlanningResult
        Optimal values, greedy deterministic policy, and iteration count.

    Raises
    ------
    TypeError
        If the transition model has an invalid type.
    ValueError
        If an argument has an invalid range or value.
    RuntimeError
        If value iteration exceeds ``max_iterations``.
    """

    num_states, num_actions = _validate_transition_model(transition_model)
    _validate_algorithm_parameters(gamma, theta, max_iterations)

    values = np.zeros(num_states, dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        action_values = _calculate_action_values(
            values,
            transition_model,
            gamma,
            num_states,
            num_actions,
        )
        updated_values = np.max(action_values, axis=1)

        if np.max(np.abs(updated_values - values)) < theta:
            final_action_values = _calculate_action_values(
                updated_values,
                transition_model,
                gamma,
                num_states,
                num_actions,
            )
            policy = np.argmax(final_action_values, axis=1).astype(
                np.int64,
                copy=False,
            )
            return PlanningResult(updated_values, policy, iteration)

        values = updated_values

    raise RuntimeError(
        "value iteration did not converge within "
        f"{max_iterations} iterations"
    )


def _evaluate_policy(
    policy: PolicyArray,
    transition_model: TransitionModel,
    gamma: float,
    theta: float,
    max_iterations: int,
) -> tuple[ValueArray, int]:
    values = np.zeros(policy.size, dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        updated_values = np.zeros_like(values)

        for state, action in enumerate(policy):
            for transition in transition_model[state][int(action)]:
                continuation = 0.0
                if not transition.terminated:
                    continuation = gamma * values[transition.next_state]
                updated_values[state] += transition.probability * (
                    transition.reward + continuation
                )

        if np.max(np.abs(updated_values - values)) < theta:
            return updated_values, iteration

        values = updated_values

    raise RuntimeError(
        "policy evaluation did not converge within "
        f"{max_iterations} iterations"
    )


def _calculate_action_values(
    values: ValueArray,
    transition_model: TransitionModel,
    gamma: float,
    num_states: int,
    num_actions: int,
) -> ValueArray:
    action_values = np.zeros(
        (num_states, num_actions),
        dtype=np.float64,
    )

    for state in range(num_states):
        for action in range(num_actions):
            for transition in transition_model[state][action]:
                continuation = 0.0
                if not transition.terminated:
                    continuation = gamma * values[transition.next_state]
                action_values[state, action] += transition.probability * (
                    transition.reward + continuation
                )

    return action_values


def _validate_transition_model(
    transition_model: TransitionModel,
) -> tuple[int, int]:
    if not isinstance(transition_model, Mapping):
        raise TypeError("transition_model must be a mapping")
    if not transition_model:
        raise ValueError("transition_model must contain at least one state")

    num_states = len(transition_model)
    expected_states = set(range(num_states))
    if any(type(state) is not int for state in transition_model):
        raise TypeError("transition_model state keys must be integers")
    if set(transition_model) != expected_states:
        raise ValueError(
            "transition_model state keys must be contiguous integers "
            "starting at zero"
        )

    first_actions = transition_model[0]
    if not isinstance(first_actions, Mapping):
        raise TypeError("actions for state 0 must be a mapping")
    if not first_actions:
        raise ValueError("every state must contain at least one action")

    num_actions = len(first_actions)
    expected_actions = set(range(num_actions))

    for state in range(num_states):
        actions = transition_model[state]
        if not isinstance(actions, Mapping):
            raise TypeError(f"actions for state {state} must be a mapping")
        if any(type(action) is not int for action in actions):
            raise TypeError("transition_model action keys must be integers")
        if set(actions) != expected_actions:
            raise ValueError(
                "action keys for every state must be identical contiguous "
                "integers starting at zero"
            )

        for action in range(num_actions):
            transitions = actions[action]
            if not isinstance(transitions, Sequence) or isinstance(
                transitions,
                (str, bytes),
            ):
                raise TypeError("transitions must be a sequence")
            if not transitions:
                raise ValueError(
                    f"state {state}, action {action} has no transitions"
                )

            probability_sum = 0.0
            for transition in transitions:
                if not isinstance(transition, Transition):
                    raise TypeError(
                        "transition outcomes must be Transition instances"
                    )
                probability = transition.probability
                if isinstance(probability, bool) or not isinstance(
                    probability,
                    Real,
                ):
                    raise TypeError(
                        "transition probabilities must be real numbers"
                    )
                if not isfinite(probability) or probability < 0.0:
                    raise ValueError(
                        "transition probabilities must be finite and "
                        "non-negative"
                    )
                probability_sum += probability

                if (
                    isinstance(transition.next_state, bool)
                    or not isinstance(transition.next_state, Integral)
                ):
                    raise TypeError("transition.next_state must be an integer")
                if not 0 <= transition.next_state < num_states:
                    raise ValueError(
                        f"invalid next state {transition.next_state} for "
                        f"state {state}, action {action}"
                    )
                if isinstance(transition.reward, bool) or not isinstance(
                    transition.reward,
                    Real,
                ):
                    raise TypeError("transition rewards must be real numbers")
                if not isfinite(transition.reward):
                    raise ValueError("transition rewards must be finite")
                if not isinstance(transition.terminated, bool):
                    raise TypeError("transition.terminated must be a bool")

            if not isclose(
                probability_sum,
                1.0,
                rel_tol=_PROBABILITY_REL_TOLERANCE,
                abs_tol=_PROBABILITY_ABS_TOLERANCE,
            ):
                raise ValueError(
                    f"transition probabilities for state {state}, action "
                    f"{action} must sum to 1.0; got {probability_sum}"
                )

    return num_states, num_actions


def _validate_policy(
    policy: PolicyArray,
    num_states: int,
    num_actions: int,
) -> PolicyArray:
    if not isinstance(policy, np.ndarray):
        raise TypeError("policy must be a NumPy array")
    if policy.ndim != 1 or policy.shape[0] != num_states:
        raise ValueError(
            f"policy must have shape ({num_states},); got {policy.shape}"
        )
    if not np.issubdtype(policy.dtype, np.integer):
        raise TypeError("policy must contain integer action indices")
    if np.any(policy < 0) or np.any(policy >= num_actions):
        raise ValueError(
            f"policy actions must be in the range [0, {num_actions})"
        )

    return policy.astype(np.int64, copy=False)


def _validate_values(values: ValueArray, num_states: int) -> ValueArray:
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy array")
    if values.ndim != 1 or values.shape[0] != num_states:
        raise ValueError(
            f"values must have shape ({num_states},); got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("values must contain numeric data")
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError("values must contain real numbers")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite numbers")

    return values.astype(np.float64, copy=False)


def _validate_algorithm_parameters(
    gamma: float,
    theta: float,
    max_iterations: int,
) -> None:
    _validate_gamma(gamma)
    if isinstance(theta, bool) or not isinstance(theta, Real):
        raise TypeError("theta must be a real number")
    if not isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be a finite positive number")
    _validate_max_iterations(max_iterations, "max_iterations")


def _validate_gamma(gamma: float) -> None:
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise TypeError("gamma must be a real number")
    if not isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be a finite number in the range [0, 1]")


def _validate_max_iterations(value: int, parameter_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{parameter_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{parameter_name} must be positive")
