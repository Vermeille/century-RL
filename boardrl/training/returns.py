"""Utilities for computing rewards, returns and scores for game logs."""

from functools import partial
from typing import Callable, Iterable
from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.training.sample import TrainingSample
from boardrl.utils import chunk

import torch


def discount(rews: Iterable, discount_factor: float) -> float:
    """Return the discounted sum of rewards."""
    return sum(discount_factor**i * r.reward for i, r in enumerate(rews))


def set_next(history: list) -> None:
    """Link each log in ``history`` to its successor via ``next`` attribute."""

    for i, log in enumerate(history[:-1]):
        log.next = history[i + 1]


def set_rewards(history: list, scale: float = 1.0) -> None:
    """Populate scaled point-delta rewards without modifying recorded points."""

    history[-1].reward = 0
    for i in range(len(history) - 1):
        history[i].reward = scale * (
            history[i + 1].current_diff_points - history[i].current_diff_points
        )


def set_episodic_rewards(history: list) -> None:
    """Populate a terminal outcome reward without changing point scores."""

    for log in history:
        log.reward = 0.0
    if history[-1].terminal and len(history) > 1:
        history[-2].reward = history[-1].episodic_utility


def entropy_reward(history: list, strength: float) -> None:
    """Add an entropy bonus to each step in ``history``.

    The bonus encourages exploration by penalising low-entropy action
    distributions.
    """

    for log in history[:-1]:
        log.reward += (
            strength
            * -torch.log_softmax(log.action_distribution, dim=0)[log.action_idx]
        )


def set_returns(history: list, discount_factor: float) -> None:
    """Compute discounted returns for each log.

    Properly terminal histories receive ordinary discounted returns. A
    truncated history is deliberately left without a terminal return: its
    endpoint must be bootstrapped from a value estimate by
    :func:`annotate_with_model`.
    """

    if history[-1].terminal:
        for i in range(len(history) - 1):
            history[i].returns = discount(history[i:], discount_factor)
    else:
        for i in range(len(history) - 1):
            history[i].returns = float("nan")


def set_score(history: list) -> None:
    """Attach the final score to each log if the game terminated."""

    if history[-1].terminal:
        for log in history:
            log.score = history[-1].current_diff_points
    else:
        for log in history:
            log.score = float("nan")


def compute_returns(
    games: SelfPlayResults,
    discount_factor: float,
    *,
    entropy_reward_scale: float | None = None,
    reward_fn: Callable[[list], None] = set_rewards,
) -> None:
    """Compute rewards, returns and scores for a batch of games.

    The function mutates the histories in ``games`` in-place. Steps are applied
    declaratively via a list of helper functions that operate on each history.
    """

    for game in games:
        for history in game:
            if len(history) == 0:
                continue

            fns: list[Callable[[list], None]] = []
            fns.extend([set_next, reward_fn])
            if entropy_reward_scale is not None:
                fns.append(partial(entropy_reward, strength=entropy_reward_scale))
            fns.append(partial(set_returns, discount_factor=discount_factor))
            fns.append(set_score)

            for fn in fns:
                fn(history)


def annotate_with_model(
    model,
    trainset: list[TrainingSample],
    bs,
    gamma,
    gae_lambda,
    value_lambda,
    *,
    use_cached_rollout=False,
):
    with torch.no_grad():

        def eval_states(states):
            out = []
            for batch in chunk(states, bs):
                out.extend(model(batch).unbatched())
            return out

        if use_cached_rollout:
            missing_samples = [
                sample
                for sample in trainset
                if any(
                    value is None
                    for value in (
                        sample.reference_policy,
                        sample.reference_value,
                        sample.reference_value_stddev,
                        sample.reference_max_q,
                    )
                )
            ]
        else:
            missing_samples = list(trainset)

        preds = eval_states([sample.state for sample in missing_samples])
        for sample, pv in zip(missing_samples, preds):
            sample.reference_policy = pv.policy[0]
            sample.reference_value = pv.value.mean.item()
            sample.reference_value_stddev = pv.value.stddev.item()
            sample.reference_max_q = pv.q_value()[0].max().item()

        truncated = [
            sample
            for sample in trainset
            if sample.next is not None and sample.next.truncated
        ]
        truncated_preds = eval_states([sample.next.state for sample in truncated])
        for sample, pv in zip(truncated, truncated_preds):
            sample.next_reference_value = pv.value.mean.item()
            sample.next_reference_max_q = pv.q_value()[0].max().item()

        for sample in trainset:
            if sample.next is None:
                continue

            if sample.next.terminal:
                sample.next_reference_value = 0.0
                sample.next_reference_max_q = 0.0
            elif sample.next.truncated:
                pass
            else:
                if not isinstance(sample.next, TrainingSample):
                    raise TypeError("non-terminal next state must be a TrainingSample")
                sample.next_reference_value = sample.next.reference_value
                sample.next_reference_max_q = sample.next.reference_max_q

            if sample.reference_value is None or sample.next_reference_value is None:
                raise ValueError("reference values must be populated before advantages")
            if sample.reward is None:
                raise ValueError("reward must be populated before advantages")
            sample.advantage = (
                sample.reward
                + gamma * sample.next_reference_value
                - sample.reference_value
            )

        def compute_gae(sample: TrainingSample):
            if sample.gae is not None:
                return sample.gae
            if sample.advantage is None:
                raise ValueError("advantage must be populated before GAE")

            if (
                sample.next is None
                or sample.next.terminal
                or sample.next.truncated
            ):
                tail = 0.0
            else:
                if not isinstance(sample.next, TrainingSample):
                    raise TypeError("non-terminal next state must be a TrainingSample")
                tail = compute_gae(sample.next)
            sample.gae = sample.advantage + gae_lambda * gamma * tail
            return sample.gae

        def compute_td_lambda(sample: TrainingSample):
            if sample.td_lambda is not None:
                return sample.td_lambda
            if sample.reward is None or sample.next_reference_value is None:
                raise ValueError("reward and next value must be populated before TD(lambda)")

            if sample.next is None or sample.next.terminal:
                tail = 0.0
            elif sample.next.truncated:
                tail = sample.next_reference_value
            else:
                if not isinstance(sample.next, TrainingSample):
                    raise TypeError("non-terminal next state must be a TrainingSample")
                tail = compute_td_lambda(sample.next)

            sample.td_lambda = (
                sample.reward
                + gamma * (1 - value_lambda) * sample.next_reference_value
                + gamma * value_lambda * tail
            )
            return sample.td_lambda

        for sample in trainset:
            if sample.next is not None:
                compute_td_lambda(sample)
                compute_gae(sample)

        gae_values = torch.tensor([sample.gae for sample in trainset])
        # Use the complete on-policy batch to keep the normalized advantages
        # at unit scale. Estimating the scale after trimming the tails amplified
        # rare wins against strong opponents instead of making them robust.
        mean = gae_values.mean().item()
        std = gae_values.std().item()
        for sample in trainset:
            sample.normalized_gae = (sample.gae - mean) / (std + 1e-4)
