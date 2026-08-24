"""Utilities for computing rewards, returns and scores for game logs."""

from functools import partial
from typing import Callable, Iterable
from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.utils import chunk

import torch


def trimmed_mean_std(
    values: torch.Tensor, trim_fraction: float = 0.025
) -> tuple[float, float]:
    """Estimate mean and standard deviation after symmetric tail trimming.

    The returned parameters are computed after removing the same fraction from
    both sorted tails. The original values remain unchanged and are all
    normalized with these parameters. Small batches retain all values rather
    than removing fewer than one item from each tail.
    """

    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")

    trim_count = int(values.numel() * trim_fraction)
    if trim_count == 0 or 2 * trim_count >= values.numel() - 1:
        retained = values
    else:
        retained = values.sort().values[trim_count:-trim_count]

    return retained.mean().item(), retained.std().item()


def discount(rews: Iterable, discount_factor: float) -> float:
    """Return the discounted sum of rewards."""
    return sum(discount_factor**i * r.reward for i, r in enumerate(rews))


def rescale(history: list, scale: float) -> None:
    """Scale ``current_diff_points`` of each log by ``scale``."""

    for log in history:
        log.current_diff_points *= scale


def set_next(history: list) -> None:
    """Link each log in ``history`` to its successor via ``next`` attribute."""

    for i, log in enumerate(history[:-1]):
        log.next = history[i + 1]


def set_rewards(history: list) -> None:
    """Populate ``reward`` fields from ``current_diff_points`` differences."""

    history[-1].reward = 0
    for i in range(len(history) - 1):
        history[i].reward = (
            history[i + 1].current_diff_points - history[i].current_diff_points
        )


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
    reward_rescale: float | None = None,
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
            if reward_rescale is not None:
                fns.append(partial(rescale, scale=reward_rescale))
            fns.extend([set_next, set_rewards])
            if entropy_reward_scale is not None:
                fns.append(partial(entropy_reward, strength=entropy_reward_scale))
            fns.append(partial(set_returns, discount_factor=discount_factor))
            fns.append(set_score)

            for fn in fns:
                fn(history)


def annotate_with_model(
    model,
    trainset,
    bs,
    gamma,
    gae_lambda,
    value_lambda,
    *,
    use_cached_rollout=False,
):
    with torch.no_grad():

        def is_terminal(sample):
            return sample.terminal

        def is_truncated(sample):
            return sample.truncated

        def eval_states(states):
            out = []
            for batch in chunk(states, bs):
                out.extend(model(batch).unbatched())
            return out

        # The final state of a truncated trace is not itself a training
        # sample, but it is the bootstrap state for the preceding action.
        # Evaluate it alongside the rollout samples when its cached value is
        # unavailable.
        evaluation_samples = list(trainset)
        evaluation_samples.extend(
            sample.next
            for sample in trainset
            if sample.next is not None and is_truncated(sample.next)
        )
        unique_samples = list(
            {id(sample): sample for sample in evaluation_samples}.values()
        )
        if use_cached_rollout:
            missing_samples = [
                s
                for s in unique_samples
                if not all(
                    hasattr(s, key)
                    for key in (
                        "reference_policy",
                        "reference_value",
                        "reference_max_q",
                    )
                )
            ]
        else:
            missing_samples = unique_samples
        preds = eval_states([s.state for s in missing_samples])
        for sample, pv in zip(missing_samples, preds):
            sample.reference_policy = pv.policy[0]
            sample.reference_value = pv.value.mean.item()
            sample.reference_max_q = pv.q_value()[0].max().item()

        for sample in trainset:
            if sample.next is None:
                continue
            if is_terminal(sample.next):
                sample.next.reference_value = 0
                sample.next.reference_max_q = 0
                sample.next.advantage = 0
                sample.next.td_lambda = 0
                sample.next.gae = 0
                sample.next.normalized_gae = 0
            elif is_truncated(sample.next):
                # At a cutoff the game has not ended. The endpoint value is
                # therefore the base case for TD(lambda), while its GAE is
                # zero because there is no action/advantage at the endpoint.
                sample.next.td_lambda = sample.next.reference_value
                sample.next.gae = 0
                sample.next.normalized_gae = 0

        def compute_gae(s):
            if hasattr(s, "gae"):
                return s.gae
            s.gae = s.advantage + gae_lambda * gamma * compute_gae(s.next)
            return s.gae

        def compute_td_lambda(s):
            if hasattr(s, "td_lambda"):
                return s.td_lambda
            s.td_lambda = (
                s.reward
                + gamma * (1 - value_lambda) * s.next_reference_value
                + gamma * value_lambda * compute_td_lambda(s.next)
            )
            return s.td_lambda

        for sample in trainset:
            if sample.next:
                sample.next_reference_value = sample.next.reference_value
                sample.next_reference_max_q = sample.next.reference_max_q
                sample.advantage = (
                    sample.reward
                    + gamma * sample.next_reference_value
                    - sample.reference_value
                )
        for sample in trainset:
            if sample.next:
                compute_td_lambda(sample)
                compute_gae(sample)

        gae_values = torch.tensor([s.gae for s in trainset])
        mean, std = trimmed_mean_std(gae_values)
        for sample in trainset:
            sample.normalized_gae = (sample.gae - mean) / (std + 1e-4)
