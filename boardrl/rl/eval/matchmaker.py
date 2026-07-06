from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence, List
from boardrl.utils.registerbyname import RegisterByName

from boardrl.rl.eval.selfplay import SelfPlayResults, pit, self_play2

meta_strategies = RegisterByName()


def resolve_strat(s, meta_strategies, state, **kwargs):
    if s in meta_strategies:
        return meta_strategies(
            s,
            meta_strategies=meta_strategies,
            state=state,
            kwargs=kwargs,
        )
    return s


@meta_strategies.register("meta_random")
def meta_random(meta_strategies, candidates, state, kwargs):
    import random

    c = random.choices(
        [c["strat"] for c in candidates],
        weights=[c.get("weight", 1.0) for c in candidates],
    )[0]
    return resolve_strat(c, meta_strategies, state, **kwargs)


@meta_strategies.register("meta_best_opponent")
def meta_best_opponent(meta_strategies, to, state, kwargs):
    winrates = state["matrix"]
    for op, op2 in winrates.items():
        if len(op2) == 0:
            return resolve_strat(op, meta_strategies, state, **kwargs)
    candidates = list(winrates[to].items())
    values = torch.tensor([c[1] for c in candidates])
    values = torch.distributions.Categorical(logits=-40 * values)
    sampled = values.sample((1,))
    new = candidates[sampled.item()][0]
    return resolve_strat(new, meta_strategies, state, **kwargs)


@dataclass
class PairStats:
    """Aggregated head-to-head results for two strategies."""

    wins: float = 0.0
    games: int = 0

    def update(self, result: float) -> None:
        self.games += 1
        self.wins += result

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    def __repr__(self) -> str:
        return str(self.win_rate)


class MatchMaker:
    """Creates strategies, runs games and tracks aggregated outcomes."""

    def __init__(self, game_desc, model_pool, discount_factor: float):
        self._game_desc = game_desc
        self._model_pool = model_pool
        self._discount_factor = discount_factor
        self._win_matrix: MutableMapping[str, MutableMapping[str, PairStats]] = {}

    @property
    def win_matrix(self) -> Mapping[str, Mapping[str, float]]:
        """Return the current win-rate matrix."""

        return {
            outer: {inner: stats.win_rate for inner, stats in inner_map.items()}
            for outer, inner_map in self._win_matrix.items()
        }

    def head_to_head(self, first: str, second: str) -> PairStats:
        """Return cumulative head-to-head stats between two strategies."""

        stats = self._win_matrix.get(first, {}).get(second)
        if stats is None:
            return PairStats()
        return PairStats(wins=stats.wins, games=stats.games)

    def reset_history(self) -> None:
        self._win_matrix.clear()

    def run_self_play(
        self,
        strategy_names: Sequence[str | dict],
        num_games: int,
        max_len: int,
        *,
        rotate: bool = True,
        desc: str = "playing games",
    ) -> SelfPlayResults:
        strategies_names = [
            [
                resolve_strat(
                    s,
                    meta_strategies=meta_strategies,
                    state={"matrix": self.win_matrix},
                )
                for s in strategy_names
            ]
            for _ in range(num_games)
        ]
        results = self_play2(
            self._game_desc.make_game,
            [self._make_strategies(s) for s in strategies_names],
            max_len,
            rotate=rotate,
            desc=desc,
        )
        self._record_outcomes(strategies_names, results)
        return results

    def run_pit(
        self,
        strategy_names: Sequence[str],
        num_games: int,
        max_len: int,
        *,
        rotate: bool = True,
    ) -> SelfPlayResults:
        results = pit(
            self._game_desc.make_game,
            self._make_strategies(strategy_names),
            num_games,
            max_len,
            rotate=rotate,
        )
        return results

    def _make_strategies(self, strategy_names: Sequence[str | dict]):
        return [
            self._game_desc.strategy_from_string(
                strategy,
                model=self._model_pool,
                discount_factor=self._discount_factor,
            )
            for strategy in strategy_names
        ]

    def _record_outcomes(
        self, strategy_names: Sequence[Sequence[str | dict]], results: SelfPlayResults
    ) -> None:
        if not results:
            return

        if len(strategy_names) != len(results):
            strategy_names = [strategy_names for _ in results]

        for s_name, game in zip(strategy_names, results):
            scores: dict[str, float] = {}
            for trace in game.by_strategy:
                name = s_name[trace.strategy_id]
                end_state = trace[-1]
                score = end_state.current_diff_points
                scores[name] = float(score)

            if len(scores) < 2:
                # Nothing to aggregate if only one strategy participated.
                continue

            for name in scores:
                self.ensure_strategy_registered(name)

            names = list(scores)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if a == b:
                        continue

                    result_a, result_b = self._pair_result(scores[a], scores[b])
                    self._update_pair_stats(a, b, result_a, result_b)

    def ensure_strategy_registered(self, name: str) -> None:
        if name not in self._win_matrix:
            self._win_matrix[name] = {}

    def _get_pair_stats(self, first: str, second: str) -> PairStats:
        inner = self._win_matrix.setdefault(first, {})
        stats = inner.get(second)
        if stats is None:
            stats = PairStats()
            inner[second] = stats
        return stats

    def _update_pair_stats(
        self, first: str, second: str, result_first: float, result_second: float
    ) -> None:
        self._get_pair_stats(first, second).update(result_first)
        self._get_pair_stats(second, first).update(result_second)

    @staticmethod
    def _pair_result(score_a: float, score_b: float) -> tuple[float, float]:
        if score_a == score_b:
            return 0.5, 0.5
        if score_a > score_b:
            return 1.0, 0.0
        return 0.0, 1.0
