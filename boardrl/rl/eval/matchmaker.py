from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from boardrl.rl.eval.selfplay import SelfPlayResults, pit, self_play2


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

    def __init__(
        self, game_desc, model_pool, discount_factor: float, *, elo_k: float = 32.0
    ):
        self._game_desc = game_desc
        self._model_pool = model_pool
        self._discount_factor = discount_factor
        self._elo_k = elo_k
        self._win_matrix: MutableMapping[str, MutableMapping[str, PairStats]] = {}
        self._elo: dict[str, float] = {}

    @property
    def win_matrix(self) -> Mapping[str, Mapping[str, float]]:
        """Return the current win-rate matrix."""

        return {
            outer: {inner: stats.win_rate for inner, stats in inner_map.items()}
            for outer, inner_map in self._win_matrix.items()
        }

    @property
    def elo(self) -> Mapping[str, float]:
        """Return the current Elo ratings for each seen strategy."""

        return dict(self._elo)

    def head_to_head(self, first: str, second: str) -> PairStats:
        """Return cumulative head-to-head stats between two strategies."""

        stats = self._win_matrix.get(first, {}).get(second)
        if stats is None:
            return PairStats()
        return PairStats(wins=stats.wins, games=stats.games)

    def reset_history(self) -> None:
        self._win_matrix.clear()
        self._elo.clear()

    def run_self_play(
        self,
        strategy_names: Sequence[str],
        num_games: int,
        max_len: int,
        *,
        rotate: bool = True,
        desc: str = "playing games",
    ) -> SelfPlayResults:
        def x(strats):
            ss = strats[:]
            for i, s in enumerate(ss):
                if not s.startswith("opponent,to="):
                    continue

                new = None
                for p1, op in self._win_matrix.items():
                    if len(op) == 0:
                        new = p1
                        break

                if new is None:
                    to = int(s[len("opponent,to=") :])
                    base = strats[to]
                    if base not in self._win_matrix:
                        continue
                    candidates = list(self._win_matrix[base].items())
                    values = torch.tensor([c[1].win_rate for c in candidates])
                    values = torch.distributions.Categorical(logits=-20 * values)
                    sampled = values.sample((1,))
                    new = candidates[sampled.item()][0]
                ss[i] = new
            return ss

        strategy_names = [x(strategy_names) for _ in range(num_games)]
        results = self_play2(
            self._game_desc.make_game,
            [self._make_strategies(s) for s in strategy_names],
            max_len,
            rotate=rotate,
            desc=desc,
        )
        self._record_outcomes(strategy_names, results)
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

    def _make_strategies(self, strategy_names: Sequence[str]):
        return [
            self._game_desc.strategy_from_string(
                strategy,
                model=self._model_pool,
                discount_factor=self._discount_factor,
            )
            for strategy in strategy_names
        ]

    def _record_outcomes(
        self, strategy_names: Sequence[Sequence[str]], results: SelfPlayResults
    ) -> None:
        if not results:
            return

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

            pairwise_deltas: dict[str, float] = {name: 0.0 for name in scores}
            names = list(scores)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if a == b:
                        continue

                    result_a, result_b = self._pair_result(scores[a], scores[b])
                    self._update_pair_stats(a, b, result_a, result_b)

                    expected_a = self._expected_score(self._elo[a], self._elo[b])
                    expected_b = 1.0 - expected_a
                    pairwise_deltas[a] += self._elo_k * (result_a - expected_a)
                    pairwise_deltas[b] += self._elo_k * (result_b - expected_b)

            for name, delta in pairwise_deltas.items():
                self._elo[name] += delta

    def ensure_strategy_registered(self, name: str) -> None:
        if name not in self._win_matrix:
            self._win_matrix[name] = {}
        if name not in self._elo:
            self._elo[name] = 1500.0

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

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + 10**exponent)
