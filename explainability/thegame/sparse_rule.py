"""Forward-select short additive move costs using joint-choice imitation loss.

Unlike independent candidate classification, every fit compares the chosen
move directly with its alternatives. Selection never reads final-test labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from explainability.thegame.pile_tree import PileRule, agreement, prepare


NAMES = (
    'cost', 'signed_log_cost', 'positive_cost', 'log_positive_cost',
    'cost_squared', 'mean_remaining_cost', 'worst_remaining_cost',
    'cheapest_remaining_cost', 'dead_remaining_cards', 'remaining_placements',
    'card_legal_piles', 'card_single_pile', 'separation_change',
    'separation_after', 'open_space_after', 'reverse_setup',
    'fraction_of_space', 'next_same_pile_gap', 'cost_times_progress',
    'worst_cost_times_progress', 'mean_cost_times_progress',
)


def engineer(features):
    cost = features[:, :, 0]
    progress = 1 - features[:, :, 30] / 84
    values = [cost, np.sign(cost) * np.log1p(abs(cost)), np.maximum(cost, 0),
              np.log1p(np.maximum(cost, 0)), cost ** 2,
              features[:, :, 6], features[:, :, 7], features[:, :, 5],
              features[:, :, 3], features[:, :, 4], features[:, :, 2],
              (features[:, :, 2] == 1).astype(float), features[:, :, 11],
              features[:, :, 10], features[:, :, 12], features[:, :, 9],
              features[:, :, 29], features[:, :, 28], cost * progress,
              features[:, :, 7] * progress, features[:, :, 6] * progress]
    return np.stack(values, axis=2)


class SparseRule(PileRule):
    def scores(self, features):
        x = engineer(features)
        return -np.einsum('ncf,f->nc', x[:, :, self.specification['columns']], self.specification['weights'])


def fit_weights(x, labels, unique):
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    # Center within states: state-constant offsets cannot affect rankings.
    centered = x - x.mean(axis=1, keepdims=True)
    scale = np.maximum(centered.std(axis=(0, 1)), 1e-6)
    normalized = centered / scale
    target = (labels & unique).astype(float)
    target /= target.sum(axis=1, keepdims=True)

    def objective(weights):
        logits = -np.einsum('ncf,f->nc', normalized, weights)
        logits = np.where(unique, logits, -1e8)
        logprob = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -(target * logprob).sum(axis=1).mean() + 1e-4 * np.dot(weights, weights)
        error = np.exp(logprob) - target
        gradient = -np.einsum('nc,ncf->f', error, normalized) / len(x) + 2e-4 * weights
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[-1]), jac=True, method='L-BFGS-B', options={'maxiter':100})
    return result.x / scale, float(result.fun)


def fit(args):
    data = np.load(args.data)
    features = prepare(data['x'])
    x = engineer(features).astype(np.float64)
    labels = data['exact'][:, 7:11]
    indices = data['indices'][:, 7:11]
    unique = np.ones(indices.shape, dtype=bool)
    for i in range(1, 4):
        unique[:, i] = (indices[:, i:i+1] != indices[:, :i]).all(axis=1)
    train = (data['episode'] < 512) & labels.any(axis=1)
    validation = (data['episode'] >= 512) & (data['episode'] < 640)
    selected_rows = np.flatnonzero(train)[::3]
    columns = [0]
    specs = {}
    for terms in range(1, 7):
        if terms > 1:
            proposals = []
            for candidate in range(1, len(NAMES)):
                if candidate in columns:
                    continue
                proposed = columns + [candidate]
                _, loss = fit_weights(x[selected_rows][:, :, proposed], labels[selected_rows], unique[selected_rows])
                proposals.append((loss, candidate))
            columns.append(min(proposals)[1])
        weights, loss = fit_weights(x[train][:, :, columns], labels[train], unique[train])
        spec = {'kind':'sparse', 'columns':columns[:], 'features':[NAMES[i] for i in columns], 'weights':weights.tolist(), 'train_loss':loss}
        spec['validation'] = agreement(data,validation,SparseRule(spec),features)
        specs[f'sparse_{terms}'] = spec
        print(f'sparse_{terms}',json.dumps(spec),flush=True)
        args.output.write_text(json.dumps({'models':specs,'features':NAMES},indent=2))
    # Human-readable rounding, chosen before opening the final-test results.
    rounded = {
        'rounded_4': ([0, 6, 12, 16], [1, .25, -.2, 8]),
        'rounded_5': ([0, 6, 12, 16, 11], [1, .25, -.2, 5.5, -4]),
    }
    for name, (columns, weights) in rounded.items():
        spec = {'kind':'sparse', 'columns':columns, 'features':[NAMES[i] for i in columns], 'weights':weights}
        spec['validation'] = agreement(data,validation,SparseRule(spec),features)
        specs[name] = spec
        print(name,json.dumps(spec),flush=True)
    args.output.write_text(json.dumps({'models':specs,'features':NAMES},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
