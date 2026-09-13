"""Archive Omni stopping-rule extraction and isolated gameplay evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np

from explainability.thegame.omni_stopping import FEATURE_NAMES,StopRule,classification
from explainability.thegame.summarize_semantic import fingerprint
from explainability.thegame.summarize_human import paired_difference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--greedy',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    data=np.load(args.data);models=json.loads(args.models.read_text())
    evaluation=json.loads(args.evaluation.read_text());greedy=json.loads(args.greedy.read_text())
    test=data['episode']>=768;x=data['x'][test];y=data['y'][test]
    fitted=StopRule(models['models']['own_8']).predict(x)
    reduced=StopRule(models['models']['phase_exact']).predict(x)
    assert np.array_equal(fitted,reduced),'The simple phase rule must exactly reduce the fitted own_8 tree.'
    rates={}
    for phase,mask in [('early',x[:,5]>61),('middle',(x[:,5]>3)&(x[:,5]<=61)),('near_empty',x[:,5]<=3)]:
        rates[phase]={}
        for label,low,high in [('reverse',-10,-10),('1_to_2',1,2),('3_to_4',3,4),('5_to_7',5,7),('8_plus',8,100)]:
            selected=mask&(x[:,0]>=low)&(x[:,0]<=high)
            rates[phase][label]={'n':int(selected.sum()),'continue_rate':float(y[selected].mean()) if selected.any() else None}
    for name in evaluation['protocol']['selected']:
        evaluation[name]['vs_always_stop']=paired_difference(evaluation[name]['scores'],evaluation['always_stop']['scores'])
    archive={'protocol':{'corpus_games':len(data['scores']),'optional_decisions':len(data['y']),
                         'corpus_seed':2840000,'train_games':512,'validation_games':256,'test_games':256,
                         'test_decisions':int(test.sum()),'checkpoint_sha256':fingerprint(Path(evaluation['protocol']['checkpoint'])),
                         'corpus_sha256':fingerprint(args.data),'features':FEATURE_NAMES},
             'models':models,'hybrids':evaluation,'greedy':greedy,'teacher_conditional_continue_rates_test':rates}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'omni_stopping.json').write_text(json.dumps(archive,indent=2)+'\n')
    lines=[
        '# Omni: when to play another card', '',
        '## Short practical rule', '',
        '**After meeting the turn minimum, keep playing while you have a legal move costing '
        'at most 3. Otherwise end the turn. Reconsider after every card.** A backwards-ten '
        'jump has cost −10, so it qualifies automatically.', '',
        'This is a small empirical surrogate, not an exact reconstruction of the checkpoint. '
        'The minimum is two cards while the draw pile is nonempty and one after it empties. '
        'Ending early is never allowed. The rule can produce several extra plays, not just one.', '',
        '## What the fitted tree found', '',
        'An eight-leaf classifier reduces exactly to:', '',
        '| Cards left in draw pile | Continue if cheapest legal move costs at most |',
        '|---|---:|',
        '| 62 or more | 2 |',
        '| 4–61 | 4 |',
        '| 0–3 | 7 |', '',
        'The early/middle/end pattern describes the fitted classifier, not a proven reason '
        'inside the network. The exact boundaries 61 and 3 were learned. A rounded version '
        'uses more than 60 / 1–60 / empty instead. Counting the deck precisely may be less '
        'convenient than the constant-three rule.', '',
        'Trees of up to eight leaves given the other hand and upcoming cards still chose '
        'only own minimum cost and deck size. This does **not** show that the neural policy '
        'ignores the other hand, draws, or memory; unmodeled exceptions remain.', '',
        '## Isolated stopping-rule test', '',
        'All rows below use the checkpoint’s card ranking. Only the decision to play a '
        'card versus end the turn is replaced. If continuing is forced when the model would '
        'stop, select its highest-logit legal card. Thus a minimum-cost gate does not necessarily '
        'force the model to choose that cheapest card.', '',
        '| Stopping decision | Mean points | SD | Worst 5% mean | Wins | Held-out stop/play agreement |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    names={'teacher':'Original Omni policy','always_stop':'Always stop after minimum','cost_3':'Cheapest cost ≤3',
           'cost_4':'Cheapest cost ≤4','phase_exact':'Fitted 2 / 4 / 7 phases','phase_rounded':'Rounded phases'}
    for name,label in names.items():
        row=evaluation[name];s=row['summary']
        accuracy=row.get('test',{'accuracy':1})['accuracy']
        lines.append(f'| {label} | {s["mean"]:.2f} | {s["std"]:.2f} | {s["worst_5_percent_mean"]:.2f} | {100*s["win_rate"]:.2f}% | {100*accuracy:.2f}% |')
    lines += ['',f'These are {evaluation["protocol"]["games"]:,} fresh matched deals, seeds '
              f'`{evaluation["protocol"]["seed"]} + i`. Both seats use the same policy. '
              'Checkpoint: Omni YOLO-yay2 `step-900.pth`, argmax, using historical source '
              '`run-source/YOLO-yay2-20260901T062612Z-604c1b8083e4`.', '']
    for name in ['cost_3','phase_exact']:
        row=evaluation[name];s=row['summary'];d=row['vs_always_stop']
        lines.append(f'- {names[name]} versus the original policy: {s["delta_vs_teacher"]:+.2f} points, '
                     f'paired 95% CI [{s["delta_ci95"][0]:+.2f}, {s["delta_ci95"][1]:+.2f}]. '
                     f'Versus forced stopping: {d["mean"]:+.2f}, CI [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}].')
    lines += ['', 'The phase rule has higher held-out decision agreement, but did not produce '
              'a higher mean gameplay score than the simpler constant-three rule in this batch. '
              'This is another reason to distinguish faithful imitation from useful advice.']
    lines += ['', '## Fully non-neural check', '',
              'Here both card selection and stopping are explicit rules: choose the cheapest '
              'card, and use the stopping rule below. These rules work without the other '
              'hand or upcoming draws; tests also check identical Free/Omni trajectories '
              'for the own-information rule. They were evaluated on a separate 2,048-deal batch, '
              'seeds `3040000 + i`; compare rows within this table, not raw scores across tables.', '',
              '| Stopping rule with lowest-cost cards | Mean points | Gain versus forced stopping |',
              '|---|---:|---:|']
    for name in ['always_stop','cost_3','cost_4','phase_exact','phase_rounded']:
        row=greedy['results'][name]
        lines.append(f'| {names[name]} | {row["summary"]["mean"]:.2f} | {row["vs_stop"]["mean"]:+.2f} |')
    d=greedy['results']['cost_3']['vs_stop']
    lines += ['',f'Constant-three improvement over greedy forced stopping: **{d["mean"]:+.2f}** '
              f'points, paired approximate 95% CI **[{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}]**.', '',
              'This is the directly tested calculator-free advice. It does not establish the '
              'best stopping rule when card selection follows the separate awkward-card tactic; '
              'that combination would need its own evaluation.', '',
              '## Observed neural continuation frequencies', '',
              'Percent of held-out optional decisions where the original policy played another '
              'card, conditioned on minimum available cost and deck phase. These are conditional '
              'observations, not interventions.', '',
              '| Deck phase | Reverse-ten | Cost 1–2 | Cost 3–4 | Cost 5–7 | Cost ≥8 |',
              '|---|---:|---:|---:|---:|---:|']
    for phase,values in rates.items():
        cells=[f'{100*v["continue_rate"]:.1f}% (n={v["n"]})' if v['n'] else '—' for v in values.values()]
        lines.append('| '+phase+' | '+' | '.join(cells)+' |')
    lines += ['', '## Method and limits', '',
              f'- Collected {len(data["y"]):,} optional decisions from 1,024 teacher games. '
              f'Whole-game split: 512 train / 256 validation / 256 test; {int(test.sum()):,} test decisions. '
              'Compulsory card plays and states without a choice between card and stop are excluded.',
              '- Fit Boolean decision trees with 2/3/4/6/8 leaves, plus simple cost thresholds. '
              'Own-only, full-Omni, and model-preferred-card feature groups were compared. '
              'The tested threshold and phase candidates were frozen before opening final test '
              'and gameplay results. All evaluated candidates are reported.',
              '- Features are parsed from displayed text. Memory is not a surrogate feature. '
              'Model inference stays on CUDA with no silent CPU fallback.',
              '- Frozen rules are evaluated over complete trajectories, not just teacher states. '
              'Matched initial games are constructed with independent `random.seed(seed + i)` '
              'before rollouts, preserving deal pairing across strategies.',
              '- Uncertainty intervals are approximate paired-game normal intervals and are '
              'not adjusted for multiple comparisons. Similar means are not proof of equivalence.',
              '- Results are for the repository’s two-player rules. More players, human '
              'communication, and human implementation errors have not been evaluated.', '',
              'Scripts and commands are in `../README.md`. Full per-game scores, rule exports, '
              'classification metrics, conditional rates, and provenance are in `omni_stopping.json`.', '']
    (args.output_dir/'OMNI_STOPPING.md').write_text('\n'.join(lines))
    print(json.dumps({'hybrid_cost3':evaluation['cost_3']['summary'],'greedy_cost3':greedy['results']['cost_3']['vs_stop']},indent=2))


if __name__ == '__main__':
    main()
