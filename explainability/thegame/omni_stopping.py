"""Explain Omni's optional play-versus-stop decision with small Boolean rules.

Hybrids keep the checkpoint's highest-logit card move and replace only the
decision to take that card versus x, after ending the turn becomes legal.
"""
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boardrl.games import games_library
from boardrl.games.strategies import one_hot
from explainability.thegame.semantic_tree import load_teacher, placements, play, score_summary
from explainability.thegame.summarize_human import paired_difference


OWN_FEATURES = ('own_min_cost','own_second_card_cost','own_hand_size','own_playable_cards',
                'own_reverse_moves','deck_count','cards_played_this_turn','own_dead_cards')
OMNI_FEATURES = ('other_min_cost','other_second_card_cost','other_hand_size','other_dead_cards',
                 'own_minus_other_min','next_draw_min_cost','refill_best_cost',
                 'greedy_blocks_other_cards','other_best_cost_after_greedy')
FEATURE_NAMES = OWN_FEATURES+OMNI_FEATURES+('model_card_cost',)


def hand_costs(hand,piles):
    options=[placements(card,piles) for card in hand]
    best=sorted(min(cost for _,cost in choices) for choices in options if choices)
    return best, sum(cost == -10 for choices in options for _,cost in choices), sum(not choices for choices in options)


@dataclass(frozen=True)
class StopObservation:
    piles: tuple[int,...]
    own: tuple[int,...]
    others: tuple[int,...]
    deck: tuple[int,...]
    deck_count: int
    played: int
    moves: tuple[str,...]

    @classmethod
    def parse(cls,text):
        lines=text.splitlines()
        def field(prefix):
            return next(line[len(prefix):].strip() for line in lines if line.startswith(prefix))
        hands=[tuple(map(int,line[5:].split())) for line in lines if line.startswith('Hand:')]
        deck_text=next((line[5:].strip() for line in lines if line.startswith('Deck:')),'')
        return cls(tuple(map(int,field('Piles:').split())),hands[0],
                   tuple(card for hand in hands[1:] for card in hand),
                   tuple(map(int,deck_text.split())),int(field('Cards:')),
                   int(field('Round:').rsplit('Action:',1)[1]),
                   tuple(line[1:] for line in lines if line.startswith('@')))

    def card_cost(self,move):
        card,pile=map(int,move.split('->'))
        return card-self.piles[pile] if pile < 2 else self.piles[pile]-card

    def features(self,preferred_card):
        own,reverse,dead=hand_costs(self.own,self.piles)
        other,_,other_dead=hand_costs(self.others,self.piles)
        draw,_,_=hand_costs(self.deck[:1],self.piles)
        refill_count=min(7-len(self.own),self.deck_count)
        refill,_,_=hand_costs(self.own+self.deck[:refill_count],self.piles)
        card_moves=[move for move in self.moves if '->' in move]
        greedy=min(card_moves,key=self.card_cost)
        card,pile=map(int,greedy.split('->'))
        after=list(self.piles);after[pile]=card
        after_other,_,after_dead=hand_costs(self.others,after)
        def first(values):
            return values[0] if values else 100
        def second(values):
            return values[1] if len(values)>1 else 100
        return np.asarray([first(own),second(own),len(self.own),len(own),reverse,
                           self.deck_count,self.played,dead,first(other),second(other),len(self.others),other_dead,
                           first(own)-first(other),first(draw),first(refill),after_dead-other_dead,
                           first(after_other),self.card_cost(preferred_card)],dtype=np.float32)


def tree_predict(node,x):
    if 'continue' in node:
        return np.full(len(x),node['continue'],dtype=bool)
    left=x[:,node['index']] <= node['threshold']
    result=np.empty(len(x),dtype=bool)
    result[left]=tree_predict(node['left'],x[left])
    result[~left]=tree_predict(node['right'],x[~left])
    return result


class StopRule:
    def __init__(self,specification):
        self.specification=specification

    def predict(self,x):
        spec=self.specification
        methods={'tree':self.tree,'threshold':self.threshold,'constant':self.constant,'phase':self.phase}
        return methods[spec['kind']](x)

    def tree(self,x):
        return tree_predict(self.specification['tree'],x)

    def threshold(self,x):
        return x[:,self.specification.get('index',0)] <= self.specification['cost']

    def constant(self,x):
        return np.full(len(x),self.specification['continue'],dtype=bool)

    def phase(self,x):
        spec=self.specification
        limit=np.where(x[:,5]>spec['early_deck'],2,np.where(x[:,5]<=spec['late_deck'],7,4))
        return x[:,0] <= limit


class StopHybrid:
    def __init__(self,processor,rule=None,records=None,game_ids=None):
        self.processor,self.rule,self.records,self.game_ids=processor,rule,records,game_ids
        self.optional=self.continued=0

    async def __call__(self,game):
        text=game.display_with_moves()
        output=await self.processor(text)
        logits=output.policy[0].detach().cpu().numpy()
        action=int(logits.argmax())
        card_indices=[i for i,move in enumerate(game.moves) if '->' in move]
        if 'x' in game.moves and card_indices:
            x_index=game.moves.index('x')
            card_index=max(card_indices,key=lambda index:float(logits[index]))
            observation=StopObservation.parse(text)
            features=observation.features(game.moves[card_index])
            if self.records is not None:
                self.records.append((features,action != x_index,self.game_ids[id(game)],
                                     float(logits[card_index]-logits[x_index])))
            if self.rule is not None:
                action=card_index if self.rule.predict(features[None,:])[0] else x_index
            self.optional+=1
            self.continued+=action != x_index
        return one_hot(action,len(game.moves)).log(),{}


class GreedyWithStop:
    """A fully non-neural policy, for checking practical transfer of stop rules."""
    def __init__(self,rule):
        self.rule=rule

    def action(self,text):
        observation=StopObservation.parse(text)
        cards=[move for move in observation.moves if '->' in move]
        if not cards:
            return observation.moves.index('x')
        card=min(cards,key=observation.card_cost)
        if 'x' in observation.moves and not self.rule.predict(observation.features(card)[None,:])[0]:
            return observation.moves.index('x')
        return observation.moves.index(card)


def collect(args):
    inference=load_teacher(args.checkpoint)
    records,game_ids=[],{}
    teacher=StopHybrid(inference.processor,records=records,game_ids=game_ids)
    scores=play(teacher,args.games,args.seed,game_ids=game_ids,mode='omni')
    x,y,episode,margin=zip(*records)
    np.savez_compressed(args.data,x=x,y=y,episode=episode,margin=margin,scores=scores)
    print(json.dumps({'games':args.games,'optional_decisions':len(records),'continue_rate':float(np.mean(y)),
                      'mean_score':float(scores.mean()),'features':FEATURE_NAMES},indent=2),flush=True)


def classification(y,pred):
    tp=int(np.sum(y & pred));fp=int(np.sum(~y & pred));fn=int(np.sum(y & ~pred))
    return {'n':len(y),'accuracy':float(np.mean(y==pred)), 'continue_rate':float(np.mean(y)),
            'predicted_continue_rate':float(np.mean(pred)), 'continue_precision':tp/(tp+fp) if tp+fp else None,
            'continue_recall':tp/(tp+fn) if tp+fn else None}


def export(tree,columns,node=0):
    if tree.children_left[node] == -1:
        return {'continue':bool(np.argmax(tree.value[node,0]) == 1)}
    index=int(columns[tree.feature[node]])
    return {'index':index,'feature':FEATURE_NAMES[index],'threshold':float(tree.threshold[node]),
            'left':export(tree,columns,tree.children_left[node]),'right':export(tree,columns,tree.children_right[node])}


def render(node,indent=''):
    if 'continue' in node:
        return indent+('PLAY ANOTHER CARD' if node['continue'] else 'END TURN')+'\n'
    return (indent+f"if {node['feature']} <= {node['threshold']:.4g}:\n"+render(node['left'],indent+'    ')
            +indent+'else:\n'+render(node['right'],indent+'    '))


def fit(args):
    from sklearn.tree import DecisionTreeClassifier
    data=np.load(args.data);x=data['x'];y=data['y']
    train=data['episode']<512
    validation=(data['episode']>=512)&(data['episode']<768)
    models={}
    for scope,count in [('own',len(OWN_FEATURES)),('omni',len(OWN_FEATURES)+len(OMNI_FEATURES)),('model_card',len(FEATURE_NAMES))]:
        for leaves in [2,3,4,6,8]:
            tree=DecisionTreeClassifier(max_leaf_nodes=leaves,min_samples_leaf=100,random_state=31)
            tree.fit(x[train,:count],y[train])
            spec={'kind':'tree','scope':scope,'leaves':int(tree.get_n_leaves()),'tree':export(tree.tree_,list(range(count)))}
            spec['text']=render(spec['tree'])
            spec['validation']=classification(y[validation],StopRule(spec).predict(x[validation]))
            name=f'{scope}_{leaves}';models[name]=spec
            print(name,spec['validation'],spec['text'],flush=True)
    for cost in [-10,0,1,2,3,4,5,8,10]:
        spec={'kind':'threshold','cost':cost}
        spec['validation']=classification(y[validation],StopRule(spec).predict(x[validation]))
        models[f'cost_{cost}']=spec
        print(f'cost_{cost}',spec['validation'],flush=True)
    models['always_stop']={'kind':'constant','continue':False}
    models['always_continue']={'kind':'constant','continue':True}
    for name,early,late in [('phase_exact',61,3),('phase_rounded',60,0)]:
        spec={'kind':'phase','early_deck':early,'late_deck':late}
        spec['validation']=classification(y[validation],StopRule(spec).predict(x[validation]))
        models[name]=spec
    args.output.write_text(json.dumps({'models':models,'features':FEATURE_NAMES},indent=2)+'\n')


def evaluate(args):
    specs=json.loads(args.models.read_text())['models']
    data=np.load(args.data);test=data['episode']>=768
    inference=load_teacher(args.checkpoint)
    teacher=StopHybrid(inference.processor)
    baseline=play(teacher,args.games,args.seed,mode='omni')
    results={'protocol':{'mode':'omni','checkpoint':str(args.checkpoint.resolve()),'seed':args.seed,'games':args.games,
                         'selected':args.selected,'card_policy':'highest checkpoint logit among legal card moves',
                         'source_tag':'run-source/YOLO-yay2-20260901T062612Z-604c1b8083e4'},
             'teacher':{'scores':baseline.tolist(),'summary':score_summary(baseline,baseline),
                        'optional_decisions':teacher.optional,'continue_rate':teacher.continued/teacher.optional}}
    print('teacher',results['teacher']['summary'],flush=True)
    for name in args.selected:
        rule=StopRule(specs[name]);hybrid=StopHybrid(inference.processor,rule)
        scores=play(hybrid,args.games,args.seed,mode='omni')
        results[name]={'specification':specs[name],'scores':scores.tolist(),'summary':score_summary(scores,baseline),
                       'test':classification(data['y'][test],rule.predict(data['x'][test])),
                       'optional_decisions':hybrid.optional,'continue_rate':hybrid.continued/hybrid.optional}
        print(name,results[name]['summary'],results[name]['test'],flush=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')


def greedy(args):
    specs=json.loads(args.models.read_text())['models']
    descriptor=games_library('thegame,mode=omni')
    results={'protocol':{'games':args.games,'seed':args.seed,'mode':'omni','card_policy':'lowest cost; no neural inference'},'results':{}}
    for name in ['always_stop']+args.selected:
        policy=GreedyWithStop(StopRule(specs[name]));scores=[]
        for index in range(args.games):
            random.seed(args.seed+index)
            game=descriptor.make_game(num_players=2)
            actions=0
            while not game.ended():
                game.play_idx(policy.action(game.display_with_moves()))
                actions+=1
                if actions>300:
                    raise RuntimeError('Unexpected non-terminating card game')
            scores.append(game.points())
        baseline=results['results'].get('always_stop',{'scores':scores})['scores']
        results['results'][name]={'scores':scores,'summary':score_summary(np.asarray(scores),np.asarray(baseline)),
                                  'vs_stop':paired_difference(scores,baseline)}
        # Here the reference is an always-stop greedy policy, not a teacher.
        summary=results['results'][name]['summary']
        summary['delta_vs_always_stop']=summary.pop('delta_vs_teacher')
        print(name,summary,flush=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['collect','fit','evaluate','greedy'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--models',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--games',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=2840000)
    parser.add_argument('--selected',nargs='+')
    args=parser.parse_args()
    {'collect':collect,'fit':fit,'evaluate':evaluate,'greedy':greedy}[args.stage](args)


if __name__ == '__main__':
    main()
