import torch.multiprocessing as mp
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import pyximport

pyximport.install(setup_args={"script_args": ['--cython-cplus']})
from engine import *


class Illegal(BaseException):
    pass


class AlternativeEncoder(nn.Module):

    def __init__(self, n_layers, dim):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(1, dim),
                    nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)),
                nn.Sequential(
                    nn.GroupNorm(1, dim),
                    nn.Conv1d(dim, dim * 4, 1),
                    nn.ReLU(True),
                    nn.Conv1d(dim * 4, dim, 1),
                )
            ]) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = x.transpose(2, 1)
        for m in self.layers:
            x = m[0](x).add_(x)
            x = m[1](x).add_(x)
        x = x.transpose(2, 1)
        return x


class AttentionPool1d(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.query = nn.Parameter(
            torch.randn(embed_dim) / math.sqrt(embed_dim))
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # BLC -> LBC
        x, _ = F.multi_head_attention_forward(
            query=self.query.expand(1, x.shape[1], x.shape[2]),
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxlen = 512
        dim = 256
        self.in_embed = nn.Embedding(128, dim)
        self.pos_enc_out = nn.Parameter(
            torch.randn(self.maxlen, dim) / math.sqrt(dim))
        self.pos_enc = nn.Parameter(
            torch.randn(self.maxlen, dim) / math.sqrt(dim))
        self.pos_enc2 = nn.Parameter(
            torch.randn(self.maxlen, dim) / math.sqrt(dim))

        self.encode = nn.Sequential(
            AlternativeEncoder(4, dim),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,
                                                             dim // 32,
                                                             dim * 4,
                                                             dropout=0.,
                                                             batch_first=True),
                                  num_layers=2,
                                  norm=nn.LayerNorm(dim)))
        self.decode = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            dim, dim // 32, dim * 4, dropout=0., batch_first=True),
                                            num_layers=2,
                                            norm=nn.LayerNorm(dim))
        self.to_char = nn.Linear(dim, 128)
        self.reward = nn.Sequential(nn.LayerNorm(dim, 1),
                                    AttentionPool1d(dim, dim // 32, dim),
                                    nn.Linear(dim, 1))

        #for p in self.parameters(): nn.init.normal_(p, std=0.02)

    def text_encode(self, txts, maxlen, pad=False):

        def do_pad(l):
            if pad:
                return l + [0] * (maxlen - len(l))
            else:
                return l

        txts = [
            torch.LongTensor(do_pad([ord(c) for c in txt][:maxlen]))
            for txt in txts
        ]
        return nn.utils.rnn.pad_sequence(txts, batch_first=True).to(
            self.in_embed.weight.device)

    def mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)))
        return mask.to(self.in_embed.weight.device)

    def text_embed(self, txts, maxlen, pos, pad=False):
        txts = self.text_encode(txts, maxlen, pad=pad)
        txts = self.in_embed(txts)
        return txts + pos[:min(txts.shape[1], maxlen)]

    def forward(self, games, outs=None, win=None):
        enc = self.encode(
            self.text_embed(games, self.maxlen, self.pos_enc, pad=True))
        enc += self.pos_enc_out[:enc.shape[1]]

        if outs is not None:
            outs_pad = [chr(1) + out for out in outs]
            outs_enc = self.text_embed(outs_pad, 25, self.pos_enc2)
            pred = self.to_char(
                self.decode(outs_enc,
                            enc,
                            tgt_mask=self.mask(outs_enc.shape[1])))
            outs = [out + '\n' for out in outs]
            loss = nn.functional.cross_entropy(pred.transpose(1, 2),
                                               self.text_encode(outs, 25),
                                               ignore_index=0,
                                               reduction='none',
                                               label_smoothing=0.)
            policy_loss = loss.mean()
            losses = {'policy': policy_loss.item()}
            loss = policy_loss
            if win is not None:
                win = torch.tensor(win, device=loss.device, dtype=torch.float)
                r = self.reward(enc).squeeze(1)
                reward_loss = nn.functional.binary_cross_entropy_with_logits(
                    r,
                    torch.sign(win) * 0.5 + 0.5)
                losses['reward_loss'] = reward_loss.item()
                loss += reward_loss
            return loss, losses
        else:
            assert not self.training
            #outs = self.stochastic_decode(enc, 0.25)
            #outs = self.nucleus_decode(enc, 0.9)
            outs = self.greedy_decode(enc)
            r = self.reward(enc).squeeze(1)
            return [o[1:(o + '\n').index('\n')] for o in outs], r

    def greedy_decode(self, enc):
        outs = [chr(1) for _ in range(len(enc))]
        with torch.no_grad():
            for _ in range(25):
                o = self.to_char(
                    self.decode(self.text_embed(outs, 25, self.pos_enc2), enc))
                o = o[:, -1, :]
                o = o.argmax(dim=1)
                for i, c in enumerate(o):
                    outs[i] += chr(int(c.item()))
        return outs

    def nucleus_decode(self, enc, thresh):
        outs = [chr(1) for _ in range(len(enc))]
        with torch.no_grad():
            for _ in range(25):
                o = self.to_char(
                    self.decode(self.text_embed(outs, 25, self.pos_enc2), enc))
                o = nn.functional.softmax(o[:, -1, :], dim=1)

                o, idx = torch.sort(o, dim=1, descending=True)
                #print(o[0, 0], idx[0, 0])
                o[:, 1:][o.cumsum(dim=1)[:, 1:] > thresh] = 0
                o = torch.multinomial(o, num_samples=1)
                for i, c in enumerate(o[:, -1]):
                    outs[i] += chr(int(idx[i, c.item()]))
        print(outs[0])
        return outs

    def stochastic_decode(self, enc, T):
        outs = [chr(1) for _ in range(len(enc))]
        with torch.no_grad():
            for _ in range(25):
                o = self.to_char(
                    self.decode(self.text_embed(outs, 25, self.pos_enc2),
                                enc)) / T
                o = nn.functional.softmax(o[:, -1, :], dim=1)
                o = torch.multinomial(o, num_samples=1)
                for i, c in enumerate(o[:, -1]):
                    outs[i] += chr(int(c.item()))
        return outs


class GamesData:

    def __init__(self, data):
        self.data = data

    def to_trainset(self):
        out = []
        for d in self.data:
            for log in d['history']:
                out.append([log['state'], log['action'], log['winner']])
        return out

    def avg_len(self):
        return sum(len(d['history']) for d in self.data) / len(self.data)

    def avg_points(self):
        return sum(max(d['p0'], d['p1']) for d in self.data) / (len(self.data))

    def stats_cause(self):
        illeg = sum(1 for d in self.data if d['cause'] == 'illegal')
        proper = sum(1 for d in self.data if d['cause'] == 'proper')
        toolong = sum(1 for d in self.data if d['cause'] == 'toolong')
        n = len(self.data)
        return {
            'illegal': illeg / n,
            'proper': proper / n,
            'toolong': toolong / n
        }

    def avg_move_summary(self):
        movs = {}
        for typ in 'HRVA':
            v = sum(1 for d in self.data for h in d['history']
                    if len(h['action']) > 0 and h['action'][0] == typ)
            movs[typ] = v / len(self.data)
        return movs

    def prompt_size(self):
        v = sum(
            sum(len(h['state']) for h in d['history']) / len(d['history'])
            for d in self.data) / len(self.data)
        return v

    def metrics(self):
        return {
            'prompt_size': int(self.prompt_size()),
            'avg_len': self.avg_len(),
            'avg_points': self.avg_points(),
            'causes': self.stats_cause(),
            'avg_move_summary': self.avg_move_summary()
        }

    def dump(self):
        with open('game.txt', 'w') as f:
            for i, d in enumerate(self.data):
                print(f'== GAME {i} ==', file=f)
                for log in d['history']:
                    print(log['state'], file=f)
                    print('>', log['action'], ','.join(log['notes']), file=f)
                    print(file=f)


def chunk(data, size):
    i = 0
    while i < len(data):
        yield data[i:i + size]
        i += size


def flatten(list_of_lists):
    out = []
    for l in list_of_lists:
        out += l
    return out


def autobatch(model, input, bs=None):
    if bs is None:
        bs = len(input)
    assert bs > 0

    try:
        with torch.no_grad():
            return flatten([model(x) for x in chunk(input, bs)])
    except Exception as e:
        print(e)
        return autobatch(model, input, bs // 2)


def self_play(model, n_games, dropout, max_len, device):
    model.eval()
    print(device)
    if device is not None:
        model.to(device)
    data = [{'history': []} for _ in range(n_games)]
    running = [True for _ in range(n_games)]
    games = [Game() for _ in range(n_games)]
    temps = [random.random() * 10 + 10 for _ in range(n_games)]

    for i_mov in range(max_len):
        if not any(running):
            break

        for i in range(n_games):
            if not running[i]:
                continue

            winner = None
            g = games[i]
            log = {'state': g.display(), 'notes': []}
            if random.uniform(0, 1) < dropout:
                with torch.no_grad():
                    mov, debug = g.gen_neural_move(model, T=temps[i])
                log['notes'] += [str(x) for x in debug]
            log['action'] = mov
            data[i]['history'].append(log)
            try:
                g.play_str(mov)
            except Illegal:
                running[i] = False
                log['notes'].append('illegal')
                data[i]['winner'] = 1 - g.state
                data[i]['cause'] = 'illegal'
                data[i]['p0'] = g.p0.points()
                data[i]['p1'] = g.p1.points()

            if g.ended():
                data[i]['history'].append({'state': g.display()})
                data[i]['history'].append(
                    {'state': g.display(force=1 - g.state)})
                running[i] = False
                data[i]['winner'] = 0 if g.p0.points() > g.p1.points() else 1
                data[i]['cause'] = 'proper'
                data[i]['p0'] = g.p0.points()
                data[i]['p1'] = g.p1.points()

    for i in range(n_games):
        if running[i]:
            data[i]['history'].append({
                'state': games[i].display(),
                'notes': [],
                'action': ''
            })
            data[i]['history'].append({
                'state':
                games[i].display(force=1 - g.state),
                'notes': [],
                'action':
                ''
            })
            data[i]['winner'] = 0 if games[i].p0.points() > games[i].p1.points(
            ) else 1
            data[i]['cause'] = 'toolong'
            data[i]['p0'] = games[i].p0.points()
            data[i]['p1'] = games[i].p1.points()

        hist = data[i]['history']
        points = abs(data[i]['p0'] - data[i]['p1'])
        for j in range(len(hist)):
            if j % 2 == data[i]['winner']:
                hist[j]['winner'] = points
            else:
                hist[j]['winner'] = -points
            hist[j]['notes'] += ['reward: ' + str(hist[j]['winner'])]

    return GamesData(data)


import math


class CosineWarmup:

    def __init__(self, warmup_steps, total_steps, opt):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.i = 0
        self.opt = opt
        for group in opt.param_groups:
            group['init_lr'] = group['lr']

    def step(self):
        if self.i <= self.warmup_steps:
            scale = (-math.cos(math.pi * self.i / self.warmup_steps) + 1) / 2
        else:
            scale = (math.cos(math.pi * (self.i - self.warmup_steps) /
                              (self.total_steps - self.warmup_steps)) + 1) / 2
        for group in self.opt.param_groups:
            group['lr'] = scale * group['init_lr']
        self.i += 1


def bootstrap_gen(num_games, maxlen):
    trainset = []
    for _ in range(num_games):
        game = Game()
        for __ in range(maxlen):
            x = game.display()
            y = random.choice(game.gen_move())
            trainset.append([x, y, True])
            game.play_str(y)
    return trainset


from visdom import Visdom


def bootstrap(model, opt, num_games, maxlen, batchsize, epochs):
    trainset = bootstrap_gen(num_games, maxlen)
    print('maxlen', max(len(x) for x, _, __ in trainset))

    print(Counter(x[1][0] for x in trainset))
    print(Counter(x[1] for x in trainset))
    viz = Visdom(env='century-3l-' + str(opt.param_groups[0]['lr']))
    viz.close()
    sched = CosineWarmup(
        5000,  #len(trainset) // batchsize * 5,
        len(trainset) // batchsize * epochs,
        opt)
    ii = 0
    for epoch in range(epochs):
        model.train()
        for i in range(len(trainset) // batchsize):
            X, Y, W = zip(
                *random.sample(trainset, min(batchsize, len(trainset))))
            opt.zero_grad()
            loss = model(X, Y, W)
            loss.backward()
            #print(i, loss.item())
            sched.step()
            opt.step()

            if ii % 10 == 0:
                viz.line(torch.tensor([loss.item()]),
                         torch.tensor([ii]),
                         win='loss',
                         update='append',
                         opts={'title': 'loss'})
            ii += 1
        try:
            data = self_play(m, 400, 0., 20)
            data.dump()
            metrics = data.metrics()
            print('bootstrap', epoch, metrics)
        except Exception as e:
            print(e)
        torch.save(m.state_dict(), 'bootstrap.pth')
        trainset = bootstrap_gen(num_games, maxlen)


def load(model, file):
    try:
        ckpt = torch.load(file, map_location='cpu')
        d = model.state_dict()
        for k in d.keys():
            if k in d and k in ckpt:
                try:
                    d[k].copy_(ckpt[k])
                except Exception as e:
                    print(e)
        print('loaded')
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':
    from collections import Counter
    import sys
    import time

    if False:
        start = time.time()

        g = Game()
        print('RUN')
        g.gen_good_move(100)

        print((time.time() - start))
        sys.exit(0)

    m = Model()
    m.to('cuda:0')
    prev_points = 0
    num_games = 20
    EPOCHS = 10

    # bootstrap
    #if not load(m, 'bootstrap.pth'):
    if not load(m, 'rl-48.pth'):
        pass  #bootstrap(m, opt, num_games=12000, maxlen=50, batchsize=100, epochs=30)

    viz = Visdom(env='century-rl')
    viz.close()
    # self play
    mp.set_start_method('spawn')
    ii = 0
    for epoch in range(3000):
        print('EPOCH', epoch)
        with mp.Pool(4) as pool:
            data = [
                pool.apply_async(self_play,
                                 args=(m, num_games, 1, 50, f'cuda:{dev}'))
                for dev in range(4)
            ]
            data = GamesData(flatten([d.get().data for d in data]))
        data.dump()
        metrics = data.metrics()
        if False and metrics['avg_points'] < prev_points:
            print('NOPE, REVERT', metrics['avg_points'], prev_points)
            if False:
                with torch.no_grad():
                    for dst, src in zip(m.state_dict().values(),
                                        prev_m.state_dict().values()):
                        dst.copy_(src)
            data = self_play(m, num_games, 1, 20)
            metrics = data.metrics()
            trainset += data.to_trainset()
        else:
            prev_m = copy.deepcopy(m)
            prev_points = metrics['avg_points']
            trainset = data.to_trainset()

        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    viz.line(torch.tensor([vv]),
                             torch.tensor([ii]),
                             win=k + '.' + kk,
                             update='append',
                             opts={'title': k + '.' + kk})
            else:
                viz.line(torch.tensor([v]),
                         torch.tensor([ii]),
                         win=k,
                         update='append',
                         opts={'title': k})
        print(metrics)
        print(len(trainset), 'samples')
        if len(trainset) > 0:
            print(Counter(list(zip(*trainset))[1]))
        m.train()

        opt = torch.optim.AdamW(m.parameters(),
                                lr=float(sys.argv[1]),
                                betas=(0.9, 0.999))
        sched = CosineWarmup(min(5000,
                                 len(trainset) // 64 * 1),
                             len(trainset) // 64 * EPOCHS, opt)

        for e in range(EPOCHS):
            random.shuffle(trainset)
            for batch in chunk(trainset, 64):
                X, Y, W = zip(*batch)
                opt.zero_grad()
                loss, losses = m(X, Y, W)
                loss.backward()
                opt.step()
                sched.step()
                ii += 1
                if ii % 10 == 0:
                    print('lr', opt.param_groups[0]['lr'])
                    for k, v in losses.items():
                        viz.line(torch.tensor([v]),
                                 torch.tensor([ii]),
                                 win='loss' + k,
                                 update='append',
                                 opts={'title': 'loss.' + k})
            print()

        torch.save(m.state_dict(), f'rl-{epoch}.pth')
