import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


def P_win(r1, r2, D):
    return 1 / (1 + 10 ** ((r2 - r1) / D))


def main():
    # Load the model
    with open("games.json", "r") as f:
        games = json.load(f)
    players = list(set(g["player1"] for g in games) | set(g["player2"] for g in games))
    p_to_i = {p: i for i, p in enumerate(players)}

    D = 10
    P = nn.Embedding(len(players), 1)
    with torch.no_grad():
        P.weight.data.fill_(10)

    i1 = torch.tensor([p_to_i[g["player1"]] for g in games])
    i2 = torch.tensor([p_to_i[g["player2"]] for g in games])
    random_buy = torch.tensor([p_to_i["random_buy"]])
    random = torch.tensor([p_to_i["random"]])

    sgd = SGD([P.weight], lr=100 * D, momentum=0.8)
    for _ in range(60):
        sgd.zero_grad()
        r1 = P(i1).squeeze()
        r2 = P(i2).squeeze()
        results = torch.tensor([g["points"] >= 0 for g in games], dtype=torch.float32)
        # qa = Q(r1, D)
        # qb = Q(r2, D)
        # p_win = qa / (qa + qb)
        p_win = P_win(r1, r2, D)
        # log_p_win = log_P_win(r1, r2, D)
        loss = F.binary_cross_entropy(
            p_win,
            results,
            reduction="mean",
        )
        # loss = torch.mean((results * 2 - 1) * log_p_win)
        loss.backward()
        sgd.step()
        with torch.no_grad():
            P.weight -= P.weight[random]
        ranking = P.weight.squeeze().argsort(descending=True)
        print(loss.item(), P.weight.squeeze().int())
        print(
            [(players[i], P.weight[i].item()) for i in ranking[:5]],
            "...",
            [(players[i], P.weight[i].item()) for i in ranking[-5:]],
        )
        print(
            "random_buy:",
            P(random_buy).item(),
            "no_action_random_buy:",
            P(torch.tensor([p_to_i["no_actions_random_buy"]])).item(),
            "Pwin(first, last)=",
            P_win(P(ranking[0]), P(ranking[-1]), D).item(),
        )
        print(
            list(
                zip(
                    torch.linspace(-5 * D, 5 * D, 11).tolist(),
                    torch.sigmoid(torch.linspace(-5 * D, 5 * D, 11) / D).tolist(),
                )
            ),
            torch.sigmoid(torch.tensor([P(random_buy) - P(random)]) / D).tolist(),
        )


if __name__ == "__main__":
    main()
