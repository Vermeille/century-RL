"""Integration tests verifying that the entire RL model can learn simple tasks."""

import torch
import torch.nn.functional as F
import torch.distributions as dist

from boardrl.rl.model.model import Model


def test_model_learns_policy_and_value():
    """Overfit the entire model on a trivial set of text games.

    Each example is a short string containing two action markers, ``"@0"`` and
    ``"@1"``.  ``targets`` specifies which of those two actions is correct for
    that string while ``values`` holds the desired value prediction.  After a few
    hundred updates the model should perfectly reproduce both the selected action
    and the value for every training example.
    """
    dist.Distribution.set_default_validate_args(False)
    torch.manual_seed(0)

    games = ["a@0b@1", "b@0c@1", "c@0d@1", "d@0e@1"]
    targets = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([1.0, -1.0, 0.5, -0.5])

    model = Model(dim=32, num_layers=2, head_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(300):
        out = model(games)
        logits = torch.stack(out.policy)
        loss_policy = F.nll_loss(logits, targets)
        loss_value = F.mse_loss(out.value.mean, values)
        (loss_policy + loss_value).backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        out = model(games)
        preds = torch.stack(out.policy).argmax(dim=1)
        assert torch.equal(preds, targets)
        assert torch.allclose(out.value.mean, values, atol=0.1)


def test_model_learns_from_context():
    """Model must use the surrounding letters to choose the correct action.

    We create six alphabet fragments where the two markers ``"@0"`` and
    ``"@1"`` are placed at different positions for each fragment.  The policy
    should output ``0`` whenever the letter *before* ``"@0"`` is a vowel and ``1``
    otherwise.  Because the markers shift around, a solution based solely on
    positional embeddings will fail; the transformer needs to look at nearby
    characters to succeed.
    """
    dist.Distribution.set_default_validate_args(False)
    torch.manual_seed(0)

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    vowels = set("aeiou")
    games, targets, values = [], [], []
    for shift in range(6):
        seq = alphabet[shift : shift + 6]
        pos0 = shift % 5
        pos1 = (shift * 2 + 1) % 5
        letters = seq.copy()
        if pos0 < pos1:
            letters.insert(pos0 + 1, "@0")
            letters.insert(pos1 + 2, "@1")
        else:
            letters.insert(pos1 + 1, "@1")
            letters.insert(pos0 + 2, "@0")
        games.append("".join(letters))
        targets.append(0 if seq[pos0] in vowels else 1)
        values.append(1.0 if targets[-1] == 0 else -1.0)

    targets = torch.tensor(targets)
    values = torch.tensor(values)
    model = Model(dim=64, num_layers=3, head_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    for _ in range(500):
        out = model(games)
        logits = torch.stack(out.policy)
        loss_policy = F.nll_loss(logits, targets)
        loss_value = F.mse_loss(out.value.mean, values)
        (loss_policy + loss_value).backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        out = model(games)
        preds = torch.stack(out.policy).argmax(dim=1)
        assert torch.equal(preds, targets)
        assert torch.allclose(out.value.mean, values, atol=0.1)
