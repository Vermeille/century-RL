import torch
from boardrl.training import TrainingSample


def test_collate_basic():
    s1 = TrainingSample(a=1, b=torch.tensor([1, 2]))
    s2 = TrainingSample(a=3, b=torch.tensor([3, 4]))
    batch = TrainingSample.collate([s1, s2])

    assert torch.equal(batch.a, torch.tensor([1, 3]))
    assert torch.equal(batch.b, torch.tensor([[1, 2], [3, 4]]))


def test_collate_mismatched_tensor_list():
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([3])
    s1 = TrainingSample(x=t1)
    s2 = TrainingSample(x=t2)

    batch = TrainingSample.collate([s1, s2])

    assert isinstance(batch.x, list)
    assert torch.equal(batch.x[0], t1)
    assert torch.equal(batch.x[1], t2)
