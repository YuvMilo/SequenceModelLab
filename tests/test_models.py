

def test_RecurrentDiagSMMCalcStrategy_1D_output():
    import torch
    from src.models.strategies.calc import RecurrentDiagSMMCalcStrategy

    x = torch.zeros([2, 2, 1])
    x[0, 0, 0] = 1
    A = torch.Tensor([2, 1])
    B = torch.Tensor([1, 1])
    C = torch.Tensor([1, 1])
    D = torch.Tensor([100])

    s = RecurrentDiagSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[102], [103]], [[100], [100]]]))


def test_RecurrentDiagSMMCalcStrategy():
    import torch
    from src.models.strategies.calc import RecurrentDiagSMMCalcStrategy

    x = torch.zeros([1, 1, 2])
    x[0, 0, :] = torch.Tensor([1, 0])
    A = torch.Tensor([0])
    B = torch.zeros([1, 2])
    C = torch.zeros([2, 1])
    D = torch.Tensor([1, 100])

    s = RecurrentDiagSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[1, 100]]]))

def test_RecurrentSMMCalcStrategy1D():
    import torch
    from src.models.strategies.calc import RecurrentSMMCalcStrategy

    x = torch.zeros([1, 2, 1])
    x[0, 0, 0] = 1
    A = torch.Tensor([[0, 0], [1, 0]])
    B = torch.Tensor([1, 0])
    C = torch.Tensor([0, 1])
    D = torch.Tensor([100])

    s = RecurrentSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[100], [101]]]))
