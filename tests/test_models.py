

def test_RecurrentDiagSMMCalcStrategy_1D_output():
    import torch
    from src.models.strategies.calc import RecurrentDiagSMMCalcStrategy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.zeros([2, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([2, 1]).to(device)
    B = torch.Tensor([1, 1]).to(device)
    C = torch.Tensor([1, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    s = RecurrentDiagSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[102], [103]], [[100], [100]]]).to(device))


def test_RecurrentDiagSMMCalcStrategy():
    import torch
    from src.models.strategies.calc import RecurrentDiagSMMCalcStrategy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([1, 1, 2]).to(device)
    x[0, 0, :] = torch.Tensor([1, 0])
    A = torch.Tensor([0]).to(device)
    B = torch.zeros([1, 2]).to(device)
    C = torch.zeros([2, 1]).to(device)
    D = torch.Tensor([1, 100]).to(device)

    s = RecurrentDiagSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[1, 100]]]).to(device))

def test_RecurrentSMMCalcStrategy1D():
    import torch
    from src.models.strategies.calc import RecurrentSMMCalcStrategy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([1, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([[0, 0], [1, 0]]).to(device)
    B = torch.Tensor([1, 0]).to(device)
    C = torch.Tensor([0, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    s = RecurrentSMMCalcStrategy()
    ret = s.calc(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[100], [101]]]).to(device))
