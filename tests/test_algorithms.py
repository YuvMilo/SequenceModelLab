

def test_recurrent_diag_ssm_calculation_1D_output():
    import torch
    from src.algorithms.ssm import recurrent_diag_ssm_calculation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.zeros([2, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([2, 1]).to(device)
    B = torch.Tensor([1, 1]).to(device)
    C = torch.Tensor([1, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = recurrent_diag_ssm_calculation(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[102], [103]], [[100], [100]]]).to(device))


def test_recurrent_diag_ssm_calculation():
    import torch
    from src.algorithms.ssm import recurrent_diag_ssm_calculation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([1, 1, 2]).to(device)
    x[0, 0, :] = torch.Tensor([1, 0])
    A = torch.Tensor([0]).to(device)
    B = torch.zeros([1, 2]).to(device)
    C = torch.zeros([2, 1]).to(device)
    D = torch.Tensor([1, 100]).to(device)

    ret = recurrent_diag_ssm_calculation(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[1, 100]]]).to(device))


def test_recurrent_diag_ssm_calculation():
    import torch
    from src.algorithms.ssm import recurrent_ssm_calculation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([1, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([[0, 0], [1, 0]]).to(device)
    B = torch.Tensor([1, 0]).to(device)
    C = torch.Tensor([0, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = recurrent_ssm_calculation(x, A, B, C, D)

    assert torch.equal(ret, torch.Tensor([[[100], [101]]]).to(device))


def test_calc_kernel():
    import torch
    from src.algorithms.ssm import calc_kernel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.Tensor([[0, 0], [1, 0]]).to(device)
    B = torch.Tensor([1, 0]).to(device)
    C = torch.Tensor([0, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = calc_kernel(A, B, C, D, ker_len=3)

    assert torch.equal(ret, torch.Tensor([100, 101, 100]).to(device))

def test_calc_kernel_diag():
    import torch
    from src.algorithms.ssm import calc_kernel_diag

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.Tensor([0]).to(device)
    B = torch.Tensor([1]).to(device)
    C = torch.Tensor([1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = calc_kernel_diag(A, B, C, D, ker_len=3)

    assert torch.equal(ret, torch.Tensor([101, 100, 100]).to(device))



