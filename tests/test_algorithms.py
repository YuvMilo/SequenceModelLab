def test_recurrent_diag_ssm_calculation_1D_output():
    import torch
    from src.algorithms.ssm_calc import recurrent_diag_ssm_calculation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.zeros([2, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([2, 1]).to(device)
    B = torch.Tensor([1, 1]).to(device)
    C = torch.Tensor([1, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    identity = lambda x: x
    hidden, outs = recurrent_diag_ssm_calculation(x, A, B, C, D, identity)

    assert torch.equal(outs, torch.Tensor([[[102], [103]], [[100], [100]]]).to(device))
    assert torch.equal(hidden, torch.Tensor([[[1, 1], [2, 1]], [[0, 0], [0, 0]]]).to(device))


def test_recurrent_diag_ssm_calculation():
    import torch
    from src.algorithms.ssm_calc import recurrent_ssm_calculation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([1, 2, 1]).to(device)
    x[0, 0, 0] = 1
    A = torch.Tensor([[0, 0], [1, 0]]).to(device)
    B = torch.Tensor([1, 0]).to(device)
    C = torch.Tensor([0, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    identity = lambda x: x
    hiddens, outs = recurrent_ssm_calculation(x, A, B, C, D, identity)

    assert torch.equal(outs, torch.Tensor([[[100], [101]]]).to(device))
    assert torch.equal(hiddens, torch.Tensor([[[1, 0], [0, 1]]]).to(device))

def test_calc_kernel():
    import torch
    from src.algorithms.ssm_calc import calc_kernel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.Tensor([[0, 0], [1, 0]]).to(device)
    B = torch.Tensor([1, 0]).to(device)
    C = torch.Tensor([0, 1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = calc_kernel(A, B, C, D, ker_len=3)

    assert torch.equal(ret, torch.Tensor([100, 101, 100]).to(device))


def test_calc_kernel_diag():
    import torch
    from src.algorithms.ssm_calc import calc_kernel_diag

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.Tensor([0]).to(device)
    B = torch.Tensor([1]).to(device)
    C = torch.Tensor([1]).to(device)
    D = torch.Tensor([100]).to(device)

    ret = calc_kernel_diag(A, B, C, D, ker_len=3)

    assert torch.equal(ret, torch.Tensor([101, 100, 100]).to(device))


def test_get_rot_ssm_one_over_n():
    import torch
    import numpy as np
    from src.algorithms.ssm_init1D import get_rot_ssm_one_over_n_init

    A, _, _, _ = get_rot_ssm_one_over_n_init(num_hidden_state=6, radii=0.5,
                                             off_diagonal_ratio=5,
                                             main_diagonal_diff=8)
    eig = torch.linalg.eig(A).eigenvalues.numpy()

    wanted_eig = np.array([-0.5,
                           -0.5,
                           0.5,
                           0.5,
                           0.5 * (np.cos(2 * np.pi / 3) + np.sin(2 * np.pi / 3) * 1j),
                           0.5 * (np.cos(2 * np.pi / 3) - np.sin(2 * np.pi / 3) * 1j)])
    wanted_eig = sorted(wanted_eig)
    eig = sorted(eig)

    assert np.allclose(wanted_eig, eig,  atol=1e-3)
    assert np.abs(A[0, 0] - A[1, 1]) == 8
    assert max(np.abs(A[2, 3] / A[3, 2]), np.abs(A[3, 2] / A[2, 3])) == 5


def test_get_rot_ssm_equally_spaced():
    import torch
    import numpy as np
    from src.algorithms.ssm_init1D import get_rot_ssm_equally_spaced_init

    A, _, _, _ = get_rot_ssm_equally_spaced_init(num_hidden_state=6, radii=0.5,
                                                 angle_shift=0,
                                                 off_diagonal_ratio=5,
                                                 main_diagonal_diff=8)
    eig = torch.linalg.eig(A).eigenvalues.numpy()

    wanted_eig = np.array([
        0.5,
        0.5,
        0.5j,
        -0.5j,
        -0.5,
        -0.5
    ])
    wanted_eig = sorted(wanted_eig)
    eig = sorted(eig)

    assert np.allclose(wanted_eig, eig,  atol=1e-3)
    assert np.abs(A[0, 0]-A[1, 1]) == 8
    assert max(np.abs(A[2, 3]/A[3, 2]), np.abs(A[3, 2]/A[2, 3])) == 5
    assert np.allclose(wanted_eig, eig, atol=1e-3)


def test_get_diag_ssm_plus_noise():
    import torch
    import numpy as np
    from src.algorithms.ssm_init1D import get_diag_ssm_plus_noise_init

    A, B, C, D = get_diag_ssm_plus_noise_init(num_hidden_state=2,
                                              A_diag=0.9,
                                              A_noise_std=0,
                                              B_init_std=0,
                                              C_init_std=0)

    assert torch.norm(B) == 0
    assert torch.norm(C) == 0
    assert torch.norm(D) == 0
    assert np.allclose(A, torch.eye(2)*0.9)


def test_get_hippo_cont():
    import torch
    import numpy as np
    from src.algorithms.ssm_init1D import get_hippo_cont_init

    A, B, C, D = get_hippo_cont_init(num_hidden_state=2,
                                     C_init_std=0)

    assert torch.norm(C) == 0
    assert torch.norm(D) == 0
    assert np.allclose(B, [[1], [3**0.5]])
    assert np.allclose(A[1, 0], -1*3**0.5)

def test_get_hippo_cont():
    import torch
    import numpy as np
    from src.algorithms.misc import matrix_to_real_2x2block_matrix_with_same_eigenvalues

    n = 20
    A = torch.randn([n, n])
    B = matrix_to_real_2x2block_matrix_with_same_eigenvalues(A)
    sorted_eig_A = np.sort_complex(torch.linalg.eig(A)[0].tolist())
    sorted_eig_B = np.sort_complex(torch.linalg.eig(B)[0].tolist())
    assert np.allclose(sorted_eig_B, sorted_eig_A)
    assert B.dtype==A.dtype