

top_fssm_res_linear = [
    {'lr': '0.001', 'noise': '0.001',
     'diag_init': '0.93', 'BC_std': '0.1', 'opt': 'adam',
     'model_name': 'ssm'},
    {'lr': '0.001', 'noise': '0.01',
     'diag_init': '0.9', 'BC_std': '0.1', 'opt': 'adam',
     'model_name': 'ssm'},
    {'lr': '0.01', 'noise': '0.01',
     'diag_init': '0.9', 'BC_std': '0.1', 'opt': 'SGD',
     'model_name': 'ssm'},
    {'lr': '0.01', 'noise': '0.01',
     'diag_init': '0.93', 'BC_std': '0.001', 'opt': 'SGD',
     'model_name': 'ssm'},
]

top_hippo_res_linear = [
    {'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo'},
    {'lr': '0.01', 'dt': '0.05',
     'opt': 'adam', 'model_name': 'hippo'},
    {'lr': '0.0001', 'dt': '0.05',
     'opt': 'SGD', 'model_name': 'hippo'},
    {'lr': '5e-05', 'dt': '0.1',
     'opt': 'SGD', 'model_name': 'hippo'}
]

top_hippo_no_res_linear = [
    {'no': 'a', 'lr': '0.1', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_a'},
    {'no': 'ac', 'hidden': '64', 'lag': '60', 'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_ac'},
]

top_rot_res_linear = [
    {'rot_type': 'eq', 'lr': '0.001',
     'opt': 'adam', 'model_name': 'rot'},
    {'rot_type': 'eq', 'lr': '0.005',
     'opt': 'SGD', 'model_name': 'rot'}
]


top_fssm_res = [
    {'lr': 0.001, 'noise': 0.001, 'diag_init': 0.9,
     'BC_std': 0.1, 'opt': 'adam', 'model_name': 'ssm'},
    {'lr': 0.001, 'noise': 0.0001, 'diag_init': 0.9,
     'BC_std': 0.1, 'opt': 'adam', 'model_name': 'ssm'},
]

top_rot_res = [
    {'lr': 0.001, 'rot_type': 'one_over',
     'opt': 'adam', 'model_name': 'rot_one_over'},
    {'lr': 0.001, 'rot_type': 'eq',
     'opt': 'adam', 'model_name': 'rot_eq'}
]

top_deep_rot_res = [
    {'lr': 0.001, 'rot_type': 'eq',
     'opt': 'adam', 'model_name': 'hippo_eq',
     'depth': 1},
    {'lr': 0.0001, 'rot_type': 'one_over',
     'opt': 'adam', 'model_name': 'hippo_one_over',
     'depth': 2},
    {'lr': 0.0001, 'rot_type': 'one_over',
     'opt': 'adam', 'model_name': 'hippo_one_over',
     'depth': 3},
    {'lr': 0.001, 'rot_type': 'one_over',
     'opt': 'adam', 'model_name': 'hippo_one_over',
     'depth': 1},
    {'lr': 0.001, 'rot_type': 'eq',
     'opt': 'adam', 'model_name': 'hippo_eq',
     'depth': 2},
    {'lr': 0.001, 'rot_type': 'eq',
     'opt': 'adam', 'model_name': 'hippo_eq',
     'depth': 3}
]

top_deep_fssm_res = [
    {'lr': 0.001, 'noise': 0.001,
     'diag_init': 1, 'BC_std': 0.01,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 2},
    {'lr': 0.001, 'noise': 0.01,
     'diag_init': 0.95, 'BC_std': 0.1,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 2},
    {'lr': 0.001, 'noise': 0.0001,
     'diag_init': 0.95, 'BC_std': 0.05,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 3},
    {'lr': 0.001, 'noise': 0.001,
     'diag_init': 1.05, 'BC_std': 0.05,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 3},
    {'lr': 0.001, 'noise': 0.01,
     'diag_init': 0.9, 'BC_std': 0.1,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 1},
    {'lr': 0.001, 'noise': 0.01,
     'diag_init': 0.95, 'BC_std': 0.05,
     'opt': 'adam', 'model_name': 'ssm',
     'depth': 1}
]
