

top_fssm_res = [
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

top_hippo_res = [
    {'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo'},
    {'lr': '0.01', 'dt': '0.05',
     'opt': 'adam', 'model_name': 'hippo'},
    {'lr': '0.0001', 'dt': '0.05',
     'opt': 'SGD', 'model_name': 'hippo'},
    {'lr': '5e-05', 'dt': '0.1',
     'opt': 'SGD', 'model_name': 'hippo'}
]

top_hippo_no_res = [
    {'no': 'a', 'lr': '0.1', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_a'},
    {'no': 'ac', 'hidden': '64', 'lag': '60', 'lr': '0.001', 'dt': '0.1',
     'opt': 'adam', 'model_name': 'hippo_no_ac'},
]

top_rot_res = [
    {'rot_type': 'eq', 'lr': '0.001',
     'opt': 'adam', 'model_name': 'rot'},
    {'rot_type': 'eq', 'hidden': '64', 'lag': '60', 'lr': '0.005',
     'opt': 'SGD', 'model_name': 'rot'}
]
