from visualizations.vis_utils import vis_hippo_as_a_matrix,\
    animate_SSM_state_dynamics, A_hippo_init_func, B_hippo_init_func

if __name__ == "__main__":
    #vis_hippo_as_a_matrix()
    A = A_hippo_init_func(64)
    B = B_hippo_init_func(64)
    save_path = r"..\\results\\HippoStateDynamic2.mp4"
    animate_SSM_state_dynamics(A, B, save_path=save_path,max_time=0.1)
