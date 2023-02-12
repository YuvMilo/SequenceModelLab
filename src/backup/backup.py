"""Some unused functions that might be useful in the future"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

# TODO - Output this into a util that is used by the strategy
def disc_dynamics(A, B, dt):
    eye = np.eye(A.shape[0])
    left_disc_const = np.linalg.inv(eye - A / 2 * dt)
    A = np.matmul(left_disc_const, (eye + A / 2 * dt))
    B = np.matmul(left_disc_const, B * dt)
    return A, B

def plot_single_state_frame(state, title):
    plt.title(title)
    x = np.array([i for i in range(state.shape[0])])
    plt.scatter(x, state, color="orange")
    ax = plt.gca()


def animate_SSM_state_dynamics(A, B, save_path, max_time=2, time_dilation=15,
                            fps=40):
    dt = 1/(fps*time_dilation)
    iterations = int(fps*time_dilation*max_time)

    A, B = disc_dynamics(A, B, dt)

    writer = FFMpegWriter(fps=fps)
    fig = plt.figure()

    x = B
    with writer.saving(fig, save_path, 100):
        for i in tqdm(range(iterations)):
            fig.clear()

            x = np.matmul(A, x)
            title = "time={:.2f},ts=%d".format(dt*(i+1),i)
            plot_single_state_frame(x[:,0], title)
            writer.grab_frame()

def A_hippo_init_func(N):
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)

    return A


def B_hippo_init_func(N):
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    B = B.copy()
    return B

def vis_hippo_as_a_matrix():
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    A1 = A_hippo_init_func(32)
    B1 = B_hippo_init_func(32)
    A2, B2 = disc_dynamics(A1, B1, 0.005)
    A3, B3 = disc_dynamics(A1, B1, 0.01)
    A4, B4 = disc_dynamics(A1, B1, 0.1)
    ax1.title.set_text('continuous')
    ax2.title.set_text('dt=0.005')
    ax3.title.set_text('dt=0.01')
    ax4.title.set_text('dt=0.1')

    im = ax1.imshow(A1, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax1)
    im = ax2.imshow(A2, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax2)
    im = ax3.imshow(A3, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax3)
    im = ax4.imshow(A4, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text('continuous')
    ax2.title.set_text('dt=0.005')
    ax3.title.set_text('dt=0.01')
    ax4.title.set_text('dt=0.1')
    im = ax1.imshow(B1, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax1)
    im = ax2.imshow(B2, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax2)
    im = ax3.imshow(B3, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax3)
    im = ax4.imshow(B4, cmap='hot', interpolation='nearest');
    fig.colorbar(im, ax=ax4)

    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

