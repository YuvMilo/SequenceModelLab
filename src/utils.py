import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

def safe_complex_mm(m1, m2):
    """
    A matrix multiplication function that works if either of m1 or m2 are complex
    (current torch.mm works only if m1 and m2 are of the same type)
    """
    if m1.type() == m2.type():
        return torch.mm(m1, m2)
    else:
        m1 = m1.type(torch.cfloat)
        m2 = m2.type(torch.cfloat)
        return torch.mm(m1, m2)


def plot_log(name, s=0, e=None):
    path = r"C:\Users\yuvmi\PycharmProjects\SequenceModelLab\src\results\lag_exp\\"
    path += name
    history = pickle.load(open(path, "rb"))
    figure, axis = plt.subplots(2, 1)
    if e is None:
        e = len(history["loss"])
    losses = [e.entity for e in history["loss"]][s:e]
    axis[0].plot(losses)
    k = history["kernel"][e-1].entity
    axis[1].plot(k)
    plt.show()
    print(np.min(losses))
