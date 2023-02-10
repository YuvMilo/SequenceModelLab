import torch


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
