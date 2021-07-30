"""Utilities for feature pipelines."""


def batch_list(a, n=4):
    out = []
    tmp = []
    for i in range(len(a)):
        tmp.append(a[i])
        if (i + 1) % n == 0:
            out.append(tmp)
            tmp = []
    if len(tmp) > 0:
        out.append(tmp)
    return out
