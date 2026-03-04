import numpy as np


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def stability_jaccard(neigh_a, neigh_b, query_ids):
    values = []
    for pid in query_ids:
        values.append(jaccard(neigh_a[pid], neigh_b[pid]))

    return float(np.mean(values)) if values else float("nan")