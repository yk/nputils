import numpy as np
import matplotlib.pyplot as plt


def simplex_volume(simplex_points):
    return np.abs(np.linalg.det(np.transpose(simplex_points[1:] - simplex_points[0])))


def calculate_sparsity(data):
    nonz = np.count_nonzero(data)
    return float(data.size - nonz) / data.size


def average_auc(scores):
    return np.mean(scores)


def sign(a):
    s = np.sign(a)
    s[s == 0] = 1
    return s


def msqrt(M):
    (u, s, vt) = np.linalg.svd(M)
    ms = np.dot(u, np.dot(np.diag(np.sqrt(s)), vt))
    return ms


def score_log_plot(results, title, subtract_min=True):
    """
    results must be a list of tuples
    (label, xs, means, stds)
    """
    minScore = np.min([r[2] for r in results]) + 1e-13 if subtract_min else 0.0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for result in results:
        meanScores = result[2]
        meanScores -= minScore
        stdScores = result[3]
        epochs = result[1]
        base_line, = ax.plot(epochs, meanScores, label=str(result[0]))
        ax.fill_between(epochs, meanScores - stdScores, meanScores + stdScores, alpha=0.5,
                        facecolor=base_line.get_color())
    ax.set_yscale('log')
    ax.legend(loc=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective Value")
    ax.set_title("title")

