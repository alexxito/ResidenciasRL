import numpy as np
import matplotlib.pyplot as plt


def learning_curve(x, scores, file):
    avg = np.zeros(len(scores))
    for i in range(len(avg)):
        avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

        plt.plot(x, avg)
        plt.title('Promedio de los 100 puntajes previos')
        plt.savefig(file)
