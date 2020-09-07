from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def similarity_between_docs(doc1, doc2, is_1d=False):
    if is_1d:
        v1 = np.reshape(doc1, (1, -1))
        v2 = np.reshape(doc2, (1, -1))
    else:
        d1 = np.mean(doc1, axis=0)
        d2 = np.mean(doc2, axis=0)
        v1 = np.reshape(d1, (1, -1))
        v2 = np.reshape(d2, (1, -1))
    return cosine_similarity(v1, v2)[0][0]

def plot_1d_heatmap(vec, name):
    v = vec.reshape(1, -1)
    plt.figure(figsize=(20, 2))
    sns.heatmap(v).set_title(name)
    plt.show()
    return 