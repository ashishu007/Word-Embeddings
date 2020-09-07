from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

def similarity_between_vecs(vec1, vec2):
    v1 = np.reshape(vec1, (1, -1))
    v2 = np.reshape(vec2, (1, -1))
    return cosine_similarity(v1, v2)[0][0]

def similarity_between_docs(doc1, doc2, is_2d=False):
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
    plt.rcParams["figure.figsize"] = 5,2

    x = np.linspace(-1,1)
    y = vec

    fig, (ax) = plt.subplots(nrows=1, sharex=True)

    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    plt.title(name)
    plt.tight_layout()
    plt.show()
    return 