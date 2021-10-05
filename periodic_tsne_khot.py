from ase.data import chemical_symbols
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import numpy as np
from khot_embeddings import KHOT_EMBEDDINGS as khot
from qmof_khot_embeddings import QMOF_KHOT_EMBEDDINGS as qmof
from ase.data import atomic_numbers, atomic_names
import json

x = np.array(list(khot.values()))
initializations = [['CGCNN', np.array(list(khot.values()))], [
    'QMOF', np.array(list(qmof.values())[:100])]]
transforms = [['Unscaled', None], ]
# min_max_scaler = preprocessing.MinMaxScaler()
# normalized_table = pd.DataFrame(x_scaled)
transform = None
n_components = 2
seeds = [111, 121, 150, 500]
for seed in seeds:
    rs = np.random.RandomState(seed)
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, method='exact', random_state=rs)
    umap = UMAP(n_components=n_components, random_state=rs)
    embeddings = [['PCA', pca], ['T-SNE', tsne], ['UMAP', umap]]

    fig, axs = plt.subplots(len(initializations), len(embeddings))
    fig.set_size_inches(24., 13.)
    colors = [
        'bisque',
        'black',
        'salmon',
        'red',
        'lightgrey',
        'lightslategray',
        'slategray',
        'dimgrey',
        'steelblue',
        'royalblue',
        'blue',
        'teal',
        'seagreen',
        'darkolivegreen',
        'plum',
        'mediumorchid',
        'fuchsia',
        'deeppink',
        'indigo',
    ]
    # group_colors = [mcolors.CSS4_COLORS[colors[int(i)-1]] for i in relevant_table['group_id']]
    # names = [chemical_symbols[i] + str(i) for i in relevant_table.index]
    group_colors = [
        mcolors.CSS4_COLORS[colors[int(np.where(i[:19] == 1.)[0])]] for i in x]
    names = [chemical_symbols[i+1] + str(i+1) for i in range(len(x))]
    for j, (embtitle, embedding) in enumerate(embeddings):
        # for k, (tratitle, transform) in enumerate(transforms):
        for k, (inittitle, x) in enumerate(initializations):
            ax = axs[k, j]
            x_scaled = transform.fit_transform(
                x) if transform is not None else x
            x_embedding = embedding.fit_transform(x_scaled)
            ax.scatter(x_embedding[:, 0], x_embedding[:, 1], c=group_colors)
            for i, name in enumerate(names):
                ax.annotate(name, x_embedding[i, :])
            title = '%s: %s' % (embtitle, inittitle)
            ax.title.set_text(title)
            print(title)
    fig.show()
    input()
    # fig.savefig('images/tsne_elements_%d.png' % seed)
    plt.close(fig)
