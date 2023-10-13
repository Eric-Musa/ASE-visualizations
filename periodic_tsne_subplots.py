import os
from ase.data import chemical_symbols
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import numpy as np
from elemental_encodings import KHOT_ENCODINGS as khot, QMOF_KHOT_ENCODINGS as qmof

x = np.array(list(khot.values()))
initializations = [['CGCNN', np.array(list(khot.values()))], ['QMOF', np.array(list(qmof.values())[:100])]]
transforms = [['Unscaled', None], ]
transform = None
n_components = 2

seeds = [111, 121]

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
for seed in seeds:
    rs = np.random.RandomState(seed)
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, method='exact', random_state=rs)
    umap = UMAP(n_components=n_components, random_state=rs)
    embeddings = [['PCA', pca], ['T-SNE', tsne], ['UMAP', umap]]
    for j, (embtitle, embedding) in enumerate(embeddings):
        # for k, (tratitle, transform) in enumerate(transforms):
        for k, (inittitle, x) in enumerate(initializations):
            fig, ax = plt.subplots()
            fig.set_size_inches(16., 9.)
            x_scaled = transform.fit_transform(
                x) if transform is not None else x
            x_embedding = embedding.fit_transform(x_scaled)
            ax.scatter(x_embedding[:, 0], x_embedding[:, 1], c=group_colors, s=150)
            for i, name in enumerate(names):
                ax.annotate(name, x_embedding[i, :], fontsize=16)
            title = '%s embedding of %s atom initializations' % (embtitle, inittitle)
            ax.title.set_text(title)
            ax.title.set_fontsize(30)
            print(title)
            # fig.show()
            # input()
            image_dir = 'images/%d_subplots' % seed
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            fig.savefig('%s/periodic_viz_%s-%s.png' % (image_dir, inittitle, embtitle))
            plt.close(fig)
