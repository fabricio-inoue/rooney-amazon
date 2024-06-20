
# Análise e Segmentação de Imagens com Python

Este script realiza várias tarefas de processamento de imagens usando bibliotecas Python, como `pandas`, `seaborn`, `matplotlib`, `cv2` e `sklearn`. As etapas incluem a leitura de dados, plotagem da distribuição de tags, carregamento e exibição de imagens, normalização de imagens para t-SNE e segmentação de imagens usando a clusterização K-Means.

## Requisitos

Para executar este script, você precisa das seguintes bibliotecas Python:

- `pandas`
- `seaborn`
- `matplotlib`
- `cv2` (OpenCV)
- `os`
- `numpy`
- `tqdm`
- `sklearn`

Instale-as usando pip se ainda não tiver instalado:

```bash
pip install pandas seaborn matplotlib opencv-python tqdm scikit-learn
```


## Carregamento de Dados e Plotagem da Distribuição de Tags

Primeiro, carregamos o conjunto de dados e plotamos a distribuição das tags.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('planet/planet/train_v2.csv/train_v2.csv')

df['tags'] = df['tags'].apply(lambda x: x.split(' '))
all_tags = [tag for tags in df['tags'] for tag in tags]
tag_counts = pd.Series(all_tags).value_counts()

sns.barplot(x=tag_counts.index, y=tag_counts.values)
plt.xticks(rotation=90)
plt.show()
```

## Carregamento e Exibição de Imagens

Funções para carregar e exibir imagens usando OpenCV.

```python
import cv2
import os

def get_image(name, folder='planet/planet/train-jpg', ext='jpg'):
    img_path = f'{folder}/{name}.{ext}'
    if not os.path.exists(img_path):
        print(f"Image {img_path} does not exist.")
        return None
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}.")
    return img

def show_image(img, title=''):
    if img is not None:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"Cannot display image {title} as it is None.")

example_image_name = 'train_10'
example_image = get_image(example_image_name)
show_image(example_image, title=example_image_name)
```

## Normalização de Imagens e Visualização t-SNE

Normalize imagens e visualize usando t-SNE.

```python
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

def normalize_images(image_names):
    images = []
    for name in tqdm(image_names):
        img = get_image(name)
        if img is not None:
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX).reshape(-1)
            images.append(img)
    if images:
        return np.vstack(images)
    else:
        print("No images were loaded for normalization.")
        return np.array([])

sample_images = df.sample(1000, random_state=42)['image_name']
img_matrix = normalize_images(sample_images)
if img_matrix.size > 0:
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(img_matrix)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title('t-SNE Results')
    plt.show()
else:
    print("No valid images to perform t-SNE.")
```

## Segmentação de Imagens usando K-Means

Segmente imagens e calcule as porcentagens dos clusters.

```python
from sklearn.cluster import KMeans
import numpy as np

def segment_image(image_name, n_clusters=3):
    img = get_image(image_name)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(img.shape)
        labels = kmeans.labels_
        return segmented_img, labels, img.shape
    else:
        return None, None, None

def calculate_cluster_percentages(labels, img_shape, n_clusters=3):
    total_pixels = img_shape[0] * img_shape[1]
    cluster_percentages = [(labels == i).sum() / total_pixels * 100 for i in range(n_clusters)]
    return cluster_percentages

def analyze_image(image_name):
    df = pd.read_csv('planet/planet/train_v2.csv/train_v2.csv')

    labels = df[df['image_name'] == image_name]['tags'].values[0].split(' ')

    n_clusters=len(labels)-1

    segmented_img, labels, img_shape = segment_image(image_name, n_clusters)
    if segmented_img is not None:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        original_img = get_image(image_name)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image: {image_name}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_img.astype('uint8'))
        plt.title(f'Segmented Image: {image_name}')
        plt.axis('off')

        print("Porcentagem de clusters:")

        cluster_percentages = calculate_cluster_percentages(labels, img_shape, n_clusters)
        for i, percentage in enumerate(cluster_percentages):
            print(f"Cluster {i}: {percentage:.2f}%")

        plt.show()
    else:
        print(f"Failed to analyze image {image_name}.")
```

## Analisar Imagens Aleatórias

Analise um conjunto de imagens aleatórias.

```python
import random

for i in range(10):
    analyze_image(f'train_{random.randint(0, 40000)}')
```

Este script demonstra como realizar várias tarefas de processamento de imagens, incluindo análise de tags, normalização de imagens, visualização t-SNE e segmentação de imagens usando a clusterização K-Means.

```

```
