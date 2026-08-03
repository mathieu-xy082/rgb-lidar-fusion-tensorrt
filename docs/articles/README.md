# Articles et positionnement bibliographique

Ce dossier suit la littérature utile au projet RGB-LiDAR fusion / TensorRT.

## Convention

- Les fichiers Markdown sont versionnés.
- Les PDF authentiques ne sont pas versionnés.
- Les PDF peuvent être téléchargés localement dans :

```text
docs/articles/biblio/
```

Ce sous-dossier est ignoré par Git pour éviter de gonfler le dépôt.

## Fichiers

- [`bibliographie-rgb-lidar-fusion.md`](bibliographie-rgb-lidar-fusion.md) — références clés, DOI, arXiv, liens PDF.
- [`positionnement-rgb-lidar-splatting-tensorrt.md`](positionnement-rgb-lidar-splatting-tensorrt.md) — première analyse : originalité, risques, angles publiables.

## Commandes utiles

Télécharger un PDF localement sans le versionner :

```bash
mkdir -p docs/articles/biblio
wget -O docs/articles/biblio/pointpainting-1911.10150.pdf \
  https://arxiv.org/pdf/1911.10150v2
```

Lister le cache local :

```bash
find docs/articles/biblio -maxdepth 1 -type f | sort
```
