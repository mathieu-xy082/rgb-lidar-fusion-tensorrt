# Tutorat pratique — rgb-lidar-fusion-tensorrt

Objectif : s'approprier le repo RGB-LiDAR Fusion TensorRT par la pratique, séance par séance.

Repo étudié : `.`

Pour distinguer le pipeline implémenté de l'architecture BEV cible, consulter
d'abord l'[index documentaire](../README.md).

## Parcours

1. [Séance 1 — Lancer et lire le repo](./seance-01-lancer-et-lire.md)
2. [Séance 2 — Géométrie caméra/LiDAR](./seance-02-geometrie-camera-lidar.md)
3. [Séance 3 — Cartes LiDAR sparse et enriched](./seance-03-sparse-enriched-splatting.md)
4. [Séance 4 — Mini-échantillon KITTI réel](./seance-04-mini-kitti-reel.md)
5. [Séance 5 — Vision d'ensemble du pipeline et des blocs du repo](./seance-05-vision-ensemble-pipeline.md)
6. [Séance 6 — Inspecter les `.npz` et les tenseurs LiDAR](./seance-06-inspecter-npz-lidar.md)
7. [Séance 7 — Dataset → batch → modèle](./seance-07-dataset-batch-modele.md)
8. [Séance 8 — Forward pass, mini-entraînement, ONNX et TensorRT](./seance-08-forward-training-onnx-tensorrt.md)

## Compléments interactifs

- Après les séances 2 et 4 :
  [`01_project_lidar_on_image.ipynb`](../../notebooks/01_project_lidar_on_image.ipynb)
  pour contrôler la projection d'un scan KITTI sur l'image.
- Après les séances 3 et 7 :
  [`02_visualize_hybrid_input.ipynb`](../../notebooks/02_visualize_hybrid_input.ipynb)
  pour inspecter les 6 cartes sparse, les 2 cartes splattées et l'entrée modèle.
- Pour la validation du backbone camera-depth :
  [`03_camera_depth_training_analysis.ipynb`](../../notebooks/03_camera_depth_training_analysis.ipynb).

## Méthode

Pour chaque séance :

- objectif court ;
- commandes à lancer ;
- sorties attendues ;
- interprétation ;
- mini-exercice.

Ces supports sont versionnés dans `docs/onboarding/` pour servir de parcours de prise en main pratique du repo.
