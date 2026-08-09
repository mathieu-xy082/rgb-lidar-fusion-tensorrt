# Bibliographie initiale — RGB/LiDAR fusion, depth completion, splatting, temps réel

Date de première passe : 2026-08-03.

Cette liste est volontairement légère : elle conserve les identifiants, DOI et liens
de téléchargement quand ils sont disponibles. Les PDF peuvent être téléchargés
localement dans `docs/articles/biblio/`, qui n'est pas suivi par Git.

## Références clés

| Thème | Publication | Année | DOI | arXiv / PDF |
|---|---:|---:|---|---|
| Fusion multi-vues historique | Chen et al., *Multi-View 3D Object Detection Network for Autonomous Driving* | 2017 | [10.1109/CVPR.2017.691](https://doi.org/10.1109/CVPR.2017.691) | [arXiv:1611.07759](https://arxiv.org/abs/1611.07759v3) / [PDF](https://arxiv.org/pdf/1611.07759v3) |
| Fusion image/BEV/LiDAR | Ku et al., *Joint 3D Proposal Generation and Object Detection from View Aggregation* / AVOD | 2018 | [10.1109/IROS.2018.8594049](https://doi.org/10.1109/IROS.2018.8594049) | [arXiv:1712.02294](https://arxiv.org/abs/1712.02294v4) / [PDF](https://arxiv.org/pdf/1712.02294v4) |
| Fusion séquentielle sémantique → points | Vora et al., *PointPainting: Sequential Fusion for 3D Object Detection* | 2020 | [10.1109/CVPR42600.2020.00466](https://doi.org/10.1109/CVPR42600.2020.00466) | [arXiv:1911.10150](https://arxiv.org/abs/1911.10150v2) / [PDF](https://arxiv.org/pdf/1911.10150v2) |
| Baseline LiDAR efficace | Lang et al., *PointPillars: Fast Encoders for Object Detection from Point Clouds* | 2019 | [10.1109/CVPR.2019.01298](https://doi.org/10.1109/CVPR.2019.01298) | [arXiv:1812.05784](https://arxiv.org/abs/1812.05784v2) / [PDF](https://arxiv.org/pdf/1812.05784v2) |
| RGB-D/frustum | Qi et al., *Frustum PointNets for 3D Object Detection from RGB-D Data* | 2018 | [10.1109/CVPR.2018.00102](https://doi.org/10.1109/CVPR.2018.00102) | [arXiv:1711.08488](https://arxiv.org/abs/1711.08488v2) / [PDF](https://arxiv.org/pdf/1711.08488v2) |
| Depth completion RGB + LiDAR sparse | Ma et al., *Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera* | 2019 | [10.1109/ICRA.2019.8793637](https://doi.org/10.1109/ICRA.2019.8793637) | [arXiv:1807.00275](https://arxiv.org/abs/1807.00275v2) / [PDF](https://arxiv.org/pdf/1807.00275v2) |
| Depth completion avec normales | Qiu et al., *DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image* | 2019 | [10.1109/CVPR.2019.00343](https://doi.org/10.1109/CVPR.2019.00343) | [arXiv:1812.00488](https://arxiv.org/abs/1812.00488v2) / [PDF](https://arxiv.org/pdf/1812.00488v2) |
| Fusion continue image/LiDAR | Liang et al., *Deep Continuous Fusion for Multi-Sensor 3D Object Detection* | 2018 | [10.1007/978-3-030-01270-0_39](https://doi.org/10.1007/978-3-030-01270-0_39) | Springer / ECCV |
| Fusion multi-tâche multi-capteurs | Liang et al., *Multi-Task Multi-Sensor Fusion for 3D Object Detection* | 2019 | [10.1109/CVPR.2019.00752](https://doi.org/10.1109/CVPR.2019.00752) | CVPR |
| Fusion BEV moderne | Liu et al., *BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation* | 2023 | [10.1109/ICRA48891.2023.10160968](https://doi.org/10.1109/ICRA48891.2023.10160968) | [arXiv:2205.13542](https://arxiv.org/abs/2205.13542v3) / [PDF](https://arxiv.org/pdf/2205.13542v3) |
| Fusion transformer robuste | Bai et al., *TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers* | 2022 | [10.1109/CVPR52688.2022.00116](https://doi.org/10.1109/CVPR52688.2022.00116) | [arXiv:2203.11496](https://arxiv.org/abs/2203.11496v1) / [PDF](https://arxiv.org/pdf/2203.11496v1) |
| Splat/lattice sur nuages de points | Su et al., *SPLATNet: Sparse Lattice Networks for Point Cloud Processing* | 2018 | [10.1109/CVPR.2018.00268](https://doi.org/10.1109/CVPR.2018.00268) | [arXiv:1802.08275](https://arxiv.org/abs/1802.08275v4) / [PDF](https://arxiv.org/pdf/1802.08275v4) |

## À creuser ensuite

- Depth completion temps réel et architectures légères : PENet, GuideNet, NLSPN,
  CSPN/CSPN++.
- Fusion sous contraintes embarquées : quantization-aware fusion, TensorRT, INT8,
  pruning, distillation.
- Robustesse capteurs : mauvais calibrage, occlusions, pluie/brouillard, faible
  densité LiDAR, désynchronisation caméra/LiDAR.
- Benchmarks : KITTI depth completion, KITTI 3D object detection, nuScenes,
  Waymo Open Dataset.

## Commandes de téléchargement PDF local

Exemples :

```bash
mkdir -p docs/articles/biblio
wget -O docs/articles/biblio/mv3d-1611.07759.pdf https://arxiv.org/pdf/1611.07759v3
wget -O docs/articles/biblio/avod-1712.02294.pdf https://arxiv.org/pdf/1712.02294v4
wget -O docs/articles/biblio/pointpainting-1911.10150.pdf https://arxiv.org/pdf/1911.10150v2
wget -O docs/articles/biblio/pointpillars-1812.05784.pdf https://arxiv.org/pdf/1812.05784v2
wget -O docs/articles/biblio/deeplidar-1812.00488.pdf https://arxiv.org/pdf/1812.00488v2
wget -O docs/articles/biblio/sparse-to-dense-1807.00275.pdf https://arxiv.org/pdf/1807.00275v2
wget -O docs/articles/biblio/bevfusion-2205.13542.pdf https://arxiv.org/pdf/2205.13542v3
wget -O docs/articles/biblio/transfusion-2203.11496.pdf https://arxiv.org/pdf/2203.11496v1
wget -O docs/articles/biblio/splatnet-1802.08275.pdf https://arxiv.org/pdf/1802.08275v4
```
