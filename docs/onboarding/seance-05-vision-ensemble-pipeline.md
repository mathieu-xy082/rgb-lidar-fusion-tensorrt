# Séance 5 — Vision d'ensemble du pipeline et des blocs du repo

## Objectif

Comprendre la carte globale du projet avant de descendre dans les détails des
`.npz`, du dataset, du modèle PyTorch, de l'entraînement, de l'export ONNX et de
TensorRT.

À la fin de cette séance, tu dois pouvoir répondre à ces questions :

```text
1. Où sont produites les données d'entraînement ?
2. Où sont transformées les données capteurs en tenseurs ?
3. Où est défini le modèle PyTorch ?
4. Où est gérée la boucle d'entraînement ?
5. Où est produit le fichier ONNX ?
6. Où TensorRT intervient-il ?
```

## Vue d'ensemble

```text
KITTI / données capteurs
        ↓
download_kitti_samples.py
        ↓
png_to_ppm.py
        ↓
smoke_project_lidar.py
        ↓
sparse_maps_<id>.npz
        ↓
splatted_maps_<id>.npz         futur artefact structuré dédié ML
        ↓
dataset.py
        ↓
model_batch.py
        ↓
baseline_model.py
        ↓
training.py / train_baseline.py
        ↓
export_onnx.py
        ↓
tensorrt_build_engine.py
        ↓
tensorrt_infer.py / tensorrt_benchmark.py
```

## Étape A — Se placer sur la branche onboarding

```bash
cd ~/codes/ai-projects/rgb-lidar-fusion-tensorrt/rgb-lidar-fusion-tensorrt
git pull
git status --short --branch
```

Sortie attendue approximative :

```text
## ec/onboarding...origin/ec/onboarding
```

Si tu es sur une autre branche, ce n'est pas bloquant pour lire les fichiers,
mais pour suivre les séances versionnées on privilégie `ec/onboarding`.

## Étape B — Voir les fichiers par grandes zones

```bash
find src scripts tests docs/onboarding configs \
  -path '*/__pycache__/*' -prune -o \
  -type f -print | sort
```

Cette commande ne fait que lister. Elle permet de raccrocher les fichiers aux
blocs ci-dessous.

---

# Bloc A — Données brutes / acquisition

## Rôle

Récupérer ou représenter les données source :

```text
image caméra
scan Velodyne
calibration KITTI
manifest éventuel
```

## Fichiers principaux

```text
docs/onboarding/scripts/download_kitti_samples.py
data/README.md
configs/kitti.yaml
```

## Dossiers concernés

```text
data/kitti/          données source téléchargées, hors Git
results/onboarding/  sorties générées, hors Git
```

## À retenir

```text
data/ = entrées capteurs réelles ou exemples locaux
```

Ces données ne sont pas versionnées.

---

# Bloc B — Projection géométrique LiDAR → image

## Rôle

Transformer un nuage LiDAR 3D en pixels alignés sur la caméra.

```text
calibration KITTI
+ points Velodyne 3D
+ taille/image caméra
→ points projetés dans l'image
→ cartes LiDAR sparse
```

## Fichiers principaux

```text
src/rgb_lidar_fusion/calibration.py
src/rgb_lidar_fusion/project_lidar.py
scripts/smoke_project_lidar.py
tests/test_project_lidar.py
```

## Sorties

```text
overlay_<id>.ppm          visualisation humaine/debug
sparse_maps_<id>.npz      artefact numérique sparse
```

## Réponse à la question importante

Oui : `scripts/smoke_project_lidar.py` est aujourd'hui le point central qui
produit les artefacts numériques sparse `.npz` à partir des données capteurs.

Il ne s'agit pas encore directement de PyTorch. C'est la couche qui prépare des
données numériques que le dataset/batch pourra ensuite exposer au modèle.

---

# Bloc C — Splatting / enrichissement LiDAR

## Rôle

Dilater localement les pixels LiDAR sparse pour obtenir une représentation plus
utile à un CNN.

```text
sparse depth + mask
→ depth_expanded
→ confidence
```

## Fichiers principaux

```text
src/rgb_lidar_fusion/lidar_splatting.py
tests/test_lidar_splatting.py
docs/onboarding/scripts/render_splatting_ppm.py
```

## Branche dédiée en cours

```text
ec/splatted-structured-maps
```

Objectif de cette branche : ajouter un vrai artefact :

```text
splatted_maps_<id>.npz
```

## À retenir

```text
sparse_maps_<id>.npz     = mesure brute projetée
splatted_maps_<id>.npz   = future entrée ML enrichie
```

---

# Bloc D — Dataset / préparation des batches

## Rôle

Transformer des fichiers/artefacts en samples utilisables par le modèle.

## Fichiers principaux

```text
src/rgb_lidar_fusion/dataset.py
src/rgb_lidar_fusion/model_batch.py
tests/test_kitti_dataset.py
tests/test_model_batch.py
```

## Contrat conceptuel

```python
{
    "image": [3, H, W],
    "lidar_maps": [6, H, W],        # sparse aujourd'hui
    "target": ...,
    "meta": ...,
}
```

Puis le batch prépare une entrée modèle :

```text
RGB 3 canaux + LiDAR 6 ou 8 canaux
```

## À retenir

```text
dataset.py     = lit/produit un sample
model_batch.py = empile/adapte les samples pour le modèle
```

---

# Bloc E — Modèle PyTorch

## Rôle

Définir le réseau qui consomme RGB + cartes LiDAR.

## Fichiers principaux

```text
src/rgb_lidar_fusion/baseline_model.py
tests/test_baseline_model.py
```

## Contrat actuel

```text
rgb:        Tensor[B, 3, H, W]
lidar_maps: Tensor[B, L, H, W]
```

Avec :

```text
sparse mode:   L = 6
enriched mode: L = 8
```

## À retenir

Le modèle PyTorch ne lit pas directement les fichiers KITTI. Il reçoit des
tenseurs déjà préparés par le dataset/batch.

---

# Bloc F — Entraînement

## Rôle

Faire tourner optimisation, loss, métriques et checkpoints.

## Fichiers principaux

```text
src/rgb_lidar_fusion/training.py
scripts/train_baseline.py
tests/test_training_loop.py
configs/training/synthetic_smoke.yaml
configs/training/kitti_tiny.yaml
docs/gpu-training-plan.md
```

## Sorties typiques

```text
checkpoints
metrics CSV/JSON
logs
```

Ces sorties doivent rester hors Git.

## À retenir

```text
training = boucle qui ajuste les poids du modèle
```

---

# Bloc G — Export ONNX

## Rôle

Convertir le modèle PyTorch vers un format portable.

## Fichiers principaux

```text
scripts/export_onnx.py
scripts/validate_onnx.py
tests/test_onnx_export_scripts.py
tests/test_onnx_export_readiness.py
docs/onnx-export-validation.md
```

## Sortie

```text
model.onnx
```

## À retenir

```text
ONNX = pont entre PyTorch et runtimes d'inférence
```

---

# Bloc H — TensorRT

## Rôle

Consommer l'ONNX pour construire un moteur optimisé NVIDIA.

## Fichiers principaux

```text
src/rgb_lidar_fusion/tensorrt_runtime.py
scripts/tensorrt_build_engine.py
scripts/tensorrt_infer.py
scripts/tensorrt_benchmark.py
tests/test_tensorrt_scripts.py
tests/test_tensorrt_runtime.py
docs/tensorrt-benchmark-plan.md
docs/workstreams/tensorrt-runtime-environment-check.md
```

## Sorties

```text
.engine
.plan
latency report
```

## À retenir

```text
TensorRT = inférence optimisée, pas entraînement
```

---

## Étape C — Commande de lecture guidée

Pour parcourir rapidement les fichiers principaux :

```bash
for f in \
  scripts/smoke_project_lidar.py \
  src/rgb_lidar_fusion/project_lidar.py \
  src/rgb_lidar_fusion/lidar_splatting.py \
  src/rgb_lidar_fusion/dataset.py \
  src/rgb_lidar_fusion/model_batch.py \
  src/rgb_lidar_fusion/baseline_model.py \
  src/rgb_lidar_fusion/training.py \
  scripts/export_onnx.py \
  src/rgb_lidar_fusion/tensorrt_runtime.py
 do
  echo
  echo "===== $f ====="
  sed -n '1,40p' "$f"
done
```

Le but n'est pas de tout comprendre ligne par ligne. Le but est de reconnaître
la responsabilité de chaque fichier.

## Mini-exercice

Associe chaque fichier à son bloc :

```text
1. scripts/smoke_project_lidar.py
2. src/rgb_lidar_fusion/model_batch.py
3. src/rgb_lidar_fusion/baseline_model.py
4. scripts/export_onnx.py
5. scripts/tensorrt_build_engine.py
```

Choix :

```text
A. modèle PyTorch
B. projection / production d'artefacts sparse
C. export ONNX
D. préparation batch
E. TensorRT build
```

Réponses attendues :

```text
1 → B
2 → D
3 → A
4 → C
5 → E
```
