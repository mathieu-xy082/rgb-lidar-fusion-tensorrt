# Séance 8 — Forward pass, mini-entraînement, ONNX et TensorRT

## Objectif

Faire le lien entre les tenseurs préparés et le cycle ML/deployment complet :

```text
batch
→ forward PyTorch
→ training smoke
→ export ONNX
→ préparation TensorRT
```

Cette séance sera probablement ajustée selon l'état des branches de travail et
la disponibilité des dépendances `ml` / `onnx` / TensorRT.

## Bloc A — Vérifier les dépendances ML

```bash
pdm install -G dev -G ml
```

Puis :

```bash
pdm run pytest tests/test_baseline_model.py tests/test_model_batch.py -q
```

## Bloc B — Forward pass PyTorch manuel

```bash
pdm run python - <<'PY'
import torch
from rgb_lidar_fusion.baseline_model import BaselineFusionModel

model = BaselineFusionModel(lidar_mode="enriched", output_dim=1)
rgb = torch.randn(1, 3, 64, 96)
lidar_maps = torch.randn(1, 8, 64, 96)

out = model(rgb, lidar_maps)
print("rgb:", rgb.shape)
print("lidar_maps:", lidar_maps.shape)
print("output:", out.shape)
PY
```

À retenir :

```text
le modèle reçoit deux tenseurs alignés spatialement : RGB et LiDAR maps
```

## Bloc C — Smoke training

Fichiers concernés :

```text
src/rgb_lidar_fusion/training.py
scripts/train_baseline.py
configs/training/synthetic_smoke.yaml
tests/test_training_loop.py
```

Commande probable :

```bash
pdm run python scripts/train_baseline.py \
  --config configs/training/synthetic_smoke.yaml
```

Sorties attendues :

```text
device=...
epoch=...
loss=...
checkpoint=...
metrics=...
```

Les checkpoints et métriques restent hors Git.

## Bloc D — Export ONNX

Dépendances :

```bash
pdm install -G dev -G ml -G onnx
```

Fichiers concernés :

```text
scripts/export_onnx.py
scripts/validate_onnx.py
docs/onnx-export-validation.md
```

Commandes indicatives :

```bash
pdm run python scripts/export_onnx.py --help
pdm run python scripts/validate_onnx.py --help
```

Le but pédagogique est de comprendre :

```text
PyTorch model
→ model.onnx
→ validation de shape/parité
```

## Bloc E — TensorRT

Fichiers concernés :

```text
src/rgb_lidar_fusion/tensorrt_runtime.py
scripts/tensorrt_build_engine.py
scripts/tensorrt_infer.py
scripts/tensorrt_benchmark.py
docs/tensorrt-benchmark-plan.md
```

Commande de diagnostic sûre :

```bash
pdm run python scripts/tensorrt_benchmark.py --detect
```

À retenir :

```text
TensorRT consomme un ONNX ou un engine déjà construit.
TensorRT sert l'inférence optimisée, pas l'entraînement.
```

## Mini-exercice

Remets ces étapes dans l'ordre :

```text
A. TensorRT benchmark
B. Dataset / batch
C. PyTorch training
D. ONNX export
E. Forward model
```

Réponse attendue :

```text
B → E → C → D → A
```
