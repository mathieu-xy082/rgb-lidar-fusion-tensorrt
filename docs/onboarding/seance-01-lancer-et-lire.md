# Séance 1 — Lancer et lire le repo

## Objectif

Être capable de lancer le repo, vérifier l'environnement Python/PDM, exécuter la validation, générer les artefacts demo, puis comprendre les sorties principales.

## Étape A — Vérifier l'environnement

À lancer :

```bash
cd .
git status --short --branch
PDM_IGNORE_ACTIVE_VENV=1 pdm run python -c "import sys; print(sys.executable)"
```

Sorties attendues :

```text
## main...origin/main
./.venv/bin/python
```

Interprétation :

- `main...origin/main` : branche locale alignée avec GitHub.
- Aucun fichier listé après le status : working tree propre.
- `.venv/bin/python` : PDM utilise bien l'environnement virtuel du repo.

## Étape B — Installer puis lancer la validation globale

Si `pdm run validate` répond `Command 'pytest' is not found in your PATH`, installer d'abord les dépendances du repo :

```bash
pdm install -G dev -G ml
```

Puis lancer :

```bash
pdm run validate
```

Si PDM réutilise un environnement virtuel externe au lieu de `.venv`, utiliser ponctuellement :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

Alternative confortable : exporter une fois par terminal ou créer un alias shell :

```bash
export PDM_IGNORE_ACTIVE_VENV=1
# ou
alias prun='PDM_IGNORE_ACTIVE_VENV=1 pdm run'
alias pinstall='PDM_IGNORE_ACTIVE_VENV=1 pdm install'
```

À observer :

- nombre de tests passés ;
- `projected_points` ;
- `lidar_maps_shape` ;
- `model_input_shapes`.

## Étape C — Générer les artefacts demo

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run demo-artifacts
```

Puis :

```bash
find results/demo_artifacts -maxdepth 3 -type f
```

À observer :

```text
synthetic_overlay.ppm
synthetic_sparse_maps.npz
synthetic_splatted_maps.npz
synthetic_baseline_inputs.npz
manifest.json
```

## Étape D — Lire le manifeste

```bash
python -m json.tool results/demo_artifacts/manifest.json
```

À comprendre :

- shapes des entrées ;
- nombre de points projetés ;
- nombre de pixels sparse/splatted ;
- chemins produits ;
- hashes SHA-256.

## Question de fin de séance

Quelle est la différence entre :

```text
lidar_maps_shape=(6, H, W)
```

et :

```text
model_input_shapes.lidar_maps=[1, 8, H, W]
```
