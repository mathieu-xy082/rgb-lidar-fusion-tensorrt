# Changelog

All notable changes to `rgb-lidar-fusion-tensorrt` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

- Prepare the matched BEV pilot with a single protocol, canonical saved split,
  strict source audits and identical fresh fusion/shared/head state hashes for
  LiDAR-only and fusion variants. Add a CPU-only audit CLI. Record the completed four-frame CUDA control: 15 TP,
  no FP/FN and verified unchanged encoder weights/buffers.
- Add the streaming per-variant BEV pilot runner/CLI: train-only updates,
  both-split evaluation, valid negatives/unknown reporting, strict pilot resume
  and atomic latest/best snapshots with sidecar recovery. Verify native targets,
  frozen encoders, exact CPU resume and guarded CUDA refusal; real pilot GPU
  runs and the comparison notebook remain pending.
- Add an audited frozen-encoder BEV fusion four-frame runner, CLI, CPU feature
  inspection and opt-in notebook `07`. Restrict memorization to saved training
  IDs; record task/schedule/source provenance, stream batch progress and resume
  fused Adam checkpoints. CPU tests and one real-KITTI epoch pass; GPU
  memorization/peak-memory validation and the disjoint comparison remain pending.
- Assemble `BEVFusionDiagnostic` with existing camera/lift and LiDAR encoders,
  fusion/shared backbone and the optional-by-use center head. Add target-free
  CPU batch preparation, explicit frozen encoder/BN policy and a model JSON.
  Add audited encoder-only warm starts and distinct `bev_fusion_center_v1`
  inference/resume helpers, including strict split/hash/config checks. Synthetic
  CPU backward/Adam resume and a read-only local weight audit pass; legacy
  camera ID/hash provenance remains explicitly limited. No KITTI multimodal run.
- Add `SharedBEVBackbone`: three-scale residual context at widths 32/64/96,
  aligned stride-1 output through explicit transposed-convolution sizes and
  unchanged evidence snapshots. Controlled CPU tests verify the phase-dependent
  39/43-cell receptive field; a bounded synthetic CPU probe records parameter
  storage and forward/backward timing. This block's validation is CPU-only,
  not an end-to-end latency, GPU or training-quality result.
- Add cell-local `BEVFeatureFusion`: explicit metric-grid checks, ordered
  camera/contextual-LiDAR/support concatenation and a configurable bias-free
  1x1 projection (66 to 32 by default). `FusedBEVFeatures` preserves separate
  count/mass snapshots and support masks; 81 CPU tests cover contracts,
  branch-absence controls and gradients, including existing lift/PFN/context.
  The spatial backbone and model assembly are separate components; no new
  multimodal training-quality result is claimed.
- Document the first shared BEV fusion contract and six implementation
  increments: aligned 20 cm grids, camera/LiDAR feature and evidence policies,
  ordered 66-to-32 fusion, checkpoint/split safeguards and diagnostic validation.
  Link the new workstream and mark the LiDAR prototype integrated in 1.3.0;
  the contract-only increment preceded fusion code; multimodal training remains planned.

### Fixed

- Install the existing `visualization` dependency group in GitHub validation,
  force/check headless Agg rendering, and skip only graphics-dependent BEV
  tests when Matplotlib is absent from a minimal local environment. Add a
  missing-Matplotlib regression without uninstalling the local dependency.
- Record ordered camera-depth IDs, train/validation membership, triplet SHA-256
  and effective protocol in new atomic `camera_depth_v2` checkpoints/split.json.
  Check resumes before device initialization and verify sample reads; use saved
  membership in notebook analysis/comparison. Keep legacy checkpoints readable
  but unverified, disallow verified resume and require an explicit exception
  for BEV warm starts. Historical files/results are not retroactively certified.
  Include `fix/**` pushes in CI so the dedicated correction branch is validated.
- Install PDM 2.29.4 with its locked dependency set in both CI jobs, avoiding
  the incompatible Hishel 2.x cache API. Check tooling dependencies and import
  the HTTP/cache client before installing project dependencies; no project
  dependency or lockfile changes are required.

## [1.3.0] - 2026-11-05 - Merge branch 'ec/lidar-bev-encoder' into main
### Added

- LiDAR center training on a saved disjoint KITTI frame split (410/102 for the
  local 512-frame pilot), streaming preparation, negative/unknown frame policy,
  independent eval-mode train/validation metrics, latest/best checkpoints and
  checked resume. Notebook `06` defaults to no training. Existing four-frame
  overfit checkpoints remain readable.
- Four-frame LiDAR memorization runner and notebook `05`, validated locally
  on CUDA for 200 epochs/800 steps with 13 matched centers. These are training
  frames only, not a generalization or full-3D-detection result.
- CPU-first `04_lidar_bev_analysis.ipynb` and reusable LiDAR frame inspection,
  showing KITTI boxes, input/context features, supervision domains, targets and
  explicitly untrained predictions without automatic download or training.
- Optional LiDAR-only Car-center/XY-offset head, stable masked focal/L1 objective,
  and metric center decoding. Target construction preserves full calibration,
  converts bottom centers correctly, and excludes unknown/DontCare frustums from
  negative supervision. This is not a mandatory fused-model head or full 3D AP.
- Official label-only preparation for existing KITTI frame IDs, protected
  extraction, provenance and local annotation/geometry/target audit. Labels and
  generated reports remain ignored; annotation preparation itself starts no training.
- NumPy-first raw XYZI pillar construction on the shared configurable BEV grid,
  targeting 20 cm cells. Returns nine point features, explicit padding/source
  indices, pre-cap measurement counts, and reproducible per-pillar overflow
  sampling (64 retained points by default). Empty samples/batches, finite-input
  policy, camera-grid alignment, and input/RNG preservation are tested. This
  grouping step has no learned parameters.
- Mask-aware `PillarFeatureNet` (Linear/BatchNorm/ReLU/max, 32 channels by
  default) and differentiable stride-1 LiDAR BEV scatter preserving raw
  pre-cap measurement evidence. Padding is excluded from normalization and
  pooling; empty/singleton training calls use unchanged BN running statistics.
  CPU tests and a randomly initialized KITTI forward/backward check cover this
  first trainable representation; no LiDAR training runner, detection head,
  or GPU/export validation is introduced.
- Residual LiDAR BEV context with four bias-free stride-1 convolutions at
  dilations `(1, 2, 2, 4)`, preserving the grid and raw measurement evidence.
  Controlled impulse/gradient tests verify a complete 19x19 receptive field.
  Empty/odd/non-square grids and full PFN-to-context backward are CPU-tested.
  A reproducible CPU comparison against a small multi-resolution reference
  records parameter/timing trade-offs, without detection or GPU claims.
- Opt-in joint camera/LiDAR BEV inspection on a shared configurable grid,
  keeping historical camera-only defaults/artifacts and native LiDAR outside
  camera coverage. Figures and NPZ/JSON separate measurements, splat support,
  and random-weight LiDAR features rather than claiming semantic scores.
- Bounded, incremental-log LiDAR-only CPU/CUDA diagnostics for batch-1 FP32
  PFN/scatter/context forward/backward and peak PyTorch allocation, without
  optimizer/checkpoint writes or retries after failure. The local default
  memory guard is 2048 MiB; camera preflight remains at 2500 MiB. One measured
  GPU iteration per mode completed on real KITTI after warmup; this is not an
  end-to-end real-time, detection-quality, or long-run stability result.

### Changed

- Clear notebook execution outputs/counters and restore opt-in action flags,
  retaining reviewed block-holdout and CUDA presets. Executed sessions are kept
  locally under ignored results; no dataset or trained checkpoint is committed.
- Reorganized the learning documentation around peer camera-depth and LiDAR
  branches, their planned BEV convergence and shared operators. Branch pages
  and explorers now offer consistent navigation; documentation checks cover
  local links, anchors, branch structure and SVG syntax. Model code and
  interactive calculations are unchanged.

### Validated

- Close the optional LiDAR center prototype after 20 CUDA epochs/8,200 updates
  on 410 train / 102 validation frames. Best validation-loss epoch is 9
  (2.01414): 220 TP / 87 FP / 125 FN, precision 71.66%, recall 63.77% under
  the configured masked-domain score/radius policy. Later epochs overfit;
  this is not scene-independent generalization, full 3D AP or real-time validation.
- Review validation cases `000009` (found), `000148` (missed), `000388` (FP).
  Retain local `best.pt` for prototype reuse; further auxiliary-head optimization
  is not a prerequisite for camera/LiDAR BEV integration.
- Closure regression: 633 tests passed, one CUDA-only test skipped, no notebook
  checks excluded. Versioned notebooks have no execution outputs/counters.

## [1.2.0] - 2026-11-02 - Merge branch 'ec/camera-depth-bev-lift' into main
### Added

- `CameraDepthModel.forward_features()` exposes full-resolution `[B, 32, H, W]`
  final-decoder features for the future BEV lift, preserving the depth output,
  existing checkpoint schema, and feature-to-backbone gradients.
- Full resized KITTI `P2`, camera-from-LiDAR transform, and depth scale carried
  separately through dataset, model batches, holdout, and fixed evaluation.
- Differentiable metric camera-z conversion and camera-to-LiDAR backprojection,
  preserving full `P2` translation, v1 projection resize, and existing inputs.
  Geometry tests cover frame axes, non-zero translation, resizing, per-sample
  calibration, round trips, and depth gradients.
- Explicit sparse/splatted/predicted depth selection with metric validity masks,
  separate heuristic splat confidence, and mandatory support for learned depth.
  Invalid depths are excluded with a retained mask and safe zero placeholders;
  tests cover leakage under both holdouts and finite geometry gradients. No
  change to training, checkpoints, or model architecture.
- Functional full-resolution camera-feature BEV lift with explicit metric grid,
  height/bounds filtering, masked cell means, optional heuristic splat weighting,
  per-cell counts/mass, and image support diagnostics. Feature/backbone gradients
  are tested; discrete placement does not train depth/calibration coordinates.
- Standalone CPU KITTI BEV inspection with an existing checkpoint, raw-LiDAR
  reference, PNG/JSON/NPZ outputs, and generated-fixture regression tests.
  No BEV detector, new training, GPU latency claim, or lift export is introduced.

### Changed

- Standardized calibration fields, geometry arguments, and batch keys on
  `t_vehicle_to_camera` (`T_vehicle_to_camera` in equations), with an explicit
  projection/backprojection derivation and a tested numerical example in code.
  Legacy JSON manifests using `t_camera_from_vehicle` remain readable; Python
  callers should use the new name. Transform values and model weights are unchanged.

### Validated

- Prototype closure on 2026-11-02: qualitative held-out depth inspection on
  frame 000366 and coherent sparse/unweighted-splat/weighted-splat BEV views on
  frame 000000. Broader scene and dense-depth validation remain deferred.
- Pre-integration review: 357 tests passed, one CUDA-only test skipped, and
  100 randomized BEV aggregations matched an independent NumPy reference.
  Real KITTI weighted-splat inspection completed on CPU with unchanged checkpoint
  SHA-256. This does not establish GPU performance or lift export readiness.

## [1.1.0] - 2026-10-18 - Merge branch 'ec/camera-depth-backbone' into main 
### Added

- Dense `CameraDepthModel` with deterministic sparse LiDAR holdout,
  leakage-safe resplatting, masked Smooth L1 training, and checkpoint support.
- KITTI camera-depth runner with deterministic train/validation splits, fixed
  per-frame validation masks, and held-out MAE/RMSE in meters.
- Random-point and spatial-block LiDAR holdout, with achieved holdout fractions
  and leakage-safe splatted-depth baseline errors and coverage.
- Matched 512-frame experiment configurations and cross-evaluation of both
  checkpoints under both holdout modes.
- Three notebooks for LiDAR projection, hybrid-input exploration, training,
  metric analysis, splatting ablation, and prediction inspection.
- Notebook GPU preflight with driver recovery checks, competing-process and
  free-memory guards, synchronized CUDA probes, and allocator diagnostics.
- Standalone bounded FP32 GPU diagnostic with read-only checkpoint loading,
  synchronized phase markers, incremental JSONL logs, and memory measurements.
- Optional dependency groups for notebook execution and visualization.
- Documentation index, current milestones, explicit BEV design links, and
  separated active/archive workstreams.
- Recorded camera-depth experiment results, sparse-supervision limitations,
  and feature/geometry/validity prerequisites for the future BEV lift.

### Changed

- Pinned PyTorch to `2.13.0+cu129` through the official CUDA 12.9 package index.
- Set camera-depth training to `320x1024`; YAML defaults use batch size 2,
  while the local 4 GiB GPU notebook and matched experiments use batch size 1.
- Reworked the README to distinguish the implemented pipeline, legacy baseline,
  camera-depth representation, and target BEV architecture.

### Fixed

- Restore Adam state against parameters already placed on the training device,
  avoiding mixed CPU/CUDA optimizer tensors when resuming a checkpoint.
- Persist camera-depth metrics and run metadata after each completed epoch,
  preserving completed history if a later epoch fails.
- Release unused CUDA allocator cache before the notebook's free-memory check,
  without lowering the safety threshold or deleting referenced tensors.
- Stop issuing explicit cache-cleanup calls after a failed CUDA smoke probe.

### Validated

- Camera-depth smoke training completed on 64 KITTI Object frames on CPU and
  the local NVIDIA GPU.
- Separate random-point and spatial-block GPU runs completed on 512 KITTI
  Object frames (410 train / 102 validation), five epochs and 2050 steps each.
  All four cross-evaluation combinations completed. See the
  [camera-depth workstream](docs/workstreams/camera-depth-backbone.md) for the
  scores, achieved holdout fractions, and interpretation limits.
- Sparse-versus-enriched input ablation on 64 frames improved held-out errors
  with splatted depth/confidence inputs; this is a single-seed result.
- After the local NVIDIA driver/GSP update from `575.64.03` to `615.71.09`,
  a 100-step GPU diagnostic, the remaining random-point epoch, the full block
  run, and cross-evaluation completed successfully. This does not establish
  the cause of the earlier Xid failures.
- Clean pre-merge validation on 2026-10-18: `232 passed, 1 skipped` (CUDA-only
  regression), lockfile check, deterministic projection/demo artifact checks,
  and a two-epoch CPU baseline training smoke. Remote CI was not rerun as part
  of these local checks.

### Known limitations

- Depth metrics supervise only hidden measured LiDAR pixels, not every dense
  prediction. Broad scene generalization, learned RGB semantics, and prediction uncertainty
  are not established by this single-seed pilot.
- Random and block holdout produce different target sets and achieved removal
  fractions; comparisons must use the same evaluation masks. Splat baseline
  errors include zero predictions outside splat support and require coverage
  to be interpreted correctly.
- Qualitative review remains before camera-depth v1 workstream closure.
  Mixed holdout and larger-dataset/sensitivity studies are optional follow-ups,
  not prerequisites for the first geometric BEV prototype.
- The feature-output interface, BEV lift, LiDAR BEV branch, and 3D detection
  heads are not implemented. Existing ONNX export covers the legacy baseline,
  not `CameraDepthModel`; real TensorRT engine validation remains deferred.
- The local 4 GiB GPU has limited headroom. AMP remains disabled, and the
  successful post-update runs do not guarantee future GPU stability.

## 1.0.0 - 2026-08-02

First stable portfolio/demo release of the RGB-LiDAR Fusion TensorRT prototype. This release establishes the reviewed technical foundation for an AV/robotics ML deployment pipeline: geometry, dataset contracts, enriched LiDAR representation, PyTorch model/training loop, ONNX parity scaffolding, TensorRT runtime planning, reproducible demo artifacts, and GitHub Actions CI.

### Added

- GitHub Actions CI equivalent to the former GitLab validation surface, with lockfile checking, full validation, and deterministic CPU training smoke coverage.
- PDM-based Python project structure with staged validation commands and lightweight default dependencies.
- KITTI calibration parsing and camera/LiDAR projection utilities.
- Synthetic and local KITTI smoke paths for projected LiDAR overlays and sparse LiDAR map generation.
- Lightweight KITTI dataset layer with manifest validation, source-path metadata, and clear dataset index errors.
- Deterministic local LiDAR surface splatting with `depth_expanded` and `confidence` channels.
- Baseline PyTorch RGB+LiDAR fusion model with the nominal enriched LiDAR contract:
  - `rgb: [B, 3, H, W]`
  - `lidar_maps: [B, 8, H, W]`
  - generic concatenated model-batch input `inputs: [B, 11, H, W]`.
- Dataset-to-model and model-input adapter contracts so downstream PyTorch, ONNX, demo, and training code avoid manual tensor slicing.
- ONNX export and ONNX Runtime parity validation for deterministic synthetic baseline inputs.
- Reproducible demo artifact pipeline producing overlay, sparse maps, splatted maps, enriched baseline inputs, and SHA-256 integrity metadata under ignored `results/` paths.
- TensorRT runtime environment detection, guarded build/infer/benchmark script stubs, and documented FP16 benchmark schema for future NVIDIA runtime validation.
- CPU-safe / GPU-ready baseline training loop with deterministic seed handling, explicit device diagnostics, checkpoint save/resume, metrics JSON/CSV, run metadata, and synthetic smoke configuration.
- Artifact policy for keeping datasets, checkpoints, ONNX exports, TensorRT engines, logs, and generated demo outputs out of Git.

### Validated

- GitHub Actions CI passed on the release branch and is integrated into `main`.
- Local validation on `main` passed with `91 passed` plus deterministic smoke/demo artifact generation.
- CPU training smoke completed two epochs, wrote checkpoint/metrics/run metadata outside the repository, and produced finite decreasing losses on the synthetic target.

### Deferred

- Real KITTI training target and trained checkpoint generation.
- GPU-backed training run with recorded device/runtime metadata.
- ONNX export from a trained checkpoint rather than synthetic/random baseline weights.
- TensorRT FP16 engine build and latency benchmark on a verified NVIDIA runtime.
- Detection-style head, post-processing, KITTI metrics, NMS, and real perception-quality evaluation.
