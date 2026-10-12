# HSRUIE cloud reproduction report

Date: 2026-10-11 (Asia/Shanghai)

Paper: *Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement*  
DOI: `10.1109/TIP.2025.3647323`  
Official repository: `gemyumeng/HSRUIE`  
Commit: `397da5e492b309f5c1af4f64df73c5a647e038cd`

## Scope

This run verifies the public checkpoint inference path only. It does not claim to reproduce the paper's training, UVTD/SUIM/USOD10K full evaluation, or Tables I–VIII. The official repository has no dataset, and the local material refers to a missing `HSRUIE_detailed_reproduction_guide.md`.

The paper reports UVTD (5,380 images), SUIM (1,634), USOD10K (10,255), 256x256 random crops, 50+50 epochs for stages 1 and 3, 100+100 epochs for stage 2, batch size 2, Adam, and an RTX 2070 workstation. This run uses an RTX 4090 and the released checkpoints, so it is a cloud inference validation under a different runtime.

## Cloud environment

All model loading, inference, PDF image extraction and figure generation were performed on the server under `/root/work`. The report text was assembled from those cloud logs. No local inference was run.

- GPU: NVIDIA GeForce RTX 4090, 24 GB; driver 570.124.04; system CUDA 12.8.
- Python 3.8.20; PyTorch `2.2.2+cu121`; torchvision `0.17.2+cu121`.
- CUDA check passed: `cuda available=True`, capability `(8,9)`, a CUDA 2x3 matrix multiplication succeeded.
- PyTorch 2.2.2 was used because the old 1.8.0+cu111 wheel was repeatedly CDN-limited and targets at most sm_86. The model code and checkpoints were not changed.

Full driver output, package list, runtime check, and SHA256 values are in `environment.txt`.

## Weights

LOW, HIGH2, HIGH3, SAD and NC are in `HSRUIE/checkpoints/maps_cyclegan/`. NC is 37,238,739 bytes and has SHA256 `8a530b5960698b78e638fad2ae2d3b09f789b090cc21d505abeffdf39da57fed`. VGG19 is in `HSRUIE/model_data/vgg19-dcbb9e9d.pth`; CLIP RN50 is in `~/.cache/clip/RN50.pt`. Complete hashes are in `environment.txt`.

## Inputs and preprocessing

The repository does not contain test images. On the server, eight embedded raster panels were extracted from page 12 of the supplied paper PDF and used as auditable smoke-test inputs. The original PDF pages are preserved as visual references.

The released `base_dataset.py` keeps resize/crop/flip commented out. The extracted panels were 305/306 pixels, and the first unmodified-size run failed in `models/FEM.py` with a one-pixel feature concatenation mismatch (`152` vs `153`). Following the handoff document, the eight inputs were resized with Lanczos to 256x256 in the cloud; no model source was patched. The failed log is `test_paper_failed.log`; input hashes are in `input_sha256.txt`.

## Command

```bash
cd /root/work
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hsruie
python HSRUIE/test.py \
  --dataroot ./HSRUIE/datasets/paper_test_256 \
  --dataset_mode single --name maps_cyclegan --model cycle_gan \
  --checkpoints_dir ./HSRUIE/checkpoints \
  --results_dir ./HSRUIE/results_paper256 \
  --gpu_ids 0 --num_test 8 --eval
```

The official commit loads `netLOW` but skips it in `forward()` (`fe1 = fc1`). That behavior was retained and recorded.

## Results and evidence

- `results_paper256/maps_cyclegan/test_latest/index.html`: official result page.
- `input_output_contact_sheet.png`: eight input to `I_e` output pairs.
- `metrics.csv` and `metrics.json`: per-image RGB/luminance/saturation/edge statistics. These are descriptive statistics, not PSNR or SSIM because no ground truth is available.
- `profile.json`: eight-image timing and GPU memory. Total inference time was 0.8067 s; mean including warmup was 0.1008 s/image; peak `torch.cuda.max_memory_allocated()` was 0.6377 GiB.
- `original_fig9_fig10_page12.png`, `original_fig5_page7.png`, `original_fig7_page9.png`, `original_fig11_page13.png`: extracted paper figure regions. `original_pages/` contains high-resolution page 7 through page 13 PNGs.
- `index.html`: browser page containing input/output pairs and original paper references.

The paper images are reference evidence, not pixel-identical comparisons on the same UVTD sample. The paper does not release the raw Fig. 10 input files with this repository, so the PDF-panel smoke test cannot be called a same-image reconstruction of Fig. 10.

## Acceptance

1. RTX 4090 CUDA device and actual tensor computation passed.
2. All public model weights plus VGG19 and CLIP loaded successfully.
3. After the documented 256x256 input preprocessing, the official test script exited cleanly, created `index.html`, and produced 8/8 output images.
4. PNG, HTML, CSV, JSON, hashes, environment logs, original-paper references, and this report are saved in the cloud artifact directory.
5. The evidence supports successful cloud inference with the released weights. It does not support a claim that the paper's full training and table metrics have been reproduced.
