# HSRUIE-Reproduction

《Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement》复现记录。

- **论文**：IEEE Transactions on Image Processing, vol. 35, pp. 42–56, 2026。DOI: `10.1109/TIP.2025.3647323`
- **官方代码**：[gemyumeng/HSRUIE](https://github.com/gemyumeng/HSRUIE)，commit `397da5e492b309f5c1af4f64df73c5a647e038cd`
- **本仓库定位**：组会汇报 + 云端**公开权重推理复现（Level A）**的完整存档。不包含、也不声称完成论文的全量训练复现。

## 目录结构

| 目录 | 内容 |
|---|---|
| `code/` | 官方代码克隆（commit `397da5e`，源码零改动）+ 全部 5 个权重（`code/checkpoints/maps_cyclegan/`）。其中 `latest_net_NC.pth` 官方仅发布于 Google Drive，本仓库收录了校验过的副本；其余 4 个随官方仓库发布 |
| `artifacts/` | 云端推理结果与证据：输出图、指标（`metrics.csv/json`）、输入输出对照表、与论文原图对比、可视化报告 HTML、环境与运行日志、复现报告 `HSRUIE_reproduction_report.md` |
| `docs/` | `HSRUIE_复现交接文档.md`（权重清单与哈希、环境搭建、运行命令、验收检查）；`resume_nc.sh`（NC 权重的 Google Drive 断点续传下载脚本） |
| `paper/` | 论文出版版 PDF 与全文文本。**出版商版权材料，仅限个人科研使用** |
| `slides/` | 组会文献汇报 PPT（30 页） |
| `archive/` | 云端复现产物打包 `HSRUIE_cloud_reproduction_artifacts.tar.gz` |

## 复现结论摘要

**已完成**（详见 `artifacts/HSRUIE_reproduction_report.md`）：

- 云端 RTX 4090（PyTorch 2.2.2+cu121，官方 pin 的 1.8.0+cu111 因 CDN 受限改用，模型代码与权重未改动）跑通官方 `test.py` 推理路径；
- 测试输入为论文 PDF 第 12 页提取的 8 幅面板图（Lanczos resize 到 256×256，原因见报告），8/8 正常产出增强图，脚本干净退出；
- 8 图总推理耗时 0.8067 s，峰值显存 0.6377 GiB；全部权重（含 VGG19、CLIP RN50）加载校验通过。

**未完成 / 不声称**：

- 论文训练流程：官方仓库**不含 stage 1–2 的训练代码**（`cycle_gan_model.py` 中优化器目标 `netUNET` 从未创建、`define_D` 从未调用、无 ranking/Lsep 等损失实现），全量复现需按论文正文重写；
- UVTD / SUIM / USOD10K 全量评测与 Tables I–VIII。

**重要注意事项**：官方 commit 的 `forward()` 跳过 `netLOW`（`fe1 = fc1`），即公开推理路径绕过了三级层次结构中的一级。因此本仓库结果应表述为"**公开版本推理结果**"，而非论文完整管线的复现。

## 权重与校验

| 文件 | 来源 | 校验 |
|---|---|---|
| `latest_net_NC.pth`（37,238,739 B） | Google Drive（File ID `1SgiK6fBHXd2yICBhNNfoncUWjSUxslYe`） | SHA256 `8a530b5960698b78e638fad2ae2d3b09f789b090cc21d505abeffdf39da57fed` |
| `latest_net_LOW/HIGH2/HIGH3/SAD.pth` | 官方仓库 | 随 commit `397da5e` 发布 |
| `vgg19-dcbb9e9d.pth` | `download.pytorch.org/models/`（torchvision 标准 VGG19） | 574,673,361 B |
| CLIP `RN50.pt` | `openaipublic.azureedge.net`（URL 见 `code/CLIP/clip/clip.py`） | 255,827,503 B |

完整哈希见 `artifacts/environment.txt` 与 `artifacts/input_sha256.txt`。

## 运行方法（简述）

```bash
python test.py \
  --dataroot ./datasets/paper_test_256 \
  --dataset_mode single --name maps_cyclegan --model cycle_gan \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results_paper256 \
  --gpu_ids 0 --num_test 8 --eval
```

在 `code/` 目录内运行。环境要求与搭建步骤、输入预处理原因、验收清单见 `docs/HSRUIE_复现交接文档.md`；本次实际运行的完整命令与云端环境见 `artifacts/HSRUIE_reproduction_report.md`。

## 版权与引用

- `paper/` 目录为出版商版权材料，仅供个人科研使用；转载或引用请以 DOI 为准：
  > Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement. *IEEE Trans. Image Process.*, vol. 35, pp. 42–56, 2026, doi: 10.1109/TIP.2025.3647323.
- 代码与权重的版权归原作者及官方仓库所有；本仓库仅为复现存档。

## 贡献者

单一作者：[HaoyuJiangdasak](https://github.com/HaoyuJiangdasak)。
