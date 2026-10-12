# HSRUIE 复现交接文档（推理版）

**论文**：Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement
**出处**：IEEE Transactions on Image Processing, vol. 35 (2026), pp. 42–56，DOI `10.1109/TIP.2025.3647323`
**官方代码**：https://github.com/gemyumeng/HSRUIE
**核验 commit**：`397da5e492b309f5c1af4f64df73c5a647e038cd`

**文档核验日期**：2026-10-11
**核验方式**：本地 clone 全部源码逐行静态核对 + 权重文件二进制解剖 + 网络可达性实测。**未执行过模型推理**，因此本文所有"预期结果"均为静态推断，标注为【待实测】。

**本文范围**：只做**公开权重推理**——租一台卡、装环境、跑通、出增强图。
**不含**：从头训练、下游任务评测、论文表格复现。这些内容见 `HSRUIE_详细复现指南.md`。

---

## 0. 结论与工期

> **权重已全部备齐，无阻塞。租一台 RTX 3090，半天内可出增强图。**

| 项目 | 值 |
|---|---|
| 人力 | **0.5–1 天** |
| GPU 机时 | **< 1 小时** |
| 租机费用 | **约 ¥5–15**（按小时租即可，不必包月） |
| 交付物 | 增强图 + `index.html` 对照页 + 环境/权重记录 |

---

## 1. 权重清单（已全部落实）

| 文件 | 大小 (B) | SHA256 | 来源 | 放置位置 |
|---|---|---|---|---|
| `latest_net_LOW.pth` | 551,277 | `2be0932616dc6b09fd78a333af00ac574c7f468da922bf5e11f4967b2371fa19` | 仓库自带 | `checkpoints/maps_cyclegan/` |
| `latest_net_HIGH2.pth` | 3,223,196 | `a464cff319a17e8cebfbc200f70846925b9964640f79386505e7c6cde7b3f2af` | 仓库自带 | `checkpoints/maps_cyclegan/` |
| `latest_net_HIGH3.pth` | 4,344,399 | `ddbb0f17a1881a14fd4c8823a84e18e8a3c8d181fad3401b2ed0dea179a13eb5` | 仓库自带 | `checkpoints/maps_cyclegan/` |
| `latest_net_SAD.pth` | 5,941,340 | `0d66726fbf23257dead2328bd41afa90fcc145cdc5a41b9d4585a80315c1b63b` | 仓库自带 | `checkpoints/maps_cyclegan/` |
| **`latest_net_NC.pth`** | **37,238,739** | **`8a530b5960698b78e638fad2ae2d3b09f789b090cc21d505abeffdf39da57fed`** | **✅ 已下载**，在本机 `weights_dl/` | `checkpoints/maps_cyclegan/` |
| `vgg19-dcbb9e9d.pth` | 574,673,361 | 【待实测】 | `download.pytorch.org/models/`（**无需 Google Drive**） | `model_data/` |
| CLIP `RN50.pt` | 255,827,503 | 【待实测】 | `openaipublic.azureedge.net`（**无需 Google Drive**） | `~/.cache/clip/` |

**只有 NC 一个文件必须上传**，其余两个由服务器直连下载即可，不必从本机传 575 MB + 256 MB。

### 1.1 NC 权重完整性验证（已完成）

对 `latest_net_NC.pth` 做过 ZIP 二进制解剖，与源码静态推算**精确吻合**：

| 检查项 | 实测值 |
|---|---|
| ZIP 合法性 | ✅ 110 entries |
| 张量存储条目 | 108 个（= 18 ResBlock × 6 张量） |
| 压缩方式 | `0`（stored，未压缩） |
| 原始张量字节 | **37,201,920** |
| dtype | `FloatStorage`（fp32） |
| 参数量 | **9,300,480** = 37,201,920 ÷ 4 |
| 键名 | `conv11.0.conv1.weight` … `conv13.5.gn2.bias` |
| `data.pkl` | 15,131 B（ZIP 总大小 = 37,201,920 + 36,819 头部开销） |

**推算依据**：`FE` = `conv11`(6×ResBlock(64,64)) + `conv12`(6×ResBlock(128,128)) + `conv13`(6×ResBlock(256,256))，`ResBlock(c)` = `c²×9 + 2c + c²×9 + 2c`：
6×73,984 + 6×295,424 + 6×1,180,672 = **9,300,480** ✓

> 这条验证同时证明：仓库当前的 `FE` 定义就是训练 NC 时所用的版本，未被改动。

### 1.2 网络可达性实测（2026-10-11，本机 Windows）

| 域名 | 直连 | 经 Clash 代理 `127.0.0.1:7890` |
|---|---|---|
| `drive.google.com` | ❌ 超时（HTTP 000 / 8s） | ✅ HTTP 302 / 1.1s |
| `download.pytorch.org` | ✅ HTTP 403（根路径正常） | — |
| `openaipublic.azureedge.net` | ✅ HTTP 400（根路径正常） | — |
| `github.com` | ✅ HTTP 200 | — |

**若日后需重新获取 NC**（File ID `1SgiK6fBHXd2yICBhNNfoncUWjSUxslYe`）：
1. 必须显式 `export https_proxy=http://127.0.0.1:7890`，否则 curl 不走代理，表现为"Google 不通"。
2. >25 MB 的文件返回 HTML 病毒扫描拦截页，需解析 `confirm` + `uuid` 后请求 `https://drive.usercontent.google.com/download?id=...&export=download&confirm=t&uuid=...`。
3. 服务器支持 `Accept-Ranges: bytes`，可 `curl -C -` 续传；该节点实测仅 ~29 KB/s。
4. **`--max-time` 要给足**——首次用 600 s 被静默截断在 17,171,782 B，文件缺 ZIP EOCD，`zipfile` 报 "not a zip file"。

---

## 2. 算力选型

### 2.1 硬约束：CUDA 架构

`torch==1.8.0+cu111` 官方 wheel 编译进的架构上限是 **sm_86**。

| GPU | 算力 | 判断 |
|---|---|---|
| **RTX 3090（24 GB）** | **8.6** | ✅ **首选** |
| RTX 2080Ti / 3080 / A4000 / A5000 | 7.5 / 8.6 | ✅ 原生支持，显存偏小见 2.2 |
| RTX 4090 / 4080 / L20 / L40S | 8.9 | ⚠️ **无原生 kernel**；理论上驱动可前向 JIT compute_86 的 PTX，但**未验证，别拿租机时长赌** |
| H100 / H800 | 9.0 | ❌ 需 CUDA 12.0+ |
| RTX 5090 / RTX 5060 Laptop | 12.0 | ❌ 需 CUDA 12.8+ |

> **本机 RTX 5060 Laptop = sm_120（8 GB，驱动 592.15），本地无法运行，这是租服务器的唯一原因。**
> ⚠️ **国内云平台默认推荐 4090，必须手动改选 3090。**

### 2.2 显存：唯一的 OOM 风险点

`data/base_dataset.py:85-104` 的 resize / crop / flip **全部被注释掉**，图片会按**原始分辨率**进入 VGG19 与 SAD。SAD 内含大量 64 通道 `Dense_Block_IN` / `Trans_Down` / `Trans_Up` 在全分辨率上运算，1080p 输入在 8–11 GB 卡上可能爆显存。

**对策**：
- 优先选 **24 GB** 卡；
- 冒烟测试先用 **256×256** 小图，确认链路通了再上大尺寸；
- 若正式跑必须缩放，**记录缩放方式**，且评测时同步处理 mask / 检测框坐标。

### 2.3 其他租用要求

1. **镜像选 Miniconda 基础镜像**，不要选预装 PyTorch 2.x 的。
2. **宿主驱动无需降级**——CUDA 驱动向后兼容，cu111 的 wheel 在 CUDA 12.x 驱动上照常运行（要求 ≥ 455，现役均远超）。
3. **磁盘 20 GB 足够**（权重 ~615 MB + CLIP 256 MB + 环境 ~5 GB + 测试图与输出）。
4. **出网需可达** `download.pytorch.org` 与 `openaipublic.azureedge.net`；Google Drive **不需要**。

---

## 3. 环境配置

### 3.1 获取代码（目录名必须是 `HSRUIE`）

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/gemyumeng/HSRUIE.git HSRUIE
cd HSRUIE
git checkout 397da5e492b309f5c1af4f64df73c5a647e038cd
cd ~/work        # 之后所有命令都从 ~/work 执行，不是在 HSRUIE 内
```

> **目录名不能改**：`models/cycle_gan_model.py:71,75` 和 `models/vggloss.py:17` 硬编码了 `HSRUIE/` 前缀。目录名对了 + 从父目录运行，**就不需要改任何源码**。

### 3.2 创建 conda 环境

**不要直接用 `conda env create -f environment.yml`**：
1. 该文件末尾有作者机器路径 `prefix: /home/server1-cvlab710/anaconda3/envs/clip`；
2. `torch==1.8.0+cu111` / `torchvision==0.9.0+cu111` 不在 PyPI 默认索引，而 YAML 未声明 `extra-index` / `find-links`，**大概率一次失败**。

改为手动安装：

```bash
conda create -n hsruie python=3.8.20 -y
conda activate hsruie

# torch 三件套：必须指定官方历史 wheel 索引
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 其余依赖（版本取自 environment.yml，已剔除 torch 三行）
pip install numpy==1.24.4 scipy==1.10.1 pillow==9.4.0 opencv-python==4.10.0.84 \
  pytorch-wavelets==1.3.0 visdom==0.2.4 dominate==2.9.1 thop==0.1.1-2209072238 \
  scikit-image==0.21.0 matplotlib==3.7.5 ftfy==6.2.3 regex==2024.9.11 \
  tqdm==4.66.5 imageio==2.35.1 networkx==3.1 pywavelets==1.4.1 tifffile==20230710
```

**CLIP 不需要 pip 安装**——仓库已 vendor 在 `HSRUIE/CLIP/`，代码用 `from CLIP import clip` 导入。

### 3.3 环境验收（三条必查，装完立刻跑）

```bash
# ① 版本与 CUDA 可用性
python -c "import sys,torch,torchvision as tv;print(sys.version);print(torch.__version__,tv.__version__);print('cuda available:',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

# ② 【关键】架构列表必须包含所租卡的算力
python -c "import torch;print(torch.cuda.get_arch_list())"

# ③ 实跑一次张量运算，确认不是"能 import 但无 kernel"
python -c "import torch;x=torch.randn(2,3).cuda();print((x@x.T).sum().item())"
```

**② 的判读**：期望输出含 `sm_86`，形如 `['sm_37','sm_50','sm_60','sm_70','sm_75','sm_80','sm_86']`。
- 误租 4090（sm_89）→ 列表最高只有 `sm_86`，③ 可能靠 PTX JIT 侥幸通过但极慢，也可能直接报 `no kernel image is available for execution on the device` → **立刻退租换机**
- ③ 报错而 ① 显示 `cuda available: True` → 就是架构不匹配的典型症状

**三条命令的完整终端输出必须存档。**

### 3.4 放置权重

**NC 已在本机下载并验证完毕，两份副本都在**（内容相同，SHA256 一致）：
- `weights_dl/latest_net_NC.pth`
- `HSRUIE_repo/checkpoints/maps_cyclegan/latest_net_NC.pth` ← 已放入仓库，5 个权重齐备

**上传方式二选一：**

```bash
# 方式 A（推荐）：服务器上重新 clone，只传 NC 这一个文件（37 MB）
scp "C:/Users/Lenovo/Desktop/HSRUIE组会和复现/weights_dl/latest_net_NC.pth" \
    <server>:~/work/HSRUIE/checkpoints/maps_cyclegan/

# 方式 B：整包上传本地 HSRUIE_repo/，但落地后必须改名
scp -r "C:/Users/Lenovo/Desktop/HSRUIE组会和复现/HSRUIE_repo" <server>:~/work/
ssh <server> "cd ~/work && mv HSRUIE_repo HSRUIE"     # ← 改名是硬性要求，见 3.1
```

> 本机曾尝试把 `HSRUIE_repo` 直接改名为 `HSRUIE`，被 Windows 拒绝（`Permission denied`，`.git` 与 IDE 文件监视持有目录句柄）。**这不影响任何流程**：本机 GPU 是 sm_120，本来就不会在本地跑推理；改名只在服务器上需要，而方式 A 是重新 clone，名字天生正确。若走方式 B，记得落地后执行上面那条 `mv`。
>
> `model_data/` 已在本机建好但为空——`vgg19` 由服务器直连下载，不必上传 575 MB。

```bash
cd ~/work
mkdir -p HSRUIE/model_data      # 仓库里没有这个目录，必须自己建

# ② VGG19：服务器直连下载，不走 Drive
wget -c https://download.pytorch.org/models/vgg19-dcbb9e9d.pth \
  -O HSRUIE/model_data/vgg19-dcbb9e9d.pth

# ③ CLIP RN50：触发自动下载到 ~/.cache/clip/
python -c "import sys;sys.path.insert(0,'HSRUIE');from CLIP import clip;clip.load('RN50');print('CLIP OK')"
```

> **路径以源码为准，不要照 README**：README 写的是 `./model_data/maps_cyclegan/`，但 `models/vggloss.py:17` 实际读 `HSRUIE/model_data/vgg19-dcbb9e9d.pth`。`model_data/` 目录仓库里**不存在，必须自己建**。

**核对哈希**：

```bash
sha256sum HSRUIE/checkpoints/maps_cyclegan/*.pth HSRUIE/model_data/*.pth
```

逐项对照第 1 节。NC 必须是 `8a530b59…d4d6a50dd6cd`，大小正好 37,238,739 B。

### 3.5 目标目录结构

```text
~/work/                              ← 运行时的工作目录（父目录）
└── HSRUIE/                          ← 目录名不可改
    ├── test.py
    ├── CLIP/                        ← 仓库自带，无需 pip
    ├── datasets/maps/testA/         ← 自己放待增强图
    ├── checkpoints/maps_cyclegan/
    │   ├── latest_net_LOW.pth       ← 仓库自带
    │   ├── latest_net_HIGH2.pth     ← 仓库自带
    │   ├── latest_net_HIGH3.pth     ← 仓库自带
    │   ├── latest_net_SAD.pth       ← 仓库自带
    │   └── latest_net_NC.pth        ← 上传
    ├── model_data/
    │   └── vgg19-dcbb9e9d.pth       ← 服务器下载
    └── results/maps_cyclegan/test_latest/   ← 输出
```

---

## 4. 执行推理

### 4.1 三个必带参数（都不用改源码）

| # | 问题 | 位置 | 规避 |
|---|---|---|---|
| 1 | `gpu_ids` 默认 `'4'` | `options/base_options.py:25` | 显式传 `--gpu_ids 0` |
| 2 | `__len__` 引用未定义的 `B_size` | `data/unaligned_dataset.py:59` | 用 `--dataset_mode single` 绕开 |
| 3 | 硬编码 `HSRUIE/` 前缀 | `cycle_gan_model.py:71,75`；`vggloss.py:17` | 目录名取 `HSRUIE` + 从父目录运行 |

第 2 项已核实可行：`models/__init__.py` 对 `dataset_mode` **无任何断言**；`data/single_dataset.py:25-36` 返回 `{'A','A_paths'}`，与 `CycleGANModel.set_input`（只读 `input['A']`、`input['A_paths']`）完全匹配。

### 4.2 命令

```bash
cd ~/work
mkdir -p HSRUIE/datasets/maps/testA
# 先放 5–10 张水下图进 testA 做冒烟测试（建议 256×256，见 2.2）

python HSRUIE/test.py \
  --dataroot ./HSRUIE/datasets/maps/testA \
  --dataset_mode single \
  --name maps_cyclegan \
  --model cycle_gan \
  --checkpoints_dir ./HSRUIE/checkpoints \
  --results_dir ./HSRUIE/results \
  --gpu_ids 0 \
  --num_test 10 \
  --eval
```

**输出**：`HSRUIE/results/maps_cyclegan/test_latest/`，含 `index.html` 与增强图（可视化键为 `I_e`，见 `cycle_gan_model.py:52`）。

⚠️ **`--num_test` 必须显式设置**：`options/test_options.py:17` 默认 1000，图数超过 1000 会**静默只处理前 1000 张**。
⚠️ `test.py` 硬编码 `batch_size=1`（测试脚本限制），**与论文训练 batch=2 不矛盾，不要写成冲突**。

### 4.3 验收标准

- [ ] 无异常退出，`results/maps_cyclegan/test_latest/` 下有 `index.html`
- [ ] 每张输入都有对应输出，**输出图数 == 输入图数**
- [ ] 输出尺寸与输入一致（resize/crop 被注释，理论上应完全相同）
- [ ] 颜色正常：不偏色、不发灰、无全黑/全白
- [ ] 肉眼可见增强效果（对比度 / 色彩恢复）
- [ ] 记录单张推理耗时与显存峰值

任一项不通过 → 先定位，不要继续往下跑大图。

---

## 5. ⚠️ 报告口径限制（必读）

`models/cycle_gan_model.py:109-121` 的 `forward()`：

```python
self.f_raw = self.netVgg19(self.input)
self.f_c   = self.netNC(self.f_raw)
[self.fc1, self.fc2, self.fc3]       = self.f_c
[self.fraw1, self.fraw2, self.fraw3] = self.f_raw
# self.fe1 = self.netLOW(self.input, self.fc1)      ← 被注释
self.fe1 = self.fc1                                  ← 替代实现
self.fe2 = self.netHIGH2(self.input, self.fc2)
self.fe3 = self.netHIGH3(self.fc3)
self.f_e = [self.fe1, self.fe2, self.fe3]
self.I_e = self.netSAD(self.f_e)
```

**浅层 DEM（`netLOW`）被跳过**，`fe1` 直接取 `fc1`。`latest_net_LOW.pth` 仍会被 `load_networks` 加载进 `self.netLOW`，但**在 forward 中从未被调用**。而 `models/DenseNet.py` 里的 `Mynet.forward` 却正常调用了 `netLOW`——**两处不一致**，是消融设置还是漏提交，无从判断。

| ✅ 允许的表述 | ❌ 禁止的表述 |
|---|---|
| "公开版本权重的推理结果" | "复现了论文的图 X" |
| 附 patch，说明每处改动 | 笼统写成"作者官方结果" |
| 论文未给的设置标注为 **"本次复现设置"** | 把自己的设置混入作者报告值 |

---

## 6. 收尾

### 6.1 必须归档

- [ ] 代码 commit hash（`397da5e…`）+ 本地修改的 patch（`git diff > my_changes.patch`）
- [ ] `pip freeze` 完整环境清单
- [ ] `nvidia-smi` 输出：GPU 型号、算力、驱动版本
- [ ] 第 3.3 节三条验收命令的完整终端输出
- [ ] 全部权重的**文件名 + 大小 + SHA256**（对照第 1 节）
- [ ] 测试图片列表（逐张文件名）与分辨率
- [ ] 原图 / 增强图对照（`index.html` 或直接拼图）
- [ ] 单张推理耗时、显存峰值

### 6.2 清理

- 服务器：跑完即可释放，按小时计费不必留机。
- **本机 `weights_dl/latest_net_NC.pth`（37 MB）必须保留并另存备份**——这是唯一无法从非 Google 渠道获取的文件，重下要再花十几分钟且受代理限速。
- 可删：`weights_dl/resume_nc.sh`（续传脚本，已完成使命）。

---

## 7. 风险登记册

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 误租 4090/H100，装完环境才发现无 kernel | 中 | 损失半天 + 租费 | **装环境前先跑第 3.3 节 ② ③ 两条命令** |
| 原始分辨率导致 OOM | 中 | 推理跑不完 | 先用 256×256 小图；或选 24 GB 卡 |
| `download.pytorch.org` / `azureedge` 在服务器上不通 | 低 | 环境装不上 | 用平台"学术加速"；或本机下载后上传 |
| NC 上传后大小/哈希不符 | 低 | 加载失败或结果异常 | 上传后立刻 `sha256sum` 对照第 1 节 |
| 输出被当成"论文复现结果"引用 | 中 | **学术诚信问题** | 严格执行第 5 节口径表 |

---

## 8. 附录

### 8.1 命令速查

```bash
# 环境自检（一行）
python -c "import torch;print(torch.__version__);print(torch.cuda.get_arch_list());print(torch.cuda.get_device_name(0))"

# 推理（冒烟测试）
cd ~/work && python HSRUIE/test.py --dataroot ./HSRUIE/datasets/maps/testA \
  --dataset_mode single --name maps_cyclegan --model cycle_gan \
  --checkpoints_dir ./HSRUIE/checkpoints --results_dir ./HSRUIE/results \
  --gpu_ids 0 --num_test 10 --eval

# 权重哈希核对
sha256sum HSRUIE/checkpoints/maps_cyclegan/*.pth HSRUIE/model_data/*.pth

# 显存峰值
python -c "import torch;print(torch.cuda.max_memory_allocated()/1024**3,'GiB')"
```

### 8.2 关键源码位置索引

| 内容 | 位置 |
|---|---|
| 推理 forward（含 `netLOW` 注释） | `models/cycle_gan_model.py:109-121` |
| 可视化键 `I_e` | `models/cycle_gan_model.py:52` |
| 硬编码 SAD 路径 | `models/cycle_gan_model.py:71` |
| 硬编码 NC 路径 | `models/cycle_gan_model.py:75` |
| 硬编码 VGG19 路径 | `models/vggloss.py:17` |
| `FE` / `SAD` / `ResBlock` 定义 | `models/DenseNet.py:207-247` / `:250+` / `:190-205` |
| `gpu_ids` 默认 `'4'` | `options/base_options.py:25` |
| `num_test` 默认 1000 | `options/test_options.py:17` |
| `B_size` bug | `data/unaligned_dataset.py:59` |
| 被注释的 resize/crop/flip | `data/base_dataset.py:85-104` |
| `SingleDataset` 返回结构 | `data/single_dataset.py:25-36` |
| CLIP RN50 URL 与缓存路径 | `CLIP/clip/clip.py:31` / `:120` |

### 8.3 资料来源

- 论文 PDF：p.5 §IV.A 实现细节（batch 2、RTX 2070、Adam）
- 官方 README：https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/README.md
- 官方环境：https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/environment.yml
- 核心模型：https://github.com/gemyumeng/HSRUIE/blob/397da5e492b309f5c1af4f64df73c5a647e038cd/models/cycle_gan_model.py
- 本机材料：`HSRUIE_详细复现指南.md`（含训练与评测的完整分析）、`HSRUIE_文献汇报_30页.pptx`、论文 PDF、`HSRUIE_repo/`（已 clone）、`weights_dl/latest_net_NC.pth`（已验证）

---

**文档结束。** 执行顺序：第 2 节租卡 → 第 3 节装环境 → 第 3.3 节验收 → 第 4 节推理 → 第 5 节口径 → 第 6 节归档。
