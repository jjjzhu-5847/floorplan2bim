# 项目计划：从 2D 平面图到 BIM——自动算量、估价与造价风险分析

> 这份文档是项目的唯一"总说明"。给 Claude Code 用时，先让它完整读一遍本文件和 `CLAUDE.md`，再按"第 8 节 本周任务"逐项执行。
> 最后更新：2026-10-31（在 claude.ai 中完成可行性测试，并把代码整理成本仓库的结构）。
>
> **整理后已测试**：合成平面全流程（工程量与造价和参考值完全一致）、IFC 输出（46 个构件全部可三角化）、误差预演、价格回测与开工月份区间（用模拟数据跑通）、网页生成。
> **2026-10-31 本机验证**：`run_pipeline.py --image` 与 `evaluate_cubicasa.py`（验证集前 10 张，`failed.csv` 为空）已跑通；需 transformers ≥ 5.22（权重使用新版参数命名），并修复了 OpenCV 5 兼容与自相交多边形两处崩溃。`web/app.py` 尚未运行。

---

## 1. 背景与课程要求

- 课程：**Data Analytics on Construction Management**（University of Houston，2026 秋季）。
- 题目范围（教授原话）：any topic related to civil engineering, or construction engineering, or construction management。
- 交付物：**15 页报告** + **15 分钟展示**。
- 截止时间：未确认（本计划按"本周内全部完成"安排，见第 8 节）。
- 团队：个人还是小组未确认；如有组员，按第 8 节末尾的分工建议拆分。

**本项目为什么符合要求：** 主体是工程量计算（quantity takeoff）、估价、BIM 和造价风险，都是 construction management 的核心内容。CV 只是获取数据的手段，报告主线要始终落在"算量和估价"上。

---

## 2. 项目目标

### 一句话

输入一张住宅平面图，系统自动生成 **3D/BIM 模型、工程量清单和造价**，并显示**不同开工月份的造价区间**；报告用数据回答：**算得有多准、误差从哪来、开工晚一点风险有多大。**

### 研究问题（报告的主线）

| 编号 | 问题 | 用什么回答 |
|---|---|---|
| RQ1 | 自动生成的工程量和造价，与标准答案相比误差多大？ | CubiCasa5K 测试集，条件 E vs A（第 9 节） |
| RQ2 | 误差主要来自识别、矢量化还是尺度标定？ | 误差归因实验，条件 B/C/D/E（第 9 节） |
| RQ3 | 按 AACE 估算分级，系统精度能用于哪个阶段的估价？ | `evaluation/aace.py` |
| RQ4 | 建材价格能否预测？延迟开工的造价风险有多大？ | 价格指数回测 + 按开工月份的造价区间 |

### 不做的事（范围边界）

- **不自己训练识别模型**（已决定，2026-10-21）。直接用公开的 SegFormer-B0 预训练模型。若时间富余，可作为可选任务 4.3（用 RTX 4070 Super 训练改进版）。
- 造价只覆盖**墙体、地面、门、窗**；不含结构、基础、屋顶、水电、装修做法。报告里写明为"部分范围估算"。
- 默认只评估**单层**平面图（一张图上画了多层的，单独统计或排除）。
- 人工成本不随时间变化（只有材料部分跟随价格指数）。

---

## 3. 系统流程

```
平面图图片
  │ ① 识别  recognition/segformer.py         SegFormer-B0 → 每个像素分 5 类：背景/房间/墙/门/窗（512×512）
  ▼
像素掩码
  │ ② 矢量化 vectorization/vectorize.py      骨架 + Hough 直线 → 横竖墙线、门洞、窗洞、房间多边形
  │          vectorization/transform.py     512 像素坐标 → 原图像素坐标
  ▼
"原始平面"（raw plan，像素单位）
  │ ③ 尺度   bim_generation/scale.py         门宽中位数 ÷ 0.9 m → 像素/米
  ▼
  │ ④ 建模   bim_generation/geometry.py      墙中心线 + 厚度，门窗挂到所在墙上（米）
  │          bim_generation/viewer_export.py → 网页 3D 数据
  │          bim_generation/ifc_export.py    → IFC4 BIM 文件
  ▼
  │ ⑤ 算量   takeoff_cost/quantities.py      墙长/墙面积/体积、门窗数量与面积、地面面积
  │   计价   takeoff_cost/cost.py            × 单价（config.UNIT_RATES）
  ▼
  │ ⑥ 价格   price_forecast/                 7 个 BLS 价格指数 → 回测 → 按开工月份的造价区间
  ▼
  │ ⑦ 展示   web/                            静态演示页 + 现场上传演示（Flask）
  │ 评估     evaluation/                     与 CubiCasa5K 标准答案对比、误差归因、AACE、出图
```

### 目录结构（每个文件干什么）

```
floorplan2bim/
├── PROJECT_PLAN.md            ← 本文件：目标、决策、任务、验收标准、报告与展示大纲
├── CLAUDE.md                  ← 给 Claude Code 的工作约定
├── README.md                  ← 快速上手命令
├── requirements.txt
├── config.py                  ← 所有参数与假设（层高、门宽、单价、价格指数、AACE 区间……）只在这里改
├── pipeline.py                ← 一张图/一个 SVG 跑完整流程：run_image / run_svg / save_plan
│
├── recognition/               ① 识别
│   └── segformer.py           加载预训练模型、letterbox 到 512、推理、彩色叠加图
├── vectorization/             ② 矢量化
│   ├── vectorize.py           掩码 → 墙/门/窗/房间（第一版，本周重点改进）
│   ├── svg_io.py              读 CubiCasa model.svg（标准答案）、写 SVG、把平面栅格化成掩码
│   └── transform.py           512 坐标 ↔ 原图坐标
├── bim_generation/            ③④ 尺度与建模
│   ├── scale.py               门宽 → 像素/米（本周要加 OCR 与融合）
│   ├── geometry.py            原始平面 → 米制模型（墙中心线、厚度、门窗位置）
│   ├── viewer_export.py       米制模型 → 网页 3D 数据（墙体分块、门扇、玻璃、房间）
│   └── ifc_export.py          米制模型 → IFC4（IfcWall/Opening/Door/Window/Space/Slab + 工程量属性）
├── takeoff_cost/              ⑤ 算量计价
│   ├── quantities.py          算量规则（见文件头注释，报告里照抄规则说明）
│   └── cost.py                工程量 × 单价
├── price_forecast/            ⑥ 价格风险
│   ├── backtest.py            滚动回测：朴素/漂移/ARIMA/ETS，1–12 个月误差
│   └── cost_by_month.py       ARIMA 中心线 + 历史误差联合重放 → 未来 12 个月造价 P10/P50/P90
├── evaluation/                评估与出图
│   ├── evaluate_cubicasa.py   核心实验：条件 A–E、IoU、尺度误差 → results.csv / summary.json
│   ├── aace.py                误差分布对照 AACE 估算等级
│   ├── error_propagation_demo.py  在测试平面上模拟各类识别错误对造价的影响
│   └── make_figures.py        报告图表 → outputs/figures/
├── web/                       ⑦ 展示
│   ├── template.html          网页模板（three.js 3D、工程量表、造价、开工月份区间、回测表）
│   ├── build_page.py          把 outputs/plans/* 打包成 outputs/web/index.html
│   └── app.py                 现场演示：上传图片 → 实时生成 3D 与造价（Flask）
├── samples/
│   ├── make_sample_svg.py     生成合成测试平面 samples/sample_plan/model.svg（两室一厅，100 单位=1 m）
│   └── demo_images/           5 张真实 CubiCasa 平面图（512 px 副本），完整数据到位前用于演示
├── scripts/
│   ├── run_pipeline.py        命令行：一张图或一个 SVG → outputs/plans/<name>/
│   └── build_demo.py          一条命令生成演示页
├── data/                      cubicasa5k/（不提交，约 5 GB，各自下载）、ppi/（提交）
├── models/                    （提交）segformer_best.pt
└── outputs/                   （提交）plans/、evaluation/、price_forecast/、figures/、web/
```

---

## 4. 已确定的关键决策

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-09-24 | 选题：平面图 → BIM → 算量计价 | 有挑战性、不泛滥、数据和标准答案现成、结论可验证、贴合施工管理 |
| 2026-09-24 | 加入建材价格预测与"按开工月份的造价区间" | 增加数据分析比重；造价上涨是真实的施工管理风险 |
| 2026-09-24 | 价格以**区间**呈现，不给单一价格 | 回测显示木材价格不可预测；ARIMA 整体只略好于"价格不变" |
| 2026-10-21 | 不自己训练识别模型，用公开 SegFormer-B0 | 控制工作量；误差预演显示瓶颈更可能在尺度和矢量化 |
| 2026-10-31 | 本周内完成全部工作；代码按功能分文件夹 | 用户要求 |

---

## 5. 数据与模型

### 5.1 CubiCasa5K（主数据集，必须下载）

- 下载：Zenodo record 2613548（DOI 10.5281/zenodo.2613548），约 5 GB zip。许可：CC BY-NC 4.0（课程用途可以）。
- 解压到 `data/cubicasa5k/`，使其中直接包含 `train.txt`、`val.txt`、`test.txt` 和 `high_quality_architectural/`、`high_quality/`、`colorful/` 等子目录。
- 每个平面图文件夹里有：`F1_original.png`（原图）、`F1_scaled.png`（与 SVG 坐标对齐的缩放图）、`model.svg`（人工标注的标准答案）。
- 规模：5,000 张，官方划分约 4,200 训练 / 400 验证 / 400 测试。
- **SVG 比例：100 个单位 = 1 米**（来自 TakeoffLens 项目的报告，需在任务 0.3 中验证）。官方加载器把 SVG 坐标直接画在 `F1_scaled.png` 上，所以 `F1_scaled.png` 的 1 像素应当 = 1 SVG 单位 = 1 cm（同样要在 0.3 中验证）。
- 注意：一张图可能包含多层（SVG 里多个 `<g class="Floor">`），默认评估时排除。

### 5.2 识别模型（预训练，不训练）

- SegFormer-B0（NVIDIA 2021 提出的 Transformer 语义分割模型），在 CubiCasa5K 上训练，5 类：background / room / wall / door / window。
- 来源：github.com/alifarzadjamali/floor-plan-document-intelligence（MIT 许可），文件 `results/segmentation/phase2_dev/checkpoints/best.pt`，44.8 MB，sha256 `12fa671d…2bf7`，已提交为 `models/segformer_best.pt`。
- 作者公布的测试集指标（400 张）：room IoU 0.83；wall IoU 0.53（召回 0.85，精度 0.59，偏多判）；door IoU 0.20（开启弧线、楼梯符号常被判成门）；window IoU 0.41。
- 该模型的训练设置：只用了 1,200 张训练图、18 轮、整图缩到 512 px。
- 加载方式：用 `SegformerConfig(num_labels=5)`（默认即 MiT-b0）本地重建结构，再 `load_state_dict(strict=True)`；不需要访问 Hugging Face。

### 5.3 建材价格指数（BLS 生产者价格指数，经 FRED 下载）

| FRED 编号 | 含义 | 用于 |
|---|---|---|
| WPU0811 | 软木木材 | 内外墙（木骨架） |
| WPU137 | 石膏制品 | 内外墙（石膏板） |
| WPU1333 | 预拌混凝土 | 备用（当前造价范围不含基础） |
| WPU082 | 木制品（millwork） | 门 |
| PCU321911321911 | 木门窗制造 | 门、窗 |
| WPU10710313 | 金属窗 | 窗 |
| WPUIP2311001 | 住宅建设投入品综合 | 外墙、地面 |

`data/ppi/ppi_monthly.csv` 已提交在仓库中：2010-01 至 2026-08（金属窗截至 2026-06），2026-10-31 从 FRED 网页手动下载（本机网络无法用脚本自动下载）。更新数据时，同样手动下载上表 7 个序列，按相同列名合并成这个文件。

### 5.4 演示用小图

`samples/demo_images/input_{1316,10026,10029,12539,13736}.png`：来自 github.com/daegeun-kim/neural_floorplan 的 512 px 真实 CubiCasa 平面图副本，**没有对应的标准答案**，只用于演示。

---

## 6. 已完成与已验证（可行性测试，2026-09-24）

1. **真实识别能跑**：预训练 SegFormer 在 5 张真实平面图上推理成功，结构加载 strict 通过。
2. **矢量化第一版**：外墙基本都能抓到；部分内墙和门会漏；两层并排的图纸、阳台、楼梯会造成错误。→ 本周重点改进。
3. **3D 与 IFC**：在合成平面上手算核对过——11 面墙、6 门、8 窗、14 个洞口、6 个空间、1 块楼板全部可三角化；每面墙开洞后的体积与手算完全一致。
4. **算量手算核对**（合成平面）：外墙中心线长 36.0 m；净地面面积 72.79 m²（= 内部净面积 75.56 − 隔墙占地 2.78）；窗面积 13.56 m²；造价合计 33,943 美元（占位单价），逐项乘法核对无误。
5. **误差传导预演**（合成平面，每种情况 500 次，总造价平均绝对误差）：
   - 漏检 10% 内墙：2.2%
   - 漏检 10% 窗：1.5%
   - 漏检 10% 门：0.7%
   - 用门宽标定尺度（门宽 0.8/0.9/1.0 m 混合）：3.3%，最坏 5% 情况高估 13% → **尺度标定是最大风险**
6. **价格回测**（2016–2025 滚动起点，12 个月后的平均绝对误差）：
   - 全部指数合计：ARIMA 7.0% vs 价格不变 7.4%；3 个月时打平（3.0% vs 3.0%）
   - ARIMA 更好：混凝土 2.7 vs 4.4、木门窗 3.6 vs 4.9、金属窗 6.2 vs 7.5、石膏、木制品
   - ARIMA 更差：软木木材 20.6 vs 18.2
7. **按开工月份的造价**（合成平面，材料占比 50%，人工不变）：今天 33.9k → 2027-08 开工预计 34.8k，80% 区间 33.4k–37.4k。结论：**挑不出明显更便宜的月份，但能量化"越晚开工风险越大"。**

---

## 7. 假设一览（报告"方法"与"局限"章节要逐条写）

| 假设 | 值 | 位置 | 备注 |
|---|---|---|---|
| 层高 | 2.7 m | config.H_WALL | 平面图没有高度信息 |
| 门高 | 2.1 m | config.H_DOOR | |
| 窗台 / 窗顶 | 0.9 / 2.1 m | config.SILL / H_WIN_TOP | |
| 标准门宽 | 0.9 m | config.STD_DOOR_W | 尺度标定用；真实门宽 0.8–1.0 m |
| 单价 | 占位值 | config.UNIT_RATES | **任务 3.1 必须替换并注明来源** |
| 材料占比 | 50% | config.MATERIAL_SHARE | 需找文献依据或做敏感性分析 |
| 人工成本 | 不变 | | 局限 |
| 指数权重 | 见 config.COST_BASKETS | | 木骨架住宅的假设 |
| AACE 区间 | 见 config.AACE_CLASSES | | **引用前对照 AACE RP 18R-97 原文核实** |
| 数据来源 | 芬兰住宅，套美国单价 | | 局限 |

---

## 8. 本周任务（10/31 周六 – 11/6 周五）

> 原则：**先让整条流程在真实数据上出数，再改进**。每完成一项，在本节对应条目后标 ✅ 并写一行结果。
> 如果进度落后，按优先级砍：P0 必须完成，P1 应该完成，P2 有时间再做。

### 第 1 天（周六 10/31）：环境、数据、跑通

- **0.1 环境（P0）**：Python 3.11；按 `requirements.txt` 安装，PyTorch 装 CUDA 版（4070 Super）。
  - 验收：`python -c "import torch; print(torch.cuda.is_available())"` 输出 True。
  - ✅ 本机 Python 3.13 + torch 2.9.1+cu126，CUDA 可用；transformers 需 ≥ 5.22。
- **0.2 数据（P0）**：权重、演示图、价格指数已提交在仓库中；只需手动下载 CubiCasa5K 并解压到 `data/cubicasa5k/`。
  - 验收：四类数据都在；`data/cubicasa5k/test.txt` 存在。
  - ✅ CubiCasa5K 5,000 张齐全（4200/400/400），无缺文件；验证集与测试集 800 个 SVG 全部可解析。多层图占测试集 25%（99/400）、验证集 20%（81/400）。
- **0.3 验证标准答案的对齐和比例（P0，非常重要）**：
  - 任选 5 个测试集平面：用 `vectorization/svg_io.rasterize()` 把 `model.svg` 画成掩码，叠加到 `F1_scaled.png` 上，确认墙线重合。
  - 在 3–5 张印有房间面积（如 "MH 11.7" = 11.7 m² 卧室）的图上，用 SVG 房间多边形面积 ÷ 10,000 对比印刷面积，确认"100 单位 = 1 米"。误差应在 1–2% 以内。
  - 检查 `model.svg` 里外墙的 class 名称（是否有 "External"）；若没有，`parse_svg` 会用几何规则判断外墙——抽查几张确认判断合理。
  - 验收：把叠加图存到 `outputs/checks/`，在本文件写下结论。若比例或对齐不对，先修 `config.CUBICASA_UNITS_PER_M` 或坐标变换，再往下做。
- **0.4 跑通演示（P0）**：`python scripts/build_demo.py`，打开 `outputs/web/index.html`。
  - 验收：页面能切换 3 个平面、看 3D、表格有数、开工月份区间有图。

### 第 2 天（周日 11/1）：基线评估（第一批真实数字）

- **1.1 小规模试跑（P0）**：`python evaluation/evaluate_cubicasa.py --split test --limit 30`。
  - 查看 `outputs/evaluation/failed.csv`，修复崩溃（例如无门导致无比例、空掩码、非矩形墙多边形）。
  - 查看 `outputs/evaluation/examples/` 里的三联图（原图 / 标准答案掩码 / 预测掩码），确认合理。
- **1.2 全测试集基线（P0）**：`--split test` 全量运行，保存为基线：把 `outputs/evaluation/` 复制为 `outputs/evaluation_baseline/`。
  - 验收：`summary.json` 有条件 A–E 的误差、IoU、尺度误差、AACE 结果；在本文件记录关键数字（条件 E 的造价中位绝对误差、哪一步误差最大）。
- **1.3 出第一版图（P1）**：`python evaluation/make_figures.py`。

### 第 3–4 天（周一 11/2 – 周二 11/3）：改进矢量化与尺度（技术核心）

> **只在验证集（`--split val`）上调参**，最后才在测试集上跑一次，防止"对测试集调参"。
> 用条件 C（标准答案掩码 → 矢量化）衡量矢量化本身，用 D/E 衡量整体。

- **2.1 门（P0）**：只保留落在墙线带内的门像素；门洞宽度取墙线方向上的投影长度；忽略开启弧线。楼梯被误判为门的问题：门洞必须位于两段共线墙之间。
- **2.2 内墙召回（P0）**：可选做法（择一或组合）：
  - 在 512 掩码上放宽参数（`config.VEC_*`），降低最短墙长；
  - 对大图做**切块推理**（把原图切成重叠的 512 块分别推理再拼接），细墙不会因缩放丢失——预期收益最大；
  - 墙端点吸附到附近垂直墙，闭合 L/T 接头（已有初版 `snap`）。
- **2.3 房间（P1）**：房间区域用墙 + 门洞切分；去掉过小碎片；一张图多层并排时，按连通块分成多个建筑轮廓。
- **2.4 尺度融合（P0）**：在 `bim_generation/scale.py` 中加入：
  - 门宽法（已有）；
  - OCR 读房间面积标签（如 "MH 11.7"，可用 pytesseract 或 easyocr），与对应房间的像素面积求比例；
  - 融合：多个估计取中位数或按可靠度加权。
  - 验收：在验证集上报告三种方法的尺度误差中位数，融合法应 ≤ 单一方法。
- **2.5 测试集复跑（P0）**：改进后在测试集跑一次，与 1.2 基线对比（改进前后的条件 C/D/E 误差）。这张对比表是报告的重要结果。

### 第 5 天（周三 11/4）：单价、价格、AACE

- **3.1 真实单价（P0）**：替换 `config.UNIT_RATES`，并把 `UNIT_RATE_SOURCE` 写成引用。来源优先级：学校图书馆的 RSMeans（Building Construction Costs / Square Foot Costs）> 政府或行业公开单价资料 > 其他有出处的估价资料。每个单价记录来源与年份。
- **3.2 价格模块复跑（P0）**：`python price_forecast/backtest.py`（价格数据见 5.3 节） → 用 `cost_by_month.cost_band()` 给演示平面加区间。核对结论是否与第 6 节一致（新数据会让数字略变）。
- **3.3 敏感性分析（P1）**：材料占比 30% / 50% / 70% 下的区间宽度；写进报告讨论。
- **3.4 AACE（P0）**：核实 AACE RP 18R-97 各级精度区间后更新 `config.AACE_CLASSES`；用 `evaluation/aace.py` 得出"系统达到第几级"。

### 第 6 天（周四 11/5）：展示与（可选）训练

- **4.1 演示页定稿（P0）**：`python web/build_page.py --plans …` 选 3–4 个有代表性的测试集平面（好、一般、差各一个），并加入合成平面。
- **4.2 现场演示（P1）**：`pip install flask && python web/app.py`，在浏览器上传图片，确认 10 秒内出结果。展示当天用它做开场。
- **4.3 自训练模型（P2，可选）**：用 4070 Super 训练 SegFormer-B0（全部 4,200 张训练图、切块/更高分辨率、50 轮，约 3–10 小时，晚上挂着跑）。训练脚本可复用上述 GitHub 仓库的 `scripts/train_segmentation.py` 与 `scripts/build_masks.py`。产出：新旧模型的 IoU 对比 + 条件 D/E 造价误差对比。若不做，在报告中作为"未来工作"。

### 第 7 天（周五 11/6）：报告与幻灯片

- **5.1 出图（P0）**：`python evaluation/make_figures.py`，并补充第 10 节图表清单中缺的图。
- **5.2 报告初稿（P0）**：按第 10 节大纲写，数字全部来自 `outputs/` 里的文件，不手抄估计值。
- **5.3 幻灯片（P0）**：按第 11 节大纲做。
- **5.4 自查（P0）**：报告中每个数字能追溯到某个输出文件；假设表完整；局限写清楚；相关工作有引用。

### 分工建议（如果是 3 人小组）

- A：识别 + 矢量化（任务 2.1–2.3），可选训练 4.3
- B：尺度融合与 OCR（2.4）、单价（3.1）、价格模块（3.2–3.3）、网页（4.1–4.2）
- C：评估与出图（1.x、2.5、3.4、5.1）、报告主体（5.2）

---

## 9. 评估方案细节（RQ1–RQ3）

对测试集每张单层平面图，计算五个版本：

| 条件 | 几何来自 | 尺度 | 衡量什么 |
|---|---|---|---|
| A 参考 | 标准答案 SVG | 真实（100 单位/m） | "正确"的工程量与造价 |
| B 仅尺度 | 标准答案 SVG | 门宽估计 | 尺度标定带来的误差 |
| C 仅矢量化 | 标准答案掩码 → 我们的矢量化 | 真实 | 矢量化算法本身的误差 |
| D 识别+矢量化 | SegFormer 掩码 → 矢量化 | 真实 | 识别与矢量化合计误差 |
| E 完整系统 | SegFormer 掩码 → 矢量化 | 门宽估计（或融合） | 用户实际得到的误差 |

- 误差 = (版本值 ÷ A 值 − 1) × 100%，带符号；汇总时报告中位绝对误差、平均绝对误差、P10/P90。
- 工程量指标：外墙长、内墙长、外/内墙净面积、地面面积、窗面积、内门数、入户门数、窗数；以及总造价。
- 识别指标：各类 IoU（预测掩码 vs 栅格化的标准答案，512 尺寸）。
- 归因的读法：B 的误差 ≈ 尺度贡献；C ≈ 矢量化贡献；D − C ≈ 识别贡献；E 是叠加后的整体。
- 公平性：调参只用验证集；测试集只在 1.2（基线）和 2.5（改进后）各跑一次。
- AACE：取条件 E 造价误差的 P10–P90，看落在哪一级区间内（规则写在 `evaluation/aace.py` 注释里，报告照写）。

---

## 10. 报告大纲（15 页）

1. **引言**（1.5 页）：人工算量慢且易错；既有建筑只有 2D 图纸、改造需要 BIM；建材价格波动增加估价风险。研究问题 RQ1–RQ4。贡献。
2. **文献综述**（2 页）：施工算量与估价、AACE 分级；平面图识别（CubiCasa5K、SegFormer）；BIM/IFC；建材价格上涨与预测。相关开源项目与本项目的区别（第 13 节）。
3. **数据**（1.5 页）：CubiCasa5K、价格指数、单价来源、假设表（第 7 节）。
4. **方法**（3.5 页）：总流程图；识别；矢量化；尺度标定与融合；BIM 生成与算量规则；计价与价格指数映射；误差归因设计（第 9 节）；价格回测与区间方法。
5. **结果**（4 页）：5.1 识别 IoU；5.2 工程量误差；5.3 造价误差与 AACE（核心图）；5.4 误差归因；5.5 改进前后对比；5.6 价格回测与开工月份区间。
6. **讨论**（1.5 页）：系统能用于哪个估价阶段、能省多少时间；价格风险对业主和承包商（价格调整条款）的意义；最该改进哪一步。
7. **局限与结论**（1 页）。

### 图表清单（文件名 → 用在哪）

| 图/表 | 文件 | 章节 |
|---|---|---|
| 系统总流程图 | 自己画（可参考第 3 节） | 4 |
| 平面图 → 识别 → 矢量化 → 3D 示例 | 网页截图 + `outputs/evaluation/examples/` | 4 / 5.1 |
| 各类 IoU | `outputs/figures/fig_iou.png` | 5.1 |
| 工程量误差 | `fig_quantity_errors.png` | 5.2 |
| 造价误差 vs AACE（核心图） | `fig_aace.png` | 5.3 |
| 误差归因 | `fig_error_attribution.png` | 5.4 |
| 改进前后对比表 | `evaluation_baseline/summary.json` vs `evaluation/summary.json` | 5.5 |
| 价格指数走势 | 新增（用 ppi_monthly.csv 画） | 3 / 5.6 |
| 回测结果 | `fig_price_backtest.png` | 5.6 |
| 开工月份造价区间 | 网页截图或新增图 | 5.6 |
| 网页系统截图 | `outputs/web/index.html` | 4 / 附录 |

---

## 11. 展示大纲（15 分钟）

1. （1 分钟）开场现场演示：上传一张平面图 → 几秒后出现 3D 模型和造价表。
2. （1.5 分钟）问题：人工算量的成本、2D 图纸转 BIM 的需求、价格风险。
3. （1 分钟）研究问题 RQ1–RQ4。
4. （3 分钟）方法流程（一张流程图 + 每步一句话），误差归因设计。
5. （4 分钟）结果：IoU → 工程量误差 → 造价误差与 AACE（核心图）→ 误差归因。
6. （2 分钟）价格：哪些能预测、哪些不能；开工月份区间。
7. （1.5 分钟）管理意义与局限。
8. （1 分钟）结论 + 提问。

---

## 12. 已知问题与风险

- **矢量化**是最大的技术难点；内墙漏检会低估内墙长度与面积。
- **门的识别**很差（IoU 0.20），既影响门的计数，也影响门宽法尺度标定 → 尺度融合（2.4）很关键。
- **多层并排图纸**会让外墙判断和建筑轮廓出错 → 默认排除，报告说明。
- **单价与 AACE 区间**目前是占位或凭记忆，必须核实后再写入报告。
- **数据跨国**：芬兰住宅 + 美国单价。
- **时间风险**：一周完成很紧；按 P0/P1/P2 取舍。

---

## 13. 相关工作（报告要引用并说明区别）

- **CubiCasa5K**：Kalervo et al., 2019，"CubiCasa5K: A Dataset and an Improved Multi-task Model for Floorplan Image Analysis"。
- **SegFormer**：Xie et al., NeurIPS 2021。
- **floor-plan-document-intelligence**（GitHub, alifarzadjamali）：分割 → 结构化 JSON + OCR；本项目使用其预训练权重。
- **TakeoffLens**（GitHub, hassanabbas16）：OCR + LLM 读取房间名称与面积；发现 SVG 比例 100 单位 = 1 m。
- **neural_floorplan**（GitHub, daegeun-kim）：矢量化实验。
- **本项目的区别**：输出 BIM/IFC；做到造价并用 AACE 定位；系统性的误差归因；建材价格风险分析。

---

## 14. 结果记录区（执行时填写）

- 0.3 对齐与比例验证：
- 1.2 基线（测试集）：条件 E 造价中位绝对误差 = ；最大误差来源 = ；AACE 等级 =
- 2.4 尺度：门宽法 / OCR / 融合 的尺度误差中位数 =
- 2.5 改进后（测试集）：
- 3.1 单价来源：
- 3.2 回测（新数据）：
- 4.3 自训练模型（如做）：
