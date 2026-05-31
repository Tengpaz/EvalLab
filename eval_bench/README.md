# Eval Bench: 通用新视角合成 / Virtual Camera Benchmark

这是一个独立的评测工程，用来评测多种 novel view synthesis / virtual camera 模型，而不是写死到某一个模型或某一份数据。它参考 Stable Virtual Camera benchmark 的数据组织和评测习惯，但模型侧可以是 Python adapter，也可以是任意黑盒命令行。

## 目录

```text
configs/                # dataset/model/run YAML 示例
scripts/                # validate, inference, metrics, aggregate CLI
eval_bench/             # Python package
adapters/               # 用户模型 adapter 示例
outputs/                # run 输出
```

## 环境安装

最小 smoke test 只需要 Python 3.10+ 和 PyYAML：

```bash
cd eval_bench
python -m pip install -r requirements.txt
```

如需 JPEG 读取、更接近论文评测的 SSIM、或 LPIPS：

```bash
python -m pip install -r requirements-optional.txt
```

当前实现内置了无依赖 PNG 读写、PSNR 和一个全局 SSIM，方便先把 pipeline 跑通。正式报告指标时，建议固定图像后处理规则，并在你的环境中安装 `torch`、`lpips`、`numpy`。

## 官方 Benchmark 格式要点

建议先把参考仓库放到 `external/stable-virtual-camera`：

```bash
git clone https://github.com/Stability-AI/stable-virtual-camera.git external/stable-virtual-camera
```

本工程兼容的官方风格 scene 结构是：

```text
scene_id/
  images/
    000.png
    001.png
  transforms.json
  train_test_split_1.json
  train_test_split_3.json
  train_test_split_6.json
  train_test_split_32.json
```

关键规则：

- `transforms.json` 使用 NeRF 风格 `frames`，每帧包含 `file_path` 和 4x4 `transform_matrix`。
- camera convention 保持 OpenGL camera-to-world，不在 adapter 里偷偷转换。
- 如果 frame 没有显式 `id` / `frame_id`，会按 `file_path` 排序后分配稳定 id。split id 必须和这个排序一致。
- split 文件支持 `input_ids/target_ids`、`train_ids/test_ids`、或包含 `splits` 的简单格式。
- 指标只计算 target views，不把 input views 混进指标。
- crop/resize 不硬编码在模型或数据集里。用 YAML 的 `image_preprocess` 和 `metric_postprocess` 显式配置，例如 resize 到 256、center crop、或尺寸不一致时报错。

## 配置一个官方数据集

编辑 `configs/datasets/seva_official_example.yaml`：

```yaml
type: seva_benchmark
name: re10k
root: /data/benchmarks/seva/re10k
split_file: train_test_split_3.json
num_inputs: 3
sort_images: true
image_preprocess:
  mode: center_crop
  size: 576
metric_postprocess:
  mismatch: error
```

验证：

```bash
python scripts/validate_dataset.py \
  --dataset-config configs/datasets/seva_official_example.yaml
```

## 接入一个新数据集

如果你已有 NeRF 风格数据，用 `generic_transforms`：

```yaml
type: generic_transforms
name: my_dataset
root: /data/my_dataset
scene_glob: "*"
transforms_file: transforms.json
image_dir: images
auto_split:
  strategy: first_k_as_input
  k: 3
```

也可以提供手写 split：

```yaml
split_file: train_test_split_3.json
```

支持的自动 split：

- `first_k_as_input`: 前 k 帧为 input，剩余为 target。
- `fixed_input_ids`: 手写 input ids，target ids 可省略为其余帧。
- `every_n`: 每隔 n 帧作为 input，其余为 target。

检查 split：

```bash
python scripts/inspect_split.py \
  --dataset-config configs/datasets/generic_transforms_example.yaml
```

## 接入一个 Python 模型

YAML：

```yaml
model:
  type: python
  adapter: adapters/example_model_adapter.py:ExampleModelAdapter
  weights: /path/to/model_weights.ckpt
  config: /path/to/model_config.yaml
  device: cuda:0
  extra_args:
    cfg: 6.0
    camera_scale: 1.0
```

Adapter 类需要实现：

```python
class MyModelAdapter:
    def setup(self, model_config):
        ...

    def predict(self, batch):
        # return {target_id: image_path_or_ImageData_or_PIL_or_numpy}
        ...
```

`batch` 至少包含：

- `scene_id`
- `input_images`
- `input_image_paths`
- `input_cameras`
- `target_cameras`
- `target_ids`
- `output_dir`
- `metadata`

示例 mock adapter 在 `adapters/example_model_adapter.py`，可复制第一张 input 或输出灰图。

## 接入命令行模型

YAML：

```yaml
model:
  type: command
  command_template: >
    python /path/to/infer.py
    --input_images {input_images_json}
    --input_cameras {input_cameras_json}
    --target_cameras {target_cameras_json}
    --weights {weights}
    --output_dir {output_dir}
  weights: /path/to/weights.ckpt
  output_pattern: "{target_id}.png"
  env:
    CUDA_VISIBLE_DEVICES: "0"
```

每个 scene/split 会生成：

- `input_images.json`
- `input_cameras.json`
- `target_cameras.json`
- `metadata.json`

命令执行后会检查 `{target_id}.png` 是否完整，缺图会列出缺失 target id。

## Smoke Test

这个 smoke test 会生成 3 张假图和假 camera，跑完整链路：

```bash
cd eval_bench
python scripts/prepare_benchmark.py --make-tiny --out outputs/tiny_dataset
python scripts/validate_dataset.py --dataset-config configs/datasets/tiny_smoke.yaml
python scripts/run_inference.py --run-config configs/runs/tiny_smoke.yaml
python scripts/compute_metrics.py --run-dir outputs/tiny_smoke_python_copy
python scripts/aggregate_results.py \
  --runs outputs/tiny_smoke_python_copy \
  --out outputs/aggregate_tiny.csv
```

输出：

```text
outputs/tiny_smoke_python_copy/
  predictions/tiny_smoke/train_test_split_1/scene_000/{1,2}.png
  metadata/resolved_config.yaml
  metadata/environment.json
  metadata/per_scene_status.jsonl
  metrics/per_image_metrics.jsonl
  metrics/per_scene_metrics.csv
  metrics/summary.csv
outputs/aggregate_tiny.csv
```

## 完整评测

准备一个 run config，例如 `configs/runs/example_run.yaml`，然后：

```bash
python scripts/run_inference.py --run-config configs/runs/example_run.yaml
python scripts/compute_metrics.py --run-dir outputs/my_model_re10k_p3
```

多 run 汇总：

```bash
python scripts/aggregate_results.py \
  --runs outputs/run_a outputs/run_b \
  --out outputs/aggregate.csv
```

每次 inference 会保存：

- `metadata/resolved_config.yaml`
- `metadata/environment.json`
- `metadata/per_scene_status.jsonl`
- `metadata/failures.jsonl`，如果有失败样本

## 常见错误

Camera convention 不一致：
确认 `transform_matrix` 是 OpenGL camera-to-world。如果模型内部需要 OpenCV/world-to-camera，请在模型 adapter 里显式转换，并记录在 adapter/config 中。

Split id 对不上：
如果 frame 没有显式 `id`，本工程按 `file_path` 排序分配 id。官方 split 通常依赖稳定排序；自建 split 时先跑 `inspect_split.py`。

图像尺寸不一致：
`validate_dataset.py` 会给 warning。评测时设置 `metric_postprocess.mismatch` 为 `error`、`resize_pred_to_gt`、`resize_gt_to_pred` 或 `center_crop_common`。

LPIPS 不可用：
安装 `torch lpips numpy`，或从 `metrics.enabled` 移除 `lpips`。如果 `skip_unavailable: true`，LPIPS 会被跳过并在 CSV 中留空。

CUDA 显存不足：
降低模型配置里的分辨率、input 数量、batch size，或使用 `CommandAdapter.env.CUDA_VISIBLE_DEVICES` 指定设备。run 支持 resume，已有预测图可跳过。

官方数据没有下载：
本工程不会假设数据存在。`root`、`split_file`、`transforms_file` 都由 YAML 控制；缺文件时会给出具体路径和修复建议。
