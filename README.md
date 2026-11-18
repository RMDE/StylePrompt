# StylePrompt

## Idea

- Motivation：图像生成提示词中的风格描述会改变图像的纹理、色彩、构图和抽象程度，这些变化可能影响下游视觉模型的判断。
- Target：通过选择和组合风格修饰符，在尽量保持主体语义的同时，测试视觉模型的鲁棒性。
- Tasks：图像分类、目标检测、图像分割、视觉理解和图像内容安全检测。

## Method

- 数据集构建：从提示词中提取风格片段，标注机制类别、效应属性和短语粒度，经过清洗与去重形成风格修饰符数据集。
- 单修饰符测试：分别加入一个风格修饰符，生成图像并记录目标模型的输出与评分。
- 组合攻击：根据单修饰符评分选择候选，执行类内组合搜索或类间组合优化。候选分为 `mechanism` 和 `effect` 两类。

## Dataset

数据位于 `style_prompt/dataset_experiments/`：

- `prompt_level_records_streaming_8w.jsonl`：73,410 条提示词记录。
- `fragment_level_records.jsonl`：41,802 条风格片段，包含来源 ID、机制类别、八维效应属性和粒度标签。
- `selected_descriptors/`：按机制与效应筛选的候选修饰符。

最终实验词库见 `style_prompt/style_modifier_design.py`。原始图像、第三方工具源码和模型权重需另行准备。

## Usage

安装数据处理依赖，并进入代码目录。实际生成与评估还需安装对应模型的依赖，将外部模型资源目录配置为 `STYLEPROMPT_TOOLS_ROOT`。

```bash
pip install -r requirements.txt
export STYLEPROMPT_TOOLS_ROOT=/path/to/model_assets
cd style_prompt
```

### Single-modifier experiments

- 分类：`experiment_single_fragment_classification.py`
- 检测：`experiment_single_fragment_extraction.py`
- 分割：`experiment_single_fragment_segmentation.py`
- 视觉理解：`experiment_single_fragment_understanding.py`
- 内容安全：`experiment_single_fragment.py`

以 SD1.5 和 ResNet50 为例，先运行一个条件进行检查：

```bash
python experiment_single_fragment_classification.py \
  --models sd15 --classifiers resnet50 --limit 1
```

完成单修饰符实验后，整理评分，供组合攻击读取：

```bash
python process_modifier_results.py
```

### Combination attack

最终入口为 `experiment_multi_fragment.py`，命令行默认执行类间搜索；类内搜索函数为 `optimize_cross_false`。

```bash
python experiment_multi_fragment.py \
  --model sd15 --task image_classification --target-model resnet50 \
  --candidate-type mechanism --device cuda:0 --path ./img/attack_resnet50
```

攻击需要先生成对应的 SD1.5 单修饰符评分表。`--path` 应使用专用目录，评估过程会清空并重建该目录。结果保存到 `experiment_outputs/multi_fragment/`。

批量运行入口为 `run_multi_fragment_batch.py`，可先查看执行计划：

```bash
python run_multi_fragment_batch.py --dry-run --tasks image_classification
```

### Dataset analysis

```bash
python dataset_summary.py
python dateset_selection.py --device cpu --allow-model-download
```
