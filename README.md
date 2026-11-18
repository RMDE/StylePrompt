# StylePrompt

## Idea

- Motivation: Style descriptions in image generation prompts can change texture, color, composition, and abstraction, potentially affecting downstream visual model predictions.
- Target: Evaluate visual model robustness by selecting and combining style modifiers while aiming to preserve the meaning of the main subject.
- Tasks: Image classification, object detection, image segmentation, visual understanding, and image safety detection.

## Method

- Dataset construction: Extract style fragments from prompts, annotate their mechanism categories, effect profiles, and phrase granularity, then clean and deduplicate the records.
- Single-modifier experiments: Add one style modifier at a time, generate images, and record target model outputs and scores.
- Combination attack: Select candidates using single-modifier scores and search for combinations within or across categories. Candidates belong to the `mechanism` or `effect` family.

## Dataset

The dataset is located in `style_prompt/dataset_experiments/`:

- `prompt_level_records_streaming_8w.jsonl`: 73,410 prompt records.
- `fragment_level_records.jsonl`: 41,802 style fragments with source IDs, mechanism categories, eight-dimensional effect profiles, and granularity labels.
- `selected_descriptors/`: Candidate modifiers selected by mechanism and effect.

The final experiment vocabulary is defined in `style_prompt/style_modifier_design.py`. Original images, third-party tool source code, and model weights must be obtained separately.

## Usage

Install the data processing dependencies and enter the code directory. Image generation and evaluation also require the dependencies of the selected models. Set `STYLEPROMPT_TOOLS_ROOT` to the external model assets directory.

```bash
pip install -r requirements.txt
export STYLEPROMPT_TOOLS_ROOT=/path/to/model_assets
cd style_prompt
```

### Single-modifier experiments

- Classification: `experiment_single_fragment_classification.py`
- Object detection: `experiment_single_fragment_extraction.py`
- Segmentation: `experiment_single_fragment_segmentation.py`
- Visual understanding: `experiment_single_fragment_understanding.py`
- Image safety: `experiment_single_fragment.py`

For a small check with SD1.5 and ResNet50, run one condition:

```bash
python experiment_single_fragment_classification.py \
  --models sd15 --classifiers resnet50 --limit 1
```

After completing the single-modifier experiments, process the scores for use by the combination attack:

```bash
python process_modifier_results.py
```

### Combination attack

The final entry point is `experiment_multi_fragment.py`. Its command-line interface runs cross-category search by default. Within-category search is implemented in `optimize_cross_false`.

```bash
python experiment_multi_fragment.py \
  --model sd15 --task image_classification --target-model resnet50 \
  --candidate-type mechanism --device cuda:0 --path ./img/attack_resnet50
```

The attack requires the corresponding SD1.5 single-modifier score tables. Use a dedicated directory for `--path`: evaluation clears and recreates this directory. Results are saved to `experiment_outputs/multi_fragment/`.

Use `run_multi_fragment_batch.py` for batch runs. Preview the execution plan with:

```bash
python run_multi_fragment_batch.py --dry-run --tasks image_classification
```

### Dataset analysis

```bash
python dataset_summary.py
python dateset_selection.py --device cpu --allow-model-download
```
