# MSIIP HEADLINE 2026 inference

This repository contains the MSIIP inference pipeline for **Entity-Aware Head CT Report Generation via Classification-Derived Label Priors**. It uses the soft-tissue reconstruction, ensembles eight multi-label classifiers, and conditions a Qwen3.5-4B report generator on CT slices and predicted entities. The released generator uses full language-module fine-tuning with the vision encoder and MLP projector frozen.

The runtime is based on the [HEADLINE participant starter kit](https://github.com/CCI-Bonn/headline26-starter-kit). The classifier label vocabulary covers the 50 most frequent findings and 50 most frequent diagnoses. This inference snapshot restricts the prompt to positive predictions among the first 10 labels in each group. It retains the separate classifier and generator preprocessing settings in the submission code.

## Run the inference container

Use Linux Docker with NVIDIA GPU support. Download the [published model files](https://huggingface.co/yssvyss/HEADLINE-MSIIP) into a model directory containing `multilabel/` and `vlm/`. The input directory must follow the HEADLINE socket layout, including the soft-tissue volume and geometry JSON. Create a writable output directory before running.

```sh
docker build -t msiip-headline .
docker run --rm --gpus all --network none \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/model:/opt/ml/model:ro" \
  -v "$PWD/output:/output" \
  msiip-headline
```

The container writes `/output/radiology-text-reports.json`. CT studies, annotations, reports and optimizer states are not included. Use the models for research.

## License

MSIIP's original contributions are released under [Apache-2.0](LICENSE). This grant does not relicense third-party material, including portions inherited from the HEADLINE starter kit. Any upstream notices and applicable terms continue to apply.
