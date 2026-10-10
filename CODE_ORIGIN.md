# Code origin and packaging changes

The experiment implementations were copied from the retained PFPA project sources:

- `scripts/probe_transformer_block_plasticity.py`, `scripts/probe_function_preserving_update.py`, and `scripts/pfpa_confirmatory_runner_v2.py` derive from the retained SegFormer-B0 intervention and confirmatory runner scripts.
- `scripts/swin_t_model.py` and `scripts/run_swin_interventions.py` derive from the retained Swin-T + UPerNet model and intervention runner.

Packaging changes are limited to replacing the original fixed server root and physical GPU index with `PFPA_DATA_ROOT`, `PFPA_OUTPUT_ROOT`, `PFPA_SWIN_SOURCE_CHECKPOINT`, and `PFPA_VISIBLE_GPU`; locating helper modules beside the installed scripts; accepting an explicit local pretrained-model directory in the B0 confirmatory runner; computing the runner/protocol hashes from the packaged inputs; and reporting the currently visible CUDA device instead of probing a machine-specific GPU 3.

The scoring equations, gradient projection and update equations, candidate blocks, training/evaluation operators, target/source inputs, and random seeds were retained. The scripts still require external datasets and matching checkpoints. No checkpoints or data were copied into this package.
