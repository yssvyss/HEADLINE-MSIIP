# HEADLINE-MSIIP

MSIIP's project page for **Entity-Aware Head CT Report Generation via Classification-Derived Label Priors**, developed for the HEADLINE 2026 challenge.

Our pipeline ensembles multi-label classifiers to predict clinical entities from head CT, then uses those predictions as auxiliary inputs to a Qwen3.5-4B report generator. The generator's language module was fully fine-tuned while its vision encoder and MLP projector were frozen.

## Release status

The inference code and model weights are being prepared for public release. This repository currently contains the project overview. No challenge images or reports are included.
