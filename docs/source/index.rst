.. MultiMeditron documentation master file, created by
   sphinx-quickstart on Wed Oct  1 14:49:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Placeholder for a cooler banner

.. image:: _static/multimeditron_dark.png
    :alt: MultiMeditron
    :align: center
    :width: 800px
    :class: dark-only

.. image:: _static/multimeditron_light.png
    :alt: MultiMeditron
    :align: center
    :width: 800px
    :class: light-only

.. raw:: html

   <div style="text-align: center; font-size: 25px">
   <b>A scalable, modular, multimodal training pipeline</b>
   </div>

|
|
|

🎉 Latest Updates
=================

v1.0.0 (2025/10/01):
    - MultiMeditron v1.0.0 published.

v1.1.0 (2026/04/15):
    - Added full 7-expert vs 5-expert evaluation analysis (GMAI, SLAKE, PathVQA).
    - Documented PathVQA root cause: severe binary "No" bias in the 7-expert model.
    - Added a proper train/val/test split to gating training (`test_split`).

v1.1.1 (2026/04/16):
    - Added PathVQA routing analysis (500 test images) for 5-expert and 7-expert gating.
    - Documented routing shift from Generalist to Skin/MRI on histopathology images.

v1.1.2 (2026/04/21):
    - Clarified release wording in this documentation page.


✨ Overview
===========

MultiMeditron is a scalable and modular pipeline to train multimodal models.

Features:

- **Modular modality**: Designed to be easily expanded to any types of modality
- **Scalable**: Scalable to multinode training and efficient GPU memory usage using Deepspeed
- **Configurable**: Trainings can be configured using a single YAML configuration file
- **Easy to install**: We provide Docker images for easier reproducibility


� Related Papers
=================

**Foundation Models**

- `The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783>`_ — Grattafiori et al., Meta AI, 2024.
- `MEDITRON-70B: Scaling Medical Pretraining for Large Language Models <https://arxiv.org/abs/2311.16079>`_ — Chen et al., EPFL, 2023.
- `Learning Transferable Visual Models From Natural Language Supervision <https://arxiv.org/abs/2103.00020>`_ — Radford et al., OpenAI, 2021. (CLIP ViT-B/32, used as expert encoders)

**Multimodal & MoE Architecture**

- `BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific figure-caption pairs <https://arxiv.org/abs/2303.00915>`_ — Zhang et al., Microsoft Research, 2023.
- `Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer <https://arxiv.org/abs/1701.06538>`_ — Shazeer et al., Google Brain, 2017.

**Evaluation Benchmarks**

- `GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI <https://arxiv.org/abs/2408.03361>`_ — Chen et al., 2024.
- `PathVQA: 30000+ Questions for Medical Visual Question Answering <https://arxiv.org/abs/2003.10286>`_ — He et al., 2020.
- `SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering <https://arxiv.org/abs/2102.09542>`_ — Liu et al., 2021.


�📚 Documentation
================

.. toctree::
    :glob:
    :maxdepth: 2
    :includehidden:

    User Guide <guides/guide>
    Reference <ref/modules>


