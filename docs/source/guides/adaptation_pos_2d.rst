.. role:: python(code)
   :language: python

.. _adaptation_pos_2d_label:

Adaptation of Positional Encodings for 2D Modalities
====================================================

This guide explains the adaptation of positional embeddings for 2D modalities, such as images, in MultiMeditron. The default behavior of MultiMeditron is to use 1D positional embeddings, which may not be optimal for 2D data. This guide describes how to adapt the positional embeddings to better suit 2D modalities.

Overview
--------

While many approaches exist to adapt 1D positional embeddings to 2D data, they often presuppose that the model was trained with 2D positional embeddings from the start. However, this is not the case for MultiMeditron, which is based on the :code:`Llava` architecture. This guide presents a novel approach that allows the adaptation of 1D positional embeddings to 2D data without the need for retraining the entire model from scratch. This approach has already been implemented for the llama family model prior to Llama 4, which supports 2D positional embeddings natively.

Implementation Details
----------------------

The implementation details of this approach have been made available on a [fork](https://github.com/EPFLiGHT/transformers/tree/rotary-adapt) of the [transformers library](https://github.com/huggingface/transformers), which can be found on GitHub. This fork is specifically designed to support the adaptation of 1D positional embeddings to 2D data.


To use this approach, you need to follow these steps:

1. Clone the forked transformers repository and install it in your environment. You can do this by running the following commands:

   .. code-block:: bash
        git clone https://github.com/EPFLiGHT/transformers.git -b rotary-adapt transformers-rotary-adapt
        cd transformers-rotary-adapt
        pip install -e .

2. Configure MultiMeditron to use 2D positional embeddings. To do this, you need to modify the configuration file used for training/evaluation and set the argument :code:`use_2d_position_ids` to :code:`true` in the :code:`training_args` section. When evaluating or writing your own code, you should also set this argument on the :code:`DataCollatorForMultimodal`.

3. Run the training/evaluation as usual with MultiMeditron. The model will now use 2D positional embeddings for the LLM, which should improve performance on tasks involving 2D data.


