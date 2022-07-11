# Recurrent Memory Transformer
RMT is a memory-augmented segment-level recurrent Transformer.

## Code

Scripts for running language modeling, algorithmic and mathematical experiments are located in pytorch/ folder.

Our code is forked from the Transformer-XL repository https://github.com/kimiyoung/transformer-xl.
We made changes to the Transformer-XL model in PyTorch scripts to implement Recurrent Memory Transformer.

All experiments results were produced with this repository.

Raw experiments results that were used in the paper are in experiment_results/ folder:

- short_synthetic+MT.csv -- experiments with copy/reverse/associative retrieval from figure 4.
- long_copy_reverse.csv -- results for sequence length up to 1080 for copy/reverse, figure 5a, 5b and 7c.
- results_wt103_enwik8_all_parameters.csv -- full set of hyperparameters for enwik8/wt103 runs.
- results_wt103_enwik8.csv -- enwik8/wt103 with selected subset of hyperparameters.

## Data

To obtain datasets used in the paper:

- LM on WT103 and enwik8:
  - follow the instructions from the Transformer-XL repository.
- Copy & Reverse data:
  - generation/algorithmic.ipynb
- Associative retrieval data:
  - data generation code: https://github.com/GokuMohandas/fast-weights/blob/539fb10e3c384d5f782af2560bf28631cd0eaa61/fw/data_utils.py
- Quadratic equations data:
  - generation/square_equations.ipynb
