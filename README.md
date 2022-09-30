# Recurrent Memory Transformer

>Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev

RMT is a memory-augmented segment-level recurrent Transformer. It achieves state-of-the art results on Hyperpartisan dataset and beats Transformer-XL on algorithmic tasks and LM with limited input and memory size.

Recurrent Memory Transformer is implemented as follows:

![**RMT**](img/RMT_simple.png?raw=True)

We implement our memory mechanism with no changes to Transformer model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

## Performance
Task | Dataset | Metric | Transformer | Transformer-XL | RMT
-- | -- | -- | -- | -- | -- 
LM* | WT-103 | ppl | 29.95 | 24.12 | **23.99**
LM* | enwik8 | bpc | 1.39 | 1.283 | **1.228**
step-by-step | qudaratic equations | acc |  | 93.4 | **99.8**
Classification | (Hyperpartisan) | acc | 94.9 |  | **98.1** 

\* - limited input and memory size
 

## Code

Scripts for running language modeling, algorithmic and mathematical experiments can be found in the core [pytorch folder](./pytorch/).

Our code is based on the [Transformer-XL repository](https://github.com/kimiyoung/transformer-xl).
The recurrent memory mechanism is implemented by updating the Transformer-XL PyTorch code.

All LM and algorithmic experiments from our paper were conducted using this repository.
Raw experiments results from the paper can be found in the [experiments folder](./experiment_results/):

- `short_synthetic+MT.csv` -- experiments with copy/reverse/associative retrieval from figure 4.
- `long_copy_reverse.csv` -- results for sequence length up to 1080 for copy/reverse, figure 5a, 5b and 7c.
- `results_wt103_enwik8_all_parameters.csv` -- full set of hyperparameters for enwik8/wt103 runs.
- `results_wt103_enwik8.csv` -- enwik8/wt103 with selected subset of hyperparameters.

## Reproduce results

To obtain datasets used in the paper:

- LM on WT103 and enwik8:
  - follow the instructions from the Transformer-XL repository.
- Copy & Reverse data:
  - produced with `./generation/algorithmic.ipynb`
- Quadratic equations data:
  - produced with  `./generation/square_equations.ipynb`

Example usage: 

`bash <script_name>.sh train`
## Citation
If you find our work useful, please cite the [NeurIPS 2022 paper]():
```
@article{}
