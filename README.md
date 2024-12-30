# MAZE: Multi-Agent Zero-shot coordination by coEvolution

This is the code for the paper "Heterogeneous Multi-agent Zero-Shot Coordination by Coevolution".


## Train

Our implementation is based on the code at https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019

To run the experiment, please merge the given file in the corresponding folder in the above link and refer to its installation instructions.

Run MAZE on the *CR* layout:

```
python maze.py with layout_name="simple" EX_NAME="pbt_simple" SEEDS="[1000]" ENTROPY_POOL=0.01 PAIR_INTERVAL=5 ARCHIVE_SIZE=20 POPULATION_SIZE=5
```

Run MAZE on the *AA* layout:

```
python maze.py with layout_name="unident_s" EX_NAME="pbt_unident_s" SEEDS="[1000]" ENTROPY_POOL=0.01 PAIR_INTERVAL=5 ARCHIVE_SIZE=20 POPULATION_SIZE=5
```

Run MAZE on the *AA-2* layout:

```
python maze.py with layout_name="unident" EX_NAME="pbt_unident" SEEDS="[1000]" ENTROPY_POOL=0.01 PAIR_INTERVAL=5 ARCHIVE_SIZE=20 POPULATION_SIZE=5
```

Run MAZE on the *FC* layout:

```
python maze.py with layout_name="random0" EX_NAME="pbt_random0" SEEDS="[1000]" ENTROPY_POOL=0.04 PAIR_INTERVAL=5 ARCHIVE_SIZE=20 POPULATION_SIZE=5
```

To run MAZE on the *CR-2* layout, please first replace the `resolve_interacts()` function by `resolve_interacts_2()` in `overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py`, then run:

```
python maze.py with layout_name="simple" EX_NAME="pbt_simple" SEEDS="[1000]" ENTROPY_POOL=0.01 PAIR_INTERVAL=5 ARCHIVE_SIZE=20 POPULATION_SIZE=5
```


## Test
Test models are saved in `./test_model`, which include random, sp, and maze partner models.

For human proxy, please use the model provided in  https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019.

The test script is the .py and .sh file in the `human_aware_rl/experiments/` of https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019, and just replace the model names.

## Citation
```
@article{maze,
    author = {Ke Xue, Yutong Wang, Cong Guan, Lei Yuan, Haobo Fu, Qiang Fu, Chao Qian, Yang Yu},
    title = {Heterogeneous Multi-Agent Zero-Shot Coordination by Coevolution.},
    journal = {IEEE Transactions on Evolutionary Computation.},
    year = {in press},
}

@article{maze-arxiv,
    author = {Ke Xue, Yutong Wang, Cong Guan, Lei Yuan, Haobo Fu, Qiang Fu, Chao Qian, Yang Yu},
    title = {Heterogeneous Multi-Agent Zero-Shot Coordination by Coevolution.},
    year={2022},
    journal={arXiv:2208.04957},
}
```
