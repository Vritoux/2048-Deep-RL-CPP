# 2048 Reinforcement Learning Agent

This project is a C++ implementation of a Reinforcement Learning agent for the game 2048. It is based on academic research describing the application of N-Tuple Networks, Expectimax search, and Temporal Coherence Learning to this environment.

## Overview

- **State Representation**: The 4x4 board is encoded as a 64-bit integer (`uint64_t`). State transitions are handled via pre-calculated Look-Up Tables.
- **Search Algorithm**: Expectimax search is used to account for the stochastic tile generation (90% chance of '2', 10% chance of '4').
- **Evaluation Function**: The agent uses an N-Tuple Network consisting of 4 asymmetric shapes of 7 connected tiles to approximate state values.
- **Training Method**: The model is trained using Multi-Stage Temporal Difference (MS-TD) Learning with Temporal Coherence (TC). The training process is parallelized across multiple CPU cores using OpenMP.
- **Transposition Table**: A 32 MB cache stores Expectimax evaluations to avoid redundant calculations during inference.

## References

- M. Szubert and W. Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048* (2014)
- I. C. Wu et al., *Multi-Stage Temporal Difference Learning for 2048* (2014)

## Usage

### Compilation
The code requires a compiler with C++17 and OpenMP support.
```bash
g++ -O3 -fopenmp -std=c++17 fast_train_tc.cpp -o fast_train_tc
g++ -O3 -std=c++17 evaluate_ultimate.cpp -o evaluate_ultimate
```

### Training
The training executable requires approximately 13 GB of RAM due to the size of the 7-Tuple network and the Temporal Coherence metadata.
```bash
# Usage: ./fast_train_tc <num_episodes> [optional_resume_checkpoint.bin]
./fast_train_tc 50000000
```
*Note: Training can be safely interrupted using `Ctrl+C`. The program will save a checkpoint file and a 4 GB backward-compatible model file for evaluation.*

### Evaluation
The evaluator runs the trained model against the game environment.
```bash
# Usage: ./evaluate_ultimate <run_directory> <num_games> <search_depth>
./evaluate_ultimate runs_tc/run_XXXXXXXX_XXXXXX 100 3
```

## Benchmark Example

The following results were obtained evaluating the model after 30 million training episodes, executing 100 games at a 3-ply search depth:

- **Average Score**: 31,825
- **Max Tile Distribution**:
  - 4096: 9.0%
  - 2048: 73.0%
  - 1024: 17.0%
  - 512: 1.0%
