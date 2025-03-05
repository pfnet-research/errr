# Experience Replay with Random Reshuffling

This is the code for the paper "Experience Replay with Random Reshuffling" on arXiv: https://arxiv.org/abs/2503.02269.

## RR-C experiments

Our implementations use and adapt code from CleanRL (https://github.com/vwxyzjn/cleanrl/tree/e648ee2dc8960c59ed3ee6caf9eb0c34b497958f). Its original LICENSE file is at `LICENSE_cleanrl`. Our modifications follow the MIT license in `LICENSE`.

To run the experiments, setup CleanRL's dependencies with JAX following its instructions.

### How to run
```bash
# C51 WR
python  -m rrc.c51_atari_jax --env-id AmidarNoFrameskip-v4 --seed 0
# C51 R -m R-C
python  -m rrc.c51_atari_jax_rrc --env-id AmidarNoFrameskip-v4 --seed 0
# C51 W -m OR
python  -m rrc.c51_atari_jax_wor --env-id AmidarNoFrameskip-v4 --seed 0
# DQN W -m R
python  -m rrc.dqn_atari_jax --env-id AmidarNoFrameskip-v4 --seed 0
# DQN R -m R-C
python  -m rrc.dqn_atari_jax_rrc --env-id AmidarNoFrameskip-v4 --seed 0
```

## RR-M experiments

Our implementations use and adapt code from LAP-PAL's discrete aciton code (https://github.com/sfujim/LAP-PAL/tree/e33ed4305aefe1b401ee37e4d759f8e99c155ea1/discrete). Its original LICENSE file is at `LICENSE_LAP-PAL`. Our modifications follow the MIT license in `LICENSE`.

To run the experiments, setup LAP-PAL's dependencies following its instructions. Additionally, you need to install `cupy` (https://cupy.dev/), which we use for the efficient RR-M implementation.

### How to run
```bash
# DDQN+LAP WR
python -m rrm.main --env AmidarNoFrameskip-v4 --seed 0
# DDQN+LAP RR-M
python -m rrm.main --rrm --env AmidarNoFrameskip-v4 --seed 0
# DDQN+LAP ST
python -m rrm.main --stratified --env AmidarNoFrameskip-v4 --seed 0
# DDQN+LAP RR-M+ST
python -m rrm.main --rrm --stratified --env AmidarNoFrameskip-v4 --seed 0
```
