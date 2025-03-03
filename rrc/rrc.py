import random
from typing import Optional

import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


class RRCReplayBuffer(ReplayBuffer):
    """ReplayBuffer with RR-C."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rr_buf = []

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        batch_inds = []
        while len(batch_inds) < batch_size:
            if len(self._rr_buf) == 0:
                inds = np.arange(0, self.buffer_size)
                np.random.shuffle(inds)
                self._rr_buf = list(inds)
            ind = self._rr_buf.pop()
            if self.optimize_memory_usage:
                if ind < self.pos or (self.full and ind != self.pos):
                    batch_inds.append(ind)
            else:
                if ind < self.pos or self.full:
                    batch_inds.append(ind)
        assert len(batch_inds) == batch_size
        return self._get_samples(np.array(batch_inds), env=env)


class WORReplayBuffer(ReplayBuffer):
    """ReplayBuffer with within-minibatch without-replacement sampling."""

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        if self.full:
            batch_inds = (
                np.array(random.sample(range(1, self.buffer_size), batch_size))
                + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.array(random.sample(range(self.pos), batch_size))
        return self._get_samples(batch_inds, env=env)
