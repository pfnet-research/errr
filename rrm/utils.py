import cupy
import cupyx
import cv2
import gym
import numpy as np
import torch


def ReplayBuffer(state_dim, prioritized, is_atari, atari_preprocessing, batch_size, buffer_size, device, rrm: bool, stratified: bool):
	if is_atari: 
		return PrioritizedAtariBuffer(state_dim, atari_preprocessing, batch_size, buffer_size, device, prioritized, rrm=rrm, stratified=stratified)
	else: 
		assert not rrm
		assert not stratified
		return PrioritizedStandardBuffer(state_dim, batch_size, buffer_size, device, prioritized)


def asnumpy(ndarray) -> np.ndarray:
	if isinstance(ndarray, cupy.ndarray):
		return ndarray.get()
	else:
		return ndarray


def sample_without_replacement_from_sumtree(sumtree: "SumTree", batch_size: int) -> np.ndarray | cupy.ndarray:
	xp = sumtree.xp
	ind = sumtree.sample(batch_size)
	uniq_ids = set(asnumpy(ind))
	while len(uniq_ids) < batch_size:
		sumtree.batch_set(ind, xp.zeros_like(ind))
		ind = sumtree.sample(batch_size - len(uniq_ids))
		uniq_ids.update(asnumpy(ind))
	return xp.asarray(list(uniq_ids))


class PrioritizedAtariBuffer(object):
	def __init__(self, state_dim, atari_preprocessing, batch_size, buffer_size, device, prioritized, rrm: bool, stratified: bool):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.state_history = atari_preprocessing["state_history"]

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((
			self.max_size + 1,
			atari_preprocessing["frame_size"],
			atari_preprocessing["frame_size"]
		), dtype=np.uint8)

		self.action = np.zeros((self.max_size, 1), dtype=np.int64)
		self.reward = np.zeros((self.max_size, 1))
		
		# not_done only consider "done" if episode terminates due to failure condition
		# if episode terminates due to timelimit, the transition is not added to the buffer
		self.not_done = np.zeros((self.max_size, 1))
		self.first_timestep = np.zeros(self.max_size, dtype=np.uint8)

		self.prioritized = prioritized

		# Use cuda for efficient computation of RR-M.
		self.xp = cupy

		if self.prioritized:
			self.tree = SumTree(self.max_size, xp=self.xp)
			self.max_priority = 1.0
			self.beta = 0.4
		
		self.rrm = rrm
		if rrm:
			assert prioritized
			self.tree2 = SumTree(self.max_size, xp=self.xp)
			self._actual_sample_counts = self.xp.zeros(self.max_size, dtype=np.int64)
			self._expected_sample_counts = self.xp.zeros(self.max_size, dtype=np.float64)
			self._old_oversampled = self.xp.zeros(self.max_size, dtype=np.bool)
		self.stratified = stratified


	def add(self, state, action, next_state, reward, done, env_done, first_timestep):
		# If dones don't match, env has reset due to timelimit
		# and we don't add the transition to the buffer
		if done != env_done:
			return

		self.state[self.ptr] = state[0]
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.first_timestep[self.ptr] = first_timestep

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		if self.prioritized:
			self.tree.set(self.ptr, self.max_priority)
			if self.rrm:
				self.tree2.set(self.ptr, self.max_priority)
				if self._expected_sample_counts[self.ptr] > 0:
					self._actual_sample_counts[self.ptr] = 0
					self._expected_sample_counts[self.ptr] = 0
					# Deleting the oldest transition leads to a mismatch between the actual and expected sample counts.
					self._expected_sample_counts *= self._actual_sample_counts.sum() / self._expected_sample_counts.sum()


	def sample(self):
		if self.rrm:
			assert self.prioritized
			xp = self.xp
			oversampled = self._actual_sample_counts > self._expected_sample_counts
			new_oversampled_ids, = xp.logical_and(oversampled, xp.logical_not(self._old_oversampled)).nonzero()
			new_undersampled_ids, = xp.logical_and(xp.logical_not(oversampled), self._old_oversampled).nonzero()
			cur_priorities = self.tree.nodes[-1][:self.max_size]
			self.tree2.batch_set(
				xp.concatenate([new_oversampled_ids, new_undersampled_ids]),
				xp.concatenate([cur_priorities[new_oversampled_ids] * 1e-8, cur_priorities[new_undersampled_ids]]),
			)
			if self.stratified:
				ind = self.tree2.sample(self.batch_size, stratified=self.stratified)
			else:
				ind = sample_without_replacement_from_sumtree(self.tree2, self.batch_size)
			self._expected_sample_counts += cur_priorities * self.batch_size / cur_priorities.sum()
			self._actual_sample_counts[ind] += 1
			ind = asnumpy(ind)
			self._old_oversampled = oversampled
		else:
			if self.prioritized:
				ind = self.tree.sample(self.batch_size, stratified=self.stratified)
				ind = asnumpy(ind)
			else:
				ind = np.random.randint(0, self.size, size=self.batch_size)

		# Note + is concatenate here
		state = np.zeros(((self.batch_size, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
		next_state = np.array(state)

		state_not_done = 1.
		next_not_done = 1.
		for i in range(self.state_history):

			# Wrap around if the buffer is filled
			if self.size == self.max_size:
				j = (ind - i) % self.max_size
				k = (ind - i + 1) % self.max_size
			else:
				j = ind - i
				k = (ind - i + 1).clip(min=0)
				# If j == -1, then we set state_not_done to 0.
				state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1, 1)
				j = j.clip(min=0)

			# State should be all 0s if the episode terminated previously
			state[:, i] = self.state[j] * state_not_done
			next_state[:, i] = self.state[k] * next_not_done

			# If this was the first timestep, make everything previous = 0
			next_not_done *= state_not_done
			state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

		batch = (
			torch.ByteTensor(state).to(self.device).float(),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.ByteTensor(next_state).to(self.device).float(),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

		if self.prioritized:
			if self.xp is cupy:
				ind = cupy.asarray(ind)
				weights = self.xp.array(self.tree.nodes[-1][ind]) ** -self.beta
				weights /= weights.max()
				weights = weights.get()
			else:
				weights = self.xp.array(self.tree.nodes[-1][ind]) ** -self.beta
				weights /= weights.max()
			self.beta = min(self.beta + 4.8e-8, 1) # Hardcoded: 0.4 + 4.8e-8 * 12.5e6 = 1.0. Only used by PER.
			batch += (ind, torch.FloatTensor(weights).to(self.device).reshape(-1, 1))

		return batch


	def update_priority(self, ind, priority):
		self.max_priority = max(priority.max(), self.max_priority)
		ind = self.xp.asarray(ind)
		priority = self.xp.asarray(priority)
		self.tree.batch_set(ind, priority)
		if self.rrm:
			oversampled = self._old_oversampled[ind]
			self.tree2.batch_set(ind, self.xp.where(oversampled, 1e-8 * priority, priority))


# Replay buffer for standard gym tasks
class PrioritizedStandardBuffer():
	def __init__(self, state_dim, batch_size, buffer_size, device, prioritized):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

		self.prioritized = prioritized

		if self.prioritized:
			self.tree = SumTree(self.max_size)
			self.max_priority = 1.0
			self.beta = 0.4


	def add(self, state, action, next_state, reward, done, env_done, first_timestep):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		if self.prioritized:
			self.tree.set(self.ptr, self.max_priority)
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self):
		ind = self.tree.sample(self.batch_size) if self.prioritized \
			else np.random.randint(0, self.size, size=self.batch_size)

		batch = (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

		if self.prioritized:
			weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
			weights /= weights.max()
			self.beta = min(self.beta + 2e-7, 1) # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PER.
			batch += (ind, torch.FloatTensor(weights).to(self.device).reshape(-1, 1))

		return batch


	def update_priority(self, ind, priority):
		self.max_priority = max(priority.max(), self.max_priority)
		self.tree.batch_set(ind, priority)


class SumTree(object):
	def __init__(self, max_size, xp=np):
		self.xp = xp
		self.nodes = []
		# Tree construction
		# Double the number of nodes at each level
		level_size = 1
		for _ in range(int(np.ceil(np.log2(max_size))) + 1):
			nodes = xp.zeros(level_size)
			self.nodes.append(nodes)
			level_size *= 2


	# Batch binary search through sum tree
	# Sample a priority between 0 and the max priority
	# and then search the tree for the corresponding index
	def sample(self, batch_size, stratified: bool=False):
		xp = self.xp
		if stratified:
			segment_size = self.nodes[0][0] / batch_size
			segment_starts = xp.arange(batch_size) * segment_size
			query_value = xp.random.uniform(0, segment_size, size=batch_size) + segment_starts
		else:
			query_value = xp.random.uniform(0, self.nodes[0][0], size=batch_size)
		node_index = xp.zeros(batch_size, dtype=int)
		
		for nodes in self.nodes[1:]:
			node_index *= 2
			left_sum = nodes[node_index]
			
			is_greater = xp.greater(query_value, left_sum)
			# If query_value > left_sum -> go right (+1), else go left (+0)
			node_index += is_greater
			# If we go right, we only need to consider the values in the right tree
			# so we subtract the sum of values in the left tree
			query_value -= left_sum * is_greater
		
		return node_index


	def set(self, node_index, new_priority):
		xp = self.xp
		priority_diff = new_priority - self.nodes[-1][node_index]

		for nodes in self.nodes[::-1]:
			if xp is cupy:
				cupyx.scatter_add(nodes, node_index, priority_diff)
			else:
				xp.add.at(nodes, node_index, priority_diff)
			node_index //= 2


	def batch_set(self, node_index, new_priority):
		xp = self.xp
		# Confirm we don't increment a node twice
		node_index, unique_index = xp.unique(node_index, return_index=True)
		priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]
		
		for nodes in self.nodes[::-1]:
			if xp is cupy:
				cupyx.scatter_add(nodes, node_index, priority_diff)
			else:
				xp.add.at(nodes, node_index, priority_diff)
			node_index //= 2


# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(object):
	def __init__(
		self,
		env,
		frame_skip=4,
		frame_size=84,
		state_history=4,
		done_on_life_loss=False,
		reward_clipping=True, # Clips to a range of -1,1
		max_episode_timesteps=27000
	):
		self.env = env.env
		self.done_on_life_loss = done_on_life_loss
		self.frame_skip = frame_skip
		self.frame_size = frame_size
		self.reward_clipping = reward_clipping
		self._max_episode_steps = max_episode_timesteps
		self.observation_space = np.zeros((frame_size, frame_size))
		self.action_space = self.env.action_space

		self.lives = 0
		self.episode_length = 0

		# Tracks previous 2 frames
		self.frame_buffer = np.zeros(
			(2,
			self.env.observation_space.shape[0],
			self.env.observation_space.shape[1]),
			dtype=np.uint8
		)
		# Tracks previous 4 states
		self.state_buffer = np.zeros((state_history, frame_size, frame_size), dtype=np.uint8)


	def reset(self):
		self.env.reset()
		self.lives = self.env.ale.lives()
		self.episode_length = 0
		self.env.ale.getScreenGrayscale(self.frame_buffer[0])
		self.frame_buffer[1] = 0

		self.state_buffer[0] = self.adjust_frame()
		self.state_buffer[1:] = 0
		return self.state_buffer


	# Takes single action is repeated for frame_skip frames (usually 4)
	# Reward is accumulated over those frames
	def step(self, action):
		total_reward = 0.
		self.episode_length += 1

		for frame in range(self.frame_skip):
			_, reward, done, _ = self.env.step(action)
			total_reward += reward

			if self.done_on_life_loss:
				crt_lives = self.env.ale.lives()
				done = True if crt_lives < self.lives else done
				self.lives = crt_lives

			if done: 
				break

			# Second last and last frame
			f = frame + 2 - self.frame_skip 
			if f >= 0:
				self.env.ale.getScreenGrayscale(self.frame_buffer[f])

		self.state_buffer[1:] = self.state_buffer[:-1]
		self.state_buffer[0] = self.adjust_frame()

		done_float = float(done)
		if self.episode_length >= self._max_episode_steps:
			done = True

		return self.state_buffer, total_reward, done, [np.clip(total_reward, -1, 1), done_float]


	def adjust_frame(self):
		# Take maximum over last two frames
		np.maximum(
			self.frame_buffer[0],
			self.frame_buffer[1],
			out=self.frame_buffer[0]
		)

		# Resize
		image = cv2.resize(
			self.frame_buffer[0],
			(self.frame_size, self.frame_size),
			interpolation=cv2.INTER_AREA
		)
		return np.array(image, dtype=np.uint8)


	def seed(self, seed):
		self.env.seed(seed)


# Create environment, add wrapper if necessary and create env_properties
def make_env(env_name, atari_preprocessing):
	env = gym.make(env_name)
	
	is_atari = gym.envs.registry.spec(env_name).entry_point == 'gym.envs.atari:AtariEnv'
	env = AtariPreprocessing(env, **atari_preprocessing) if is_atari else env

	state_dim = (
		atari_preprocessing["state_history"], 
		atari_preprocessing["frame_size"], 
		atari_preprocessing["frame_size"]
	) if is_atari else env.observation_space.shape[0]

	return (
		env,
		is_atari,
		state_dim,
		env.action_space.n
	)
