import gymnasium as gym
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, size):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                'agent': gym.spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
                'target': gym.spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location}

    def _get_info(self):
        return {
            'distance': np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location

        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=np.int32
           )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminal = np.array_equal(self._agent_location, self._target_location)
        truncated = False

        reward = 1 if terminal else -1 / (self.size * 2)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminal, truncated, info

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=f"{GridWorldEnv.__module__}:{GridWorldEnv.__qualname__}",
    max_episode_steps=300,
)


if __name__ == "__main__":
    import gymnasium as gym
    from tabular_q import TabularQAgent

    SIZE = 10

    # Encode observation to match with TabularQAgent
    def encode_obs(obs, size=SIZE):
        ax, ay = obs['agent']
        tx, ty = obs['target']
        # mixed-radix encoding
        return int(ax * size ** 3 + ay * size ** 2 + tx * size + ty)

    env = gym.make("gymnasium_env/GridWorld-v0", size=SIZE)
    agent = TabularQAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount=0.95,
        epsilon_start=1,
        epsilon_end=0.05,
        epsilon_decay=0.0005,
        n_states=SIZE ** 4
    )

    def train_agent(agent, env, n_episodes):
        from tqdm import tqdm

        for _ in tqdm(range(n_episodes)):
            obs, _ = env.reset()
            done = False

            while not done:
                state = encode_obs(obs)
                action = agent.get_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, terminated, encode_obs(next_obs))
                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()


    def test_agent(agent, env, n_episodes=1000):
        rewards = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.get_action(encode_obs(obs), explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

        rewards = np.array(rewards)
        print(f"Win rate: {np.mean(rewards > 0):.1%} | Avg reward: {np.mean(rewards):.3f}")

    train_agent(agent, env, n_episodes=100_000)
    test_agent(agent, env)
