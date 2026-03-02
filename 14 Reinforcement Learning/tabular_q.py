from collections import defaultdict
import gymnasium as gym
import numpy as np


class TabularQAgent:
    def __init__(
            self,
            n_actions,
            learning_rate,
            discount,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            n_states=None
    ):
        if n_states is not None:
            self._q = np.zeros((n_states, n_actions))
            self._get_q = lambda obs: self._q[obs]
            self._set_q = lambda obs, a, v: self._q.__setitem__((obs, a), v)
        else:
            _q = defaultdict(lambda: np.zeros(n_actions))
            self._get_q = lambda obs: _q[obs]
            self._set_q = lambda obs, a, v: _q[obs].__setitem__(a, v)

        self.n_actions = n_actions
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_error = []

    def get_action(self, obs, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(obs)))

    def update(
            self,
            obs,
            action,
            reward,
            terminated,
            next_obs,
    ):
        future_q = (not terminated) * np.max(self._get_q(next_obs))
        error = reward + self.discount * future_q - self._get_q(obs)[action]

        self._set_q(obs, action, self._get_q(obs)[action] + self.lr * error)
        self.training_error.append(error)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)


# Experiment
if __name__ == "__main__":
    bj_env = gym.make("Blackjack-v1", sab=False)
    bj_env = gym.wrappers.RecordEpisodeStatistics(bj_env, buffer_length=100_000)
    fl_env = gym.make("FrozenLake-v1", is_slippery=True)
    fl_env = gym.wrappers.RecordEpisodeStatistics(fl_env)

    bj_agent = TabularQAgent(
        n_actions=bj_env.action_space.n,
        learning_rate=0.1,
        discount=1.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.00002
    )

    fl_agent = TabularQAgent(
        n_actions=fl_env.action_space.n,
        learning_rate=0.1,
        discount=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.001,
        n_states=fl_env.observation_space.n
    )


    # Train agent
    def train_agent(agent, env, n_episodes):
        from tqdm import tqdm

        for _ in tqdm(range(n_episodes)):
            obs, _ = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()


    # Test the trained agent
    def test_agent(agent, env, n_episodes=1000):
        rewards = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.get_action(obs, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

        rewards = np.array(rewards)
        print(f"Win rate: {np.mean(rewards > 0):.1%} | Avg reward: {np.mean(rewards):.3f}")


    # Test agent
    train_agent(bj_agent, bj_env, 10_000)
    test_agent(bj_agent, bj_env)
    train_agent(fl_agent, fl_env, 10_000)
    test_agent(fl_agent, fl_env)