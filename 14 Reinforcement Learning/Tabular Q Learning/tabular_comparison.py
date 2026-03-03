import numpy as np
from collections import defaultdict


class TabularAgent():
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
        super().__init__()
        if n_states is not None:
            self._q = np.zeros((n_states, n_actions))
        else:
            _q = defaultdict(lambda: np.zeros(n_actions))

        self.n_actions = n_actions
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_error = []

    def _get_q(self, obs):
        return self._q[obs]

    def _set_q(self, obs, a, v):
        self._q[obs][a] = v

    def get_action(self, obs, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(obs)))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)


class QAgent(TabularAgent):
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


class SarsaAgent(TabularAgent):
    def update(
            self,
            obs,
            action,
            reward,
            terminated,
            next_obs,
            next_action,
    ):
        future_q = (not terminated) * self._get_q(next_obs)[next_action]
        error = reward + self.discount * future_q - self._get_q(obs)[action]

        self._set_q(obs, action, self._get_q(obs)[action] + self.lr * error)
        self.training_error.append(error)


def train_q(agent, env, n_episodes):
    from tqdm import tqdm

    for _ in tqdm(range(n_episodes), desc="Q-learning"):
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()


def train_sarsa(agent, env, n_episodes):
    from tqdm import tqdm

    for _ in tqdm(range(n_episodes), desc="SARSA   "):
        obs, _ = env.reset()
        action = agent.get_action(obs)
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.get_action(next_obs)
            agent.update(obs, action, reward, terminated, next_obs, next_action)
            done = terminated or truncated
            obs, action = next_obs, next_action

        agent.decay_epsilon()


def test_agent(agent, env, n_episodes=1000):
    rewards, reached_goal = [], []
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
        reached_goal.append(terminated)  # truncated = timeout, not goal

    rewards = np.array(rewards)
    goal_rate = np.mean(reached_goal)
    return goal_rate, np.mean(rewards)


# Experiment
if __name__ == "__main__":
    import gymnasium as gym

    HYPERPARAMS = dict(
        learning_rate=0.5,
        discount=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1e-4,
    )
    N_TRAIN = 10_000
    N_TEST  = 1_000

    make_env = lambda: gym.make("CliffWalking-v1", max_episode_steps=200)

    print("Environment: CliffWalking-v1")

    for AgentClass, train_fn, label in [
        (QAgent,     train_q,     "Q-learning"),
        (SarsaAgent, train_sarsa, "SARSA"),
    ]:
        agent = AgentClass(n_actions=4, n_states=48, **HYPERPARAMS)

        train_env = gym.wrappers.RecordEpisodeStatistics(
            make_env(), buffer_length=N_TRAIN
        )
        train_fn(agent, train_env, N_TRAIN)
        train_env.close()

        test_env = make_env()
        goal_rate, avg_reward = test_agent(agent, test_env, N_TEST)
        test_env.close()

        print(f"[{label}] Goal rate: {goal_rate:.1%} | Avg reward: {avg_reward:.2f} "
              f"| Final ε: {agent.epsilon:.4f}")
