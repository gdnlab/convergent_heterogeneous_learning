from pathlib import Path

import numpy as np

from coaction import agents, loggers, games, utils
from coaction.utils.math import softmax


class AlgorithmFamily(agents.Agent):
    def __init__(
        self,
        name: str,
        seed: int,
        reward_matrix,
        transition_matrix,
        alpha,
        beta,
        gamma,
        tau,
        **kwargs,
    ):
        super().__init__(name, seed, **kwargs)
        self._R = reward_matrix  # pylint: disable=invalid-name
        self._T = transition_matrix  # pylint: disable=invalid-name

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._tau = tau

        self._n_states = reward_matrix.shape[0]
        self._n_actions = reward_matrix.shape[1]
        self._n_opponent_actions = reward_matrix.shape[2]

        self._counts = np.zeros(self._n_states, dtype=np.int_)
        self._q = np.zeros(
            (self._n_states, self._n_actions), dtype=np.float_
        )  # pylint: disable=invalid-name
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._mu = None

    @property
    def can_observe(self):
        return self.rng.integers(2)

    def reset(self):
        """Reset the agent's parameters."""
        super().reset()
        self._counts = np.zeros(self._n_states, dtype=np.int_)
        self._q = np.zeros((self._n_states, self._n_actions), dtype=np.float_)
        self.v = np.zeros(self._n_states, dtype=np.float_)

    def act(self, state):
        """Return the action to take given the current state."""
        self._mu = softmax(self._q[state], self._tau)
        return self.rng.choice(self._n_actions, p=self._mu)

    def update(
        self,
        state,
        actions,
        reward,
        next_state,
        **kwargs,
    ):
        """Update the agent's parameters."""
        if self.can_observe:
            opponent_action = actions[1]
            q_hat = (
                self._R[state, :, opponent_action]
                + self._gamma * self._T[state, :, opponent_action, :] @ self.v
            )
            alpha = self._alpha(self._counts[state])
        else:
            action = actions[0]
            q_hat = reward + self._gamma * self.v[next_state]
            alpha = min(self._alpha(self._counts[state]) / self._mu[action], 1)
        self._q[state] += alpha * (q_hat - self._q[state])
        self.v[state] += self._beta(self._counts[state]) * (
            np.dot(self._mu, self._q[state]) - self.v[state]
        )
        self._counts[state] += 1


total_episodes = 30
total_stages = int(1e5)
num_parallel_episodes = None


def alpha(t: int) -> float:
    return 1 / (t + 1) ** 0.96


def beta(t: int) -> float:
    return 1 / (t + 1)


R, T = utils.games.generate_zsmg(
    n_states=2, n_actions=[2, 2], min_reward=0, max_reward=1, seed=42
)
R = R * np.expand_dims(np.array([1, 0.2]), axis=(0, 2, 3))


gamma = 0.3
tau1 = 0.002
tau2 = 0.002


# Agent kwargs
agent_0_kwargs = {  # type: AsynchronousSmoothedFictitiousPlay
    "name": "AsyncSFP",  # type: <class 'str'>
    "seed": 29,  # type: <class 'int'>
    "alpha": alpha,  # type: collections.abc.Callable[[int], float]
    "beta": beta,  # type: collections.abc.Callable[[int], float]
    "gamma": gamma,  # type: <class 'float'>
    "tau": tau1,  # type: <class 'float'>
    "initial_Q": None,  # type: None | int | float | numpy.ndarray
    "initial_pi": None,  # type: None | int | float | numpy.ndarray
    "logged_params": None,  # type: collections.abc.Collection[str]
}

agent_1_kwargs = {  # type: IndividualQLearning
    "name": "TemporalAccess",  # type: <class 'str'>
    "seed": 7,  # type: <class 'int'>
    "alpha": alpha,  # type: collections.abc.Callable[[int], float]
    "beta": beta,  # type: collections.abc.Callable[[int], float]
    "gamma": gamma,  # type: <class 'float'>
    "tau": tau2,  # type: <class 'float'>
}

agent_types = [agents.AsynchronousSmoothedFictitiousPlay, AlgorithmFamily]
agent_kwargs = [agent_0_kwargs, agent_1_kwargs]

# Game kwargs
game_type = games.MarkovGame
game_kwargs = {  # type: MarkovGame
    "name": "game",  # type: str
    "reward_matrix": R,  # type: npt.NDArray
    "transition_matrix": T,  # type: npt.NDArray[np.float_]
    "seed": 42,  # type: int
}

# Logger kwargs
agent_logger_kwargs = {  # type: AgentLogger
    "log_each": 1,  # type: <class 'int'>
    "save_in_chunks": 100_000,  # type: <class 'bool'>
}

game_logger_kwargs = {  # type: GameLogger
    "log_each": 1,  # type: <class 'int'>
    "save_state_history": False,  # type: <class 'bool'>
    "save_action_history": False,  # type: <class 'bool'>
    "save_reward_history": True,  # type: <class 'bool'>
    "save_in_chunks": 100_000,  # type: <class 'bool'>
}

progress_logger_kwargs = {  # type: ProgressLogger
    "log_each": 10_000,  # type: <class 'int'>
}
