from pathlib import Path

import numpy as np

from coaction import agents, loggers, games, utils


total_episodes = 30
total_stages = int(1e5)
num_parallel_episodes = None


def alpha(t: int) -> float:
    return 1 / (t + 1) ** 0.96


def beta(t: int) -> float:
    return 1 / (t + 1)


def alpha2(t: int) -> float:
    return 1 / (0.92 * t + 1) ** 0.96


def beta2(t: int) -> float:
    return 1 / (0.96 * t + 1)


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

agent_1_kwargs = {  # type: AsynchronousSmoothedFictitiousPlay
    "name": "AsyncSFP2",  # type: <class 'str'>
    "seed": 7,  # type: <class 'int'>
    "alpha": alpha2,  # type: collections.abc.Callable[[int], float]
    "beta": beta2,  # type: collections.abc.Callable[[int], float]
    "gamma": gamma,  # type: <class 'float'>
    "tau": tau2,  # type: <class 'float'>
    "initial_Q": None,  # type: None | int | float | numpy.ndarray
    "initial_pi": None,  # type: None | int | float | numpy.ndarray
    "logged_params": None,  # type: collections.abc.Collection[str]
}

agent_types = [
    agents.AsynchronousSmoothedFictitiousPlay,
    agents.AsynchronousSmoothedFictitiousPlay,
]
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
