"""Plot the convergence of the value functions."""

from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from coaction.loggers.loader import LogLoader
from coaction.utils.games import shapley_value
from coaction.utils.paths import ProjectPaths


PROJECT_PATH = Path(__file__).parent
PATHS = ProjectPaths(PROJECT_PATH)
LOG_LOADER = LogLoader(PROJECT_PATH)
RUN = 5
LOG_EACH = 1
DOWNSAMPLE = 1
FONT_SIZE = 20


def agent_name_to_legend(agent_name: str) -> str:
    """Convert agent name to legend name."""
    if "fp" in agent_name.lower():
        return "Full"
    elif "indq" in agent_name.lower():
        return "None"
    elif "temporal" in agent_name.lower():
        return "Temporal"
    else:
        raise ValueError("Invalid agent name")


def calculate_boundary(d, gamma, tau1, tau2, n_actions1, n_actions2):
    """Calculate the boundary around the shapley values according to the theorem."""
    nom = 2 * d + 2 * gamma - 3 * gamma * d
    den = (1 - gamma) * (d - 2 * gamma)
    per = tau1 * np.log(n_actions1) + tau2 * np.log(n_actions2)
    return nom / den * per


def get_shapley_value(experiment_name: str) -> np.ndarray:
    """Load the shapley values from the experiment.

    Returns:
        np.ndarray: Array of shapley values.
    """
    config = LOG_LOADER.load_experiment_config(RUN, experiment_name)
    return shapley_value(config.T, config.R[0], config.gamma)


def get_agent_names(experiment_name: str):
    """Get agent names from the experiment name."""
    exp_split = experiment_name.split("_")
    agent_1 = exp_split[0].title() + exp_split[1].upper()
    agent_2 = exp_split[3].title() + exp_split[4].upper()
    if agent_1 == agent_2:
        agent_2 += "2"
    if "FP" not in agent_1 and "IndQ" not in agent_1:
        agent_1 = "TemporalAccess"
    if "FP" not in agent_2 and "IndQ" not in agent_2:
        agent_2 = "TemporalAccess"
    return agent_1, agent_2


def get_experiment_names():
    """Get experiment names."""
    exp_dirs = PATHS.with_run(RUN).get_experiment_log_dirs()
    return [
        exp_dir.name
        for exp_dir in exp_dirs
        if exp_dir.is_dir() and exp_dir.name != "configs"
    ]


def plot_svs(experiment_name: str):
    """Plot the convergence of the value functions."""
    curr_dir = PATHS.get_project_dir()
    plots_dir = curr_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir = plots_dir / f"run_{RUN}"
    run_plots_dir.mkdir(parents=True, exist_ok=True)

    # load experiment config
    config = LOG_LOADER.load_experiment_config(RUN, experiment_name)

    # load shapley values
    sv1 = get_shapley_value(experiment_name)

    # create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel("Stages ($k$)", labelpad=4, fontsize=FONT_SIZE)
    ax.set_ylabel("$v_k^i$", labelpad=0.4, fontsize=FONT_SIZE)

    ax.semilogx()
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(
        lambda x, pos: f'$10^{"{" + str(int(np.log10(x) + np.log10(DOWNSAMPLE))) + "}"}$'
    )
    ax.tick_params(axis="both", which="major", labelsize=16)

    n_actions1, n_actions2 = config.R.shape[-2:]
    boundary = calculate_boundary(
        d=0.92 if "d92" in experiment_name else 1.0,
        gamma=config.gamma,
        tau1=config.tau1,
        tau2=config.tau2,
        n_actions1=n_actions1,
        n_actions2=n_actions2,
    )

    # Load initial v's
    x = np.arange(1, 1 + 1 + config.total_stages // DOWNSAMPLE, LOG_EACH)
    ylim_min = 0
    ylim_max = 0
    agent_names = get_agent_names(experiment_name)
    for agent_idx, (agent_name, color) in enumerate(zip(agent_names, ["blue", "red"])):
        v1_runs = LOG_LOADER.load_experiment_logs(
            RUN, experiment_name, agent_name, "v"
        )[:, ::DOWNSAMPLE, :]
        shape = (v1_runs.shape[0], 1, v1_runs.shape[2])
        v1_runs = np.concatenate([np.zeros(shape), v1_runs], axis=1)
        v1_mean = np.mean(v1_runs, axis=0)
        v1_std = np.std(v1_runs, axis=0)

        ylim_min = min(ylim_min, np.min(v1_mean - v1_std))
        ylim_max = max(ylim_max, np.max(v1_mean + v1_std))

        x_arr = 1000
        for s, (v_mean, v_std, col, linestyle) in enumerate(
            zip(v1_mean.T, v1_std.T, [color, f"dark{color}"], ["solid", "dashed"])
        ):
            ax.plot(
                x,
                v_mean,
                color=col,
                label=agent_name,
                linestyle=linestyle,
            )
            ax.fill_between(
                x,
                v_mean - v_std,
                v_mean + v_std,
                color=col,
                alpha=0.2,
            )
            sgn = 1 if agent_idx == 0 else -1
            ax.axhline(
                y=sgn * sv1[s] - boundary,
                color=col,
                linestyle="dotted",
            )
            ax.axhline(
                y=sgn * sv1[s] + boundary,
                color=col,
                linestyle="dotted",
            )
            plt.annotate(
                rf"{agent_name_to_legend(agent_name)} - $v_k^{agent_idx + 1}(s^{s + 1})$",
                xy=(
                    x_arr,
                    v_mean[x_arr] - sgn * v_std[x_arr],
                ),
                xytext=(
                    x_arr - 400,
                    v_mean[x_arr] - 0.28 * sgn * config.R.max(),
                ),
                arrowprops={"arrowstyle": "->"},
                fontsize=FONT_SIZE,
            )

    # Set y_lim
    ax.set_xlim(1, config.total_stages // DOWNSAMPLE)
    ax.set_ylim(-1.05, 1.05)

    # Save figure
    ax.set_rasterized(True)

    save_dir = run_plots_dir / "svs"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        save_dir / f"conv_{experiment_name}.png",
        pad_inches=0.08,
        bbox_inches="tight",
    )
    fig.savefig(
        save_dir / f"conv_{experiment_name}.pdf", pad_inches=0.08, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    """Main function."""
    exp_names = get_experiment_names()
    for exp_name in tqdm(exp_names):
        plot_svs(exp_name)


if __name__ == "__main__":
    main()
