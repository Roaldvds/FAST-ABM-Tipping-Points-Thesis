"""
Generate model-validation figures for the experiments.

The script expects to live in the thesis/ folder and assumes the DelftBlue
results are stored in:

    <project root>/results/delftblue2/

Figures are saved to:

    <project root>/results/plots/validation/

The current configuration generates:
1. Ensemble trajectory plots for the selected validation outcomes.
2. Low/high uncertainty highlighting plots for every selected outcome and
   every uncertain parameter.
3. Time-dependent sensitivity plots for the selected validation outcomes.

All stochastic replications are averaged by uncertainty scenario for these
descriptive validation figures.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="ipyparallel not installed*",
    category=UserWarning,
)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ema_workbench.util import load_results
from ema_workbench.analysis import feature_scoring, RuleInductionType


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue2"
PLOT_DIR = PROJECT_ROOT / "thesis" / "results" / "plots" / "validation_model"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Set to True if you also want every figure to open while the script runs.
SHOW_PLOTS = False

# Save a high-resolution raster version and a vector version for LaTeX.
SAVE_FORMATS = ("png",)
PNG_DPI = 300

FLOOD_TIMESTEP = 10
HIGHLIGHT_QUANTILE = 0.2

# Thesis figure colours.
ENSEMBLE_COLOR = "#B8C2CA"
LOW_COLOR = "#2389C9"      # brighter blue
HIGH_COLOR = "#F2557D"     # brighter pink
FLOOD_COLOR = "0.35"

UNCERTAINTY_COLS = [
    "initial_adaptation",
    "per_HH_adapted_to_SN_prob_midpoint",
    "weight_SN_vs_PMT",
    "savings_rate_multiplier",
    "adaptation_cost_multiplier",
    "measure_lifetime",
    "flood_depth_multiplier",
]

PRETTY_PARAMETER_NAMES = {
    "initial_adaptation": "Initial adaptation",
    "per_HH_adapted_to_SN_prob_midpoint": "Social norm midpoint",
    "weight_SN_vs_PMT": "Weight social norm vs. PMT",
    "savings_rate_multiplier": "Savings rate multiplier",
    "adaptation_cost_multiplier": "Adaptation cost multiplier",
    "measure_lifetime": "Measure lifetime",
    "flood_depth_multiplier": "Flood-depth multiplier",
}

# Current shortlist for Section 4.1.1 Model Validation.
VALIDATION_OUTCOMES = {
    "adaptation_rate": "Adaptation rate",
    "gini_relative_burden": "Relative burden (Gini)",
    "adaptation_gap_high_low": "Adaptation gap high-low",
    "share_liquidity_constrained": "Share liquidity constrained",
    "share_adaptation_deficit": "Share with adaptation deficit",
}


# =============================================================================
# Loading and aggregation
# =============================================================================

def load_delftblue_results(results_dir):
    """Load and concatenate all seed-level EMA Workbench result files."""
    result_files = sorted(results_dir.glob("ema_fast_results_seed_*.tar.gz"))

    if not result_files:
        raise FileNotFoundError(
            f"No DelftBlue result files found in:\n{results_dir}"
        )

    print(f"Found {len(result_files)} result files in {results_dir}")

    all_exp = []
    all_outcomes = {}

    for file in result_files:
        exp, out = load_results(file)
        all_exp.append(exp)

        for name, data in out.items():
            all_outcomes.setdefault(name, []).append(data)

    experiments = pd.concat(all_exp, ignore_index=True)

    for key in all_outcomes:
        all_outcomes[key] = np.vstack(all_outcomes[key])

    return experiments, all_outcomes


def add_scenario_ids(experiments):
    """Assign one ID to each unique combination of uncertain parameters."""
    experiments = experiments.copy()

    missing = [c for c in UNCERTAINTY_COLS if c not in experiments.columns]
    if missing:
        raise KeyError(f"Missing uncertainty columns: {missing}")

    experiments["scenario_id"] = (
        experiments.groupby(UNCERTAINTY_COLS, sort=False).ngroup()
    )

    return experiments


def aggregate_outcome_by_scenario(experiments, outcome):
    """
    Average stochastic replications belonging to the same uncertainty scenario.

    Returns
    -------
    scenario_experiments : DataFrame
        One row per uncertainty scenario, sorted by scenario_id.
    aggregated : ndarray
        Mean trajectory per uncertainty scenario.
    """
    scenario_ids = experiments["scenario_id"].to_numpy()
    unique_ids = np.sort(experiments["scenario_id"].unique())

    aggregated = np.vstack([
        outcome[scenario_ids == scenario_id].mean(axis=0)
        for scenario_id in unique_ids
    ])

    scenario_experiments = (
        experiments
        .drop_duplicates("scenario_id")
        .sort_values("scenario_id")
        .reset_index(drop=True)
    )

    return scenario_experiments, aggregated


def prepare_validation_outcomes(experiments, all_outcomes):
    """Aggregate the selected validation outcomes over stochastic replications."""
    missing = [
        outcome
        for outcome in VALIDATION_OUTCOMES
        if outcome not in all_outcomes
    ]
    if missing:
        raise KeyError(
            "The following validation outcomes are missing from the results: "
            f"{missing}"
        )

    mean_outcomes = {}
    scenario_experiments = None

    for outcome_key in VALIDATION_OUTCOMES:
        scenario_experiments, mean_outcomes[outcome_key] = (
            aggregate_outcome_by_scenario(
                experiments,
                all_outcomes[outcome_key],
            )
        )

    return scenario_experiments, mean_outcomes


# =============================================================================
# Figure utilities
# =============================================================================

def save_figure(fig, filename_stem):
    """Save a figure in every configured output format."""
    for fmt in SAVE_FORMATS:
        path = PLOT_DIR / f"{filename_stem}.{fmt}"

        if fmt.lower() == "png":
            fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def add_flood_line(ax):
    """Add the common flood-event marker."""
    ax.axvline(
        FLOOD_TIMESTEP,
        color=FLOOD_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="Flood event",
    )


# =============================================================================
# 1. Ensemble trajectory plots
# =============================================================================

def plot_ensemble_trajectories(mean_outcomes):
    """Plot all scenario-mean trajectories for each validation outcome."""
    for outcome_key, data in mean_outcomes.items():
        label = VALIDATION_OUTCOMES[outcome_key]
        timesteps = np.arange(1, data.shape[1] + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        for trajectory in data:
            ax.plot(
                timesteps,
                trajectory,
                color=ENSEMBLE_COLOR,
                alpha=0.15,
                linewidth=0.7,
            )

        add_flood_line(ax)

        ax.set_xlabel("Timestep")
        ax.set_ylabel(label)
        ax.set_title(f"{label} trajectories across all scenarios")
        ax.set_xlim(0, data.shape[1])
        ax.legend(frameon=False, loc="upper right")

        fig.tight_layout()

        save_figure(
            fig,
            f"trajectory_{outcome_key}",
        )


# =============================================================================
# 2. Parameter-highlighted trajectory plots
# =============================================================================

def plot_parameter_highlights(scenario_experiments, mean_outcomes):
    """
    Highlight the lowest and highest parameter quartiles against the full
    scenario ensemble.

    This currently generates every validation outcome x uncertainty combination
    so the most informative pairs can be selected later for the thesis.
    """
    for parameter in UNCERTAINTY_COLS:
        values = scenario_experiments[parameter]

        q_low = values.quantile(HIGHLIGHT_QUANTILE)
        q_high = values.quantile(1 - HIGHLIGHT_QUANTILE)

        low_mask = (values <= q_low).to_numpy()
        high_mask = (values >= q_high).to_numpy()

        parameter_label = PRETTY_PARAMETER_NAMES[parameter]

        print(
            f"{parameter_label}: "
            f"low <= {q_low:.4g}, high >= {q_high:.4g} "
            f"({low_mask.sum()} low / {high_mask.sum()} high scenarios)"
        )

        for outcome_key, data in mean_outcomes.items():
            label = VALIDATION_OUTCOMES[outcome_key]
            timesteps = np.arange(1, data.shape[1] + 1)

            if len(data) != len(scenario_experiments):
                raise ValueError(
                    f"Scenario mismatch for {outcome_key}: "
                    f"{len(data)} trajectories versus "
                    f"{len(scenario_experiments)} scenario rows."
                )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Full ensemble in the background.
            for trajectory in data:
                ax.plot(
                    timesteps,
                    trajectory,
                    color=ENSEMBLE_COLOR,
                    alpha=0.02,
                    linewidth=0.5,
                )

            # Lowest parameter quartile.
            for trajectory in data[low_mask]:
                ax.plot(
                    timesteps,
                    trajectory,
                    color=LOW_COLOR,
                    alpha=0.35,
                    linewidth=0.8,
                )

            # Highest parameter quartile.
            for trajectory in data[high_mask]:
                ax.plot(
                    timesteps,
                    trajectory,
                    color=HIGH_COLOR,
                    alpha=0.35,
                    linewidth=0.8,
                )

            legend_elements = [
                Line2D(
                    [0], [0],
                    color=LOW_COLOR,
                    linewidth=1.5,
                    label=(
                        f"Lowest {int(HIGHLIGHT_QUANTILE * 100)}% "
                        f"{parameter_label.lower()}"
                    ),
                ),
                Line2D(
                    [0], [0],
                    color=HIGH_COLOR,
                    linewidth=1.5,
                    label=(
                        f"Highest {int(HIGHLIGHT_QUANTILE * 100)}% "
                        f"{parameter_label.lower()}"
                    ),
                ),
                Line2D(
                    [0], [0],
                    color=FLOOD_COLOR,
                    linestyle="--",
                    linewidth=1.0,
                    label="Flood event",
                ),
            ]

            add_flood_line(ax)

            ax.set_xlabel("Timestep")
            ax.set_ylabel(label)
            ax.set_title(f"{label} trajectories by {parameter_label.lower()}")
            ax.set_xlim(0, data.shape[1])
            ax.legend(
                handles=legend_elements,
                frameon=False,
                loc="upper right",
            )

            fig.tight_layout()

            save_figure(
                fig,
                f"highlight_{outcome_key}__{parameter}",
            )


# =============================================================================
# 3. Time-dependent sensitivity plots
# =============================================================================

def calculate_time_dependent_sensitivity(
    scenario_experiments,
    mean_outcomes,
):
    """Calculate Extra Trees feature importance at each model timestep."""
    X = scenario_experiments[UNCERTAINTY_COLS]
    sensitivity_results = {}

    for outcome_key, outcome_mean in mean_outcomes.items():
        print(f"Calculating sensitivity: {VALIDATION_OUTCOMES[outcome_key]}")

        scores_over_time = []

        for t in range(outcome_mean.shape[1]):
            y = outcome_mean[:, t]

            # Some derived outcomes can occasionally be constant at a timestep.
            # In that case no uncertainty can explain variance because none exists.
            if np.allclose(y, y[0], equal_nan=True):
                scores_over_time.append(
                    np.zeros(len(UNCERTAINTY_COLS))
                )
                continue

            scores, _ = feature_scoring.get_ex_feature_scores(
                X,
                y,
                mode=RuleInductionType.REGRESSION,
            )

            if set(UNCERTAINTY_COLS).issubset(scores.index):
                timestep_scores = (
                    scores
                    .loc[UNCERTAINTY_COLS]
                    .iloc[:, 0]
                )

            elif set(UNCERTAINTY_COLS).issubset(scores.columns):
                timestep_scores = (
                    scores
                    .loc[:, UNCERTAINTY_COLS]
                    .iloc[0]
                    .reindex(UNCERTAINTY_COLS)
                )

            else:
                raise ValueError(
                    f"Uncertainty names not found for {outcome_key}.\n"
                    f"Index: {scores.index.tolist()}\n"
                    f"Columns: {scores.columns.tolist()}"
                )

            scores_over_time.append(timestep_scores.to_numpy())

        # Timesteps are 1...N because the first collected observation follows
        # the first model step.
        sensitivity_results[outcome_key] = pd.DataFrame(
            scores_over_time,
            columns=UNCERTAINTY_COLS,
            index=np.arange(1, outcome_mean.shape[1] + 1),
        )
        sensitivity_results[outcome_key].index.name = "Timestep"

    return sensitivity_results


def plot_sensitivity_results(sensitivity_results):
    """Save one stacked time-dependent sensitivity plot per outcome."""
    for outcome_key, sensitivity in sensitivity_results.items():
        plot_data = sensitivity.rename(
            columns=PRETTY_PARAMETER_NAMES
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        plot_data.plot.area(
            stacked=True,
            ax=ax,
        )

        add_flood_line(ax)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Feature importance")
        ax.set_title(
            "Time-dependent sensitivity of "
            f"{VALIDATION_OUTCOMES[outcome_key].lower()}"
        )
        ax.set_xlim(1, sensitivity.index.max())
        ax.set_ylim(0, 1)

        # Keep the parameter legend below the plot.
        handles, labels = ax.get_legend_handles_labels()

        # The axvline label is also included; put it last.
        ax.legend(
            handles=handles,
            labels=labels,
            title="Uncertainty",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            frameon=False,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"sensitivity_{outcome_key}",
        )


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading DelftBlue results...")
    experiments, all_outcomes = load_delftblue_results(RESULTS_DIR)

    experiments = add_scenario_ids(experiments)

    scenario_experiments, mean_outcomes = prepare_validation_outcomes(
        experiments,
        all_outcomes,
    )

    print(
        f"Loaded {len(experiments):,} model realisations, "
        f"representing {len(scenario_experiments):,} uncertainty scenarios."
    )

    print(f"Saving figures to:\n{PLOT_DIR}")

    print("\n1/3 Plotting ensemble trajectories...")
    plot_ensemble_trajectories(mean_outcomes)

    print("\n2/3 Plotting parameter-highlighted trajectories...")
    plot_parameter_highlights(
        scenario_experiments,
        mean_outcomes,
    )

    print("\n3/3 Calculating and plotting sensitivity...")
    sensitivity_results = calculate_time_dependent_sensitivity(
        scenario_experiments,
        mean_outcomes,
    )
    plot_sensitivity_results(sensitivity_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
