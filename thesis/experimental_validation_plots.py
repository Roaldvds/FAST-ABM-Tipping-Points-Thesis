"""
Generate experimental-validation figures for the experiments.

This script evaluates two parts of the experimental design:

1. Replication convergence:
   whether the stochastic mean and variance stabilise as more seeds are added.
2. Scenario-sample convergence:
   whether descriptive output statistics and time-dependent sensitivity
   estimates stabilise as more uncertainty scenarios are included.

The script searches upward for a project root containing:

    results/delftblue2/

Figures and summary tables are saved to:

    thesis/results/plots/validation_experiments/

The current shortlist contains:
- representative cumulative-mean convergence plots;
- representative cumulative-variance convergence plots;
- global replication NRMSE plots;
- incremental replication-stability plots;
- final-outcome confidence-interval convergence plots;
- scenario-sample descriptive convergence plots;
- sensitivity-convergence plots for each validation outcome;
- one combined sensitivity-convergence plot;
- CSV summary tables for the principal diagnostics.

The script retains more diagnostics than are likely to appear in
the final thesis.
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

from ema_workbench.util import load_results
from ema_workbench.analysis import feature_scoring, RuleInductionType


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root():
    """Find a parent directory containing results/delftblue2."""
    candidates = [SCRIPT_DIR, Path.cwd().resolve()]

    seen = set()
    for start in candidates:
        for candidate in [start, *start.parents]:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)

            if (candidate / "results" / "delftblue2").exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected to find "
        "'results/delftblue2/' in the current directory or one of its parents."
    )


PROJECT_ROOT = find_project_root()

RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue2"
PLOT_DIR = (
    PROJECT_ROOT
    / "thesis"
    / "results"
    / "plots"
    / "validation_experiments"
)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = False
SAVE_FORMATS = ("png",)
PNG_DPI = 300

FLOOD_TIMESTEP = 10

# Number of representative scenarios shown in the replication convergence plots.
N_REPRESENTATIVE_SCENARIOS = 6

# Scenario-space convergence settings.
N_SUBSETS = 10
RANDOM_SEED = 12345
BASE_SAMPLE_SIZES = [100, 200, 400, 600, 800, 1000, 1200, 1500]

# Sensitivity is intentionally evaluated only at selected timesteps.
SENSITIVITY_TIMESTEPS = [1, 5, 10, 11, 15, 20, 30, 40, 50]

UNCERTAINTY_COLS = [
    "initial_adaptation",
    "per_HH_adapted_to_SN_prob_midpoint",
    "weight_SN_vs_PMT",
    "savings_rate_multiplier",
    "adaptation_cost_multiplier",
    "measure_lifetime",
    "flood_depth_multiplier",
]

OUTCOME_LABELS = {
    "adaptation_rate": "Adaptation rate",
    "adaptation_baseline_adjusted": "Baseline-adjusted adaptation",
    "adaptation_progress": "Adaptation progress",
    "gini_relative_burden": "Relative burden (Gini)",
    "gini_relative_burden_cumulative_income": (
        "Cumulative relative burden (Gini)"
    ),
    "average_relative_burden_cumulative_income": (
        "Average cumulative relative burden"
    ),
    "unmet_adaptation_demand": "Unmet adaptation demand",
    "adaptation_gap_high_low": "Adaptation gap high-low",
    "share_liquidity_constrained": "Share liquidity constrained",
    "average_worry": "Average worry",
    "average_savings": "Average savings",
    "average_perceived_risk": "Average perceived risk",
    "average_flood_damage": "Average potential flood damage",
    "share_adaptation_deficit": "Share with adaptation deficit",
    "share_optimal_to_adapt": "Economically optimal share",
    "share_over_adapted": "Share over-adapted",
}

# Same core outcome set used in the model-validation section.
VALIDATION_OUTCOMES = [
    "adaptation_rate",
    "gini_relative_burden",
    "share_liquidity_constrained",
    "adaptation_gap_high_low",
    "share_adaptation_deficit",
]


# =============================================================================
# File and figure utilities
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


def save_table(df, filename):
    """Save a diagnostic table as CSV."""
    df.to_csv(PLOT_DIR / filename, index=False)


def add_flood_line(ax):
    """Add the common flood-event marker."""
    ax.axvline(
        FLOOD_TIMESTEP,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="Flood event",
    )


def pretty_outcome(outcome):
    return OUTCOME_LABELS.get(outcome, outcome)


def safe_scale(array):
    """Return a non-zero scale for normalization."""
    array = np.asarray(array)

    scale = np.nanmax(array) - np.nanmin(array)

    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(array)

    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    return scale


def normalized_rmse(current, reference):
    """RMSE normalized by the range of the reference curve."""
    return (
        np.sqrt(np.nanmean((current - reference) ** 2))
        / safe_scale(reference)
    )


# =============================================================================
# Loading and stacking replications
# =============================================================================

def load_replications(results_dir):
    """Load one EMA Workbench result file per stochastic replication."""
    result_files = sorted(results_dir.glob("ema_fast_results_seed_*.tar.gz"))

    if not result_files:
        raise FileNotFoundError(
            f"No DelftBlue result files found in:\n{results_dir}"
        )

    print(f"Found {len(result_files)} result files in {results_dir}")

    replications = []

    for path in result_files:
        experiments, outcomes = load_results(path)

        if "seed" not in experiments.columns:
            raise ValueError(f"No seed column found in {path.name}")

        seeds_in_file = experiments["seed"].unique()

        if len(seeds_in_file) != 1:
            raise ValueError(
                f"Expected one seed in {path.name}, found {seeds_in_file}"
            )

        replications.append(
            {
                "seed": int(seeds_in_file[0]),
                "experiments": experiments.reset_index(drop=True),
                "outcomes": outcomes,
            }
        )

    replications = sorted(replications, key=lambda item: item["seed"])

    print(
        "Loaded seeds:",
        [item["seed"] for item in replications],
    )

    return replications


def verify_and_prepare_scenarios(replications):
    """
    Verify that every seed contains the same uncertainty scenarios in the
    same order, then return one scenario-level experiments DataFrame.
    """
    reference = (
        replications[0]["experiments"][UNCERTAINTY_COLS]
        .reset_index(drop=True)
    )

    for replication in replications[1:]:
        current = (
            replication["experiments"][UNCERTAINTY_COLS]
            .reset_index(drop=True)
        )

        same = np.allclose(
            reference.to_numpy(dtype=float),
            current.to_numpy(dtype=float),
            equal_nan=True,
        )

        if not same:
            raise ValueError(
                "Scenario order differs between seeds. "
                "Align scenarios before continuing."
            )

    scenario_experiments = reference.copy()
    scenario_experiments["scenario_id"] = np.arange(len(reference))

    return scenario_experiments


def stack_time_series_outcomes(replications):
    """
    Stack common 2D outcomes into:
        replication × scenario × timestep
    """
    common_outcomes = set(replications[0]["outcomes"])

    for replication in replications[1:]:
        common_outcomes &= set(replication["outcomes"])

    time_series_outcomes = {}

    for name in sorted(common_outcomes):
        arrays = [
            np.asarray(replication["outcomes"][name])
            for replication in replications
        ]

        if all(array.ndim == 2 for array in arrays):
            time_series_outcomes[name] = np.stack(arrays, axis=0)

    return time_series_outcomes


# =============================================================================
# Representative scenario selection
# =============================================================================

def select_representative_scenarios(Y, n_scenarios_to_select=6):
    """
    Select scenarios spanning low/median/high final means and stochastic
    variance.
    """
    full_mean = np.nanmean(Y, axis=0)
    full_variance = np.nanvar(Y, axis=0, ddof=1)

    final_mean = full_mean[:, -1]
    integrated_variance = np.nanmean(full_variance, axis=1)

    candidate_ids = []

    for values in (final_mean, integrated_variance):
        for quantile in (0.1, 0.5, 0.9):
            target = np.nanquantile(values, quantile)
            scenario_id = int(
                np.nanargmin(np.abs(values - target))
            )
            candidate_ids.append(scenario_id)

    selected = list(dict.fromkeys(candidate_ids))

    if len(selected) < n_scenarios_to_select:
        ranked = np.argsort(integrated_variance)

        for scenario_id in ranked:
            scenario_id = int(scenario_id)

            if scenario_id not in selected:
                selected.append(scenario_id)

            if len(selected) == n_scenarios_to_select:
                break

    return selected[:n_scenarios_to_select]


def make_selection_table(
    scenario_experiments,
    Y,
    selected_scenarios,
    outcome_name,
):
    """Create a table describing the representative scenarios."""
    full_mean = np.nanmean(Y, axis=0)
    full_variance = np.nanvar(Y, axis=0, ddof=1)

    final_mean = full_mean[:, -1]
    integrated_variance = np.nanmean(full_variance, axis=1)

    table = scenario_experiments.loc[
        selected_scenarios,
        ["scenario_id"] + UNCERTAINTY_COLS,
    ].copy()

    table["outcome"] = outcome_name
    table["final_mean"] = final_mean[selected_scenarios]
    table["mean_time_series_variance"] = (
        integrated_variance[selected_scenarios]
    )

    return table


# =============================================================================
# 1. Representative cumulative mean / variance convergence
# =============================================================================

def cumulative_mean(data):
    """data shape: replication × timestep."""
    return {
        n: np.nanmean(data[:n], axis=0)
        for n in range(1, data.shape[0] + 1)
    }


def cumulative_variance(data):
    """data shape: replication × timestep."""
    return {
        n: np.nanvar(data[:n], axis=0, ddof=1)
        for n in range(2, data.shape[0] + 1)
    }


def common_mean_ylim(Y, selected_scenarios):
    """
    Determine one y-axis range for all representative cumulative-mean panels
    belonging to the same outcome.
    """
    values = []

    for scenario_id in selected_scenarios:
        curves = cumulative_mean(Y[:, scenario_id, :])

        for curve in curves.values():
            finite = curve[np.isfinite(curve)]
            values.extend(finite)

    ymin = np.min(values)
    ymax = np.max(values)

    y_range = ymax - ymin
    padding = 0.05 * y_range if y_range > 0 else 0.05

    return ymin - padding, ymax + padding


def plot_representative_mean_convergence(
    Y,
    selected_scenarios,
    outcome_name,
):
    """
    Plot six representative scenarios in one 2×3 figure.

    All panels use common y-axis limits for the outcome, making cross-scenario
    differences directly comparable.
    """
    n_replications = Y.shape[0]
    timesteps = np.arange(1, Y.shape[2] + 1)

    common_ylim = common_mean_ylim(Y, selected_scenarios)

    ncols = 3
    nrows = int(np.ceil(len(selected_scenarios) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 7.5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    replication_counts = np.arange(1, n_replications + 1)
    norm = plt.Normalize(
        replication_counts.min(),
        replication_counts.max(),
    )
    cmap = plt.colormaps["viridis"]

    for ax, scenario_id in zip(axes, selected_scenarios):
        curves = cumulative_mean(Y[:, scenario_id, :])

        for n, curve in curves.items():
            is_final = n == n_replications

            ax.plot(
                timesteps,
                curve,
                color=cmap(norm(n)),
                linewidth=2.4 if is_final else 1.0,
                alpha=1.0 if is_final else 0.55,
                zorder=3 if is_final else 2,
            )

        add_flood_line(ax)
        ax.set_title(f"Scenario {scenario_id}")
        ax.set_ylim(common_ylim)
        ax.grid(alpha=0.15)

    for ax in axes[len(selected_scenarios):]:
        ax.remove()

    fig.supxlabel("Timestep")
    fig.supylabel(pretty_outcome(outcome_name))
    fig.suptitle(
        f"Cumulative mean convergence: {pretty_outcome(outcome_name)}",
        y=1.02,
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes[:len(selected_scenarios)].tolist(),
        pad=0.02,
        fraction=0.025,
    )
    cbar.set_label("Number of replications")

    preferred_ticks = [
        n for n in [1, 5, 10, 20, 30, 40]
        if n <= n_replications
    ]
    if n_replications not in preferred_ticks:
        preferred_ticks.append(n_replications)

    cbar.set_ticks(sorted(set(preferred_ticks)))

    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.09,
        top=0.90,
        wspace=0.18,
        hspace=0.25,
    )

    save_figure(
        fig,
        f"replication_mean_convergence_{outcome_name}",
    )


def plot_representative_variance_convergence(
    Y,
    selected_scenarios,
    outcome_name,
):
    """
    Plot representative stochastic-variance convergence.

    Variance panels retain their natural y-axis scales because stochastic
    variance can differ substantially between representative scenarios.
    """
    n_replications = Y.shape[0]
    timesteps = np.arange(1, Y.shape[2] + 1)

    ncols = 3
    nrows = int(np.ceil(len(selected_scenarios) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 7.5),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    replication_counts = np.arange(2, n_replications + 1)
    norm = plt.Normalize(
        replication_counts.min(),
        replication_counts.max(),
    )
    cmap = plt.colormaps["viridis"]

    for ax, scenario_id in zip(axes, selected_scenarios):
        curves = cumulative_variance(Y[:, scenario_id, :])

        for n, curve in curves.items():
            is_final = n == n_replications

            ax.plot(
                timesteps,
                curve,
                color=cmap(norm(n)),
                linewidth=2.4 if is_final else 1.0,
                alpha=1.0 if is_final else 0.55,
                zorder=3 if is_final else 2,
            )

        add_flood_line(ax)
        ax.set_title(f"Scenario {scenario_id}")
        ax.grid(alpha=0.15)

    for ax in axes[len(selected_scenarios):]:
        ax.remove()

    fig.supxlabel("Timestep")
    fig.supylabel("Variance across replications")
    fig.suptitle(
        f"Cumulative variance convergence: {pretty_outcome(outcome_name)}",
        y=1.02,
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes[:len(selected_scenarios)].tolist(),
        pad=0.02,
        fraction=0.025,
    )
    cbar.set_label("Number of replications")

    preferred_ticks = [
        n for n in [2, 5, 10, 20, 30, 40]
        if n <= n_replications
    ]
    if n_replications not in preferred_ticks:
        preferred_ticks.append(n_replications)

    cbar.set_ticks(sorted(set(preferred_ticks)))

    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.09,
        top=0.90,
        wspace=0.22,
        hspace=0.25,
    )

    save_figure(
        fig,
        f"replication_variance_convergence_{outcome_name}",
    )


# =============================================================================
# 2. Global replication convergence diagnostics
# =============================================================================

def calculate_global_replication_convergence(Y):
    """Compare cumulative mean/variance estimates with the full seed set."""
    n_replications = Y.shape[0]

    reference_mean = np.nanmean(Y, axis=0)
    reference_variance = np.nanvar(Y, axis=0, ddof=1)

    mean_scale = safe_scale(reference_mean)
    variance_scale = safe_scale(reference_variance)

    rows = []

    for n in range(2, n_replications + 1):
        current_mean = np.nanmean(Y[:n], axis=0)
        current_variance = np.nanvar(Y[:n], axis=0, ddof=1)

        rows.append(
            {
                "replications": n,
                "mean_nrmse": (
                    np.sqrt(
                        np.nanmean(
                            (current_mean - reference_mean) ** 2
                        )
                    )
                    / mean_scale
                ),
                "variance_nrmse": (
                    np.sqrt(
                        np.nanmean(
                            (current_variance - reference_variance) ** 2
                        )
                    )
                    / variance_scale
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_global_replication_convergence(df, outcome_name):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        df["replications"],
        df["mean_nrmse"],
        marker="o",
        label="Cumulative mean",
    )
    ax.plot(
        df["replications"],
        df["variance_nrmse"],
        marker="o",
        label="Cumulative variance",
    )

    ax.set_xlabel("Number of replications")
    ax.set_ylabel("Normalized RMSE against full set")
    ax.set_title(
        f"Global replication convergence: {pretty_outcome(outcome_name)}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        f"replication_global_nrmse_{outcome_name}",
    )


def calculate_incremental_replication_change(Y):
    """Quantify the change caused by adding the latest replication."""
    n_replications = Y.shape[0]

    reference_variance = np.nanvar(Y, axis=0, ddof=1)

    mean_scale = safe_scale(np.nanmean(Y, axis=0))
    variance_scale = safe_scale(reference_variance)

    rows = []

    for n in range(3, n_replications + 1):
        mean_previous = np.nanmean(Y[:n - 1], axis=0)
        mean_current = np.nanmean(Y[:n], axis=0)

        variance_previous = np.nanvar(
            Y[:n - 1],
            axis=0,
            ddof=1,
        )
        variance_current = np.nanvar(
            Y[:n],
            axis=0,
            ddof=1,
        )

        rows.append(
            {
                "replications": n,
                "mean_change": (
                    np.sqrt(
                        np.nanmean(
                            (mean_current - mean_previous) ** 2
                        )
                    )
                    / mean_scale
                ),
                "variance_change": (
                    np.sqrt(
                        np.nanmean(
                            (variance_current - variance_previous) ** 2
                        )
                    )
                    / variance_scale
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_incremental_replication_change(df, outcome_name):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        df["replications"],
        df["mean_change"],
        marker="o",
        label="Mean change",
    )
    ax.plot(
        df["replications"],
        df["variance_change"],
        marker="o",
        label="Variance change",
    )

    ax.axhline(
        0.01,
        linestyle="--",
        linewidth=1,
        label="Illustrative 1% reference",
    )
    ax.axhline(
        0.05,
        linestyle=":",
        linewidth=1,
        label="Illustrative 5% reference",
    )

    ax.set_xlabel("Number of replications after adding one seed")
    ax.set_ylabel("Normalized change")
    ax.set_title(
        f"Incremental replication stability: {pretty_outcome(outcome_name)}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        f"replication_incremental_change_{outcome_name}",
    )


def calculate_ci_convergence(Y):
    """Calculate 95% CI half-widths for final outcomes across scenarios."""
    n_replications = Y.shape[0]

    rows = []

    for n in range(2, n_replications + 1):
        final_values = Y[:n, :, -1]

        standard_error = (
            np.nanstd(final_values, axis=0, ddof=1)
            / np.sqrt(n)
        )
        half_width = 1.96 * standard_error

        rows.append(
            {
                "replications": n,
                "median_half_width": np.nanmedian(half_width),
                "p90_half_width": np.nanquantile(
                    half_width,
                    0.9,
                ),
                "max_half_width": np.nanmax(half_width),
            }
        )

    return pd.DataFrame(rows)


def plot_ci_convergence(df, outcome_name):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        df["replications"],
        df["median_half_width"],
        marker="o",
        label="Median scenario",
    )
    ax.plot(
        df["replications"],
        df["p90_half_width"],
        marker="o",
        label="90th percentile scenario",
    )
    ax.plot(
        df["replications"],
        df["max_half_width"],
        marker="o",
        label="Worst scenario",
    )

    ax.set_xlabel("Number of replications")
    ax.set_ylabel("95% CI half-width")
    ax.set_title(
        f"Final-outcome precision: {pretty_outcome(outcome_name)}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        f"replication_ci_convergence_{outcome_name}",
    )


def build_replication_summary(time_series_outcomes, outcomes):
    """Create one compact final-seed stability table across outcomes."""
    rows = []

    for outcome_name in outcomes:
        Y = time_series_outcomes[outcome_name]
        n_replications = Y.shape[0]

        reference_mean = np.nanmean(Y, axis=0)
        reference_variance = np.nanvar(Y, axis=0, ddof=1)

        mean_previous = np.nanmean(Y[:-1], axis=0)
        mean_current = np.nanmean(Y, axis=0)

        variance_previous = np.nanvar(
            Y[:-1],
            axis=0,
            ddof=1,
        )
        variance_current = np.nanvar(
            Y,
            axis=0,
            ddof=1,
        )

        final_values = Y[:, :, -1]
        ci_half_width = (
            1.96
            * np.nanstd(final_values, axis=0, ddof=1)
            / np.sqrt(n_replications)
        )

        rows.append(
            {
                "outcome": outcome_name,
                "last_mean_change": (
                    np.sqrt(
                        np.nanmean(
                            (mean_current - mean_previous) ** 2
                        )
                    )
                    / safe_scale(reference_mean)
                ),
                "last_variance_change": (
                    np.sqrt(
                        np.nanmean(
                            (variance_current - variance_previous) ** 2
                        )
                    )
                    / safe_scale(reference_variance)
                ),
                "median_final_ci_half_width": (
                    np.nanmedian(ci_half_width)
                ),
                "p90_final_ci_half_width": (
                    np.nanquantile(ci_half_width, 0.9)
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["last_variance_change", "last_mean_change"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 3. Scenario-sample descriptive convergence
# =============================================================================

def generate_nested_subset_orders(
    n_scenarios,
    n_subsets=N_SUBSETS,
    random_seed=RANDOM_SEED,
):
    """
    Generate reproducible random scenario orders.

    Within each order, the first N scenarios form nested subsets.
    """
    rng = np.random.default_rng(random_seed)

    return [
        rng.permutation(n_scenarios)
        for _ in range(n_subsets)
    ]


def time_series_descriptives(array):
    """array shape: scenario × timestep."""
    return {
        "mean": np.nanmean(array, axis=0),
        "std": np.nanstd(array, axis=0, ddof=1),
        "q25": np.nanquantile(array, 0.25, axis=0),
        "q50": np.nanquantile(array, 0.50, axis=0),
        "q75": np.nanquantile(array, 0.75, axis=0),
    }


def calculate_descriptive_sample_convergence(
    scenario_mean_outcomes,
    outcomes,
    subset_orders,
    sample_sizes,
):
    rows = []

    for outcome_name in outcomes:
        Z = scenario_mean_outcomes[outcome_name]

        full_stats = time_series_descriptives(Z)
        full_iqr = full_stats["q75"] - full_stats["q25"]

        for repeat_id, order in enumerate(subset_orders):
            for n in sample_sizes:
                subset = Z[order[:n]]
                subset_stats = time_series_descriptives(subset)
                subset_iqr = (
                    subset_stats["q75"] - subset_stats["q25"]
                )

                rows.append(
                    {
                        "outcome": outcome_name,
                        "repeat": repeat_id,
                        "sample_size": n,
                        "mean_nrmse": normalized_rmse(
                            subset_stats["mean"],
                            full_stats["mean"],
                        ),
                        "std_nrmse": normalized_rmse(
                            subset_stats["std"],
                            full_stats["std"],
                        ),
                        "iqr_nrmse": normalized_rmse(
                            subset_iqr,
                            full_iqr,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def summarize_descriptive_convergence(df):
    return (
        df
        .groupby(["outcome", "sample_size"], as_index=False)
        .agg(
            mean_nrmse_median=("mean_nrmse", "median"),
            mean_nrmse_p10=(
                "mean_nrmse",
                lambda x: np.quantile(x, 0.1),
            ),
            mean_nrmse_p90=(
                "mean_nrmse",
                lambda x: np.quantile(x, 0.9),
            ),
            std_nrmse_median=("std_nrmse", "median"),
            std_nrmse_p10=(
                "std_nrmse",
                lambda x: np.quantile(x, 0.1),
            ),
            std_nrmse_p90=(
                "std_nrmse",
                lambda x: np.quantile(x, 0.9),
            ),
            iqr_nrmse_median=("iqr_nrmse", "median"),
            iqr_nrmse_p10=(
                "iqr_nrmse",
                lambda x: np.quantile(x, 0.1),
            ),
            iqr_nrmse_p90=(
                "iqr_nrmse",
                lambda x: np.quantile(x, 0.9),
            ),
        )
    )


def plot_descriptive_sample_convergence(summary, outcome_name):
    plot_data = summary[
        summary["outcome"] == outcome_name
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        plot_data["sample_size"],
        plot_data["mean_nrmse_median"],
        marker="o",
        label="Mean",
    )
    ax.plot(
        plot_data["sample_size"],
        plot_data["std_nrmse_median"],
        marker="o",
        label="Standard deviation",
    )
    ax.plot(
        plot_data["sample_size"],
        plot_data["iqr_nrmse_median"],
        marker="o",
        label="Interquartile range",
    )

    ax.set_xlabel("Number of uncertainty scenarios")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title(
        f"Scenario-sample convergence: {pretty_outcome(outcome_name)}"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        f"scenario_descriptive_convergence_{outcome_name}",
    )


# =============================================================================
# 4. Scenario-sample sensitivity convergence
# =============================================================================

def calculate_time_dependent_sensitivity(
    scenario_experiments,
    mean_outcomes,
):
    """Calculate Extra Trees feature importance at each supplied timestep."""
    X = scenario_experiments[UNCERTAINTY_COLS]
    sensitivity_results = {}

    for outcome_key, outcome_mean in mean_outcomes.items():
        scores_over_time = []

        for t in range(outcome_mean.shape[1]):
            y = outcome_mean[:, t]

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

            scores_over_time.append(
                timestep_scores.to_numpy()
            )

        sensitivity_results[outcome_key] = pd.DataFrame(
            scores_over_time,
            columns=UNCERTAINTY_COLS,
        )

    return sensitivity_results


def calculate_sensitivity_sample_convergence(
    scenario_experiments,
    scenario_mean_outcomes,
    outcomes,
    subset_orders,
    sample_sizes,
):
    """
    Compare Extra Trees feature-importance estimates from smaller scenario
    samples with the full-ensemble reference.
    """
    max_timestep = min(
        array.shape[1]
        for array in scenario_mean_outcomes.values()
    )

    sensitivity_timesteps = [
        t for t in SENSITIVITY_TIMESTEPS
        if t <= max_timestep
    ]
    timestep_idx = np.array(sensitivity_timesteps) - 1

    print(
        "Sensitivity convergence timesteps:",
        sensitivity_timesteps,
    )

    full_sensitivity = calculate_time_dependent_sensitivity(
        scenario_experiments,
        {
            outcome: scenario_mean_outcomes[outcome][
                :, timestep_idx
            ]
            for outcome in outcomes
        },
    )

    rows = []

    for outcome_name in outcomes:
        print(
            f"Calculating sensitivity convergence: "
            f"{pretty_outcome(outcome_name)}"
        )

        Z = scenario_mean_outcomes[outcome_name]

        reference = (
            full_sensitivity[outcome_name]
            .loc[:, UNCERTAINTY_COLS]
            .to_numpy()
        )

        for repeat_id, order in enumerate(subset_orders):
            print(
                f"  subset order "
                f"{repeat_id + 1}/{len(subset_orders)}"
            )

            for n in sample_sizes:
                subset_ids = order[:n]

                subset_experiments = (
                    scenario_experiments
                    .iloc[subset_ids]
                    .reset_index(drop=True)
                )

                subset_outcomes = {
                    outcome_name: Z[subset_ids][:, timestep_idx]
                }

                subset_sensitivity = (
                    calculate_time_dependent_sensitivity(
                        subset_experiments,
                        subset_outcomes,
                    )[outcome_name]
                )

                current = (
                    subset_sensitivity
                    .loc[:, UNCERTAINTY_COLS]
                    .to_numpy()
                )

                rows.append(
                    {
                        "outcome": outcome_name,
                        "repeat": repeat_id,
                        "sample_size": n,
                        "sensitivity_rmse": np.sqrt(
                            np.mean(
                                (current - reference) ** 2
                            )
                        ),
                        "sensitivity_max_error": np.max(
                            np.abs(current - reference)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def summarize_sensitivity_convergence(df):
    return (
        df
        .groupby(["outcome", "sample_size"], as_index=False)
        .agg(
            sensitivity_rmse_median=(
                "sensitivity_rmse",
                "median",
            ),
            sensitivity_rmse_p10=(
                "sensitivity_rmse",
                lambda x: np.quantile(x, 0.1),
            ),
            sensitivity_rmse_p90=(
                "sensitivity_rmse",
                lambda x: np.quantile(x, 0.9),
            ),
            sensitivity_max_error_median=(
                "sensitivity_max_error",
                "median",
            ),
        )
    )


def plot_sensitivity_sample_convergence(summary, outcome_name):
    plot_data = summary[
        summary["outcome"] == outcome_name
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        plot_data["sample_size"],
        plot_data["sensitivity_rmse_median"],
        marker="o",
    )
    ax.fill_between(
        plot_data["sample_size"],
        plot_data["sensitivity_rmse_p10"],
        plot_data["sensitivity_rmse_p90"],
        alpha=0.2,
    )

    ax.set_xlabel("Number of uncertainty scenarios")
    ax.set_ylabel("RMSE in feature importance")
    ax.set_title(
        f"Sensitivity convergence: {pretty_outcome(outcome_name)}"
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        f"scenario_sensitivity_convergence_{outcome_name}",
    )


def plot_combined_sensitivity_convergence(summary, outcomes):
    fig, ax = plt.subplots(figsize=(9, 5))

    for outcome_name in outcomes:
        plot_data = summary[
            summary["outcome"] == outcome_name
        ]

        ax.plot(
            plot_data["sample_size"],
            plot_data["sensitivity_rmse_median"],
            marker="o",
            label=pretty_outcome(outcome_name),
        )

    ax.set_xlabel("Number of uncertainty scenarios")
    ax.set_ylabel("RMSE in feature importance")
    ax.set_title(
        "Convergence of time-dependent sensitivity estimates"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    save_figure(
        fig,
        "scenario_sensitivity_convergence_all_outcomes",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading DelftBlue replications...")
    replications = load_replications(RESULTS_DIR)

    scenario_experiments = verify_and_prepare_scenarios(
        replications
    )
    time_series_outcomes = stack_time_series_outcomes(
        replications
    )

    n_replications = len(replications)
    n_scenarios = len(scenario_experiments)

    outcomes = [
        name
        for name in VALIDATION_OUTCOMES
        if name in time_series_outcomes
    ]

    missing = [
        name
        for name in VALIDATION_OUTCOMES
        if name not in time_series_outcomes
    ]

    if missing:
        print(
            "Warning: the following requested validation outcomes "
            f"were not found and will be skipped: {missing}"
        )

    if not outcomes:
        raise KeyError(
            "None of the requested validation outcomes are available."
        )

    print(
        f"Loaded {n_replications} stochastic replications "
        f"for {n_scenarios:,} uncertainty scenarios."
    )
    print("Validation outcomes:")
    for outcome in outcomes:
        print(f" - {pretty_outcome(outcome)}")

    print(f"Saving figures and tables to:\n{PLOT_DIR}")

    # -------------------------------------------------------------------------
    # Replication convergence
    # -------------------------------------------------------------------------
    print("\n1/4 Representative replication convergence...")

    selection_tables = []
    global_tables = []
    incremental_tables = []
    ci_tables = []

    for outcome_name in outcomes:
        print(f"  {pretty_outcome(outcome_name)}")

        Y = time_series_outcomes[outcome_name]

        selected_scenarios = select_representative_scenarios(
            Y,
            N_REPRESENTATIVE_SCENARIOS,
        )

        selection_tables.append(
            make_selection_table(
                scenario_experiments,
                Y,
                selected_scenarios,
                outcome_name,
            )
        )

        plot_representative_mean_convergence(
            Y,
            selected_scenarios,
            outcome_name,
        )
        plot_representative_variance_convergence(
            Y,
            selected_scenarios,
            outcome_name,
        )

        global_df = calculate_global_replication_convergence(Y)
        global_df.insert(0, "outcome", outcome_name)
        global_tables.append(global_df)
        plot_global_replication_convergence(
            global_df,
            outcome_name,
        )

        incremental_df = calculate_incremental_replication_change(Y)
        incremental_df.insert(0, "outcome", outcome_name)
        incremental_tables.append(incremental_df)
        plot_incremental_replication_change(
            incremental_df,
            outcome_name,
        )

        ci_df = calculate_ci_convergence(Y)
        ci_df.insert(0, "outcome", outcome_name)
        ci_tables.append(ci_df)
        plot_ci_convergence(
            ci_df,
            outcome_name,
        )

    save_table(
        pd.concat(selection_tables, ignore_index=True),
        "representative_scenarios.csv",
    )
    save_table(
        pd.concat(global_tables, ignore_index=True),
        "replication_global_convergence.csv",
    )
    save_table(
        pd.concat(incremental_tables, ignore_index=True),
        "replication_incremental_change.csv",
    )
    save_table(
        pd.concat(ci_tables, ignore_index=True),
        "replication_ci_convergence.csv",
    )

    replication_summary = build_replication_summary(
        time_series_outcomes,
        outcomes,
    )
    save_table(
        replication_summary,
        "replication_convergence_summary.csv",
    )

    # -------------------------------------------------------------------------
    # Scenario-sample convergence
    # -------------------------------------------------------------------------
    print("\n2/4 Preparing scenario-mean outcomes...")

    # These are the quantities used in subsequent analyses:
    # each uncertainty scenario is represented by its mean across stochastic
    # replications.
    scenario_mean_outcomes = {
        name: np.nanmean(
            time_series_outcomes[name],
            axis=0,
        )
        for name in outcomes
    }

    sample_sizes = sorted(
        {
            n for n in [*BASE_SAMPLE_SIZES, n_scenarios]
            if n <= n_scenarios
        }
    )

    subset_orders = generate_nested_subset_orders(
        n_scenarios=n_scenarios,
        n_subsets=N_SUBSETS,
        random_seed=RANDOM_SEED,
    )

    print("Scenario sample sizes:", sample_sizes)

    print("\n3/4 Descriptive scenario-sample convergence...")

    descriptive_convergence = (
        calculate_descriptive_sample_convergence(
            scenario_mean_outcomes,
            outcomes,
            subset_orders,
            sample_sizes,
        )
    )
    descriptive_summary = summarize_descriptive_convergence(
        descriptive_convergence
    )

    save_table(
        descriptive_convergence,
        "scenario_descriptive_convergence_raw.csv",
    )
    save_table(
        descriptive_summary,
        "scenario_descriptive_convergence_summary.csv",
    )

    for outcome_name in outcomes:
        plot_descriptive_sample_convergence(
            descriptive_summary,
            outcome_name,
        )

    print("\n4/4 Sensitivity scenario-sample convergence...")
    print(
        "This is the computationally expensive part of the script."
    )

    sensitivity_convergence = (
        calculate_sensitivity_sample_convergence(
            scenario_experiments,
            scenario_mean_outcomes,
            outcomes,
            subset_orders,
            sample_sizes,
        )
    )
    sensitivity_summary = summarize_sensitivity_convergence(
        sensitivity_convergence
    )

    save_table(
        sensitivity_convergence,
        "scenario_sensitivity_convergence_raw.csv",
    )
    save_table(
        sensitivity_summary,
        "scenario_sensitivity_convergence_summary.csv",
    )

    for outcome_name in outcomes:
        plot_sensitivity_sample_convergence(
            sensitivity_summary,
            outcome_name,
        )

    plot_combined_sensitivity_convergence(
        sensitivity_summary,
        outcomes,
    )

    print("\nDone.")
    print(f"Outputs saved to:\n{PLOT_DIR}")


if __name__ == "__main__":
    main()
