"""Clustering analysis and thesis/appendix output generation.

1. visual comparison of candidate clustering outcomes;
2. basic outcome-screening and redundancy tables;
3. multivariate DTW k-means clustering for k = 3..6;
4. per-cluster trajectory figures (no mean trajectories);
5. cluster-size tables;
6. DTW silhouette scores and a compact silhouette figure.


All generated output is written to:
    PROJECT_ROOT / "thesis" / "plots" / "clustering"
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="ipyparallel not installed*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ema_workbench.util import load_results
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw


# =============================================================================
# Configuration
# =============================================================================

RESULT_GLOB = "ema_fast_results_seed_*.tar.gz"
K_VALUES = (3, 4, 5, 6)
RANDOM_STATE = 42
FLOOD_TIMESTEP = 10
FIG_DPI = 300
COMPUTE_SILHOUETTE = True

UNCERTAINTY_COLS = [
    "initial_adaptation",
    "per_HH_adapted_to_SN_prob_midpoint",
    "weight_SN_vs_PMT",
    "savings_rate_multiplier",
    "adaptation_cost_multiplier",
    "measure_lifetime",
    "flood_depth_multiplier",
]

# Longlist retained only for basic screening/redundancy output.
CANDIDATE_CLUSTERING_OUTCOMES = [
    "adaptation_rate",
    "share_liquidity_constrained",
    "gini_relative_burden",
    "adaptation_gap_high_low",
    "share_adaptation_deficit",
    "unmet_adaptation_demand",
    "share_over_adapted",
    "average_fraction_connections_adapted",
]

# Multivariate candidate sets to compare in the actual clustering.
CANDIDATE_OUTCOME_SETS = {
    "adaptation_liquidity": [
        "adaptation_rate",
        "share_liquidity_constrained",
    ],
    "adaptation_liquidity_gap": [
        "adaptation_rate",
        "share_liquidity_constrained",
        "adaptation_gap_high_low",
    ],
    "adaptation_liquidity_gini": [
        "adaptation_rate",
        "share_liquidity_constrained",
        "gini_relative_burden",
    ],
    "adaptation_liquidity_deficit": [
        "adaptation_rate",
        "share_liquidity_constrained",
        "share_adaptation_deficit",
    ],
}

OUTCOME_NAMES = {
    "adaptation_rate": "Adaptation rate",
    "share_liquidity_constrained": "Liquidity constrained",
    "adaptation_gap_high_low": "Adaptation gap high-low",
    "gini_relative_burden": "Relative burden (Gini)",
    "share_adaptation_deficit": "Adaptation deficit",
    "unmet_adaptation_demand": "Unmet adaptation demand",
    "share_over_adapted": "Over-adapted households",
    "average_fraction_connections_adapted": "Adapted network connections",
}

SET_NAMES = {
    "adaptation_liquidity": "Adaptation + liquidity",
    "adaptation_liquidity_gap": "Adaptation + liquidity + gap",
    "adaptation_liquidity_gini": "Adaptation + liquidity + Gini",
    "adaptation_liquidity_deficit": "Adaptation + liquidity + deficit",
}

# Five outcomes that survived the first conceptual/visual screening and are used
# in the candidate multivariate sets.
OUTCOME_VISUAL_SHORTLIST = [
    "adaptation_rate",
    "share_liquidity_constrained",
    "adaptation_gap_high_low",
    "gini_relative_burden",
    "share_adaptation_deficit",
]


# =============================================================================
# Paths
# =============================================================================


def find_project_root(start: Path) -> Path:
    """Find the nearest ancestor containing results/delftblue2."""
    start = start.resolve()
    candidates = [start] + list(start.parents)

    for candidate in candidates:
        if (candidate / "results" / "delftblue2").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected an ancestor containing "
        "'results/delftblue2'. Place this script somewhere inside the project "
        "tree or adjust find_project_root()."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue2"
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "results" / "plots" / "clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility functions
# =============================================================================


def save_table(df: pd.DataFrame, stem: str, index: bool = False) -> None:
    """Save a table as CSV and tex."""
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    tex_path = OUTPUT_DIR / f"{stem}.tex"

    df.to_csv(csv_path, index=index)
    tex_path.write_text(
        df.to_latex(index=index, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, stem: str, pdf: bool = False) -> None:
    """Save a figure as a high-resolution PNG and optionally as PDF."""
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    if pdf:
        fig.savefig(
            OUTPUT_DIR / f"{stem}.pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


# =============================================================================
# Load and aggregate results
# =============================================================================


def load_ensemble():
    result_files = sorted(RESULTS_DIR.glob(RESULT_GLOB))
    if not result_files:
        raise FileNotFoundError(
            f"No result files matching '{RESULT_GLOB}' found in {RESULTS_DIR}"
        )

    print(f"Found {len(result_files)} result files in {RESULTS_DIR}")

    all_exp = []
    all_outcomes: dict[str, list[np.ndarray]] = {}

    for file in result_files:
        exp, out = load_results(file)
        all_exp.append(exp)

        for name, data in out.items():
            all_outcomes.setdefault(name, []).append(data)

    experiments = pd.concat(all_exp, ignore_index=True)
    all_outcomes = {
        key: np.vstack(value)
        for key, value in all_outcomes.items()
    }

    missing_uncertainties = [
        col for col in UNCERTAINTY_COLS if col not in experiments.columns
    ]
    if missing_uncertainties:
        raise KeyError(
            "Missing uncertainty columns in experiment table: "
            + ", ".join(missing_uncertainties)
        )

    scenario_id = (
        experiments[UNCERTAINTY_COLS]
        .round(10)
        .astype(str)
        .agg("_".join, axis=1)
    )

    scenario_experiments = (
        experiments
        .assign(scenario_id=scenario_id)
        .drop_duplicates("scenario_id")
        .reset_index(drop=True)
    )

    n_seeds = len(result_files)
    n_scenarios = len(scenario_experiments)
    expected_rows = n_seeds * n_scenarios

    mean_outcomes: dict[str, np.ndarray] = {}

    for outcome_name, outcome in all_outcomes.items():
        # Retain only time-series outcomes with one row per seed-scenario run.
        if outcome.ndim != 2 or outcome.shape[0] != expected_rows:
            continue

        outcome_by_seed = outcome.reshape(
            n_seeds,
            n_scenarios,
            outcome.shape[1],
        )

        mean_outcomes[outcome_name] = np.nanmean(
            outcome_by_seed,
            axis=0,
        )

    print(
        f"Aggregated {expected_rows:,} model runs into "
        f"{n_scenarios:,} replication-mean scenario trajectories."
    )

    return scenario_experiments, mean_outcomes


# =============================================================================
# Outcome selection outputs
# =============================================================================


def validate_required_outcomes(mean_outcomes: dict[str, np.ndarray]) -> None:
    required = set(CANDIDATE_CLUSTERING_OUTCOMES)
    for outcomes in CANDIDATE_OUTCOME_SETS.values():
        required.update(outcomes)

    missing = sorted(required - set(mean_outcomes))
    if missing:
        raise KeyError(
            "The following required time-series outcomes were not found: "
            + ", ".join(missing)
        )



def create_outcome_screening_table(mean_outcomes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Basic descriptive screening only; not treated as a behaviour score."""
    rows = []

    for outcome in CANDIDATE_CLUSTERING_OUTCOMES:
        X = mean_outcomes[outcome]

        rows.append(
            {
                "outcome": outcome,
                "overall_range": np.nanmax(X) - np.nanmin(X),
                "mean_between_scenario_std": np.nanmean(np.nanstd(X, axis=0)),
                "mean_within_scenario_temporal_std": np.nanmean(np.nanstd(X, axis=1)),
                "mean_absolute_timestep_change": np.nanmean(np.abs(np.diff(X, axis=1))),
            }
        )

    table = pd.DataFrame(rows)
    save_table(table.round(4), "outcome_screening_descriptive", index=False)
    return table



def create_redundancy_outputs(mean_outcomes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman redundancy table and heatmap for the clustering-outcome longlist."""
    flattened = pd.DataFrame(
        {
            outcome: mean_outcomes[outcome].reshape(-1)
            for outcome in CANDIDATE_CLUSTERING_OUTCOMES
        }
    )
    corr = flattened.corr(method="spearman")

    corr_export = corr.copy()
    corr_export.index.name = "outcome"
    save_table(corr_export.round(3), "outcome_redundancy_spearman", index=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Spearman correlation")

    labels = [OUTCOME_NAMES.get(x, x) for x in corr.columns]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=70, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Redundancy between candidate clustering outcomes")

    fig.tight_layout()
    save_figure(fig, "outcome_redundancy_spearman", pdf=True)

    return corr



def create_candidate_trajectory_figure(mean_outcomes: dict[str, np.ndarray]) -> None:
    """Visual outcome-selection figure: level and shape variation across scenarios."""
    n_outcomes = len(OUTCOME_VISUAL_SHORTLIST)
    n_cols = 2
    n_rows = int(np.ceil(n_outcomes / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    for ax, outcome in zip(axes_flat, OUTCOME_VISUAL_SHORTLIST):
        X = mean_outcomes[outcome]
        timesteps = np.arange(1, X.shape[1] + 1)

        for trajectory in X:
            ax.plot(
                timesteps,
                trajectory,
                color=default_color,
                alpha=0.06,
                linewidth=0.55,
            )

        ax.axvline(
            FLOOD_TIMESTEP,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_title(OUTCOME_NAMES[outcome])
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_outcomes:]:
        ax.axis("off")

    fig.suptitle("Candidate clustering-outcome trajectories", y=1.01)
    fig.tight_layout()
    save_figure(fig, "candidate_outcome_trajectories")


# =============================================================================
# Clustering
# =============================================================================


def prepare_clustering_data(
    mean_outcomes: dict[str, np.ndarray],
    selected_outcomes: list[str],
):
    """Stack and globally standardise each outcome across scenarios and time."""
    X_original = np.stack(
        [mean_outcomes[outcome] for outcome in selected_outcomes],
        axis=2,
    )

    X_scaled = np.empty_like(X_original, dtype=float)
    scalers = {}

    for j, outcome in enumerate(selected_outcomes):
        data = mean_outcomes[outcome]
        scaler = StandardScaler()
        scaled_flat = scaler.fit_transform(data.reshape(-1, 1))
        X_scaled[:, :, j] = scaled_flat.reshape(data.shape)
        scalers[outcome] = scaler

    return X_original, X_scaled, scalers



def run_candidate_clusterings(mean_outcomes: dict[str, np.ndarray]):
    cluster_results = {}

    for set_name, selected_outcomes in CANDIDATE_OUTCOME_SETS.items():
        print(f"\nClustering outcome set: {SET_NAMES[set_name]}")

        X_original, X_scaled, scalers = prepare_clustering_data(
            mean_outcomes,
            selected_outcomes,
        )

        cluster_results[set_name] = {}

        for k in K_VALUES:
            print(f"  k={k}")
            model = TimeSeriesKMeans(
                n_clusters=k,
                metric="dtw",
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(X_scaled)

            cluster_results[set_name][k] = {
                "model": model,
                "labels": labels,
                "X_original": X_original,
                "X_scaled": X_scaled,
                "scalers": scalers,
                "outcomes": selected_outcomes,
            }

    return cluster_results


# =============================================================================
# Cluster figures and tables
# =============================================================================


def create_cluster_trajectory_figure(
    X_original: np.ndarray,
    labels: np.ndarray,
    selected_outcomes: list[str],
    set_name: str,
    k: int,
) -> None:
    """Plot all member trajectories per cluster, without a mean trajectory."""
    n_clusters = len(np.unique(labels))
    n_outcomes = X_original.shape[2]
    timesteps = np.arange(1, X_original.shape[1] + 1)

    y_limits = []
    for outcome_index in range(n_outcomes):
        ymin = np.nanmin(X_original[:, :, outcome_index])
        ymax = np.nanmax(X_original[:, :, outcome_index])
        margin = 0.03 * (ymax - ymin)
        if margin == 0:
            margin = 0.01
        y_limits.append((ymin - margin, ymax + margin))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        n_outcomes,
        n_clusters,
        figsize=(3.7 * n_clusters, 2.8 * n_outcomes),
        sharex=True,
        squeeze=False,
    )

    for cluster in range(n_clusters):
        cluster_data = X_original[labels == cluster]
        cluster_color = default_colors[cluster % len(default_colors)]

        for outcome_index in range(n_outcomes):
            ax = axes[outcome_index, cluster]

            for trajectory in cluster_data:
                ax.plot(
                    timesteps,
                    trajectory[:, outcome_index],
                    color=cluster_color,
                    alpha=0.12,
                    linewidth=0.6,
                )

            ax.axvline(
                FLOOD_TIMESTEP,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
            ax.set_ylim(*y_limits[outcome_index])
            ax.grid(alpha=0.18)

            if outcome_index == 0:
                ax.set_title(
                    f"Cluster {cluster + 1}\n"
                    f"(n={len(cluster_data)})"
                )

            if cluster == 0:
                ax.set_ylabel(OUTCOME_NAMES[selected_outcomes[outcome_index]])

            if outcome_index == n_outcomes - 1:
                ax.set_xlabel("Timestep")

    fig.suptitle(
        f"{SET_NAMES[set_name]} — k={k}",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, f"clusters_{set_name}_k{k}")



def create_cluster_count_table(cluster_results) -> pd.DataFrame:
    rows = []

    for set_name, by_k in cluster_results.items():
        for k, result in by_k.items():
            labels = result["labels"]
            counts = np.bincount(labels, minlength=k)

            for cluster, count in enumerate(counts, start=1):
                rows.append(
                    {
                        "outcome_set": SET_NAMES[set_name],
                        "k": k,
                        "cluster": cluster,
                        "n": int(count),
                        "fraction": count / len(labels),
                    }
                )

    table = pd.DataFrame(rows)
    save_table(table.round(4), "cluster_counts", index=False)
    return table


# =============================================================================
# Silhouette evaluation
# =============================================================================


def create_silhouette_outputs(cluster_results) -> pd.DataFrame:
    rows = []

    for set_name, by_k in cluster_results.items():
        print(f"Computing DTW silhouette scores: {SET_NAMES[set_name]}")

        # The scaled data is identical across k for a given outcome set.
        X_scaled = by_k[K_VALUES[0]]["X_scaled"]
        distance_matrix = cdist_dtw(X_scaled)

        for k in K_VALUES:
            labels = by_k[k]["labels"]
            score = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed",
            )
            rows.append(
                {
                    "outcome_set": SET_NAMES[set_name],
                    "outcome_set_key": set_name,
                    "k": k,
                    "silhouette": score,
                }
            )

    table = pd.DataFrame(rows)
    save_table(
        table[["outcome_set", "k", "silhouette"]].round(4),
        "silhouette_scores",
        index=False,
    )

    pivot = table.pivot(
        index="outcome_set",
        columns="k",
        values="silhouette",
    )
    pivot.index.name = "Outcome set"
    save_table(pivot.round(3), "silhouette_scores_pivot", index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for set_name in CANDIDATE_OUTCOME_SETS:
        subset = table[table["outcome_set_key"] == set_name].sort_values("k")
        ax.plot(
            subset["k"],
            subset["silhouette"],
            marker="o",
            label=SET_NAMES[set_name],
        )

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("DTW silhouette score")
    ax.set_title("Internal validity of candidate clustering solutions")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "silhouette_scores", pdf=True)

    return table


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    scenario_experiments, mean_outcomes = load_ensemble()
    validate_required_outcomes(mean_outcomes)

    # Outcome-selection material (mainly appendix / methodological support).
    create_outcome_screening_table(mean_outcomes)
    create_redundancy_outputs(mean_outcomes)
    create_candidate_trajectory_figure(mean_outcomes)

    # Record the candidate multivariate specifications as a small table.
    candidate_set_table = pd.DataFrame(
        [
            {
                "outcome_set": SET_NAMES[set_name],
                "outcomes": ", ".join(outcomes),
            }
            for set_name, outcomes in CANDIDATE_OUTCOME_SETS.items()
        ]
    )
    save_table(candidate_set_table, "candidate_outcome_sets", index=False)

    # Multivariate DTW k-means clustering.
    cluster_results = run_candidate_clusterings(mean_outcomes)

    # Thesis/appendix candidate figures: one grid per outcome set and k.
    for set_name, by_k in cluster_results.items():
        for k, result in by_k.items():
            create_cluster_trajectory_figure(
                result["X_original"],
                result["labels"],
                result["outcomes"],
                set_name,
                k,
            )

    # Compact quantitative support only.
    create_cluster_count_table(cluster_results)

    if COMPUTE_SILHOUETTE:
        create_silhouette_outputs(cluster_results)

    print("\nDone. Generated thesis/appendix material in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
