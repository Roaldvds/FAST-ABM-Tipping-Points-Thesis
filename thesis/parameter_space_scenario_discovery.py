"""Parameter-space scenario-discovery analysis.

This script reproduces the multivariate behavioural clustering, applies
one-vs-rest PRIM to the seven sampled uncertainties, derives candidate
parameter-space tipping conditions from PRIM-induced subspace boundaries, and
exports the figures and tables.

The final figure set is intentionally limited to evidence that serves one of
five purposes:

1. PRIM quality: coverage-density peeling trajectories for all four regimes.
2. Parameter-space structure: the selected PRIM restrictions across all seven
   uncertain parameters.
3. Observed regime geometry: clean sample-only scatter plots in the three focal
   two-dimensional parameter subspaces.
4. Tipping interpretation: the same focal subspaces with all 2,000 sampled
   scenarios and all four selected PRIM boxes overlaid. Candidate tipping
   conditions are discussed in the text/tables and are deliberately not
   annotated on the figures.
5. Credibility: full-dimensional PRIM-box overlap and peel-alpha robustness.

All generated output is written to:
    PROJECT_ROOT / "thesis" / "results" / "plots" / "parameter space sd"
"""

from __future__ import annotations

import inspect
import itertools
import os
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ema_workbench import ema_logging
from ema_workbench.analysis import prim
from ema_workbench.util import load_results
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans


# =============================================================================
# Configuration
# =============================================================================

RESULT_GLOB = "ema_fast_results_seed_*.tar.gz"
FIG_DPI = 300
SAVE_PDF = False

UNCERTAINTY_COLS = [
    "initial_adaptation",
    "per_HH_adapted_to_SN_prob_midpoint",
    "weight_SN_vs_PMT",
    "savings_rate_multiplier",
    "adaptation_cost_multiplier",
    "measure_lifetime",
    "flood_depth_multiplier",
]

UNCERTAINTY_NAMES = {
    "initial_adaptation": "Initial adaptation",
    "per_HH_adapted_to_SN_prob_midpoint": "Social-norm midpoint",
    "weight_SN_vs_PMT": "Weight social norm vs PMT",
    "savings_rate_multiplier": "Savings-rate multiplier",
    "adaptation_cost_multiplier": "Adaptation-cost multiplier",
    "measure_lifetime": "Measure lifetime",
    "flood_depth_multiplier": "Flood-depth multiplier",
}

# Final clustering specification.
FINAL_OUTCOMES = [
    "adaptation_rate",
    "share_liquidity_constrained",
    "adaptation_gap_high_low",
]
FINAL_K = 4
CLUSTER_RANDOM_STATE = 42
EXPECTED_CLUSTER_COUNTS = [302, 926, 246, 526]

CLUSTER_NAMES = {
    1: "Declining adaptation",
    2: "Low adaptation",
    3: "High adaptation + unequal",
    4: "High adaptation + limited/reversed inequality",
}

# Standard PRIM specification.
PEEL_ALPHA = 0.05
MASS_MIN = 0.05
DENSITY_TARGET = 0.80
COVERAGE_TARGET = 0.80

# Robustness analysis.
PEEL_ALPHAS = [0.025, 0.05, 0.10]

# Constrained PCA groupings used only as a sensitivity diagnostic.
CPCA_SUBSETS = {
    "behavioural_social": [
        "initial_adaptation",
        "per_HH_adapted_to_SN_prob_midpoint",
        "weight_SN_vs_PMT",
    ],
    "financial": [
        "savings_rate_multiplier",
        "adaptation_cost_multiplier",
    ],
    "physical_adaptation": [
        "measure_lifetime",
        "flood_depth_multiplier",
    ],
}

# Generate exhaustive pairwise appendix figures as well as the compact pair grid.
GENERATE_ALL_PAIRWISE_SCATTERS = False
GENERATE_ALL_PAIRWISE_BOX_PROJECTIONS = False

# Suggested focal pairs for the main text / targeted interpretation.
FOCAL_PARAMETER_PAIRS = [
    (
        "initial_adaptation",
        "per_HH_adapted_to_SN_prob_midpoint",
        "Broad adaptation regimes",
    ),
    (
        "savings_rate_multiplier",
        "adaptation_cost_multiplier",
        "Financial conditions",
    ),
    (
        "per_HH_adapted_to_SN_prob_midpoint",
        "weight_SN_vs_PMT",
        "Social-influence conditions",
    ),
]


# PRIM-derived phase-subspace analysis.
#
# These plots deliberately derive the phase regions and candidate tipping
# conditions from the selected PRIM boxes. Sampled cluster points can be shown
# as a diagnostic overlay, but they are not used to construct the boundaries.
TIPPING_SUBSPACES = [
    {
        "id": "broad_adaptation",
        "x": "initial_adaptation",
        "y": "per_HH_adapted_to_SN_prob_midpoint",
        "clusters": [1, 2, 3, 4],
        "title": "PRIM phase subspace: broad adaptation regimes",
    },
    {
        "id": "high_adaptation_distribution",
        "x": "savings_rate_multiplier",
        "y": "adaptation_cost_multiplier",
        "clusters": [1, 2, 3, 4],
        "title": "PRIM phase subspace: distribution of high adaptation",
    },
    {
        "id": "social_feedback",
        "x": "per_HH_adapted_to_SN_prob_midpoint",
        "y": "weight_SN_vs_PMT",
        "clusters": [1, 2, 3, 4],
        "title": "PRIM phase subspace: social-feedback conditions",
    },
]

# A single-axis candidate tipping boundary is retained when two projected PRIM
# boxes are separated along one parameter while overlapping along the other.
# The normalised gap is reported for all cases. This threshold only determines
# which near-adjacent boundaries are promoted to the concise tipping-point
# table/annotations; it does not alter the PRIM boxes themselves.
TIPPING_MAX_NORMALIZED_GAP = 0.10

# Exhaustive box-only phase projections are useful in the appendix.
GENERATE_ALL_PRIM_PHASE_SUBSPACES = False


# =============================================================================
# Paths and general helpers
# =============================================================================


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest ancestor containing results/delftblue2."""
    override = os.environ.get("FAST_PROJECT_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if (candidate / "results" / "delftblue2").exists():
            return candidate
        raise FileNotFoundError(
            f"FAST_PROJECT_ROOT={candidate} does not contain results/delftblue2"
        )

    if start is None:
        start = Path(__file__).resolve().parent

    start = start.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "results" / "delftblue2").exists():
            return candidate

    # Also allow launching the script from the project root while the script
    # itself is somewhere else.
    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "results" / "delftblue2").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root. Expected an ancestor containing "
        "'results/delftblue2'. Place the script inside the project tree, run "
        "it from the project tree, or set FAST_PROJECT_ROOT."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue2"
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "results" / "plots" / "parameter space sd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ema_logging.log_to_stderr(ema_logging.INFO)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)


def save_figure(fig: plt.Figure, stem: str, pdf: bool = SAVE_PDF) -> None:
    """Save a high-resolution PNG and, optionally, a PDF."""
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



def save_table(df: pd.DataFrame, stem: str, index: bool = False) -> None:
    """Save tables in CSV and LaTeX form."""
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=index)
    try:
        latex = df.to_latex(
            index=index,
            float_format=lambda x: f"{x:.3f}",
            escape=True,
        )
        (OUTPUT_DIR / f"{stem}.tex").write_text(latex, encoding="utf-8")
    except Exception as err:
        print(f"Warning: could not save LaTeX table {stem}: {err}")



def get_cluster_colors() -> dict[int, str]:
    """Use the active Matplotlib default colour cycle."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        cluster: cycle[(cluster - 1) % len(cycle)]
        for cluster in range(1, FINAL_K + 1)
    }


CLUSTER_COLORS = get_cluster_colors()


def cluster_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=6,
            color=CLUSTER_COLORS[c],
            label=f"C{c}: {CLUSTER_NAMES[c]}",
        )
        for c in range(1, FINAL_K + 1)
    ]


# =============================================================================
# Load and aggregate the stochastic ensemble
# =============================================================================


def load_ensemble() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load DelftBlue files and aggregate stochastic replications exactly as clustering."""
    result_files = sorted(RESULTS_DIR.glob(RESULT_GLOB))
    if not result_files:
        raise FileNotFoundError(
            f"No result files matching '{RESULT_GLOB}' found in {RESULTS_DIR}"
        )

    print(f"Found {len(result_files)} result files in {RESULTS_DIR}")

    all_exp: list[pd.DataFrame] = []
    all_outcomes: dict[str, list[np.ndarray]] = {}

    for file in result_files:
        exp, out = load_results(file)
        all_exp.append(exp)
        for name, data in out.items():
            all_outcomes.setdefault(name, []).append(data)

    experiments = pd.concat(all_exp, ignore_index=True)
    all_outcomes_stacked = {
        key: np.vstack(value)
        for key, value in all_outcomes.items()
    }

    missing_uncertainties = [
        col for col in UNCERTAINTY_COLS if col not in experiments.columns
    ]
    if missing_uncertainties:
        raise KeyError(
            "Missing uncertainty columns: " + ", ".join(missing_uncertainties)
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

    for outcome_name, outcome in all_outcomes_stacked.items():
        if outcome.ndim != 2 or outcome.shape[0] != expected_rows:
            continue

        outcome_by_seed = outcome.reshape(
            n_seeds,
            n_scenarios,
            outcome.shape[1],
        )
        mean_outcomes[outcome_name] = np.nanmean(outcome_by_seed, axis=0)

    print(
        f"Aggregated {expected_rows:,} model runs into "
        f"{n_scenarios:,} replication-mean scenario trajectories."
    )

    return scenario_experiments, mean_outcomes


# =============================================================================
# Reproduce the final multivariate clustering
# =============================================================================


def prepare_final_clustering_data(
    mean_outcomes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in FINAL_OUTCOMES if name not in mean_outcomes]
    if missing:
        raise KeyError("Missing final clustering outcomes: " + ", ".join(missing))

    X_original = np.stack(
        [mean_outcomes[outcome] for outcome in FINAL_OUTCOMES],
        axis=2,
    )
    X_scaled = np.empty_like(X_original, dtype=float)

    for j, outcome in enumerate(FINAL_OUTCOMES):
        data = mean_outcomes[outcome]
        scaler = StandardScaler()
        scaled_flat = scaler.fit_transform(data.reshape(-1, 1))
        X_scaled[:, :, j] = scaled_flat.reshape(data.shape)

    return X_original, X_scaled



def reproduce_final_clustering(
    scenario_experiments: pd.DataFrame,
    mean_outcomes: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    _, X_scaled = prepare_final_clustering_data(mean_outcomes)

    model = TimeSeriesKMeans(
        n_clusters=FINAL_K,
        metric="dtw",
        random_state=CLUSTER_RANDOM_STATE,
    )
    labels = model.fit_predict(X_scaled) + 1

    counts = pd.Series(labels).value_counts().sort_index()
    cluster_check = pd.DataFrame({
        "cluster": range(1, FINAL_K + 1),
        "cluster_name": [CLUSTER_NAMES[i] for i in range(1, FINAL_K + 1)],
        "n": [int(counts.get(i, 0)) for i in range(1, FINAL_K + 1)],
    })
    cluster_check["fraction"] = cluster_check["n"] / len(labels)
    save_table(cluster_check, "cluster_assignment_check")

    found = cluster_check["n"].tolist()
    if found != EXPECTED_CLUSTER_COUNTS:
        raise RuntimeError(
            "Cluster counts do not match the final thesis solution. "
            f"Expected {EXPECTED_CLUSTER_COUNTS}, found {found}. "
            "Do not interpret PRIM results until the discrepancy is resolved."
        )

    out = scenario_experiments.copy()
    out["cluster"] = labels
    out["cluster_name"] = out["cluster"].map(CLUSTER_NAMES)

    print("Cluster counts exactly match final thesis solution:")
    print(cluster_check.to_string(index=False))
    return out, labels


# =============================================================================
# PRIM helpers
# =============================================================================


def make_prim(
    x: pd.DataFrame,
    y: np.ndarray,
    peel_alpha: float = PEEL_ALPHA,
    mass_min: float = MASS_MIN,
):
    """Instantiate EMA Workbench PRIM compatibly across versions."""
    kwargs = {
        "peel_alpha": peel_alpha,
        "mass_min": mass_min,
    }

    signature = inspect.signature(prim.Prim)
    if "threshold" in signature.parameters:
        threshold_param = signature.parameters["threshold"]
        if threshold_param.default is inspect._empty:
            # Do not use threshold to select a box; inspect the full trajectory.
            kwargs["threshold"] = 0.0

    return prim.Prim(x, y, **kwargs)



def trajectory_with_ids(box) -> pd.DataFrame:
    traj = box.peeling_trajectory.copy().reset_index(drop=True)
    if "id" not in traj.columns:
        traj.insert(0, "id", np.arange(len(traj), dtype=int))
    return traj



def select_density_candidate(
    traj: pd.DataFrame,
    density_target: float = DENSITY_TARGET,
) -> pd.Series:
    """First peeling step reaching target density, else maximum-density step."""
    eligible = traj[traj["density"] >= density_target]
    if not eligible.empty:
        row = eligible.iloc[0].copy()
        row["selection"] = f"first density >= {density_target:.0%}"
        row["target_reached"] = True
    else:
        row = traj.loc[traj["density"].idxmax()].copy()
        row["selection"] = "highest density available"
        row["target_reached"] = False
    return row



def select_coverage_candidate(
    traj: pd.DataFrame,
    coverage_target: float = COVERAGE_TARGET,
) -> pd.Series:
    """Highest-density step retaining at least the target coverage."""
    eligible = traj[traj["coverage"] >= coverage_target]
    if not eligible.empty:
        row = eligible.loc[eligible["density"].idxmax()].copy()
        row["selection"] = f"highest density with coverage >= {coverage_target:.0%}"
        row["target_reached"] = True
    else:
        row = traj.loc[traj["coverage"].idxmax()].copy()
        row["selection"] = "highest coverage available"
        row["target_reached"] = False
    return row



def candidate_boxes(traj: pd.DataFrame) -> pd.DataFrame:
    density_row = select_density_candidate(traj)
    density_row["candidate"] = "density_80"

    coverage_row = select_coverage_candidate(traj)
    coverage_row["candidate"] = "coverage_80"

    return pd.DataFrame([density_row, coverage_row])



def limits_for_dimension(
    box_lims: pd.DataFrame,
    dimension: str,
    full_data: pd.DataFrame,
) -> tuple[float, float]:
    """Return selected PRIM lower and upper limits for one dimension."""
    if dimension in box_lims.columns:
        values = box_lims[dimension]
        return float(np.real(values.iloc[0])), float(np.real(values.iloc[1]))

    if dimension in box_lims.index:
        row = box_lims.loc[dimension]
        return float(np.real(row.iloc[0])), float(np.real(row.iloc[1]))

    return (
        float(np.real(full_data[dimension].min())),
        float(np.real(full_data[dimension].max())),
    )



def get_box_limits(box, box_id: int) -> pd.DataFrame:
    return box.box_lims[int(box_id)].copy()



def scenarios_inside_box(
    data: pd.DataFrame,
    box_lims: pd.DataFrame,
) -> np.ndarray:
    mask = np.ones(len(data), dtype=bool)
    for dimension in data.columns:
        lower, upper = limits_for_dimension(box_lims, dimension, data)
        values = np.real(data[dimension].to_numpy())
        mask &= (values >= lower) & (values <= upper)
    return mask



def rule_limits_table(
    prim_results: dict[int, dict],
    X_prim: pd.DataFrame,
    candidate: str = "density_80",
) -> pd.DataFrame:
    full_min = X_prim.min()
    full_max = X_prim.max()
    rows: list[dict] = []

    for cluster in range(1, FINAL_K + 1):
        result = prim_results[cluster]
        selected = result["candidates"].loc[
            result["candidates"]["candidate"] == candidate
        ].iloc[0]
        box_lims = get_box_limits(result["box"], int(selected["id"]))

        for parameter in UNCERTAINTY_COLS:
            lower, upper = limits_for_dimension(box_lims, parameter, X_prim)
            tol = 1e-10
            restricted = (
                lower > float(full_min[parameter]) + tol
                or upper < float(full_max[parameter]) - tol
            )
            rows.append({
                "cluster": cluster,
                "cluster_name": CLUSTER_NAMES[cluster],
                "parameter": parameter,
                "parameter_name": UNCERTAINTY_NAMES[parameter],
                "lower": lower,
                "upper": upper,
                "full_lower": float(full_min[parameter]),
                "full_upper": float(full_max[parameter]),
                "restricted": restricted,
            })

    return pd.DataFrame(rows)



def format_rule_value(parameter: str, value: float) -> str:
    if parameter == "measure_lifetime":
        return f"{value:.2f}"
    return f"{value:.3f}"



def build_rule_text(
    limits: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cluster in range(1, FINAL_K + 1):
        sub = limits[(limits["cluster"] == cluster) & limits["restricted"]]
        rules = []
        for _, row in sub.iterrows():
            lower_restricted = row["lower"] > row["full_lower"] + 1e-10
            upper_restricted = row["upper"] < row["full_upper"] - 1e-10
            name = row["parameter_name"]
            p = row["parameter"]
            if lower_restricted and upper_restricted:
                rules.append(
                    f"{format_rule_value(p, row['lower'])} <= {name} <= "
                    f"{format_rule_value(p, row['upper'])}"
                )
            elif lower_restricted:
                rules.append(f"{name} >= {format_rule_value(p, row['lower'])}")
            elif upper_restricted:
                rules.append(f"{name} <= {format_rule_value(p, row['upper'])}")

        rows.append({
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES[cluster],
            "rule": "; ".join(rules) if rules else "No restrictions",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Standard PRIM analysis
# =============================================================================


def run_standard_prim(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[dict[int, dict], pd.DataFrame]:
    prim_results: dict[int, dict] = {}
    candidate_rows: list[dict] = []

    for cluster in range(1, FINAL_K + 1):
        y = (labels == cluster).astype(int)
        base_rate = float(y.mean())

        algorithm = make_prim(X_prim.copy(), y)
        box = algorithm.find_box()
        traj = trajectory_with_ids(box)
        candidates = candidate_boxes(traj)

        prim_results[cluster] = {
            "algorithm": algorithm,
            "box": box,
            "trajectory": traj,
            "candidates": candidates,
        }

        save_table(traj, f"prim_peeling_trajectory_cluster_{cluster}")

        for _, row in candidates.iterrows():
            density = float(row["density"])
            candidate_rows.append({
                "cluster": cluster,
                "cluster_name": CLUSTER_NAMES[cluster],
                "n": int(y.sum()),
                "base_rate": base_rate,
                "candidate": row["candidate"],
                "selection": row["selection"],
                "target_reached": bool(row["target_reached"]),
                "box_id": int(row["id"]),
                "coverage": float(row["coverage"]),
                "density": density,
                "mass": float(row["mass"]),
                "restricted_dimensions": int(row["res_dim"]),
                "lift_over_base_rate": density / base_rate if base_rate > 0 else np.nan,
            })

    candidate_summary = pd.DataFrame(candidate_rows)
    save_table(candidate_summary, "prim_candidate_summary_all")
    return prim_results, candidate_summary


# =============================================================================
# Standard PRIM plots
# =============================================================================


def plot_tradeoff_single(
    cluster: int,
    result: dict,
) -> None:
    traj = result["trajectory"]
    candidates = result["candidates"]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(traj["coverage"], traj["density"], marker="o", markersize=3)

    for _, row in candidates.iterrows():
        ax.scatter(
            row["coverage"], row["density"],
            s=90, facecolors="none", edgecolors="black", linewidths=1.4,
        )
        ax.annotate(
            str(row["candidate"]),
            (row["coverage"], row["density"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
        )

    ax.axhline(DENSITY_TARGET, linestyle="--", linewidth=1)
    ax.axvline(COVERAGE_TARGET, linestyle="--", linewidth=1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Cluster {cluster}: {CLUSTER_NAMES[cluster]}")
    fig.tight_layout()
    save_figure(fig, f"prim_tradeoff_cluster_{cluster}")



def plot_tradeoff_combined(prim_results: dict[int, dict]) -> None:
    """Combined PRIM peeling trajectories for the four behavioural regimes.

    Peeling steps are coloured by the number of restricted dimensions, so the
    coverage-density trade-off and the accompanying loss of interpretability can
    be read from the same figure. The two thesis reference boxes use different
    marker shapes.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    scatter = None

    candidate_styles = {
        "density_80": {
            "marker": "o",
            "label": "First density >= 0.80",
        },
        "coverage_80": {
            "marker": "s",
            "label": "Max density with coverage >= 0.80",
        },
    }

    for cluster, ax in zip(range(1, FINAL_K + 1), axes):
        traj = prim_results[cluster]["trajectory"]
        candidates = prim_results[cluster]["candidates"]

        # Faint line preserves the order of the PRIM peeling trajectory.
        ax.plot(
            traj["coverage"],
            traj["density"],
            color="0.68",
            linewidth=1.0,
            zorder=1,
        )

        # Point colour shows how many uncertainty dimensions are restricted.
        scatter = ax.scatter(
            traj["coverage"],
            traj["density"],
            c=traj["res_dim"],
            cmap="viridis",
            vmin=0,
            vmax=len(UNCERTAINTY_COLS),
            s=24,
            edgecolors="none",
            zorder=2,
        )

        for _, row in candidates.iterrows():
            style = candidate_styles[row["candidate"]]
            ax.scatter(
                row["coverage"],
                row["density"],
                s=92,
                marker=style["marker"],
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )

        ax.axhline(
            DENSITY_TARGET,
            linestyle="--",
            linewidth=0.9,
            color="0.45",
        )
        ax.axvline(
            COVERAGE_TARGET,
            linestyle="--",
            linewidth=0.9,
            color="0.45",
        )

        ax.set_title(
            f"C{cluster}: {CLUSTER_NAMES[cluster]}",
            fontsize=11,
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Coverage", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.10, linewidth=0.5)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=candidate_styles["density_80"]["marker"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=candidate_styles["density_80"]["label"],
        ),
        Line2D(
            [0],
            [0],
            marker=candidate_styles["coverage_80"]["marker"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=candidate_styles["coverage_80"]["label"],
        ),
    ]

    fig.suptitle(
        "PRIM coverage-density peeling trajectories",
        y=0.985,
        fontsize=13,
    )

    fig.legend(
        handles=marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.47, 0.935),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # Dedicated colourbar prevents it from squeezing the 2x2 panels.
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.60])
    cbar = fig.colorbar(
        scatter,
        cax=cax,
        ticks=np.arange(0, len(UNCERTAINTY_COLS) + 1),
    )
    cbar.set_label(
        "Restricted dimensions",
        fontsize=10,
    )
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.86,
        hspace=0.28,
        wspace=0.22,
    )

    save_figure(fig, "prim_tradeoffs_all_clusters")


def plot_selected_box_metric_summary(candidate_summary: pd.DataFrame) -> None:
    selected = candidate_summary[candidate_summary["candidate"] == "density_80"].copy()
    x = np.arange(len(selected))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x - width / 2, selected["coverage"], width, label="Coverage")
    ax.bar(x + width / 2, selected["density"], width, label="Density")
    ax.axhline(DENSITY_TARGET, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"C{c}" for c in selected["cluster"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Selected high-density PRIM boxes")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "selected_box_coverage_density")



def plot_interpretability_summary(candidate_summary: pd.DataFrame) -> None:
    selected = candidate_summary[candidate_summary["candidate"] == "density_80"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        [f"C{c}" for c in selected["cluster"]],
        selected["restricted_dimensions"],
    )
    ax.set_ylim(0, len(UNCERTAINTY_COLS) + 0.5)
    ax.set_ylabel("Restricted dimensions")
    ax.set_title("Interpretability of selected PRIM boxes")
    fig.tight_layout()
    save_figure(fig, "selected_box_restricted_dimensions")


# =============================================================================
# Parameter-space visualisation
# =============================================================================


def plot_parameter_pair_grid(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    n = len(UNCERTAINTY_COLS)
    fig, axes = plt.subplots(n, n, figsize=(16, 16), squeeze=False)

    for row, y_col in enumerate(UNCERTAINTY_COLS):
        for col, x_col in enumerate(UNCERTAINTY_COLS):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue

            if row == col:
                for cluster in range(1, FINAL_K + 1):
                    values = X_prim.loc[labels == cluster, x_col]
                    ax.hist(
                        values, bins=22, alpha=0.42, density=True,
                        color=CLUSTER_COLORS[cluster],
                    )
            else:
                for cluster in range(1, FINAL_K + 1):
                    mask = labels == cluster
                    ax.scatter(
                        X_prim.loc[mask, x_col],
                        X_prim.loc[mask, y_col],
                        s=7, alpha=0.30,
                        color=CLUSTER_COLORS[cluster],
                        rasterized=True,
                    )

            if row == n - 1:
                ax.set_xlabel(UNCERTAINTY_NAMES[x_col], fontsize=7)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(UNCERTAINTY_NAMES[y_col], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    fig.legend(
        handles=cluster_legend_handles(),
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Behavioural regimes across the sampled parameter space", y=1.012)
    fig.tight_layout()
    save_figure(fig, "parameter_pair_grid_by_cluster")



def plot_cluster_scatter(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str,
    stem: str,
    title: str,
) -> None:
    """Plot all sampled scenarios in a two-parameter subspace by regime."""
    fig, ax = plt.subplots(
        figsize=(8.2, 6.1),
    )
    ax.set_box_aspect(0.8)

    for cluster in range(1, FINAL_K + 1):
        mask = labels == cluster
        ax.scatter(
            X_prim.loc[mask, x_col],
            X_prim.loc[mask, y_col],
            s=14,
            alpha=0.28,
            color=CLUSTER_COLORS[cluster],
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel(
        UNCERTAINTY_NAMES[x_col],
        fontsize=10,
    )
    ax.set_ylabel(
        UNCERTAINTY_NAMES[y_col],
        fontsize=10,
    )
    ax.set_title(
        title,
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    ax.grid(
        alpha=0.12,
        linewidth=0.5,
    )

    ax.legend(
        handles=cluster_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        borderaxespad=0,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.28,
        top=0.88,
    )

    save_figure(fig, stem)


def plot_all_pairwise_scatters(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    if not GENERATE_ALL_PAIRWISE_SCATTERS:
        return

    for x_col, y_col in itertools.combinations(UNCERTAINTY_COLS, 2):
        plot_cluster_scatter(
            X_prim,
            labels,
            x_col,
            y_col,
            stem=f"pairwise_clusters__{x_col}__vs__{y_col}",
            title=(
                f"Behavioural regimes: {UNCERTAINTY_NAMES[x_col]} vs. "
                f"{UNCERTAINTY_NAMES[y_col]}"
            ),
        )



def plot_parameter_distributions(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    axes = axes.ravel()

    for ax, parameter in zip(axes, UNCERTAINTY_COLS):
        values = [
            X_prim.loc[labels == cluster, parameter].to_numpy()
            for cluster in range(1, FINAL_K + 1)
        ]
        bp = ax.boxplot(values, tick_labels=[f"C{c}" for c in range(1, FINAL_K + 1)], patch_artist=True)
        for patch, cluster in zip(bp["boxes"], range(1, FINAL_K + 1)):
            patch.set_facecolor(CLUSTER_COLORS[cluster])
            patch.set_alpha(0.5)
        ax.set_title(UNCERTAINTY_NAMES[parameter], fontsize=10)
        ax.tick_params(labelsize=8)

    for ax in axes[len(UNCERTAINTY_COLS):]:
        ax.axis("off")

    fig.suptitle("Parameter distributions by behavioural regime", y=1.01)
    fig.tight_layout()
    save_figure(fig, "parameter_distributions_by_cluster")


def plot_focal_sample_scatters(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    """Save clean sample-only views of the three focal parameter subspaces."""
    titles = {
        "broad_adaptation": (
            "Behavioural regimes across initial adaptation and social-norm midpoint"
        ),
        "high_adaptation_distribution": (
            "Behavioural regimes across savings capacity and adaptation cost"
        ),
        "social_feedback": (
            "Behavioural regimes across social-feedback conditions"
        ),
    }

    for subspace in TIPPING_SUBSPACES:
        plot_cluster_scatter(
            X_prim=X_prim,
            labels=labels,
            x_col=subspace["x"],
            y_col=subspace["y"],
            stem=f"sample_scatter__{subspace['id']}",
            title=titles[subspace["id"]],
        )


# =============================================================================
# PRIM box projections and rule visualisations
# =============================================================================


def selected_box_limits(
    prim_results: dict[int, dict],
    cluster: int,
    X_prim: pd.DataFrame,
    candidate: str = "density_80",
) -> pd.DataFrame:
    selected = prim_results[cluster]["candidates"].loc[
        prim_results[cluster]["candidates"]["candidate"] == candidate
    ].iloc[0]
    return get_box_limits(prim_results[cluster]["box"], int(selected["id"]))



def plot_prim_box_projection(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    prim_results: dict[int, dict],
    x_col: str,
    y_col: str,
    stem: str,
    title: str,
) -> None:
    """Plot sampled regimes with selected multidimensional PRIM-box projections."""
    fig, ax = plt.subplots(
        figsize=(8.2, 6.1),
    )
    ax.set_box_aspect(0.8)

    for cluster in range(1, FINAL_K + 1):
        mask = labels == cluster
        ax.scatter(
            X_prim.loc[mask, x_col],
            X_prim.loc[mask, y_col],
            s=14,
            alpha=0.28,
            color=CLUSTER_COLORS[cluster],
            edgecolors="none",
            rasterized=True,
        )

    for cluster in range(1, FINAL_K + 1):
        box_lims = selected_box_limits(
            prim_results,
            cluster,
            X_prim,
        )
        x_min, x_max = limits_for_dimension(
            box_lims,
            x_col,
            X_prim,
        )
        y_min, y_max = limits_for_dimension(
            box_lims,
            y_col,
            X_prim,
        )

        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=CLUSTER_COLORS[cluster],
                linewidth=2.4,
                zorder=5,
            )
        )

    ax.set_xlabel(
        UNCERTAINTY_NAMES[x_col],
        fontsize=10,
    )
    ax.set_ylabel(
        UNCERTAINTY_NAMES[y_col],
        fontsize=10,
    )
    ax.set_title(
        title,
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    ax.grid(
        alpha=0.12,
        linewidth=0.5,
    )

    ax.legend(
        handles=cluster_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        borderaxespad=0,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.28,
        top=0.88,
    )

    save_figure(fig, stem)


def plot_all_pairwise_box_projections(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    prim_results: dict[int, dict],
) -> None:
    if not GENERATE_ALL_PAIRWISE_BOX_PROJECTIONS:
        return

    for x_col, y_col in itertools.combinations(UNCERTAINTY_COLS, 2):
        plot_prim_box_projection(
            X_prim,
            labels,
            prim_results,
            x_col,
            y_col,
            stem=f"prim_projection__{x_col}__vs__{y_col}",
            title=(
                f"Selected PRIM boxes: {UNCERTAINTY_NAMES[x_col]} vs. "
                f"{UNCERTAINTY_NAMES[y_col]}"
            ),
        )



def plot_normalized_rule_intervals(limits: pd.DataFrame) -> None:
    """Show selected box intervals normalised to each uncertainty's sampled range."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    y_positions = np.arange(len(UNCERTAINTY_COLS))
    offsets = np.linspace(-0.27, 0.27, FINAL_K)

    for cluster, offset in zip(range(1, FINAL_K + 1), offsets):
        sub = limits[limits["cluster"] == cluster].set_index("parameter")

        for y, parameter in zip(y_positions, UNCERTAINTY_COLS):
            row = sub.loc[parameter]
            span = row["full_upper"] - row["full_lower"]
            lo = (row["lower"] - row["full_lower"]) / span
            hi = (row["upper"] - row["full_lower"]) / span

            ax.hlines(
                y + offset,
                lo,
                hi,
                color=CLUSTER_COLORS[cluster],
                linewidth=3.2,
            )
            ax.plot(
                [lo, hi],
                [y + offset, y + offset],
                marker="|",
                linestyle="",
                color=CLUSTER_COLORS[cluster],
                markersize=9,
            )

    ax.set_yticks(
        y_positions,
        [UNCERTAINTY_NAMES[p] for p in UNCERTAINTY_COLS],
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(
        "Selected interval within sampled parameter range (normalised)",
        fontsize=11,
    )
    ax.set_title(
        "Selected 80%-density PRIM rules",
        fontsize=13,
    )
    ax.tick_params(
        axis="x",
        labelsize=10,
    )
    ax.tick_params(
        axis="y",
        labelsize=10,
        pad=6,
    )
    ax.invert_yaxis()
    ax.grid(
        axis="x",
        alpha=0.12,
        linewidth=0.6,
    )

    # Same below-plot legend treatment as the final SSSD projections.
    ax.legend(
        handles=cluster_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        borderaxespad=0,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        bottom=0.22,
        top=0.90,
    )

    save_figure(fig, "selected_box_normalized_intervals")

def plot_normalized_rule_intervals_filtered(
        limits: pd.DataFrame,
        min_restricted_clusters: int = 3,
) -> None:
    """Compact version of the selected-rule plot, keeping only informative parameters.

    A parameter is kept if it is restricted in at least `min_restricted_clusters`
    of the four selected PRIM boxes. With the default of 3, parameters that are
    unrestricted in 2 or more clusters are omitted.
    """
    restriction_counts = (
        limits.groupby("parameter")["restricted"]
        .sum()
        .reindex(UNCERTAINTY_COLS)
    )

    kept_parameters = [
        p for p in UNCERTAINTY_COLS
        if restriction_counts.loc[p] >= min_restricted_clusters
    ]

    if not kept_parameters:
        print(
            "No parameters satisfy the filtered-rule criterion "
            f"(restricted in at least {min_restricted_clusters} clusters)."
        )
        return

    filtered_limits = limits[limits["parameter"].isin(kept_parameters)].copy()

    # Also save the retained parameter summary, useful for checking/reporting.
    retained_summary = pd.DataFrame({
        "parameter": kept_parameters,
        "parameter_name": [UNCERTAINTY_NAMES[p] for p in kept_parameters],
        "n_clusters_restricted": [int(restriction_counts.loc[p]) for p in kept_parameters],
    })
    save_table(
        retained_summary,
        f"selected_box_parameter_retention_min{min_restricted_clusters}",
    )

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(kept_parameters) + 2.2))
    y_positions = np.arange(len(kept_parameters))
    offsets = np.linspace(-0.27, 0.27, FINAL_K)

    for cluster, offset in zip(range(1, FINAL_K + 1), offsets):
        sub = filtered_limits[filtered_limits["cluster"] == cluster].set_index("parameter")
        for y, parameter in zip(y_positions, kept_parameters):
            row = sub.loc[parameter]
            span = row["full_upper"] - row["full_lower"]
            lo = (row["lower"] - row["full_lower"]) / span
            hi = (row["upper"] - row["full_lower"]) / span

            ax.hlines(
                y + offset,
                lo,
                hi,
                color=CLUSTER_COLORS[cluster],
                linewidth=3,
                )
            ax.plot(
                [lo, hi],
                [y + offset, y + offset],
                marker="|",
                linestyle="",
                color=CLUSTER_COLORS[cluster],
                markersize=8,
            )

    ax.set_yticks(y_positions, [UNCERTAINTY_NAMES[p] for p in kept_parameters])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Selected interval within sampled parameter range (normalised)")
    ax.set_title(
        "Selected 80%-density PRIM rules\n(filtered to parameters restricted in at least 3 clusters)"
    )
    ax.invert_yaxis()
    ax.legend(
        handles=cluster_legend_handles(),
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(fig, "selected_box_normalized_intervals_filtered")

def plot_prim_subspace_grid(
        limits: pd.DataFrame,
        min_restricted_clusters: int = 3,
) -> None:
    """Lower-triangle grid of selected PRIM boxes across informative parameters.

    Parameters are retained only when restricted in at least
    `min_restricted_clusters` of the four selected PRIM boxes.
    """

    # Determine which parameters are informative enough to retain.
    restriction_counts = (
        limits.groupby("parameter")["restricted"]
        .sum()
        .reindex(UNCERTAINTY_COLS)
    )

    parameters = [
        p for p in UNCERTAINTY_COLS
        if restriction_counts.loc[p] >= min_restricted_clusters
    ]

    print(
        "Parameters retained in PRIM subspace grid:",
        [UNCERTAINTY_NAMES[p] for p in parameters],
    )

    if len(parameters) < 2:
        print("Not enough parameters retained for pairwise grid.")
        return

    # Compact triangular layout:
    # columns = parameters except last
    # rows    = parameters except first
    n = len(parameters)
    fig, axes = plt.subplots(
        n - 1,
        n - 1,
        figsize=(2.8 * (n - 1), 2.8 * (n - 1)),
        squeeze=False,
        )

    for row in range(n - 1):
        y_idx = row + 1
        y_var = parameters[y_idx]

        for col in range(n - 1):
            ax = axes[row, col]
            x_idx = col

            # Only show the lower triangle: y must occur after x.
            if x_idx >= y_idx:
                ax.axis("off")
                continue

            x_var = parameters[x_idx]

            x_full_lo = float(
                limits.loc[
                    limits["parameter"] == x_var,
                    "full_lower",
                ].iloc[0]
            )
            x_full_hi = float(
                limits.loc[
                    limits["parameter"] == x_var,
                    "full_upper",
                ].iloc[0]
            )
            y_full_lo = float(
                limits.loc[
                    limits["parameter"] == y_var,
                    "full_lower",
                ].iloc[0]
            )
            y_full_hi = float(
                limits.loc[
                    limits["parameter"] == y_var,
                    "full_upper",
                ].iloc[0]
            )

            # Draw projection of each selected multidimensional PRIM box.
            for cluster in range(1, FINAL_K + 1):
                x_row = limits[
                    (limits["cluster"] == cluster)
                    & (limits["parameter"] == x_var)
                    ].iloc[0]

                y_row = limits[
                    (limits["cluster"] == cluster)
                    & (limits["parameter"] == y_var)
                    ].iloc[0]

                x_lo = float(x_row["lower"])
                x_hi = float(x_row["upper"])
                y_lo = float(y_row["lower"])
                y_hi = float(y_row["upper"])

                ax.add_patch(
                    Rectangle(
                        (x_lo, y_lo),
                        x_hi - x_lo,
                        y_hi - y_lo,
                        facecolor=CLUSTER_COLORS[cluster],
                        edgecolor=CLUSTER_COLORS[cluster],
                        alpha=0.22,
                        linewidth=1.8,
                        )
                )

            ax.set_xlim(x_full_lo, x_full_hi)
            ax.set_ylim(y_full_lo, y_full_hi)

            ax.grid(
                alpha=0.12,
                linewidth=0.5,
            )

            # Only label outer edges to keep the grid readable.
            if row == n - 2:
                ax.set_xlabel(
                    UNCERTAINTY_NAMES[x_var],
                    fontsize=9,
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(
                    UNCERTAINTY_NAMES[y_var],
                    fontsize=9,
                )
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    # Figure-level legend.
    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=CLUSTER_COLORS[c],
            edgecolor=CLUSTER_COLORS[c],
            alpha=0.35,
            label=f"C{c}: {CLUSTER_NAMES[c]}",
        )
        for c in range(1, FINAL_K + 1)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.88),
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.09,
        top=0.94,
        wspace=0.12,
        hspace=0.12,
    )

    fig.suptitle(
        "Pairwise projections of selected 80%-density PRIM regions",
        y=0.99,
    )

    save_figure(
        fig,
        "selected_box_pairwise_subspace_grid",
    )

def plot_box_parallel_coordinates(limits: pd.DataFrame) -> None:
    """Parallel-coordinate style display of lower/upper selected box limits."""
    x = np.arange(len(UNCERTAINTY_COLS))
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for cluster in range(1, FINAL_K + 1):
        sub = limits[limits["cluster"] == cluster].set_index("parameter")
        lower_norm = []
        upper_norm = []
        for parameter in UNCERTAINTY_COLS:
            row = sub.loc[parameter]
            span = row["full_upper"] - row["full_lower"]
            lower_norm.append((row["lower"] - row["full_lower"]) / span)
            upper_norm.append((row["upper"] - row["full_lower"]) / span)

        lower_norm = np.array(lower_norm, dtype=float)
        upper_norm = np.array(upper_norm, dtype=float)
        ax.fill_between(
            x, lower_norm, upper_norm,
            alpha=0.12, color=CLUSTER_COLORS[cluster],
        )
        ax.plot(x, lower_norm, color=CLUSTER_COLORS[cluster], linewidth=1.5)
        ax.plot(x, upper_norm, color=CLUSTER_COLORS[cluster], linewidth=1.5, linestyle="--")

    ax.set_xticks(x, [UNCERTAINTY_NAMES[p] for p in UNCERTAINTY_COLS], rotation=35, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Normalised parameter range")
    ax.set_title("Selected PRIM subspaces across all parameter dimensions")
    ax.legend(handles=cluster_legend_handles(), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, "selected_box_parallel_coordinates")


def save_parameter_retention_outputs(limits: pd.DataFrame) -> None:
    """Summarise which native uncertainties are retained in the final PRIM rules."""
    restricted = limits[limits["restricted"]].copy()

    frequency_rows = []
    for parameter in UNCERTAINTY_COLS:
        clusters = restricted.loc[
            restricted["parameter"] == parameter, "cluster"
        ].astype(int).tolist()
        frequency_rows.append({
            "parameter": parameter,
            "parameter_name": UNCERTAINTY_NAMES[parameter],
            "n_clusters_restricted": len(clusters),
            "clusters": ", ".join(f"C{c}" for c in clusters),
        })

    frequency = pd.DataFrame(frequency_rows).sort_values(
        ["n_clusters_restricted", "parameter_name"],
        ascending=[False, True],
    )
    save_table(frequency, "selected_box_parameter_retention_frequency")

    matrix = pd.DataFrame(
        0,
        index=[f"C{c}" for c in range(1, FINAL_K + 1)],
        columns=[UNCERTAINTY_NAMES[p] for p in UNCERTAINTY_COLS],
        dtype=int,
    )
    for _, row in restricted.iterrows():
        matrix.loc[f"C{int(row['cluster'])}", row["parameter_name"]] = 1
    save_table(matrix, "selected_box_parameter_restriction_matrix", index=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Uncertain parameter")
    ax.set_ylabel("Behavioural regime")
    ax.set_title("Parameters retained in selected 80%-density PRIM rules")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "restricted" if matrix.iloc[i, j] else "–", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.8)
    fig.tight_layout()
    save_figure(fig, "selected_box_parameter_restriction_matrix")




# =============================================================================
# PRIM-derived phase subspaces and candidate tipping conditions
# =============================================================================


def _limit_row(
    limits: pd.DataFrame,
    cluster: int,
    parameter: str,
) -> pd.Series:
    row = limits[
        (limits["cluster"] == cluster)
        & (limits["parameter"] == parameter)
    ]
    if len(row) != 1:
        raise KeyError(
            f"Expected exactly one selected-box limit for C{cluster}, "
            f"parameter '{parameter}', found {len(row)}"
        )
    return row.iloc[0]


def _parameter_interval(
    limits: pd.DataFrame,
    cluster: int,
    parameter: str,
) -> tuple[float, float]:
    row = _limit_row(limits, cluster, parameter)
    return float(row["lower"]), float(row["upper"])


def _interval_relation(
    interval_a: tuple[float, float],
    interval_b: tuple[float, float],
    tol: float = 1e-10,
) -> dict:
    """Describe whether two one-dimensional PRIM intervals overlap or separate."""
    a_lo, a_hi = interval_a
    b_lo, b_hi = interval_b

    # A lies below B.
    if a_hi <= b_lo + tol:
        gap_lo = min(a_hi, b_lo)
        gap_hi = max(a_hi, b_lo)
        return {
            "relation": "separated",
            "low_side": "a",
            "high_side": "b",
            "gap_lower": gap_lo,
            "gap_upper": gap_hi,
            "gap_width": max(0.0, gap_hi - gap_lo),
            "overlap_lower": np.nan,
            "overlap_upper": np.nan,
        }

    # B lies below A.
    if b_hi <= a_lo + tol:
        gap_lo = min(b_hi, a_lo)
        gap_hi = max(b_hi, a_lo)
        return {
            "relation": "separated",
            "low_side": "b",
            "high_side": "a",
            "gap_lower": gap_lo,
            "gap_upper": gap_hi,
            "gap_width": max(0.0, gap_hi - gap_lo),
            "overlap_lower": np.nan,
            "overlap_upper": np.nan,
        }

    # Otherwise the intervals overlap.
    return {
        "relation": "overlap",
        "low_side": None,
        "high_side": None,
        "gap_lower": np.nan,
        "gap_upper": np.nan,
        "gap_width": 0.0,
        "overlap_lower": max(a_lo, b_lo),
        "overlap_upper": min(a_hi, b_hi),
    }


def analyse_projected_prim_pair(
    limits: pd.DataFrame,
    X_prim: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    x_col: str,
    y_col: str,
    subspace_id: str,
) -> dict:
    """Analyse a pair of selected PRIM boxes in a two-dimensional projection.

    A single-parameter tipping candidate is possible when the two
    projected boxes are separated along exactly one axis and overlap along the
    other axis. The gap between the two PRIM box edges is reported as a tipping
    interval rather than collapsed immediately to a single exact threshold.
    """
    x_a = _parameter_interval(limits, cluster_a, x_col)
    x_b = _parameter_interval(limits, cluster_b, x_col)
    y_a = _parameter_interval(limits, cluster_a, y_col)
    y_b = _parameter_interval(limits, cluster_b, y_col)

    x_relation = _interval_relation(x_a, x_b)
    y_relation = _interval_relation(y_a, y_b)

    row = {
        "subspace_id": subspace_id,
        "x_parameter": x_col,
        "x_parameter_name": UNCERTAINTY_NAMES[x_col],
        "y_parameter": y_col,
        "y_parameter_name": UNCERTAINTY_NAMES[y_col],
        "cluster_a": cluster_a,
        "cluster_a_name": CLUSTER_NAMES[cluster_a],
        "cluster_b": cluster_b,
        "cluster_b_name": CLUSTER_NAMES[cluster_b],
        "x_a_lower": x_a[0],
        "x_a_upper": x_a[1],
        "x_b_lower": x_b[0],
        "x_b_upper": x_b[1],
        "y_a_lower": y_a[0],
        "y_a_upper": y_a[1],
        "y_b_lower": y_b[0],
        "y_b_upper": y_b[1],
        "x_relation": x_relation["relation"],
        "y_relation": y_relation["relation"],
        "boundary_type": "",
        "boundary_parameter": "",
        "boundary_parameter_name": "",
        "boundary_lower": np.nan,
        "boundary_upper": np.nan,
        "boundary_midpoint": np.nan,
        "boundary_gap_fraction": np.nan,
        "conditioning_parameter": "",
        "conditioning_parameter_name": "",
        "conditioning_lower": np.nan,
        "conditioning_upper": np.nan,
        "low_side_cluster": np.nan,
        "high_side_cluster": np.nan,
        "direction": "",
        "candidate_status": "not_single_axis",
    }

    # Separated on x, overlapping on y -> conditional x tipping boundary.
    if (
        x_relation["relation"] == "separated"
        and y_relation["relation"] == "overlap"
    ):
        low_cluster = (
            cluster_a if x_relation["low_side"] == "a" else cluster_b
        )
        high_cluster = (
            cluster_b if x_relation["high_side"] == "b" else cluster_a
        )

        full_span = float(X_prim[x_col].max() - X_prim[x_col].min())
        gap_fraction = (
            x_relation["gap_width"] / full_span
            if full_span > 0
            else np.nan
        )

        row.update({
            "boundary_type": "single_axis_x",
            "boundary_parameter": x_col,
            "boundary_parameter_name": UNCERTAINTY_NAMES[x_col],
            "boundary_lower": x_relation["gap_lower"],
            "boundary_upper": x_relation["gap_upper"],
            "boundary_midpoint": (
                x_relation["gap_lower"] + x_relation["gap_upper"]
            ) / 2,
            "boundary_gap_fraction": gap_fraction,
            "conditioning_parameter": y_col,
            "conditioning_parameter_name": UNCERTAINTY_NAMES[y_col],
            "conditioning_lower": y_relation["overlap_lower"],
            "conditioning_upper": y_relation["overlap_upper"],
            "low_side_cluster": low_cluster,
            "high_side_cluster": high_cluster,
            "direction": (
                f"C{low_cluster} -> C{high_cluster} as "
                f"{UNCERTAINTY_NAMES[x_col]} increases"
            ),
            "candidate_status": (
                "candidate"
                if gap_fraction <= TIPPING_MAX_NORMALIZED_GAP
                else "wide_gap"
            ),
        })

    # Separated on y, overlapping on x -> conditional y tipping boundary.
    elif (
        y_relation["relation"] == "separated"
        and x_relation["relation"] == "overlap"
    ):
        low_cluster = (
            cluster_a if y_relation["low_side"] == "a" else cluster_b
        )
        high_cluster = (
            cluster_b if y_relation["high_side"] == "b" else cluster_a
        )

        full_span = float(X_prim[y_col].max() - X_prim[y_col].min())
        gap_fraction = (
            y_relation["gap_width"] / full_span
            if full_span > 0
            else np.nan
        )

        row.update({
            "boundary_type": "single_axis_y",
            "boundary_parameter": y_col,
            "boundary_parameter_name": UNCERTAINTY_NAMES[y_col],
            "boundary_lower": y_relation["gap_lower"],
            "boundary_upper": y_relation["gap_upper"],
            "boundary_midpoint": (
                y_relation["gap_lower"] + y_relation["gap_upper"]
            ) / 2,
            "boundary_gap_fraction": gap_fraction,
            "conditioning_parameter": x_col,
            "conditioning_parameter_name": UNCERTAINTY_NAMES[x_col],
            "conditioning_lower": x_relation["overlap_lower"],
            "conditioning_upper": x_relation["overlap_upper"],
            "low_side_cluster": low_cluster,
            "high_side_cluster": high_cluster,
            "direction": (
                f"C{low_cluster} -> C{high_cluster} as "
                f"{UNCERTAINTY_NAMES[y_col]} increases"
            ),
            "candidate_status": (
                "candidate"
                if gap_fraction <= TIPPING_MAX_NORMALIZED_GAP
                else "wide_gap"
            ),
        })

    elif (
        x_relation["relation"] == "separated"
        and y_relation["relation"] == "separated"
    ):
        row["boundary_type"] = "separated_both_axes"
        row["candidate_status"] = "no_single_parameter_boundary"

    else:
        row["boundary_type"] = "overlapping_projection"
        row["candidate_status"] = "overlapping_subspaces"

    return row


def compute_prim_subspace_diagnostics(
    limits: pd.DataFrame,
    X_prim: pd.DataFrame,
    subspaces: list[dict] = TIPPING_SUBSPACES,
) -> pd.DataFrame:
    """Evaluate all pairwise PRIM-box relations in the configured focal subspaces."""
    rows = []

    for subspace in subspaces:
        clusters = subspace["clusters"]
        for cluster_a, cluster_b in itertools.combinations(clusters, 2):
            rows.append(
                analyse_projected_prim_pair(
                    limits=limits,
                    X_prim=X_prim,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    x_col=subspace["x"],
                    y_col=subspace["y"],
                    subspace_id=subspace["id"],
                )
            )

    diagnostics = pd.DataFrame(rows)
    save_table(diagnostics, "prim_focal_subspace_boundary_diagnostics")
    return diagnostics


def compute_all_pairwise_prim_boundary_diagnostics(
    limits: pd.DataFrame,
    X_prim: pd.DataFrame,
) -> pd.DataFrame:
    """Appendix table for all 21 parameter pairs and all six cluster pairs."""
    rows = []

    for x_col, y_col in itertools.combinations(UNCERTAINTY_COLS, 2):
        subspace_id = f"{x_col}__vs__{y_col}"
        for cluster_a, cluster_b in itertools.combinations(
            range(1, FINAL_K + 1), 2
        ):
            rows.append(
                analyse_projected_prim_pair(
                    limits=limits,
                    X_prim=X_prim,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    x_col=x_col,
                    y_col=y_col,
                    subspace_id=subspace_id,
                )
            )

    diagnostics = pd.DataFrame(rows)
    save_table(diagnostics, "prim_all_pairwise_boundary_diagnostics")
    return diagnostics


def extract_candidate_tipping_points(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise, non-duplicated tipping-point table from PRIM boundaries.

    The source remains the selected PRIM boxes. If the same cluster pair and
    boundary parameter appear in multiple focal projections, the first
    configured subspace is retained to avoid describing the same boundary twice.
    """
    candidates = diagnostics[
        diagnostics["candidate_status"] == "candidate"
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=[
            "identifier",
            "from_cluster",
            "to_cluster",
            "transition",
            "boundary_parameter",
            "boundary_parameter_name",
            "boundary_lower",
            "boundary_upper",
            "boundary_midpoint",
            "boundary_gap_fraction",
            "conditioning_parameter",
            "conditioning_parameter_name",
            "conditioning_lower",
            "conditioning_upper",
            "subspace_id",
            "rule",
        ])

    subspace_order = {
        s["id"]: i for i, s in enumerate(TIPPING_SUBSPACES)
    }
    candidates["_subspace_order"] = (
        candidates["subspace_id"].map(subspace_order).fillna(999)
    )

    candidates["cluster_pair_key"] = candidates.apply(
        lambda r: "-".join(
            map(
                str,
                sorted([
                    int(r["low_side_cluster"]),
                    int(r["high_side_cluster"]),
                ]),
            )
        ),
        axis=1,
    )

    candidates = (
        candidates
        .sort_values(
            [
                "_subspace_order",
                "boundary_gap_fraction",
                "cluster_pair_key",
            ]
        )
        .drop_duplicates(
            subset=["cluster_pair_key", "boundary_parameter"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    rows = []

    for i, row in candidates.iterrows():
        low_cluster = int(row["low_side_cluster"])
        high_cluster = int(row["high_side_cluster"])
        identifier = f"PSSD-TP{i + 1}"

        if np.isclose(row["boundary_lower"], row["boundary_upper"]):
            boundary_text = f"~{row['boundary_midpoint']:.3f}"
        else:
            boundary_text = (
                f"~{row['boundary_lower']:.3f}-"
                f"{row['boundary_upper']:.3f}"
            )

        condition_text = (
            f"{row['conditioning_parameter_name']} "
            f"{row['conditioning_lower']:.3f}-"
            f"{row['conditioning_upper']:.3f}"
        )

        rule = (
            f"{row['boundary_parameter_name']} crosses {boundary_text}; "
            f"within the projected overlap where {condition_text}. "
            f"Additional PRIM restrictions in other dimensions may apply."
        )

        rows.append({
            "identifier": identifier,
            "from_cluster": low_cluster,
            "from_regime": CLUSTER_NAMES[low_cluster],
            "to_cluster": high_cluster,
            "to_regime": CLUSTER_NAMES[high_cluster],
            "transition": (
                f"C{low_cluster}: {CLUSTER_NAMES[low_cluster]} -> "
                f"C{high_cluster}: {CLUSTER_NAMES[high_cluster]}"
            ),
            "boundary_parameter": row["boundary_parameter"],
            "boundary_parameter_name": row["boundary_parameter_name"],
            "boundary_lower": float(row["boundary_lower"]),
            "boundary_upper": float(row["boundary_upper"]),
            "boundary_midpoint": float(row["boundary_midpoint"]),
            "boundary_gap_fraction": float(row["boundary_gap_fraction"]),
            "conditioning_parameter": row["conditioning_parameter"],
            "conditioning_parameter_name": row[
                "conditioning_parameter_name"
            ],
            "conditioning_lower": float(row["conditioning_lower"]),
            "conditioning_upper": float(row["conditioning_upper"]),
            "subspace_id": row["subspace_id"],
            "rule": rule,
        })

    tipping = pd.DataFrame(rows)
    save_table(tipping, "candidate_parameter_space_tipping_points")

    thesis_cols = [
        "identifier",
        "transition",
        "boundary_parameter_name",
        "boundary_lower",
        "boundary_upper",
        "conditioning_parameter_name",
        "conditioning_lower",
        "conditioning_upper",
        "rule",
    ]
    save_table(
        tipping[thesis_cols].copy(),
        "candidate_parameter_space_tipping_points_thesis_table",
    )

    return tipping


def _phase_subspace_limits(
    X_prim: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(X_prim[x_col].min())
    x_max = float(X_prim[x_col].max())
    y_min = float(X_prim[y_col].min())
    y_max = float(X_prim[y_col].max())

    x_pad = 0.025 * (x_max - x_min)
    y_pad = 0.025 * (y_max - y_min)

    return (
        (x_min - x_pad, x_max + x_pad),
        (y_min - y_pad, y_max + y_pad),
    )


def plot_prim_phase_subspace(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    limits: pd.DataFrame,
    subspace: dict,
    tipping_points: pd.DataFrame | None = None,
    show_samples: bool = False,
    annotate_tipping: bool = False,
) -> None:
    """Plot selected PRIM boxes as a phase-subspace diagram.

    The boxes are the phase representation. If show_samples=True, the sampled
    cluster assignments are added only as a diagnostic overlay.
    """
    x_col = subspace["x"]
    y_col = subspace["y"]
    clusters = subspace["clusters"]

    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    ax.set_box_aspect(0.8)

    # Optional sampled points: diagnostic only, never used to infer boundaries.
    if show_samples:
        for cluster in clusters:
            mask = labels == cluster
            ax.scatter(
                X_prim.loc[mask, x_col],
                X_prim.loc[mask, y_col],
                s=11,
                alpha=0.22,
                color=CLUSTER_COLORS[cluster],
                rasterized=True,
                zorder=1,
            )

    # Selected PRIM boxes define the phase subspaces.
    for cluster in clusters:
        x_min, x_max = _parameter_interval(limits, cluster, x_col)
        y_min, y_max = _parameter_interval(limits, cluster, y_col)

        ax.add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=CLUSTER_COLORS[cluster],
            edgecolor=CLUSTER_COLORS[cluster],
            alpha=0.22 if not show_samples else 0.12,
            linewidth=2.2,
            zorder=2,
        ))

        ax.add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=CLUSTER_COLORS[cluster],
            linewidth=2.0,
            zorder=3,
        ))

    # Annotate only near-adjacent, PRIM-derived candidate boundaries.
    if (
        annotate_tipping
        and tipping_points is not None
        and not tipping_points.empty
    ):
        sub_tipping = tipping_points[
            tipping_points["subspace_id"] == subspace["id"]
        ]

        for _, tp in sub_tipping.iterrows():
            if tp["boundary_parameter"] == x_col:
                x_lo = float(tp["boundary_lower"])
                x_hi = float(tp["boundary_upper"])
                y_lo = float(tp["conditioning_lower"])
                y_hi = float(tp["conditioning_upper"])

                if np.isclose(x_lo, x_hi):
                    ax.plot(
                        [x_lo, x_lo],
                        [y_lo, y_hi],
                        linestyle="--",
                        linewidth=2.0,
                        zorder=4,
                    )
                else:
                    ax.add_patch(Rectangle(
                        (x_lo, y_lo),
                        x_hi - x_lo,
                        y_hi - y_lo,
                        fill=False,
                        hatch="////",
                        linewidth=1.4,
                        zorder=4,
                    ))

                ax.annotate(
                    tp["identifier"],
                    (
                        float(tp["boundary_midpoint"]),
                        (y_lo + y_hi) / 2,
                    ),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    zorder=5,
                )

            elif tp["boundary_parameter"] == y_col:
                y_lo = float(tp["boundary_lower"])
                y_hi = float(tp["boundary_upper"])
                x_lo = float(tp["conditioning_lower"])
                x_hi = float(tp["conditioning_upper"])

                if np.isclose(y_lo, y_hi):
                    ax.plot(
                        [x_lo, x_hi],
                        [y_lo, y_lo],
                        linestyle="--",
                        linewidth=2.0,
                        zorder=4,
                    )
                else:
                    ax.add_patch(Rectangle(
                        (x_lo, y_lo),
                        x_hi - x_lo,
                        y_hi - y_lo,
                        fill=False,
                        hatch="////",
                        linewidth=1.4,
                        zorder=4,
                    ))

                ax.annotate(
                    tp["identifier"],
                    (
                        (x_lo + x_hi) / 2,
                        float(tp["boundary_midpoint"]),
                    ),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    zorder=5,
                )

    xlim, ylim = _phase_subspace_limits(X_prim, x_col, y_col)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(UNCERTAINTY_NAMES[x_col])
    ax.set_ylabel(UNCERTAINTY_NAMES[y_col])

    title = subspace["title"].replace("PRIM phase subspace: ", "")

    if show_samples:
        title = title.capitalize()

    ax.set_title(
        title,
        fontsize=11,
    )

    handles = [
        Rectangle(
            (0, 0), 1, 1,
            facecolor=CLUSTER_COLORS[c],
            edgecolor=CLUSTER_COLORS[c],
            alpha=0.25,
            label=f"C{c}: {CLUSTER_NAMES[c]}",
        )
        for c in clusters
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        borderaxespad=0,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.28,
        top=0.88,
    )

    suffix = ""
    if show_samples:
        suffix += "__with_samples"
    if annotate_tipping:
        suffix += "__annotated_tipping"

    save_figure(
        fig,
        f"prim_phase_subspace__{subspace['id']}{suffix}",
    )


def plot_all_prim_phase_subspaces(
    X_prim: pd.DataFrame,
    limits: pd.DataFrame,
) -> None:
    """Generate box-only phase diagrams for all 21 parameter pairs."""
    if not GENERATE_ALL_PRIM_PHASE_SUBSPACES:
        return

    for x_col, y_col in itertools.combinations(UNCERTAINTY_COLS, 2):
        subspace = {
            "id": f"{x_col}__vs__{y_col}",
            "x": x_col,
            "y": y_col,
            "clusters": list(range(1, FINAL_K + 1)),
            "title": (
                f"PRIM phase subspace: {UNCERTAINTY_NAMES[x_col]} vs. "
                f"{UNCERTAINTY_NAMES[y_col]}"
            ),
        }
        # No samples and no annotations: pure PRIM representation.
        plot_prim_phase_subspace(
            X_prim=X_prim,
            labels=np.zeros(len(X_prim), dtype=int),
            limits=limits,
            subspace=subspace,
            tipping_points=None,
            show_samples=False,
            annotate_tipping=False,
        )


def plot_candidate_tipping_transition(
    X_prim: pd.DataFrame,
    limits: pd.DataFrame,
    tipping_row: pd.Series,
) -> None:
    """Create one focused two-box figure for a candidate tipping condition."""
    subspace = next(
        s for s in TIPPING_SUBSPACES
        if s["id"] == tipping_row["subspace_id"]
    )

    x_col = subspace["x"]
    y_col = subspace["y"]
    from_cluster = int(tipping_row["from_cluster"])
    to_cluster = int(tipping_row["to_cluster"])

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for cluster in [from_cluster, to_cluster]:
        x_min, x_max = _parameter_interval(limits, cluster, x_col)
        y_min, y_max = _parameter_interval(limits, cluster, y_col)

        ax.add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=CLUSTER_COLORS[cluster],
            edgecolor=CLUSTER_COLORS[cluster],
            alpha=0.25,
            linewidth=2.2,
            label=f"C{cluster}: {CLUSTER_NAMES[cluster]}",
        ))

    boundary = tipping_row["boundary_parameter"]

    if boundary == x_col:
        lo = float(tipping_row["boundary_lower"])
        hi = float(tipping_row["boundary_upper"])
        cond_lo = float(tipping_row["conditioning_lower"])
        cond_hi = float(tipping_row["conditioning_upper"])

        if np.isclose(lo, hi):
            ax.plot(
                [lo, lo], [cond_lo, cond_hi],
                linestyle="--", linewidth=2.2,
            )
        else:
            ax.add_patch(Rectangle(
                (lo, cond_lo),
                hi - lo,
                cond_hi - cond_lo,
                fill=False,
                hatch="////",
                linewidth=1.5,
            ))

    elif boundary == y_col:
        lo = float(tipping_row["boundary_lower"])
        hi = float(tipping_row["boundary_upper"])
        cond_lo = float(tipping_row["conditioning_lower"])
        cond_hi = float(tipping_row["conditioning_upper"])

        if np.isclose(lo, hi):
            ax.plot(
                [cond_lo, cond_hi], [lo, lo],
                linestyle="--", linewidth=2.2,
            )
        else:
            ax.add_patch(Rectangle(
                (cond_lo, lo),
                cond_hi - cond_lo,
                hi - lo,
                fill=False,
                hatch="////",
                linewidth=1.5,
            ))

    xlim, ylim = _phase_subspace_limits(X_prim, x_col, y_col)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(UNCERTAINTY_NAMES[x_col])
    ax.set_ylabel(UNCERTAINTY_NAMES[y_col])
    ax.set_title(
        f"{tipping_row['identifier']}: PRIM-derived candidate tipping condition"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    save_figure(
        fig,
        f"candidate_tipping_transition__{tipping_row['identifier']}",
    )


def save_prim_derived_tipping_outputs(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    limits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Derive tipping diagnostics and save the final PRIM phase-subspace plots.
    diagnostics = compute_prim_subspace_diagnostics(limits, X_prim)
    all_diagnostics = compute_all_pairwise_prim_boundary_diagnostics(
        limits,
        X_prim,
    )
    tipping_points = extract_candidate_tipping_points(diagnostics)

    for configured_subspace in TIPPING_SUBSPACES:
        subspace = dict(configured_subspace)
        subspace["clusters"] = list(range(1, FINAL_K + 1))

        plot_prim_phase_subspace(
            X_prim=X_prim,
            labels=labels,
            limits=limits,
            subspace=subspace,
            tipping_points=None,
            show_samples=True,
            annotate_tipping=False,
        )

    return tipping_points, all_diagnostics


# =============================================================================
# Overlap, separability and box composition
# =============================================================================


def compute_box_membership_analysis(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    prim_results: dict[int, dict],
) -> dict[str, pd.DataFrame | dict[int, np.ndarray]]:
    box_membership: dict[int, np.ndarray] = {}
    validation_rows = []

    for cluster in range(1, FINAL_K + 1):
        box_lims = selected_box_limits(prim_results, cluster, X_prim)
        inside = scenarios_inside_box(X_prim, box_lims)
        target = labels == cluster
        box_membership[cluster] = inside

        candidate = prim_results[cluster]["candidates"].loc[
            prim_results[cluster]["candidates"]["candidate"] == "density_80"
        ].iloc[0]

        validation_rows.append({
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES[cluster],
            "box_size": int(inside.sum()),
            "computed_coverage": float((inside & target).sum() / target.sum()),
            "PRIM_coverage": float(candidate["coverage"]),
            "computed_density": float(target[inside].mean()) if inside.any() else np.nan,
            "PRIM_density": float(candidate["density"]),
        })

    validation = pd.DataFrame(validation_rows)

    labels_short = [f"C{c}" for c in range(1, FINAL_K + 1)]
    shared_counts = pd.DataFrame(0, index=labels_short, columns=labels_short, dtype=int)
    jaccard = pd.DataFrame(0.0, index=labels_short, columns=labels_short)

    for i in range(1, FINAL_K + 1):
        for j in range(1, FINAL_K + 1):
            a = box_membership[i]
            b = box_membership[j]
            intersection = int((a & b).sum())
            union = int((a | b).sum())
            shared_counts.loc[f"C{i}", f"C{j}"] = intersection
            jaccard.loc[f"C{i}", f"C{j}"] = intersection / union if union else np.nan

    # Composition: of scenarios inside each box, what fraction belongs to each true cluster?
    composition_counts = pd.DataFrame(0, index=labels_short, columns=labels_short, dtype=int)
    composition_share = pd.DataFrame(0.0, index=labels_short, columns=labels_short)
    coverage_by_box = pd.DataFrame(0.0, index=labels_short, columns=labels_short)

    for box_cluster in range(1, FINAL_K + 1):
        inside = box_membership[box_cluster]
        box_label = f"C{box_cluster}"
        for true_cluster in range(1, FINAL_K + 1):
            true = labels == true_cluster
            count = int((inside & true).sum())
            composition_counts.loc[box_label, f"C{true_cluster}"] = count
            composition_share.loc[box_label, f"C{true_cluster}"] = (
                count / inside.sum() if inside.sum() else np.nan
            )
            coverage_by_box.loc[box_label, f"C{true_cluster}"] = count / true.sum()

    membership_count = np.sum(
        np.column_stack([box_membership[c] for c in range(1, FINAL_K + 1)]),
        axis=1,
    )
    membership_distribution = (
        pd.Series(membership_count)
        .value_counts()
        .sort_index()
        .rename_axis("number_of_selected_boxes")
        .reset_index(name="n_scenarios")
    )
    membership_distribution["fraction"] = membership_distribution["n_scenarios"] / len(labels)

    return {
        "box_membership": box_membership,
        "validation": validation,
        "shared_counts": shared_counts,
        "jaccard": jaccard,
        "composition_counts": composition_counts,
        "composition_share": composition_share,
        "coverage_by_box": coverage_by_box,
        "membership_distribution": membership_distribution,
    }



def plot_matrix_heatmap(
    matrix: pd.DataFrame,
    stem: str,
    title: str,
    fmt: str,
    xlabel: str = "PRIM box",
    ylabel: str = "PRIM box",
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix.to_numpy())
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center")

    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    save_figure(fig, stem)



def save_overlap_outputs(analysis: dict) -> None:
    """Save overlap/composition tables and one concise overlap figure."""
    save_table(
        analysis["validation"],
        "selected_box_validation",
    )
    save_table(
        analysis["shared_counts"],
        "prim_box_overlap_shared_scenarios",
        index=True,
    )
    save_table(
        analysis["jaccard"],
        "prim_box_overlap_jaccard",
        index=True,
    )
    save_table(
        analysis["composition_counts"],
        "prim_box_true_cluster_composition_counts",
        index=True,
    )
    save_table(
        analysis["composition_share"],
        "prim_box_true_cluster_composition_share",
        index=True,
    )
    save_table(
        analysis["coverage_by_box"],
        "prim_box_coverage_by_true_cluster",
        index=True,
    )
    save_table(
        analysis["membership_distribution"],
        "selected_box_membership_distribution",
    )

    # One full-dimensional overlap plot is sufficient for the appendix.
    plot_matrix_heatmap(
        analysis["jaccard"],
        "prim_box_overlap_jaccard",
        "Jaccard overlap between selected PRIM boxes",
        ".2f",
    )


# =============================================================================
# Peel-alpha robustness
# =============================================================================


def run_peel_alpha_robustness(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows = []
    bounds_rows = []
    results: dict[float, dict[int, dict]] = {}

    full_min = X_prim.min()
    full_max = X_prim.max()

    for peel_alpha in PEEL_ALPHAS:
        results[peel_alpha] = {}

        for cluster in range(1, FINAL_K + 1):
            y = (labels == cluster).astype(int)
            algorithm = make_prim(
                X_prim.copy(), y,
                peel_alpha=peel_alpha,
                mass_min=MASS_MIN,
            )
            box = algorithm.find_box()
            traj = trajectory_with_ids(box)
            selected = select_density_candidate(traj)
            box_id = int(selected["id"])
            box_lims = get_box_limits(box, box_id)

            results[peel_alpha][cluster] = {
                "box": box,
                "trajectory": traj,
                "selected": selected,
                "box_id": box_id,
                "box_lims": box_lims,
            }

            summary_rows.append({
                "peel_alpha": peel_alpha,
                "cluster": cluster,
                "cluster_name": CLUSTER_NAMES[cluster],
                "box_id": box_id,
                "target_reached": bool(selected["target_reached"]),
                "coverage": float(selected["coverage"]),
                "density": float(selected["density"]),
                "mass": float(selected["mass"]),
                "restricted_dimensions": int(selected["res_dim"]),
            })

            for parameter in UNCERTAINTY_COLS:
                lower, upper = limits_for_dimension(box_lims, parameter, X_prim)
                tol = 1e-10
                restricted = (
                    lower > float(full_min[parameter]) + tol
                    or upper < float(full_max[parameter]) - tol
                )
                bounds_rows.append({
                    "peel_alpha": peel_alpha,
                    "cluster": cluster,
                    "cluster_name": CLUSTER_NAMES[cluster],
                    "parameter": parameter,
                    "parameter_name": UNCERTAINTY_NAMES[parameter],
                    "lower": lower,
                    "upper": upper,
                    "full_lower": float(full_min[parameter]),
                    "full_upper": float(full_max[parameter]),
                    "restricted": restricted,
                })

    summary = pd.DataFrame(summary_rows)
    bounds = pd.DataFrame(bounds_rows)
    save_table(summary, "peel_alpha_robustness_summary")
    save_table(bounds, "peel_alpha_robustness_bounds")
    return summary, bounds, results



def plot_peel_alpha_robustness(summary: pd.DataFrame) -> None:
    metrics = [
        ("coverage", "Coverage", (0, 1.02)),
        ("density", "Density", (0, 1.02)),
        ("restricted_dimensions", "Restricted dimensions", (0, len(UNCERTAINTY_COLS) + 0.5)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (metric, label, ylim) in zip(axes, metrics):
        for cluster in range(1, FINAL_K + 1):
            sub = summary[summary["cluster"] == cluster].sort_values("peel_alpha")
            ax.plot(
                sub["peel_alpha"], sub[metric], marker="o",
                color=CLUSTER_COLORS[cluster], label=f"C{cluster}",
            )
        ax.set_xlabel("Peel alpha")
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.set_xticks(PEEL_ALPHAS)
        if metric == "density":
            ax.axhline(DENSITY_TARGET, linestyle="--", linewidth=1)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensitivity of selected PRIM boxes to peel alpha", y=1.02)
    fig.tight_layout()
    save_figure(fig, "peel_alpha_robustness_metrics")



def plot_peel_alpha_bound_stability(bounds: pd.DataFrame) -> None:
    """One appendix figure per cluster showing normalised selected bounds by alpha."""
    for cluster in range(1, FINAL_K + 1):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        y_positions = np.arange(len(UNCERTAINTY_COLS))
        offsets = np.linspace(-0.18, 0.18, len(PEEL_ALPHAS))

        for alpha, offset in zip(PEEL_ALPHAS, offsets):
            sub = bounds[
                (bounds["cluster"] == cluster)
                & (np.isclose(bounds["peel_alpha"], alpha))
            ].set_index("parameter")

            for y, parameter in zip(y_positions, UNCERTAINTY_COLS):
                row = sub.loc[parameter]
                span = row["full_upper"] - row["full_lower"]
                lo = (row["lower"] - row["full_lower"]) / span
                hi = (row["upper"] - row["full_lower"]) / span
                ax.hlines(y + offset, lo, hi, linewidth=2, label=None)
                ax.plot([lo, hi], [y + offset, y + offset], marker="|", linestyle="", markersize=7)

        handles = [
            Line2D([0], [0], linewidth=2, label=f"alpha={a}")
            for a in PEEL_ALPHAS
        ]
        ax.set_yticks(y_positions, [UNCERTAINTY_NAMES[p] for p in UNCERTAINTY_COLS])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Selected interval within sampled parameter range (normalised)")
        ax.set_title(f"Peel-alpha sensitivity of PRIM bounds — C{cluster}")
        ax.invert_yaxis()
        ax.legend(handles=handles, frameon=False, fontsize=8)
        fig.tight_layout()
        save_figure(fig, f"peel_alpha_bound_stability_cluster_{cluster}")


# =============================================================================
# PCA-PRIM / CPCA-PRIM sensitivity diagnostic
# =============================================================================


def selected_rotated_restrictions(result: dict) -> pd.DataFrame:
    box = result["box"]
    selected = result["candidate"]
    X_rotated = result["X"]
    box_id = int(selected["id"])
    box_lims = get_box_limits(box, box_id)

    rows = []
    for dimension in X_rotated.columns:
        lower, upper = limits_for_dimension(box_lims, dimension, X_rotated)
        full_lower = float(np.real(X_rotated[dimension].min()))
        full_upper = float(np.real(X_rotated[dimension].max()))
        restricted = (
            lower > full_lower + 1e-10
            or upper < full_upper - 1e-10
        )
        if restricted:
            rows.append({
                "dimension": dimension,
                "lower": lower,
                "upper": upper,
                "full_lower": full_lower,
                "full_upper": full_upper,
            })
    return pd.DataFrame(rows)



def run_rotation_sensitivity(
    X_prim: pd.DataFrame,
    labels: np.ndarray,
    prim_results: dict[int, dict],
) -> pd.DataFrame:
    rows = []

    for cluster in range(1, FINAL_K + 1):
        y = (labels == cluster).astype(int)

        # Standard result.
        standard = prim_results[cluster]
        candidate = standard["candidates"].loc[
            standard["candidates"]["candidate"] == "density_80"
        ].iloc[0]
        rows.append({
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES[cluster],
            "method": "Standard PRIM",
            "target_reached": bool(candidate["target_reached"]),
            "coverage": float(candidate["coverage"]),
            "density": float(candidate["density"]),
            "mass": float(candidate["mass"]),
            "restricted_dimensions": int(candidate["res_dim"]),
        })

        for method, subsets in [("PCA-PRIM", None), ("CPCA-PRIM", CPCA_SUBSETS)]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if subsets is None:
                        X_rotated, rotation = prim.pca_preprocess(X_prim.copy(), y)
                    else:
                        X_rotated, rotation = prim.pca_preprocess(
                            X_prim.copy(), y, subsets=subsets
                        )

                    algorithm = make_prim(X_rotated.copy(), y)
                    box = algorithm.find_box()
                    traj = trajectory_with_ids(box)
                    selected = select_density_candidate(traj)

                warning_text = " | ".join(str(w.message) for w in caught)
                result = {
                    "box": box,
                    "candidate": selected,
                    "X": X_rotated,
                    "rotation_matrix": rotation,
                }

                restrictions = selected_rotated_restrictions(result)
                save_table(
                    restrictions,
                    f"rotation_{method.lower().replace('-', '_')}_restrictions_cluster_{cluster}",
                )
                rotation_df = pd.DataFrame(rotation)
                save_table(
                    rotation_df,
                    f"rotation_{method.lower().replace('-', '_')}_matrix_cluster_{cluster}",
                    index=True,
                )

                rows.append({
                    "cluster": cluster,
                    "cluster_name": CLUSTER_NAMES[cluster],
                    "method": method,
                    "target_reached": bool(selected["target_reached"]),
                    "coverage": float(selected["coverage"]),
                    "density": float(selected["density"]),
                    "mass": float(selected["mass"]),
                    "restricted_dimensions": int(selected["res_dim"]),
                    "warnings": warning_text,
                })

            except Exception as err:
                rows.append({
                    "cluster": cluster,
                    "cluster_name": CLUSTER_NAMES[cluster],
                    "method": method,
                    "target_reached": False,
                    "coverage": np.nan,
                    "density": np.nan,
                    "mass": np.nan,
                    "restricted_dimensions": np.nan,
                    "warnings": f"FAILED: {err}",
                })

    comparison = pd.DataFrame(rows)
    save_table(comparison, "rotation_method_comparison")
    return comparison



def plot_rotation_comparison(comparison: pd.DataFrame) -> None:
    methods = ["Standard PRIM", "PCA-PRIM", "CPCA-PRIM"]
    clusters = range(1, FINAL_K + 1)
    x = np.arange(FINAL_K)
    width = 0.24

    for metric, ylabel, stem, ylim in [
        ("coverage", "Coverage", "rotation_method_coverage_comparison", (0, 1.05)),
        ("density", "Density", "rotation_method_density_comparison", (0, 1.05)),
        (
            "restricted_dimensions",
            "Restricted dimensions",
            "rotation_method_dimensions_comparison",
            (0, len(UNCERTAINTY_COLS) + 0.5),
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for i, method in enumerate(methods):
            sub = (
                comparison[comparison["method"] == method]
                .set_index("cluster")
                .reindex(clusters)
            )
            ax.bar(x + (i - 1) * width, sub[metric], width, label=method)

        ax.set_xticks(x, [f"C{c}" for c in clusters])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"Standard vs rotated PRIM: {ylabel.lower()}")
        ax.legend(frameon=False, fontsize=8)
        if metric == "density":
            ax.axhline(DENSITY_TARGET, linestyle="--", linewidth=1)
        fig.tight_layout()
        save_figure(fig, stem)


# =============================================================================
# Output manifest
# =============================================================================


def write_output_manifest() -> None:
    """Document the final thesis-oriented output set."""
    rows = [
        (
            "prim_tradeoffs_all_clusters",
            "figure",
            "main/appendix",
            "Demonstrates the coverage-density trade-off and selected-box quality for all four regimes.",
        ),
        (
            "selected_box_normalized_intervals",
            "figure",
            "main/appendix",
            "Shows the position and width of the selected PRIM rules across all seven uncertainties.",
        ),
        (
            "selected_box_parameter_restriction_matrix",
            "figure",
            "appendix",
            "Shows which uncertain parameters are restricted in each behavioural regime.",
        ),
        (
            "sample_scatter__broad_adaptation",
            "figure",
            "main",
            "Clean observed mapping of all 2,000 scenarios across initial adaptation and social-norm midpoint.",
        ),
        (
            "sample_scatter__high_adaptation_distribution",
            "figure",
            "appendix/main candidate",
            "Observed regime geometry in the savings-rate/adaptation-cost subspace.",
        ),
        (
            "sample_scatter__social_feedback",
            "figure",
            "appendix/main candidate",
            "Observed regime geometry in the social-norm midpoint/social-weight subspace.",
        ),
        (
            "prim_phase_subspace__broad_adaptation__with_samples",
            "figure",
            "main",
            "All four PRIM regions over the sampled scenarios in the main tipping-condition subspace.",
        ),
        (
            "prim_phase_subspace__high_adaptation_distribution__with_samples",
            "figure",
            "main/appendix",
            "All four PRIM regions over the sampled scenarios in the financial subspace.",
        ),
        (
            "prim_phase_subspace__social_feedback__with_samples",
            "figure",
            "main/appendix",
            "All four PRIM regions over the sampled scenarios in the social-feedback subspace.",
        ),
        (
            "prim_box_overlap_jaccard",
            "figure",
            "appendix",
            "Tests whether the selected PRIM phase regions remain distinct in the complete seven-dimensional space.",
        ),
        (
            "peel_alpha_robustness_metrics",
            "figure",
            "appendix",
            "Tests sensitivity of coverage, density, and dimensionality to the PRIM peeling fraction.",
        ),
        (
            "candidate_parameter_space_tipping_points",
            "table",
            "main/support",
            "Candidate tipping intervals derived from near-adjacent PRIM-box boundaries for interpretation in the text.",
        ),
        (
            "prim_focal_subspace_boundary_diagnostics",
            "table",
            "main/support",
            "Complete boundary diagnostics for all cluster pairs in the three focal subspaces.",
        ),
        (
            "selected_density80_restricted_rules",
            "table",
            "main",
            "Exact native-parameter restrictions defining the selected PRIM regions.",
        ),
        (
            "selected_density80_summary",
            "table",
            "main",
            "Coverage, density, mass, dimensionality, and lift of the selected boxes.",
        ),
        (
            "rotation_method_comparison",
            "table",
            "appendix",
            "Standard PRIM versus PCA-PRIM and CPCA-PRIM sensitivity check.",
        ),
    ]

    manifest = pd.DataFrame(
        rows,
        columns=["output", "type", "suggested_use", "description"],
    )
    save_table(manifest, "output_manifest")


# =============================================================================
# Main execution
# =============================================================================


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Results dir: ", RESULTS_DIR)
    print("Output dir:  ", OUTPUT_DIR)

    scenario_experiments, mean_outcomes = load_ensemble()
    if len(scenario_experiments) != 2000:
        raise RuntimeError(
            f"Expected 2,000 uncertainty scenarios, found {len(scenario_experiments)}"
        )

    scenario_experiments, labels = reproduce_final_clustering(
        scenario_experiments,
        mean_outcomes,
    )

    X_prim = scenario_experiments[UNCERTAINTY_COLS].copy()
    if X_prim.isna().any().any():
        raise ValueError("PRIM input contains missing values")

    # ------------------------------------------------------------------
    # Core data and standard PRIM
    # ------------------------------------------------------------------
    assignments = scenario_experiments[
        UNCERTAINTY_COLS + ["scenario_id", "cluster", "cluster_name"]
    ].copy()
    save_table(assignments, "final_cluster_assignments")
    save_table(
        X_prim.describe().T.reset_index(names="parameter"),
        "parameter_space_descriptives",
    )

    print("\nRunning standard PRIM...")
    prim_results, candidate_summary = run_standard_prim(X_prim, labels)

    save_table(
        candidate_summary[
            candidate_summary["candidate"] == "density_80"
        ].copy(),
        "selected_density80_summary",
    )
    save_table(
        candidate_summary[
            candidate_summary["candidate"] == "coverage_80"
        ].copy(),
        "selected_coverage80_summary",
    )

    limits = rule_limits_table(
        prim_results,
        X_prim,
        candidate="density_80",
    )
    save_table(limits, "selected_density80_box_limits")
    save_table(
        limits[limits["restricted"]].copy(),
        "selected_density80_restricted_rules",
    )
    save_table(
        build_rule_text(limits),
        "selected_density80_rules_text",
    )

    # Alternative high-coverage rules remain available as tables only.
    coverage_limits = rule_limits_table(
        prim_results,
        X_prim,
        candidate="coverage_80",
    )
    save_table(
        coverage_limits,
        "selected_coverage80_box_limits",
    )
    save_table(
        build_rule_text(coverage_limits),
        "selected_coverage80_rules_text",
    )

    # ------------------------------------------------------------------
    # Evidence 1: quality and native-parameter structure of PRIM boxes
    # ------------------------------------------------------------------
    print("Generating PRIM quality and rule-structure figures...")
    plot_tradeoff_combined(prim_results)
    plot_normalized_rule_intervals(limits)
    plot_normalized_rule_intervals_filtered(limits, min_restricted_clusters=3)
    plot_prim_subspace_grid(
        limits,
        min_restricted_clusters=3,
    )
    save_parameter_retention_outputs(limits)

    # ------------------------------------------------------------------
    # Evidence 2: observed parameter-space mapping before PRIM overlays
    # ------------------------------------------------------------------
    print("Generating focal sample-only parameter-space plots...")
    plot_focal_sample_scatters(X_prim, labels)

    # ------------------------------------------------------------------
    # Evidence 3: PRIM-derived phase subspaces used for tipping analysis
    # ------------------------------------------------------------------
    print("Generating PRIM phase-subspace plots and tipping diagnostics...")
    tipping_points, _ = save_prim_derived_tipping_outputs(
        X_prim=X_prim,
        labels=labels,
        limits=limits,
    )
    print(
        f"Identified {len(tipping_points)} near-adjacent "
        "PRIM-derived candidate tipping condition(s)."
    )
    print(
        "Candidate tipping conditions are saved to tables and should be "
        "discussed in the Results text; they are not annotated on figures."
    )

    # ------------------------------------------------------------------
    # Evidence 4: full-dimensional separability
    # ------------------------------------------------------------------
    print("Analysing selected-box overlap and separability...")
    overlap = compute_box_membership_analysis(
        X_prim,
        labels,
        prim_results,
    )
    save_overlap_outputs(overlap)

    # ------------------------------------------------------------------
    # Evidence 5: robustness of PRIM settings
    # ------------------------------------------------------------------
    print("Running peel-alpha robustness analysis...")
    robustness_summary, _, _ = run_peel_alpha_robustness(
        X_prim,
        labels,
    )
    plot_peel_alpha_robustness(robustness_summary)

    # PCA/CPCA remains a table-only diagnostic because rotated spaces did not
    # materially simplify the result and are harder to interpret substantively.
    print("Running PCA/CPCA PRIM sensitivity diagnostic...")
    run_rotation_sensitivity(
        X_prim,
        labels,
        prim_results,
    )

    write_output_manifest()

    print("\nDone. Final parameter-space outputs written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
