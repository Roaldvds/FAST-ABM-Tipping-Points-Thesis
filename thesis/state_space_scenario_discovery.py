"""
Final plotting and analysis script for observable-state state-space scenario discovery.

Primary methodological choices
------------------------------
1. State space = potentially policy-monitorable endogenous state levels x(t).
2. Recent one-step change x(t)-x(t-1) is a secondary sensitivity.
3. A no-current-uptake specification tests whether discrimination is merely a
   restatement of the adaptation trajectory used in clustering.
4. Primary representative PRIM box:
      highest density subject to coverage >= 0.80.
5. Sensitivity representative box:
      maximum coverage-density F1.
6. Two-dimensional PRIM projections use cluster-coloured boundaries.
   If a plotted dimension is unrestricted, only the meaningful boundary is
   drawn; if neither plotted dimension is restricted, no rectangle is drawn.

Outputs
-------
Figures are written to:

    thesis/results/plots/state space sd

Main-text, appendix and CSV outputs are distinguished by filename prefixes.
"""

from __future__ import annotations

import inspect
import warnings
from itertools import combinations
from pathlib import Path

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
# 1. CONFIGURATION
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message="ipyparallel not installed*",
    category=UserWarning,
)

ema_logging.log_to_stderr(ema_logging.INFO)

RESULT_GLOB = "ema_fast_results_seed_*.tar.gz"

UNCERTAINTY_COLS = [
    "initial_adaptation",
    "per_HH_adapted_to_SN_prob_midpoint",
    "weight_SN_vs_PMT",
    "savings_rate_multiplier",
    "adaptation_cost_multiplier",
    "measure_lifetime",
    "flood_depth_multiplier",
]

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

FLOOD_TIMESTEP = 10
N_TIMESTEPS = 50

PEEL_ALPHA = 0.05
MASS_MIN = 0.05
MIN_STD = 1e-10

OBSERVABLE_STATE_VARS = [
    "fraction_dry_proofed",
    "fraction_wet_proofed",
    "average_savings",
    "average_relative_burden_cumulative_income",
    "unmet_adaptation_demand",
    "average_damage_experienced",
    "share_households_with_expiry",
    "average_active_measure_age",
]

OBSERVABLE_STATE_VARS_NO_UPTAKE = [
    v
    for v in OBSERVABLE_STATE_VARS
    if v not in {"fraction_dry_proofed", "fraction_wet_proofed"}
]

STATE_LABELS = {
    "fraction_dry_proofed": "Share dry-proofed",
    "fraction_wet_proofed": "Share wet-proofed",
    "average_savings": "Average savings",
    "average_relative_burden_cumulative_income": "Relative burden",
    "unmet_adaptation_demand": "Unmet adaptation demand",
    "average_damage_experienced": "Average damage experienced",
    "share_households_with_expiry": "Share with expiring measure",
    "average_active_measure_age": "Average active measure age",
}

PRIMARY_SPEC = "observable_levels"
RECENT_CHANGE_SPEC = "observable_recent_change_1"
NO_UPTAKE_SPEC = "observable_levels_no_uptake"

ANALYSIS_SPECS = {
    PRIMARY_SPEC: {
        "variables": OBSERVABLE_STATE_VARS,
        "representation": "levels",
        "lag": None,
    },
    RECENT_CHANGE_SPEC: {
        "variables": OBSERVABLE_STATE_VARS,
        "representation": "recent_change",
        "lag": 1,
    },
    NO_UPTAKE_SPEC: {
        "variables": OBSERVABLE_STATE_VARS_NO_UPTAKE,
        "representation": "levels",
        "lag": None,
    },
}

PRIMARY_SELECTION = "coverage80_max_density"
SENSITIVITY_SELECTION = "max_f1"
SELECTION_STRATEGIES = [
    PRIMARY_SELECTION,
    SENSITIVITY_SELECTION,
]

KEY_TIMESTEPS = [1, 5, 9, 10, 11, 15, 20]
MAIN_SUBSPACE_TIMESTEPS = [9, 10, 11]
APPENDIX_CHECK_TIMESTEPS = [5]
PROJECTION_AXIS_TIMESTEPS = [5, 9, 10, 11]

SAVE_PNG = True
SAVE_PDF = False
DPI = 300

# Use matplotlib's standard categorical palette consistently.
_cmap = plt.get_cmap("tab10")
CLUSTER_COLORS = {
    c: _cmap(c - 1)
    for c in range(1, FINAL_K + 1)
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
    }
)


# =============================================================================
# 2. PATHS AND SAVING
# =============================================================================

def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()

    for candidate in [start] + list(start.parents):
        if (candidate / "results" / "delftblue2").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the FAST project root containing results/delftblue2."
    )


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "thesis"
    / "results"
    / "plots"
    / "state space sd"
)

# Keep every final SSSD figure and CSV in one directory. Filename prefixes distinguish main-text,
# appendix, and table outputs.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DIR = OUTPUT_DIR
APPENDIX_DIR = OUTPUT_DIR
TABLE_DIR = OUTPUT_DIR


def save_figure(fig, folder, stem):
    """Save a figure in thesis-friendly raster and vector formats."""
    if SAVE_PNG:
        fig.savefig(
            folder / f"{stem}.png",
            dpi=DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        fig.savefig(
            folder / f"{stem}.pdf",
            bbox_inches="tight",
        )


def panel_layout(fig, title, top=0.86):
    """
    Reserve explicit vertical space for the overall title and figure legend.
    This avoids the title/legend overlap seen in the notebook.
    """
    fig.suptitle(title, y=0.985)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=top,
        hspace=0.35,
        wspace=0.25,
    )


# =============================================================================
# 3. LOAD AND AVERAGE STOCHASTIC REPLICATIONS
# =============================================================================

def make_scenario_ids(exp):
    return (
        exp[UNCERTAINTY_COLS]
        .round(10)
        .astype(str)
        .agg("_".join, axis=1)
        .to_numpy()
    )


def load_ensemble_incremental():
    files = sorted(RESULTS_DIR.glob(RESULT_GLOB))

    if not files:
        raise FileNotFoundError(
            f"No files matching {RESULT_GLOB} found in {RESULTS_DIR}"
        )

    scenario_experiments = None
    reference_ids = None
    sums = {}
    counts = {}

    print(f"Found {len(files)} stochastic-replication files")

    for i, file in enumerate(files, 1):
        exp, out = load_results(file)
        ids = make_scenario_ids(exp)

        if scenario_experiments is None:
            scenario_experiments = exp.copy().reset_index(drop=True)
            scenario_experiments["scenario_id"] = ids
            reference_ids = ids.copy()

        elif not np.array_equal(ids, reference_ids):
            raise RuntimeError(
                f"Scenario ordering differs in {file.name}; "
                "replications cannot safely be averaged."
            )

        for name, data in out.items():
            arr = np.asarray(data)

            if arr.ndim != 2 or arr.shape[0] != len(exp):
                continue

            arr = np.asarray(arr, dtype=float)

            if name not in sums:
                sums[name] = np.zeros_like(arr, dtype=float)
                counts[name] = 0

            if sums[name].shape == arr.shape:
                sums[name] += arr
                counts[name] += 1

        if i == 1 or i % 5 == 0 or i == len(files):
            print(f"Accumulated {i}/{len(files)} replications")

    means = {
        name: total / counts[name]
        for name, total in sums.items()
        if counts[name] == len(files)
    }

    return scenario_experiments, means


# =============================================================================
# 4. FINAL BEHAVIOURAL CLUSTERING
# =============================================================================

def reproduce_final_clustering(mean_outcomes):
    missing = [v for v in FINAL_OUTCOMES if v not in mean_outcomes]

    if missing:
        raise KeyError(f"Missing clustering outcomes: {missing}")

    X_cluster = np.stack(
        [mean_outcomes[v] for v in FINAL_OUTCOMES],
        axis=2,
    )

    X_scaled = np.empty_like(X_cluster, dtype=float)

    for j, var in enumerate(FINAL_OUTCOMES):
        scaler = StandardScaler()
        flat = scaler.fit_transform(
            mean_outcomes[var].reshape(-1, 1)
        )
        X_scaled[:, :, j] = flat.reshape(
            mean_outcomes[var].shape
        )

    model = TimeSeriesKMeans(
        n_clusters=FINAL_K,
        metric="dtw",
        random_state=CLUSTER_RANDOM_STATE,
    )

    labels = model.fit_predict(X_scaled) + 1

    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    found = [
        int(counts.get(c, 0))
        for c in range(1, FINAL_K + 1)
    ]

    if found != EXPECTED_CLUSTER_COUNTS:
        raise RuntimeError(
            "Final clustering was not reproduced. "
            f"Found {found}, expected {EXPECTED_CLUSTER_COUNTS}."
        )

    print("Final thesis clustering reproduced exactly:", found)
    return labels


# =============================================================================
# 5. STATE MATRICES
# =============================================================================

def validate_state_vars(mean_outcomes):
    missing = [
        v
        for v in OBSERVABLE_STATE_VARS
        if v not in mean_outcomes
    ]

    if missing:
        raise KeyError(
            "Missing observable state variables: "
            + ", ".join(missing)
        )

    rows = []

    for var in OBSERVABLE_STATE_VARS:
        arr = np.asarray(mean_outcomes[var], dtype=float)
        std_t = np.nanstd(arr, axis=0)

        rows.append(
            {
                "state_variable": var,
                "state_label": STATE_LABELS[var],
                "finite": bool(np.isfinite(arr).all()),
                "std_t0": float(std_t[0]),
                "std_t1": float(std_t[1]),
                "std_t5": float(std_t[5]),
                "std_t10": float(std_t[10]),
                "std_t20": float(std_t[20]),
                "std_final": float(std_t[-1]),
            }
        )

    return pd.DataFrame(rows)


def state_matrix(mean_outcomes, timestep, spec_name):
    spec = ANALYSIS_SPECS[spec_name]
    variables = spec["variables"]
    representation = spec["representation"]
    lag = spec["lag"]

    data = {}

    for var in variables:
        current = np.asarray(
            mean_outcomes[var][:, timestep],
            dtype=float,
        )

        if representation == "levels":
            values = current

        elif representation == "recent_change":
            if timestep < lag:
                continue

            previous = np.asarray(
                mean_outcomes[var][:, timestep - lag],
                dtype=float,
            )
            values = current - previous

        else:
            raise ValueError(
                f"Unknown representation: {representation}"
            )

        data[var] = values

    X = pd.DataFrame(data)

    if X.empty:
        return X

    varying = X.std(axis=0, ddof=0) > MIN_STD
    X = X.loc[:, varying]

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Non-finite values at t={timestep}, spec={spec_name}"
        )

    return X


# =============================================================================
# 6. PRIM
# =============================================================================

def make_prim(X, y):
    kwargs = {
        "peel_alpha": PEEL_ALPHA,
        "mass_min": MASS_MIN,
    }

    signature = inspect.signature(prim.Prim)

    if (
        "threshold" in signature.parameters
        and signature.parameters["threshold"].default
        is inspect._empty
    ):
        kwargs["threshold"] = 0.0

    return prim.Prim(X, y, **kwargs)


def peeling_trajectory(box):
    tr = box.peeling_trajectory.copy().reset_index(drop=True)

    if "id" not in tr.columns:
        tr.insert(
            0,
            "id",
            np.arange(len(tr), dtype=int),
        )

    return tr


def select_box_from_trajectory(tr, strategy):
    tr = tr.copy()

    if strategy == PRIMARY_SELECTION:
        eligible = tr[
            tr["coverage"] >= 0.80
        ]

        if eligible.empty:
            row = tr.loc[
                tr["coverage"].idxmax()
            ].copy()
            note = "highest coverage available"

        else:
            row = (
                eligible
                .sort_values(
                    ["density", "coverage"],
                    ascending=[False, False],
                )
                .iloc[0]
                .copy()
            )
            note = "highest density with coverage >= 0.80"

    elif strategy == SENSITIVITY_SELECTION:
        denom = tr["coverage"] + tr["density"]

        f1 = np.where(
            denom > 0,
            2
            * tr["coverage"]
            * tr["density"]
            / denom,
            np.nan,
        )

        idx = int(np.nanargmax(f1))
        row = tr.iloc[idx].copy()
        note = "maximum coverage-density F1"

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    coverage = float(row["coverage"])
    density = float(row["density"])

    row["f1"] = (
        2
        * coverage
        * density
        / (coverage + density)
        if coverage + density > 0
        else np.nan
    )

    row["selection_strategy"] = strategy
    row["selection_note"] = note

    return row


def limits_for_dimension(box_lims, dimension, full_data):
    if dimension in box_lims.columns:
        return (
            float(box_lims[dimension].iloc[0]),
            float(box_lims[dimension].iloc[1]),
        )

    if dimension in box_lims.index:
        row = box_lims.loc[dimension]
        return (
            float(row.iloc[0]),
            float(row.iloc[1]),
        )

    return (
        float(full_data[dimension].min()),
        float(full_data[dimension].max()),
    )


def run_state_space_prim(mean_outcomes, labels):
    results = {}
    rows = []

    for spec_name in ANALYSIS_SPECS:
        print("=" * 90)
        print("PRIM specification:", spec_name)
        results[spec_name] = {}

        for t in range(N_TIMESTEPS):
            X = state_matrix(
                mean_outcomes,
                t,
                spec_name,
            )
            results[spec_name][t] = {}

            if X.shape[1] == 0:
                continue

            for c in range(1, FINAL_K + 1):
                y = (labels == c).astype(int)

                algorithm = make_prim(
                    X.copy(),
                    y,
                )
                box = algorithm.find_box()
                tr = peeling_trajectory(box)

                selected = {
                    strategy: select_box_from_trajectory(
                        tr,
                        strategy,
                    )
                    for strategy in SELECTION_STRATEGIES
                }

                results[spec_name][t][c] = {
                    "X": X,
                    "box": box,
                    "trajectory": tr,
                    "selected": selected,
                }

                for strategy, row in selected.items():
                    rows.append(
                        {
                            "spec": spec_name,
                            "timestep": t,
                            "cluster": c,
                            "cluster_name": CLUSTER_NAMES[c],
                            "strategy": strategy,
                            "box_id": int(row["id"]),
                            "coverage": float(row["coverage"]),
                            "density": float(row["density"]),
                            "mass": float(row["mass"]),
                            "restricted_dimensions": int(
                                row["res_dim"]
                            ),
                            "f1": float(row["f1"]),
                            "n_available_state_dimensions": X.shape[1],
                        }
                    )

            if t % 5 == 0 or t == N_TIMESTEPS - 1:
                print(f"  completed timestep {t}")

    return results, pd.DataFrame(rows)


# =============================================================================
# 7. RULE EXTRACTION
# =============================================================================

def extract_selected_rules(state_results):
    rows = []

    for spec_name, by_t in state_results.items():
        for t, by_cluster in by_t.items():
            for c, result in by_cluster.items():
                X = result["X"]

                for strategy in SELECTION_STRATEGIES:
                    selected = result["selected"][strategy]
                    box_id = int(selected["id"])
                    box_lims = (
                        result["box"]
                        .box_lims[box_id]
                        .copy()
                    )

                    for var in X.columns:
                        lower, upper = limits_for_dimension(
                            box_lims,
                            var,
                            X,
                        )

                        full_lower = float(
                            X[var].min()
                        )
                        full_upper = float(
                            X[var].max()
                        )

                        restricted = (
                            lower > full_lower + 1e-10
                            or upper < full_upper - 1e-10
                        )

                        if restricted:
                            rows.append(
                                {
                                    "spec": spec_name,
                                    "timestep": t,
                                    "cluster": c,
                                    "cluster_name": CLUSTER_NAMES[c],
                                    "strategy": strategy,
                                    "state_variable": var,
                                    "state_label": STATE_LABELS[var],
                                    "lower": lower,
                                    "upper": upper,
                                    "full_lower": full_lower,
                                    "full_upper": full_upper,
                                }
                            )

    return pd.DataFrame(rows)


def make_primary_rule_tables(state_rules):
    primary_rules = state_rules[
        (state_rules["spec"] == PRIMARY_SPEC)
        & (
            state_rules["strategy"]
            == PRIMARY_SELECTION
        )
    ].copy()

    rule_frequency = (
        primary_rules
        .groupby(
            ["state_variable", "state_label"]
        )
        .size()
        .rename("n_selected_rules")
        .reset_index()
        .sort_values(
            "n_selected_rules",
            ascending=False,
        )
    )

    pair_rows = []

    for c in range(1, FINAL_K + 1):
        for t in range(N_TIMESTEPS):
            vars_here = sorted(
                primary_rules[
                    (primary_rules["cluster"] == c)
                    & (
                        primary_rules["timestep"]
                        == t
                    )
                ]["state_variable"].unique()
            )

            for a, b in combinations(
                vars_here,
                2,
            ):
                pair_rows.append(
                    {
                        "cluster": c,
                        "timestep": t,
                        "var_a": a,
                        "var_b": b,
                    }
                )

    if pair_rows:
        pair_counts = (
            pd.DataFrame(pair_rows)
            .groupby(["var_a", "var_b"])
            .size()
            .rename("co_selection_count")
            .reset_index()
            .sort_values(
                "co_selection_count",
                ascending=False,
            )
        )
    else:
        pair_counts = pd.DataFrame(
            columns=[
                "var_a",
                "var_b",
                "co_selection_count",
            ]
        )

    return (
        primary_rules,
        rule_frequency,
        pair_counts,
    )


# =============================================================================
# 8. COMMON PLOT HELPERS
# =============================================================================

def metric_panel(
    state_summary,
    metric,
    ylabel,
    title,
    series_filters,
    folder,
    stem,
    ylim=None,
):
    """
    Generic 2x2 cluster panel.

    series_filters:
        list of tuples (label, boolean filter callable)
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.7),
        sharex=True,
    )
    axes = axes.ravel()

    for c, ax in zip(
        range(1, FINAL_K + 1),
        axes,
    ):
        for label, filter_fn in series_filters:
            sub = state_summary[
                filter_fn(state_summary)
                & (state_summary["cluster"] == c)
            ].sort_values("timestep")

            ax.plot(
                sub["timestep"],
                sub[metric],
                linewidth=1.8,
                label=label,
            )

        ax.axvspan(
            MAIN_SUBSPACE_TIMESTEPS[0],
            MAIN_SUBSPACE_TIMESTEPS[-1],
            color="0.92",
            alpha=0.45,
            zorder=0,
        )
        ax.axvline(
            FLOOD_TIMESTEP,
            linestyle="--",
            linewidth=1.1,
            color="0.35",
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(
            f"C{c}: {CLUSTER_NAMES[c]}"
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(
            alpha=0.18,
            linewidth=0.6,
        )

    handles, labs = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(len(labs), 3),
        frameon=False,
    )

    panel_layout(
        fig,
        title,
        top=0.84,
    )

    save_figure(
        fig,
        folder,
        stem,
    )
    plt.close(fig)


def cluster_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=CLUSTER_COLORS[c],
            markeredgecolor=CLUSTER_COLORS[c],
            alpha=0.7,
            label=f"C{c}: {CLUSTER_NAMES[c]}",
        )
        for c in range(1, FINAL_K + 1)
    ]


def selected_box_limits(
    state_results,
    spec_name,
    timestep,
    cluster,
    strategy,
):
    result = state_results[
        spec_name
    ][timestep][cluster]

    selected = result[
        "selected"
    ][strategy]

    box_id = int(
        selected["id"]
    )

    return (
        result["box"]
        .box_lims[box_id]
        .copy()
    )


def restriction_info(
    box_lims,
    var,
    X,
):
    lo, hi = limits_for_dimension(
        box_lims,
        var,
        X,
    )

    full_lo = float(X[var].min())
    full_hi = float(X[var].max())

    restricted = (
        lo > full_lo + 1e-10
        or hi < full_hi - 1e-10
    )

    return restricted, lo, hi



def pooled_projection_limits(
    mean_outcomes,
    x_var,
    y_var,
    timesteps=PROJECTION_AXIS_TIMESTEPS,
    pad_fraction=0.03,
):
    """Return common x/y limits so timestep projections are directly comparable."""
    x_values = []
    y_values = []

    for t in timesteps:
        X = state_matrix(
            mean_outcomes,
            t,
            PRIMARY_SPEC,
        )
        if x_var in X.columns and y_var in X.columns:
            x_values.append(X[x_var].to_numpy(dtype=float))
            y_values.append(X[y_var].to_numpy(dtype=float))

    if not x_values:
        return None, None

    x = np.concatenate(x_values)
    y = np.concatenate(y_values)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    x_lim = (
        x_min - pad_fraction * x_span,
        x_max + pad_fraction * x_span,
    )
    y_lim = (
        y_min - pad_fraction * y_span,
        y_max + pad_fraction * y_span,
    )

    return x_lim, y_lim


def timestep_descriptor(timestep):
    if timestep == 5:
        return "earlier pre-divergence"
    if timestep == 9:
        return "pre-divergence"
    if timestep == 10:
        return "flood / divergence onset"
    if timestep == 11:
        return "early post-divergence"
    return None


# =============================================================================
# 9. MAIN-TEXT CANDIDATE FIGURES
# =============================================================================

def plot_main_performance(state_summary):
    primary_filter = lambda df: (
        (df["spec"] == PRIMARY_SPEC)
        & (
            df["strategy"]
            == PRIMARY_SELECTION
        )
    )

    sensitivity_filter = lambda df: (
        (df["spec"] == PRIMARY_SPEC)
        & (
            df["strategy"]
            == SENSITIVITY_SELECTION
        )
    )

    metric_panel(
        state_summary,
        metric="coverage",
        ylabel="Coverage",
        title="Observable-state PRIM coverage through time",
        series_filters=[
            (
                "Coverage ≥ 0.80 / max density",
                primary_filter,
            ),
            (
                "Max F1",
                sensitivity_filter,
            ),
        ],
        folder=MAIN_DIR,
        stem="main_01_observable_state_coverage",
        ylim=(0, 1.03),
    )

    metric_panel(
        state_summary,
        metric="f1",
        ylabel="Coverage-density F1",
        title="Observable-state PRIM coverage-density balance",
        series_filters=[
            (
                "Coverage ≥ 0.80 / max density",
                primary_filter,
            ),
            (
                "Max F1",
                sensitivity_filter,
            ),
        ],
        folder=MAIN_DIR,
        stem="main_02_observable_state_f1",
        ylim=(0, 1.03),
    )


def plot_representation_sensitivity(state_summary):
    series_filters = [
        (
            "Current observable state",
            lambda df: (
                (df["spec"] == PRIMARY_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
        (
            "One-step recent change",
            lambda df: (
                (df["spec"] == RECENT_CHANGE_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
        (
            "Current state, no uptake",
            lambda df: (
                (df["spec"] == NO_UPTAKE_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
    ]

    metric_panel(
        state_summary,
        metric="density",
        ylabel="Density",
        title="Observable-state specification sensitivity",
        series_filters=series_filters,
        folder=MAIN_DIR,
        stem="main_03_state_specification_density",
        ylim=(0, 1.03),
    )


def plot_rule_frequency(rule_frequency):
    data = rule_frequency.sort_values(
        "n_selected_rules",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.4, 5.4),
    )

    ax.barh(
        data["state_label"],
        data["n_selected_rules"],
    )

    ax.set_xlabel(
        "Number of selected cluster–timestep PRIM rules"
    )
    ax.set_ylabel("")
    ax.set_title(
        "Frequency of observable states in selected PRIM rules"
    )
    ax.grid(
        axis="x",
        alpha=0.18,
        linewidth=0.6,
    )

    fig.tight_layout()

    save_figure(
        fig,
        MAIN_DIR,
        "main_04_state_variable_frequency",
    )
    plt.close(fig)


def plot_projection(
    mean_outcomes,
    labels,
    state_results,
    x_var,
    y_var,
    timestep,
    strategy,
    spec_name,
    folder,
    stem,
    x_lim=None,
    y_lim=None,
):
    X = state_matrix(
        mean_outcomes,
        timestep,
        spec_name,
    )

    if (
        x_var not in X.columns
        or y_var not in X.columns
    ):
        return

    fig, ax = plt.subplots(
        figsize=(8.2, 6.1),
    )
    ax.set_box_aspect(0.8)

    for c in range(1, FINAL_K + 1):
        mask = labels == c

        ax.scatter(
            X.loc[mask, x_var],
            X.loc[mask, y_var],
            s=14,
            alpha=0.28,
            color=CLUSTER_COLORS[c],
            edgecolors="none",
            rasterized=True,
        )

    # Draw each cluster's PRIM projection in its own cluster colour.
    for c in range(1, FINAL_K + 1):
        box_lims = selected_box_limits(
            state_results,
            spec_name,
            timestep,
            c,
            strategy,
        )

        x_r, x_lo, x_hi = restriction_info(
            box_lims,
            x_var,
            X,
        )

        y_r, y_lo, y_hi = restriction_info(
            box_lims,
            y_var,
            X,
        )

        color = CLUSTER_COLORS[c]

        if x_r and y_r:
            ax.add_patch(
                Rectangle(
                    (x_lo, y_lo),
                    x_hi - x_lo,
                    y_hi - y_lo,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.4,
                    zorder=5,
                )
            )

        elif x_r:
            ax.axvline(
                x_lo,
                color=color,
                linewidth=2.1,
                zorder=5,
            )
            ax.axvline(
                x_hi,
                color=color,
                linewidth=2.1,
                zorder=5,
            )

        elif y_r:
            ax.axhline(
                y_lo,
                color=color,
                linewidth=2.1,
                zorder=5,
            )
            ax.axhline(
                y_hi,
                color=color,
                linewidth=2.1,
                zorder=5,
            )

        # If neither axis is restricted, deliberately draw nothing.

    ax.set_xlabel(
        STATE_LABELS[x_var]
    )
    ax.set_ylabel(
        STATE_LABELS[y_var]
    )

    strategy_title = {
        PRIMARY_SELECTION:
            "max density with coverage ≥ 0.80",
        SENSITIVITY_SELECTION:
            "maximum F1",
    }[strategy]

    descriptor = timestep_descriptor(timestep)
    if descriptor is None:
        plot_title = f"Observable state space — timestep {timestep}"
    else:
        plot_title = f"Timestep {timestep} — {descriptor}"

    ax.set_title(plot_title)

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.grid(
        alpha=0.12,
        linewidth=0.5,
    )

    ax.text(
        0.01,
        0.99,
        strategy_title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )

    ax.legend(
        handles=cluster_legend_handles(),
        frameon=False,
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

    save_figure(
        fig,
        folder,
        stem,
    )
    plt.close(fig)


def plot_focal_main_projections(
    mean_outcomes,
    labels,
    state_results,
    pair_counts,
):
    """
    Save every focal timestep as its own full-size figure.

    Timesteps 9-11 remain the main-text transition window. The axes are fixed
    within each state-variable pair using pooled t=5,9,10,11 data so visual
    movement between panels is not caused by rescaling.
    """
    top_pairs = [
        (row.var_a, row.var_b)
        for row in pair_counts.head(3).itertuples()
    ]

    for pair_idx, (x_var, y_var) in enumerate(top_pairs, 1):
        x_lim, y_lim = pooled_projection_limits(
            mean_outcomes,
            x_var,
            y_var,
        )

        for t in MAIN_SUBSPACE_TIMESTEPS:
            plot_projection(
                mean_outcomes,
                labels,
                state_results,
                x_var=x_var,
                y_var=y_var,
                timestep=t,
                strategy=PRIMARY_SELECTION,
                spec_name=PRIMARY_SPEC,
                folder=MAIN_DIR,
                stem=(
                    f"main_state_space_pair{pair_idx}_"
                    f"{x_var}__vs__{y_var}_t{t}"
                ),
                x_lim=x_lim,
                y_lim=y_lim,
            )


def plot_t5_appendix_check(
    mean_outcomes,
    labels,
    state_results,
    pair_counts,
):
    """
    Earlier pre-divergence check for the appendix.

    This tests whether the focal t=9-11 interpretation was already equally
    apparent at t=5, while avoiding a return all the way to initialization.
    """
    top_pairs = [
        (row.var_a, row.var_b)
        for row in pair_counts.head(3).itertuples()
    ]

    for pair_idx, (x_var, y_var) in enumerate(top_pairs, 1):
        x_lim, y_lim = pooled_projection_limits(
            mean_outcomes,
            x_var,
            y_var,
        )

        for t in APPENDIX_CHECK_TIMESTEPS:
            plot_projection(
                mean_outcomes,
                labels,
                state_results,
                x_var=x_var,
                y_var=y_var,
                timestep=t,
                strategy=PRIMARY_SELECTION,
                spec_name=PRIMARY_SPEC,
                folder=APPENDIX_DIR,
                stem=(
                    f"appendix_t5_check_pair{pair_idx}_"
                    f"{x_var}__vs__{y_var}_t{t}"
                ),
                x_lim=x_lim,
                y_lim=y_lim,
            )


# =============================================================================
# 10. APPENDIX CANDIDATES
# =============================================================================

def plot_appendix_performance(state_summary):
    main_filter = lambda df: (
        (df["spec"] == PRIMARY_SPEC)
        & (
            df["strategy"]
            == PRIMARY_SELECTION
        )
    )

    f1_filter = lambda df: (
        (df["spec"] == PRIMARY_SPEC)
        & (
            df["strategy"]
            == SENSITIVITY_SELECTION
        )
    )

    metric_panel(
        state_summary,
        metric="density",
        ylabel="Density",
        title="Observable-state PRIM density through time",
        series_filters=[
            (
                "Coverage ≥ 0.80 / max density",
                main_filter,
            ),
            (
                "Max F1",
                f1_filter,
            ),
        ],
        folder=APPENDIX_DIR,
        stem="appendix_01_density_over_time",
        ylim=(0, 1.03),
    )

    metric_panel(
        state_summary,
        metric="restricted_dimensions",
        ylabel="Restricted dimensions",
        title="Observable-state PRIM rule complexity through time",
        series_filters=[
            (
                "Coverage ≥ 0.80 / max density",
                main_filter,
            ),
            (
                "Max F1",
                f1_filter,
            ),
        ],
        folder=APPENDIX_DIR,
        stem="appendix_02_rule_complexity",
    )

    representation_filters = [
        (
            "Current observable state",
            lambda df: (
                (df["spec"] == PRIMARY_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
        (
            "One-step recent change",
            lambda df: (
                (df["spec"] == RECENT_CHANGE_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
        (
            "Current state, no uptake",
            lambda df: (
                (df["spec"] == NO_UPTAKE_SPEC)
                & (
                    df["strategy"]
                    == PRIMARY_SELECTION
                )
            ),
        ),
    ]

    metric_panel(
        state_summary,
        metric="coverage",
        ylabel="Coverage",
        title="State-representation sensitivity: coverage",
        series_filters=representation_filters,
        folder=APPENDIX_DIR,
        stem="appendix_03_representation_coverage",
        ylim=(0, 1.03),
    )

    metric_panel(
        state_summary,
        metric="f1",
        ylabel="Coverage-density F1",
        title="State-representation sensitivity: F1",
        series_filters=representation_filters,
        folder=APPENDIX_DIR,
        stem="appendix_04_representation_f1",
        ylim=(0, 1.03),
    )


def plot_rule_timelines(primary_rules):
    for c in range(1, FINAL_K + 1):
        sub = primary_rules[
            primary_rules["cluster"] == c
        ]

        matrix = pd.DataFrame(
            0,
            index=OBSERVABLE_STATE_VARS,
            columns=range(N_TIMESTEPS),
            dtype=int,
        )

        for _, row in sub.iterrows():
            matrix.loc[
                row["state_variable"],
                int(row["timestep"]),
            ] = 1

        fig, ax = plt.subplots(
            figsize=(11.0, 4.6)
        )

        im = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            cmap="Greys",
        )

        ax.set_yticks(
            range(
                len(OBSERVABLE_STATE_VARS)
            ),
            [
                STATE_LABELS[v]
                for v in OBSERVABLE_STATE_VARS
            ],
        )

        ax.set_xticks(
            range(0, N_TIMESTEPS, 5),
            range(0, N_TIMESTEPS, 5),
        )

        ax.axvline(
            FLOOD_TIMESTEP,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_xlabel("Timestep")
        ax.set_title(
            f"C{c}: observable states restricted in selected PRIM rule"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            ticks=[0, 1],
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label(
            "Restricted in selected rule"
        )

        fig.tight_layout()

        save_figure(
            fig,
            APPENDIX_DIR,
            f"appendix_05_rule_timeline_C{c}",
        )
        plt.close(fig)


def plot_pair_frequency(pair_counts):
    if pair_counts.empty:
        return

    top = (
        pair_counts
        .head(12)
        .copy()
    )

    top["pair_label"] = (
        top["var_a"].map(STATE_LABELS)
        + " × "
        + top["var_b"].map(STATE_LABELS)
    )

    top = top.sort_values(
        "co_selection_count",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 6.2),
    )

    ax.barh(
        top["pair_label"],
        top["co_selection_count"],
    )

    ax.set_xlabel(
        "Number of cluster–timestep rules selecting both states"
    )
    ax.set_ylabel("")
    ax.set_title(
        "Most recurrent observable state-variable pairs"
    )
    ax.grid(
        axis="x",
        alpha=0.18,
        linewidth=0.6,
    )

    fig.tight_layout()

    save_figure(
        fig,
        APPENDIX_DIR,
        "appendix_06_pair_co_selection_frequency",
    )
    plt.close(fig)


def selected_variable_set(
    primary_rules,
    cluster,
    timestep,
):
    sub = primary_rules[
        (primary_rules["cluster"] == cluster)
        & (
            primary_rules["timestep"]
            == timestep
        )
    ]

    return set(
        sub["state_variable"]
    )


def compute_rule_stability(primary_rules):
    rows = []

    for c in range(1, FINAL_K + 1):
        for t in range(
            N_TIMESTEPS - 1
        ):
            a = selected_variable_set(
                primary_rules,
                c,
                t,
            )
            b = selected_variable_set(
                primary_rules,
                c,
                t + 1,
            )

            union = a | b

            jaccard = (
                len(a & b) / len(union)
                if union
                else 1.0
            )

            rows.append(
                {
                    "cluster": c,
                    "timestep": t,
                    "next_timestep": t + 1,
                    "jaccard": jaccard,
                }
            )

    return pd.DataFrame(rows)


def plot_rule_stability(rule_stability):
    fig, ax = plt.subplots(
        figsize=(9.5, 5.1),
    )

    for c in range(1, FINAL_K + 1):
        sub = rule_stability[
            rule_stability["cluster"] == c
        ]

        ax.plot(
            sub["next_timestep"],
            sub["jaccard"],
            label=f"C{c}",
            color=CLUSTER_COLORS[c],
            linewidth=1.7,
        )

    ax.axvline(
        FLOOD_TIMESTEP,
        linestyle="--",
        linewidth=1.1,
        color="0.35",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(
        "Adjacent-timestep Jaccard similarity"
    )
    ax.set_title(
        "Temporal stability of selected observable-state dimensions"
    )
    ax.legend(
        frameon=False,
        ncol=4,
    )
    ax.grid(
        alpha=0.18,
        linewidth=0.6,
    )

    fig.tight_layout()

    save_figure(
        fig,
        APPENDIX_DIR,
        "appendix_07_rule_stability",
    )
    plt.close(fig)


def plot_rule_bound_evolution(
    primary_rules,
    top_n=4,
):
    """
    Plot normalised lower/upper bounds of the most frequently selected states
    for each cluster. Normalisation is against the sampled range at each t.
    """
    for c in range(1, FINAL_K + 1):
        sub = primary_rules[
            primary_rules["cluster"] == c
        ]

        top_vars = (
            sub.groupby(
                "state_variable"
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(top_n)
            .index
            .tolist()
        )

        if not top_vars:
            continue

        fig, axes = plt.subplots(
            len(top_vars),
            1,
            figsize=(
                9.5,
                2.45 * len(top_vars),
            ),
            sharex=True,
        )

        if len(top_vars) == 1:
            axes = [axes]

        for ax, var in zip(
            axes,
            top_vars,
        ):
            rows = []

            for t in range(
                N_TIMESTEPS
            ):
                rule = sub[
                    (
                        sub["timestep"]
                        == t
                    )
                    & (
                        sub[
                            "state_variable"
                        ]
                        == var
                    )
                ]

                if rule.empty:
                    rows.append(
                        {
                            "timestep": t,
                            "lower_norm": np.nan,
                            "upper_norm": np.nan,
                        }
                    )
                    continue

                row = rule.iloc[0]
                span = (
                    row["full_upper"]
                    - row["full_lower"]
                )

                if span <= 0:
                    lo = np.nan
                    hi = np.nan
                else:
                    lo = (
                        row["lower"]
                        - row["full_lower"]
                    ) / span
                    hi = (
                        row["upper"]
                        - row["full_lower"]
                    ) / span

                rows.append(
                    {
                        "timestep": t,
                        "lower_norm": lo,
                        "upper_norm": hi,
                    }
                )

            temp = pd.DataFrame(rows)

            ax.plot(
                temp["timestep"],
                temp["lower_norm"],
                linewidth=1.6,
                label="Lower bound",
            )
            ax.plot(
                temp["timestep"],
                temp["upper_norm"],
                linewidth=1.6,
                label="Upper bound",
            )
            ax.fill_between(
                temp["timestep"],
                temp["lower_norm"],
                temp["upper_norm"],
                alpha=0.12,
            )

            ax.axvline(
                FLOOD_TIMESTEP,
                linestyle="--",
                linewidth=1.0,
                color="0.35",
            )

            ax.set_ylim(
                -0.02,
                1.02,
            )
            ax.set_ylabel(
                "Normalised\nrange"
            )
            ax.set_title(
                STATE_LABELS[var],
                loc="left",
                pad=4,
            )
            ax.grid(
                alpha=0.14,
                linewidth=0.5,
            )

        axes[-1].set_xlabel(
            "Timestep"
        )
        axes[0].legend(
            frameon=False,
            ncol=2,
            loc="upper right",
        )

        fig.suptitle(
            f"C{c}: evolution of recurring observable-state PRIM bounds",
            y=0.995,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        save_figure(
            fig,
            APPENDIX_DIR,
            f"appendix_08_bound_evolution_C{c}",
        )
        plt.close(fig)


def plot_peeling_tradeoffs(
    state_results,
    timestep,
):
    """
    Coverage-density PRIM peeling trajectories for all clusters at one t,
    showing both representative-box selections.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 8.3),
    )
    axes = axes.ravel()

    for c, ax in zip(
        range(1, FINAL_K + 1),
        axes,
    ):
        result = state_results[
            PRIMARY_SPEC
        ][timestep][c]

        tr = result["trajectory"]

        ax.plot(
            tr["coverage"],
            tr["density"],
            color=CLUSTER_COLORS[c],
            linewidth=1.8,
        )

        for strategy, marker, label in [
            (
                PRIMARY_SELECTION,
                "o",
                "Coverage ≥ 0.80 / max density",
            ),
            (
                SENSITIVITY_SELECTION,
                "s",
                "Max F1",
            ),
        ]:
            row = result[
                "selected"
            ][strategy]

            ax.scatter(
                row["coverage"],
                row["density"],
                marker=marker,
                s=58,
                color=CLUSTER_COLORS[c],
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label=label,
            )

        ax.axvline(
            0.80,
            linestyle=":",
            color="0.55",
            linewidth=1,
        )
        ax.axhline(
            0.80,
            linestyle=":",
            color="0.55",
            linewidth=1,
        )

        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Density")
        ax.set_title(
            f"C{c}: {CLUSTER_NAMES[c]}"
        )
        ax.grid(
            alpha=0.14,
            linewidth=0.5,
        )

    handles, labs = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )

    panel_layout(
        fig,
        f"PRIM coverage–density trade-offs at timestep {timestep}",
        top=0.84,
    )

    save_figure(
        fig,
        APPENDIX_DIR,
        f"appendix_09_peeling_tradeoff_t{timestep}",
    )
    plt.close(fig)


def plot_all_projection_sensitivities(
    mean_outcomes,
    labels,
    state_results,
    pair_counts,
):
    top_pairs = [
        (row.var_a, row.var_b)
        for row in pair_counts.head(3).itertuples()
    ]

    for pair_idx, (
        x_var,
        y_var,
    ) in enumerate(
        top_pairs,
        1,
    ):
        x_lim, y_lim = pooled_projection_limits(
            mean_outcomes,
            x_var,
            y_var,
        )

        for t in MAIN_SUBSPACE_TIMESTEPS:
            # Alternative representative-box rule.
            plot_projection(
                mean_outcomes,
                labels,
                state_results,
                x_var,
                y_var,
                timestep=t,
                strategy=SENSITIVITY_SELECTION,
                spec_name=PRIMARY_SPEC,
                folder=APPENDIX_DIR,
                stem=(
                    f"appendix_maxF1_pair{pair_idx}_"
                    f"{x_var}__vs__{y_var}_t{t}"
                ),
                x_lim=x_lim,
                y_lim=y_lim,
            )

            # Recent-change representation under the primary box rule.
            plot_projection(
                mean_outcomes,
                labels,
                state_results,
                x_var,
                y_var,
                timestep=t,
                strategy=PRIMARY_SELECTION,
                spec_name=RECENT_CHANGE_SPEC,
                folder=APPENDIX_DIR,
                stem=(
                    f"appendix_recent_change_pair{pair_idx}_"
                    f"{x_var}__vs__{y_var}_t{t}"
                ),
                x_lim=x_lim,
                y_lim=y_lim,
            )

            # No-uptake projections only if both axes remain available.
            if (
                x_var
                in OBSERVABLE_STATE_VARS_NO_UPTAKE
                and y_var
                in OBSERVABLE_STATE_VARS_NO_UPTAKE
            ):
                plot_projection(
                    mean_outcomes,
                    labels,
                    state_results,
                    x_var,
                    y_var,
                    timestep=t,
                    strategy=PRIMARY_SELECTION,
                    spec_name=NO_UPTAKE_SPEC,
                    folder=APPENDIX_DIR,
                    stem=(
                        f"appendix_no_uptake_pair{pair_idx}_"
                        f"{x_var}__vs__{y_var}_t{t}"
                    ),
                    x_lim=x_lim,
                    y_lim=y_lim,
                )


# =============================================================================
# 11. SAVE TABLES
# =============================================================================

def save_tables(
    state_inventory,
    state_summary,
    state_rules,
    rule_frequency,
    pair_counts,
    rule_stability,
):
    state_inventory.to_csv(
        TABLE_DIR
        / "observable_state_inventory.csv",
        index=False,
    )

    state_summary.to_csv(
        TABLE_DIR
        / "observable_state_prim_summary_all_timesteps.csv",
        index=False,
    )

    state_rules.to_csv(
        TABLE_DIR
        / "observable_state_prim_rules_all_timesteps.csv",
        index=False,
    )

    rule_frequency.to_csv(
        TABLE_DIR
        / "observable_state_rule_frequency.csv",
        index=False,
    )

    pair_counts.to_csv(
        TABLE_DIR
        / "observable_state_pair_co_selection.csv",
        index=False,
    )

    rule_stability.to_csv(
        TABLE_DIR
        / "observable_state_rule_stability.csv",
        index=False,
    )

    flood_window = state_summary[
        state_summary["timestep"].isin(
            [5, 9, 10, 11, 15]
        )
    ].copy()

    flood_window.to_csv(
        TABLE_DIR
        / "observable_state_flood_window.csv",
        index=False,
    )

    strategy_comparison = state_summary[
        (
            state_summary["spec"]
            == PRIMARY_SPEC
        )
        & (
            state_summary["timestep"]
            .isin([9, 10, 11])
        )
    ].copy()

    strategy_comparison.to_csv(
        TABLE_DIR
        / "observable_state_box_selection_sensitivity.csv",
        index=False,
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    print("Project root:", PROJECT_ROOT)
    print("Results dir: ", RESULTS_DIR)
    print("Output dir:  ", OUTPUT_DIR)

    scenario_experiments, mean_outcomes = (
        load_ensemble_incremental()
    )

    if len(scenario_experiments) != 2000:
        raise RuntimeError(
            "Expected 2,000 uncertainty scenarios, "
            f"found {len(scenario_experiments)}."
        )

    labels = reproduce_final_clustering(
        mean_outcomes
    )

    state_inventory = validate_state_vars(
        mean_outcomes
    )

    state_results, state_summary = (
        run_state_space_prim(
            mean_outcomes,
            labels,
        )
    )

    state_rules = extract_selected_rules(
        state_results
    )

    (
        primary_rules,
        rule_frequency,
        pair_counts,
    ) = make_primary_rule_tables(
        state_rules
    )

    rule_stability = compute_rule_stability(
        primary_rules
    )

    # -------------------------
    # MAIN-TEXT CANDIDATES
    # -------------------------
    plot_main_performance(
        state_summary
    )

    plot_representation_sensitivity(
        state_summary
    )

    plot_rule_frequency(
        rule_frequency
    )

    plot_focal_main_projections(
        mean_outcomes,
        labels,
        state_results,
        pair_counts,
    )

    plot_t5_appendix_check(
        mean_outcomes,
        labels,
        state_results,
        pair_counts,
    )

    # -------------------------
    # APPENDIX CANDIDATES
    # -------------------------
    plot_appendix_performance(
        state_summary
    )

    plot_rule_timelines(
        primary_rules
    )

    plot_pair_frequency(
        pair_counts
    )

    plot_rule_stability(
        rule_stability
    )

    plot_rule_bound_evolution(
        primary_rules,
        top_n=4,
    )

    for t in [9, 10, 11]:
        plot_peeling_tradeoffs(
            state_results,
            timestep=t,
        )

    plot_all_projection_sensitivities(
        mean_outcomes,
        labels,
        state_results,
        pair_counts,
    )

    save_tables(
        state_inventory,
        state_summary,
        state_rules,
        rule_frequency,
        pair_counts,
        rule_stability,
    )

    print()
    print("Finished.")
    print("All SSSD outputs:")
    print(" ", OUTPUT_DIR)


if __name__ == "__main__":
    main()
