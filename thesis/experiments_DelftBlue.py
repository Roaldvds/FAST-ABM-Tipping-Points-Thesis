import warnings
warnings.filterwarnings(
    "ignore",
    message="ipyparallel not installed*",
    category=UserWarning
)

from pathlib import Path
import os
import numpy as np

from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    Constant,
    TimeSeriesOutcome,
    ScalarOutcome,
    MultiprocessingEvaluator,
    save_results,
)
from ema_workbench.em_framework.samplers import LHSSampler
from src.model import AdaptationModel


N_STEPS = 50

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "delftblue"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_fast_model(
        seed,
        initial_adaptation,
        per_HH_adapted_to_SN_prob_midpoint,
        weight_SN_vs_PMT,
        savings_rate_multiplier,
        adaptation_cost_multiplier,
        measure_lifetime,
        flood_depth_multiplier,
):
    seed=int(seed)

    # Scale the baseline flood depths while preserving the spatial pattern
    baseline_flood_depths = np.array(
        [1.0, 0.6, 0.4, 0.0]
    )

    residential_flood_depths = (
            baseline_flood_depths
            * float(flood_depth_multiplier)
    )

    flood_depth_vector = [
        99,
        *residential_flood_depths.tolist()
    ]

    model = AdaptationModel(
        seed=seed,
        nr_households=1000,

        # Varying initial adaptation
        set_hh_adapted_at_start=float(initial_adaptation),

        # Social norm / behavioural settings
        exchange_what="threat+coping",
        social_norm_calculation="all",
        per_HH_adapted_to_SN_prob_midpoint=per_HH_adapted_to_SN_prob_midpoint,
        weight_SN_vs_PMT=weight_SN_vs_PMT,
        opinion_dynamics_model="similarity_biased",
        weight_opinion_others=0.5,
        threshold_similarity_bias=0.3,

        # Flood setup
        flood_depth_in_m=flood_depth_vector,

        # Fixed flood timing
        flood_time=[10],

        # Spacial setup
        population_density=[
            [0, 50, 50, 50, 50],
            [0, 50, 50, 50, 50],
            [0, 50, 50, 50, 50],
            [0, 50, 50, 50, 50],
            [0, 50, 50, 50, 50],
        ],

        # Network
        network_structure_file_path="data/processed/network_structures/Network_empirical.graphml",

        # Decision settings
        minimum_intention_to_consider_measure=0.1,
        consider_measure_every_x_steps=3,

        # Disable measure ageing
        measures_aging={
            "dry-proofing": int(measure_lifetime),
            "wet-proofing": int(measure_lifetime),
        },

        # Thesis extensions
        savings_rate_multiplier=savings_rate_multiplier,
        adaptation_cost_multiplier=adaptation_cost_multiplier,

        # Data collection
        transformative_threshold=0.8,
        collect_agent_data=0,
    )

    # Actual realised adaptation level immediately after initialization
    initial_adaptation_realised = (
            sum(
                1
                for agent in model.agents
                if 1 in agent.measures_taken.values()
            )
            / model.nr_households
    )

    for _ in range(N_STEPS):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    def series(name):
        if name in df.columns:
            return df[name].to_numpy()
        return np.full(N_STEPS, np.nan)

        # Original absolute adaptation trajectory
    adaptation_rate = series("fraction_adapted_any")

    # Change in adaptation compared with the realised initial level
    adaptation_baseline_adjusted = (
            adaptation_rate - initial_adaptation_realised
    )

    # Share of the initially available adaptation gap that is closed
    remaining_adaptation_gap = 1.0 - initial_adaptation_realised

    if remaining_adaptation_gap > 0:
        adaptation_progress = (
                                      adaptation_rate - initial_adaptation_realised
                              ) / remaining_adaptation_gap
    else:
        # Defensive fallback; your current uncertainty range cannot reach 1.0
        adaptation_progress = np.zeros_like(adaptation_rate)

    return {
        # Adaptation time-series outcomes
        "adaptation_rate": adaptation_rate,
        "adaptation_baseline_adjusted":
            adaptation_baseline_adjusted,
        "adaptation_progress": adaptation_progress,

        # Other time-series outcomes
        "gini_relative_burden":
            series("gini_relative_burden"),
        "gini_relative_burden_simulation_only":
            series("gini_relative_burden_simulation_only"),
        "gini_relative_burden_cumulative_income":
            series("gini_relative_burden_cumulative_income"),
        "gini_relative_burden_cumulative_income_simulation_only":
            series("gini_relative_burden_cumulative_income_simulation_only"),
        "average_relative_burden_cumulative_income":
            series("average_relative_burden_cumulative_income"),
        "unmet_adaptation_demand":
            series("share_unmet_adaptation_demand"),
        "adaptation_gap_high_low":
            series("adaptation_gap_high_low"),
        "share_liquidity_constrained":
            series("share_liquidity_constrained"),
        "average_worry": series("average_worry"),
        "average_savings": series("average_savings"),
        "average_perceived_risk":
            series("average_perceived_risk"),

        "average_social_norm_intention":
            series("avg_intention_SN"),
        "average_pmt_intention_dry_proofing":
            series("avg_intention_PMT_DP"),
        "average_pmt_intention_wet_proofing":
            series("avg_intention_PMT_WP"),
        "average_overall_intention_dry_proofing":
            series("avg_intention_overall_DP"),
        "average_overall_intention_wet_proofing":
            series("avg_intention_overall_WP"),

        "average_flood_damage":
            series("average_flood_damage"),
        "share_adaptation_deficit":
            series("share_adaptation_deficit"),
        "share_optimal_to_adapt":
            series("share_optimal_to_adapt"),
        "share_over_adapted":
            series("share_over_adapted"),

        "total_damage_experienced":
            series("total_damage_experienced"),
        "average_damage_experienced":
            series("average_damage_experienced"),
        "share_households_with_expiry":
            series("share_households_with_expiry"),
        "average_active_measure_age":
            series("average_active_measure_age"),
        "average_remaining_measure_lifetime":
            series("average_remaining_measure_lifetime"),
        "fraction_dry_proofed":
            series("fraction_dry_proofed"),
        "fraction_wet_proofed":
            series("fraction_wet_proofed"),
        "average_fraction_connections_adapted":
            series("average_fraction_connections_adapted"),

        # Adaptation scalar outcomes
        "realised_initial_adaptation":
            initial_adaptation_realised,
        "final_adaptation_rate":
            adaptation_rate[-1],
        "final_adaptation_baseline_adjusted":
            adaptation_baseline_adjusted[-1],
        "final_adaptation_progress":
            adaptation_progress[-1],

        # Other scalar outcomes
        "final_gini_relative_burden":
            series("gini_relative_burden")[-1],
        "final_unmet_adaptation_demand":
            series("share_unmet_adaptation_demand")[-1],
        "final_adaptation_gap_high_low":
            series("adaptation_gap_high_low")[-1],
        "final_share_liquidity_constrained":
            series("share_liquidity_constrained")[-1],
        "final_average_worry":
            series("average_worry")[-1],
        "final_average_savings":
            series("average_savings")[-1],
        "final_average_perceived_risk":
            series("average_perceived_risk")[-1],
        "final_average_flood_damage":
            series("average_flood_damage")[-1],
        "final_gini_relative_burden_cumulative_income":
            series("gini_relative_burden_cumulative_income")[-1],
        "final_gini_relative_burden_simulation_only":
            series("gini_relative_burden_simulation_only")[-1],
        "final_gini_relative_burden_cumulative_income_simulation_only":
            series("gini_relative_burden_cumulative_income_simulation_only")[-1],
        "final_average_relative_burden":
            series("average_relative_burden")[-1],
        "final_average_relative_burden_cumulative_income":
            series("average_relative_burden_cumulative_income")[-1],
        "final_share_adaptation_deficit":
            series("share_adaptation_deficit")[-1],
        "final_share_optimal_to_adapt":
            series("share_optimal_to_adapt")[-1],
        "final_share_over_adapted":
            series("share_over_adapted")[-1],
        "final_average_social_norm_intention":
            series("avg_intention_SN")[-1],
        "final_average_pmt_intention_dry_proofing":
            series("avg_intention_PMT_DP")[-1],
        "final_average_pmt_intention_wet_proofing":
            series("avg_intention_PMT_WP")[-1],
        "final_average_overall_intention_dry_proofing":
            series("avg_intention_overall_DP")[-1],
        "final_average_overall_intention_wet_proofing":
            series("avg_intention_overall_WP")[-1],
    }


def build_ema_model():
    ema_model = Model("FAST_ABM", function=run_fast_model)

    ema_model.uncertainties = [
        RealParameter("initial_adaptation", 0.1, 0.8),
        RealParameter("per_HH_adapted_to_SN_prob_midpoint", 0.3, 0.8),
        RealParameter("weight_SN_vs_PMT", 0.3, 0.8),
        RealParameter("savings_rate_multiplier", 0.5, 1.5),
        RealParameter("adaptation_cost_multiplier", 0.5, 1.5),
        IntegerParameter("measure_lifetime", 10, 20),
        RealParameter("flood_depth_multiplier",0.5,1.5,),
    ]

    ema_model.outcomes = [
        # Time series
        TimeSeriesOutcome("adaptation_rate"),
        TimeSeriesOutcome("adaptation_baseline_adjusted"),
        TimeSeriesOutcome("adaptation_progress"),
        TimeSeriesOutcome("gini_relative_burden"),
        TimeSeriesOutcome("gini_relative_burden_simulation_only"),
        TimeSeriesOutcome("gini_relative_burden_cumulative_income"),
        TimeSeriesOutcome("gini_relative_burden_cumulative_income_simulation_only"),
        TimeSeriesOutcome("average_relative_burden_cumulative_income"),
        TimeSeriesOutcome("unmet_adaptation_demand"),
        TimeSeriesOutcome("adaptation_gap_high_low"),
        TimeSeriesOutcome("average_worry"),
        TimeSeriesOutcome("average_savings"),
        TimeSeriesOutcome("average_perceived_risk"),
        TimeSeriesOutcome("average_social_norm_intention"),
        TimeSeriesOutcome("average_pmt_intention_dry_proofing"),
        TimeSeriesOutcome("average_pmt_intention_wet_proofing"),
        TimeSeriesOutcome("average_overall_intention_dry_proofing"),
        TimeSeriesOutcome("average_overall_intention_wet_proofing"),
        TimeSeriesOutcome("average_flood_damage"),
        TimeSeriesOutcome("total_damage_experienced"),
        TimeSeriesOutcome("average_damage_experienced"),
        TimeSeriesOutcome("share_households_with_expiry"),
        TimeSeriesOutcome("average_active_measure_age"),
        TimeSeriesOutcome("average_remaining_measure_lifetime"),
        TimeSeriesOutcome("fraction_dry_proofed"),
        TimeSeriesOutcome("fraction_wet_proofed"),
        TimeSeriesOutcome("average_fraction_connections_adapted"),
        TimeSeriesOutcome("share_liquidity_constrained"),
        TimeSeriesOutcome("share_adaptation_deficit"),
        TimeSeriesOutcome("share_optimal_to_adapt"),
        TimeSeriesOutcome("share_over_adapted"),

        # Scalars
        ScalarOutcome("final_adaptation_rate"),
        ScalarOutcome("realised_initial_adaptation"),
        ScalarOutcome("final_adaptation_baseline_adjusted"),
        ScalarOutcome("final_adaptation_progress"),
        ScalarOutcome("final_gini_relative_burden"),
        ScalarOutcome("final_gini_relative_burden_simulation_only"),
        ScalarOutcome("final_gini_relative_burden_cumulative_income"),
        ScalarOutcome("final_gini_relative_burden_cumulative_income_simulation_only"),
        ScalarOutcome("final_unmet_adaptation_demand"),
        ScalarOutcome("final_adaptation_gap_high_low"),
        ScalarOutcome("final_average_worry"),
        ScalarOutcome("final_average_savings"),
        ScalarOutcome("final_average_perceived_risk"),
        ScalarOutcome("final_average_social_norm_intention"),
        ScalarOutcome("final_average_pmt_intention_dry_proofing"),
        ScalarOutcome("final_average_pmt_intention_wet_proofing"),
        ScalarOutcome("final_average_overall_intention_dry_proofing"),
        ScalarOutcome("final_average_overall_intention_wet_proofing"),
        ScalarOutcome("final_average_flood_damage"),
        ScalarOutcome("final_share_liquidity_constrained"),
        ScalarOutcome("final_average_relative_burden"),
        ScalarOutcome("final_average_relative_burden_cumulative_income"),
        ScalarOutcome("final_share_adaptation_deficit"),
        ScalarOutcome("final_share_optimal_to_adapt"),
        ScalarOutcome("final_share_over_adapted"),
    ]

    return ema_model


if __name__ == "__main__":
    ema_model = build_ema_model()

    n_scenarios = 2000
    seed = int(os.environ.get("SEED", 12345))

    # Generate one fixed Latin Hypercube design
    sampler = LHSSampler()

    fixed_scenarios = sampler.generate_samples(
        ema_model.uncertainties,
        n_scenarios,
        rng=12345
    )

    print(f"Generated {len(fixed_scenarios)} fixed scenarios")


    print(f"Running seed {seed}")

    ema_model.constants = [
        Constant("seed", seed)
    ]

    n_processes = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            os.cpu_count() or 1
        )
    )

    print(f"Using {n_processes} worker processes")

    with MultiprocessingEvaluator(
            ema_model,
            n_processes=n_processes
    ) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=fixed_scenarios
        )

    experiments["seed"] = seed

    results = experiments, outcomes

    save_results(
        results,
        RESULTS_DIR / f"ema_fast_results_seed_{seed}.tar.gz"
    )

    experiments.to_csv(
        RESULTS_DIR / f"ema_experiments_seed_{seed}.csv",
        index=False
    )