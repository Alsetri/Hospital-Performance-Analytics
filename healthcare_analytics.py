import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "C:/Users/abals/Desktop/Coding Projects/Data/Readmissions_and_Deaths_Hospital.csv"
# Composite score weights kept at top so leadership can tune emphasis quickly.
W_READM = 0.6
W_MORT = 0.4


def load_and_explore_data(path: str) -> pd.DataFrame:
    """Load source data and print quick exploratory diagnostics."""
    # Initial load from source extract.
    df = pd.read_csv(path)

    df.info()
    print(df.head())
    print(df.shape)
    print(df.columns)

    for column in df.columns:
        print(column, df[column].nunique())

    print(df.shape[0])
    print(df.describe(include="all"))

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column cleanup, type casting, and missing-value handling."""
    # Remove fields not used in scoring/visuals to simplify the working table.
    df = df.drop(
        columns=["index", "Phone Number", "Address", "ZIP Code", "Footnote"],
        errors="ignore",
    )

    print(df.columns)

    # Display unique values after dropping columns.
    for col in df.columns:
        print(col, df[col].nunique())

    df.info()

    numeric_cols = ["Denominator", "Score", "Lower Estimate", "Higher Estimate"]
    # Coerce malformed numerics to NaN so downstream drops are explicit/traceable.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # .info() already prints details.
    df[numeric_cols].info()
    df.info()
    print(df.isnull().sum())

    print(df["County Name"].value_counts())

    # Drop rows missing county because geographic summaries require valid location data.
    county_before = len(df)
    df = df.dropna(subset=["County Name"])
    county_dropped = county_before - len(df)
    print(f"Dropped {county_dropped} rows due to missing County Name.")
    print(df["County Name"].isnull().sum())

    # Drop rows missing core metrics needed for risk logic and benchmark calculations.
    core_before = len(df)
    df = df.dropna(subset=["Denominator", "Score", "Compared to National"])
    core_dropped = core_before - len(df)
    print(
        "Dropped "
        f"{core_dropped} rows due to missing Denominator, Score, or Compared to National."
    )

    print(df.isnull().sum())
    return df


def plot_state_performance(df: pd.DataFrame) -> None:
    """Stacked distribution of performance vs national benchmark by state."""
    # Normalize to percentages so states are comparable regardless of hospital count.
    state_performance_counts = df.groupby(["State", "Compared to National"]).size().unstack(fill_value=0)
    state_performance_pct = state_performance_counts.div(state_performance_counts.sum(axis=1), axis=0)

    state_performance_pct.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Distribution of Hospital Performance vs National Average by State")
    plt.ylabel("Proportion of Hospitals")
    plt.xlabel("State")
    plt.tight_layout()
    plt.show()


def plot_readmission_views(df: pd.DataFrame) -> pd.DataFrame:
    """Create readmission-specific summaries and charts."""
    # Restrict to READM measures for readmission-focused reporting.
    readm_df = df[df["Measure ID"].str.contains("READM", case=False, na=False)].copy()

    measure_readmission_avg = (
        readm_df.groupby("Measure ID")["Score"].mean().sort_values(ascending=False)
    )

    measure_readmission_avg.plot(
        kind="barh", figsize=(12, 6), color = "lightgreen", edgecolor = "black")
    plt.title("Readmission Measures with the Highest Average Readmission Rates")
    plt.xlabel("Readmission Rate (%)")
    plt.ylabel("Measure ID")
    plt.tight_layout()
    plt.show()

    state_readmission = readm_df.groupby("State")["Score"].mean().sort_values(ascending=False)

    state_readmission.head(20).plot(kind="bar", figsize=(12, 6))
    plt.title("States with the Highest Average Readmission Rates")
    plt.ylabel("Readmission Rate (%)")
    plt.xlabel("State")
    plt.tight_layout()
    plt.show()

    return readm_df


def plot_uncertainty_by_measure(df: pd.DataFrame) -> None:
    """Chart top measures with highest average uncertainty range."""
    print(df["Lower Estimate"].nunique())
    print(df["Higher Estimate"].nunique())

    if "Uncertainty_Range" not in df.columns:
        # Confidence interval width proxy: wider range implies less certainty in the estimate.
        df["Uncertainty_Range"] = df["Higher Estimate"] - df["Lower Estimate"]

    measures_uncertainty = (
        df.groupby("Measure Name")["Uncertainty_Range"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    plt.figure(figsize=(14, 7))
    plt.barh(measures_uncertainty["Measure Name"], measures_uncertainty["Uncertainty_Range"])
    plt.title("Hospital Measures with the Greatest Uncertainty in Outcome Rates")
    plt.xlabel("Average Confidence Interval Width (%)")
    plt.ylabel("Measure")
    plt.tight_layout()
    plt.show()


def assign_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add uncertainty and percentile-based risk tiers."""
    df = df.copy()
    # Required feature for rule-based stratification.
    df["Uncertainty_Range"] = df["Higher Estimate"] - df["Lower Estimate"]

    # Data-adaptive thresholds (quartiles) rather than hard-coded constants.
    score_high = df["Score"].quantile(0.75)
    score_low = df["Score"].quantile(0.25)
    unc_high = df["Uncertainty_Range"].quantile(0.75)
    unc_low = df["Uncertainty_Range"].quantile(0.25)
    vol_low = df["Denominator"].quantile(0.25)

    # High risk: high score + high uncertainty + low volume.
    high_mask = (
        (df["Score"] >= score_high)
        & (df["Uncertainty_Range"] >= unc_high)
        & (df["Denominator"] <= vol_low)
    )
    # Low risk: low score + low uncertainty; all others are medium by design.
    low_mask = (df["Score"] <= score_low) & (df["Uncertainty_Range"] <= unc_low)

    df["Risk_Tier"] = np.select([high_mask, low_mask], ["High", "Low"], default="Medium")

    print(df["Risk_Tier"].value_counts())

    state_risk = df.groupby(["State", "Risk_Tier"]).size().unstack(fill_value=0)
    print(state_risk)

    state_risk.plot(kind="bar", stacked=True, figsize=(14, 7), edgecolor = "black")
    plt.title("Risk Tier Counts by State")
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return df


def calculate_benchmarks(df: pd.DataFrame, w_readm: float, w_mort: float) -> tuple[pd.DataFrame, pd.Series]:
    """Build hospital/state benchmark scores and rankings."""
    # Split cohorts so each domain is standardized against its own distribution.
    readm_df = df[df["Measure ID"].str.contains("READM", case=False, na=False)].copy()
    mort_df = df[df["Measure ID"].str.contains("MORT", case=False, na=False)].copy()

    readm_std = readm_df["Score"].std(ddof=0)
    mort_std = mort_df["Score"].std(ddof=0)

    # Z-scores make readmission and mortality components comparable on one scale.
    readm_df["Score_z"] = (
        (readm_df["Score"] - readm_df["Score"].mean()) / readm_std if readm_std else np.nan
    )
    mort_df["Score_z"] = (
        (mort_df["Score"] - mort_df["Score"].mean()) / mort_std if mort_std else np.nan
    )

    # Hospital-level benchmark table by averaging z-scores across each hospital's measures.
    readm_h = readm_df.groupby(["Hospital Name", "State"])["Score_z"].mean().reset_index()
    mort_h = mort_df.groupby(["Hospital Name", "State"])["Score_z"].mean().reset_index()

    bench = readm_h.merge(
        mort_h,
        on=["Hospital Name", "State"],
        how="outer",
        suffixes=("_readm", "_mort"),
    )

    # Fill missing component z-scores with 0 to keep single-domain hospitals in ranking.
    bench[["Score_z_readm", "Score_z_mort"]] = bench[["Score_z_readm", "Score_z_mort"]].fillna(0)

    # Weighted composite performance score.
    bench["Performance_Score"] = (
        w_readm * bench["Score_z_readm"] + w_mort * bench["Score_z_mort"]
    )
    # Dense rank avoids gaps in rank numbers.
    bench["Rank"] = bench["Performance_Score"].rank(ascending=False, method="dense")

    top10_hosp = bench.nlargest(10, "Performance_Score")[
        ["Hospital Name", "State", "Performance_Score", "Rank"]
    ]
    bottom10_hosp = bench.nsmallest(10, "Performance_Score")[
        ["Hospital Name", "State", "Performance_Score", "Rank"]
    ]

    print("Top 10 hospitals by Performance_Score")
    print(top10_hosp)
    print("Bottom 10 hospitals by Performance_Score")
    print(bottom10_hosp)

    state_rank = bench.groupby("State")["Performance_Score"].mean().sort_values(ascending=False)

    print("Top 10 states by average Performance_Score")
    print(state_rank.head(10))
    print("Bottom 10 states by average Performance_Score")
    print(state_rank.tail(10))

    # Persist benchmark outputs for downstream reporting/BI tools.
    bench.to_csv("hospital_benchmark_scores.csv", index=False)
    state_rank.to_csv("state_benchmark_scores.csv")

    top15 = bench.nlargest(15, "Performance_Score").sort_values("Performance_Score")
    bottom15 = bench.nsmallest(15, "Performance_Score").sort_values("Performance_Score", ascending=False)

    plt.figure(figsize=(12, 7))
    plt.barh(top15["Hospital Name"], top15["Performance_Score"])
    plt.title("Top 15 Hospitals by Composite Performance Score")
    plt.xlabel("Performance Score")
    plt.ylabel("Hospital")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.barh(bottom15["Hospital Name"], bottom15["Performance_Score"])
    plt.title("Bottom 15 Hospitals by Composite Performance Score")
    plt.xlabel("Performance Score")
    plt.ylabel("Hospital")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    state_rank.head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 States by Average Composite Performance Score")
    plt.xlabel("Average Performance Score")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()

    return bench, state_rank


def print_executive_summary(
    df: pd.DataFrame, readm_df: pd.DataFrame, bench: pd.DataFrame, state_rank: pd.Series
) -> None:
    """Generate data-driven executive insights."""
    print("\nExecutive Summary")

    # Insight 1: identify where the concentration of high-risk rows is greatest.
    state_risk_pct = (
        df.groupby(["State", "Risk_Tier"]).size().unstack(fill_value=0).pipe(
            lambda x: x.div(x.sum(axis=1), axis=0)
        )
    )
    high_risk_state = state_risk_pct["High"].idxmax()
    high_risk_pct = state_risk_pct["High"].max()

    # Insight 2: most elevated readmission measure by average score.
    measure_readmission_avg = (
        readm_df.groupby("Measure ID")["Score"].mean().sort_values(ascending=False)
    )
    top_readm_measure = measure_readmission_avg.index[0]
    top_readm_value = measure_readmission_avg.iloc[0]

    # Insight 3: hospital/state pair with the widest average uncertainty band.
    hospital_unc = (
        df.groupby(["Hospital Name", "State"])["Uncertainty_Range"]
        .mean()
        .sort_values(ascending=False)
    )
    top_unc_hospital = hospital_unc.index[0]
    top_unc_value = hospital_unc.iloc[0]

    # Insight 4: detect state-level outcome volatility via score variance.
    state_variance = df.groupby("State")["Score"].var().sort_values(ascending=False)
    top_var_state = state_variance.index[0]
    top_var_value = state_variance.iloc[0]

    # Insight 5: strongest state on composite benchmark score.
    top_state = state_rank.idxmax()
    top_state_score = state_rank.iloc[0]

    print(f"1) State with highest % High Risk: {high_risk_state} ({high_risk_pct:.1%})")
    print(
        f"2) Measure with highest average readmission score: {top_readm_measure} "
        f"({top_readm_value:.2f})"
    )
    print(
        "3) Hospital with highest average uncertainty range: "
        f"{top_unc_hospital[0]} ({top_unc_hospital[1]}) at {top_unc_value:.2f}"
    )
    print(
        f"4) State with highest variance in outcomes: {top_var_state} "
        f"(variance {top_var_value:.2f})"
    )
    print(
        f"5) Top state by benchmark performance score: {top_state} "
        f"({top_state_score:.3f})"
    )


def main() -> None:
    df = load_and_explore_data(DATA_PATH)
    df = clean_data(df)

    print(df.columns)

    plot_state_performance(df)
    readm_df = plot_readmission_views(df)

    df = assign_risk_tier(df)
    plot_uncertainty_by_measure(df)

    bench, state_rank = calculate_benchmarks(df, w_readm=W_READM, w_mort=W_MORT)

    print_executive_summary(df, readm_df, bench, state_rank)


if __name__ == "__main__":
    main()
