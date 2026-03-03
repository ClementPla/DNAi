import numpy as np
from scipy import stats

import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

from numba import njit, prange


def partover_test(
    x1=None,
    x2=None,
    x3=None,
    x4=None,
    var_equal=False,
    mu=0.0,
    alternative="two.sided",
    conf_level=None,
):
    """
    Partially overlapping samples t-test.

    Parameters:
        x1 : array-like or None
            Unpaired sample from group 1.
        x2 : array-like or None
            Unpaired sample from group 2.
        x3 : array-like or None
            Paired sample from group 1.
        x4 : array-like or None
            Paired sample from group 2 (must be same length as x3).
        var_equal : bool
            Assume equal variances if True (Student-type); otherwise Welch-type.
        mu : float
            Null hypothesis difference.
        alternative : {'two.sided', 'less', 'greater'}
            Defines the alternative hypothesis.
        conf_level : float or None
            If provided, confidence interval is computed (0<conf_level<1).

    Returns:
        dict with keys:
            statistic, parameter (df), p.value, estimate, conf.int (if requested)
    """

    # Convert all non-None inputs to numpy arrays
    def arr_or_empty(x):
        return np.array(x, dtype=float) if x is not None else np.array([], dtype=float)

    x1 = arr_or_empty(x1)
    x2 = arr_or_empty(x2)
    x3 = arr_or_empty(x3)
    x4 = arr_or_empty(x4)

    # Check inputs
    if len(x3) != len(x4):
        raise ValueError("Paired observations not of same length")
    if len(x3) < 2:
        raise ValueError("Not enough paired observations")
    if len(x1) == 0 and len(x2) == 0:
        raise ValueError("Not enough vectors specified")

    # Common statistics
    xbar1 = np.mean(np.concatenate([x1, x3])) if len(x1) + len(x3) > 0 else 0.0
    xbar2 = np.mean(np.concatenate([x2, x4])) if len(x2) + len(x4) > 0 else 0.0
    estimate = xbar1 - xbar2

    n1 = len(x1) + len(x3)
    n2 = len(x2) + len(x4)
    n12 = len(x3)

    # correlation r of paired observations
    if np.std(x3, ddof=1) == 0 or np.std(x4, ddof=1) == 0:
        r = 0.0
    else:
        r = np.corrcoef(x3, x4)[0, 1]

    # Pooled or Welch-type standard error
    if var_equal:
        s1_sq = np.var(np.concatenate([x1, x3]), ddof=1)
        s2_sq = np.var(np.concatenate([x2, x4]), ddof=1)
        spooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))

        denom1 = 1 / n1 + 1 / n2
        denom2 = 2 * r * n12 / (n1 * n2)
        denom = spooled * np.sqrt(denom1 - denom2)

        statistic = (estimate - mu) / denom if denom != 0 else np.nan
        # Degrees of freedom formula from original R code
        parameter = (n12 - 1) + (
            ((len(x1) + len(x2) + len(x3) - 1) / (len(x1) + len(x2) + 2 * len(x3)))
            * (len(x1) + len(x2))
        )
    else:
        s1_sq = np.var(np.concatenate([x1, x3]), ddof=1)
        s2_sq = np.var(np.concatenate([x2, x4]), ddof=1)
        denom1 = s1_sq / n1 + s2_sq / n2
        denom2 = (
            2
            * r
            * n12
            * np.std(np.concatenate([x1, x3]), ddof=1)
            * np.std(np.concatenate([x2, x4]), ddof=1)
            / (n1 * n2)
        )
        denom = np.sqrt(denom1 - denom2)

        statistic = (estimate - mu) / denom if denom != 0 else np.nan

        # Welch-Satterthwaite approximation
        wel_numer = (s1_sq / n1 + s2_sq / n2) ** 2
        wel_denom = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
        welapprox = wel_numer / wel_denom
        parameter = (n12 - 1) + (
            ((welapprox - n12 + 1) / (len(x1) + len(x2) + 2 * n12))
            * (len(x1) + len(x2))
        )

    # p-value
    if np.isnan(statistic):
        p_value = 1.0
    else:
        if alternative == "less":
            p_value = stats.t.cdf(statistic, df=parameter)
        elif alternative == "greater":
            p_value = stats.t.sf(statistic, df=parameter)  # 1-cdf
        elif alternative == "two.sided":
            p_value = 2 * stats.t.sf(abs(statistic), df=parameter)
        else:
            raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")

    result = {
        "statistic": statistic,
        "parameter": parameter,
        "p.value": p_value,
        "estimate": estimate,
    }

    # Confidence interval if requested
    if conf_level is not None:
        alpha = 1 - conf_level
        if alternative == "two.sided":
            tcrit = stats.t.ppf(1 - alpha / 2, df=parameter)
            lower = estimate - tcrit * denom
            upper = estimate + tcrit * denom
        elif alternative == "less":
            tcrit = stats.t.ppf(conf_level, df=parameter)
            lower = -np.inf
            upper = estimate + tcrit * denom
        elif alternative == "greater":
            tcrit = stats.t.ppf(conf_level, df=parameter)
            lower = estimate - tcrit * denom
            upper = np.inf
        result["conf.int"] = (lower, upper)

    return result


def find_grader_disagreements(
    df: pd.DataFrame, reference_type: str, alpha: float = 0.05
):
    df = df.copy()
    df["Type"] = df["Type"].astype(str)
    df["Grader"] = df["Grader"].astype(str)

    graders = sorted(df["Grader"].unique())
    assert len(graders) == 2, f"Expected exactly 2 graders, got {graders}"

    types = [t for t in df["Type"].unique() if t != reference_type]

    results = []
    for typ in types:
        row = {"Type": typ}
        sig = {}

        for grader in graders:
            mask_ref = (df["Type"] == reference_type) & (df["Grader"] == grader)
            mask_typ = (df["Type"] == typ) & (df["Grader"] == grader)

            ref_ratios = df.loc[mask_ref, "Ratio"].dropna()
            typ_ratios = df.loc[mask_typ, "Ratio"].dropna()

            row[f"n_ref_{grader}"] = len(ref_ratios)
            row[f"n_typ_{grader}"] = len(typ_ratios)

            if len(ref_ratios) < 2 or len(typ_ratios) < 2:
                row[f"U_{grader}"] = None
                row[f"p_{grader}"] = None
                row[f"significant_{grader}"] = None
                sig[grader] = None
                continue

            stat, p = mannwhitneyu(ref_ratios, typ_ratios, alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(ref_ratios), len(typ_ratios)
            r = 1 - (2 * stat) / (n1 * n2)
            row[f"effect_size_{grader}"] = r
            row[f"U_{grader}"] = stat
            row[f"p_{grader}"] = p
            row[f"significant_{grader}"] = p < alpha
            sig[grader] = p < alpha

        # Only compare when both graders have valid results
        if (
            all(v is not None for v in sig.values())
            and sig[graders[0]] != sig[graders[1]]
        ):
            sig_grader = [g for g, s in sig.items() if s][0]
            row["disagreement"] = f"{sig_grader}_only"
        else:
            row["disagreement"] = None

        results.append(row)

    result_df = pd.DataFrame(results)
    return result_df


@njit
def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    gt = 0
    lt = 0
    for i in range(nx):
        for j in range(ny):
            if x[i] > y[j]:
                gt += 1
            elif x[i] < y[j]:
                lt += 1
    return (gt - lt) / (nx * ny)


@njit(parallel=True)
def _bootstrap_deltas(x, y, idx_x, idx_y):
    n_boot, nx = idx_x.shape
    ny = idx_y.shape[1]
    deltas = np.empty(n_boot)
    for i in prange(n_boot):
        gt = 0
        lt = 0
        for j in range(nx):
            xj = x[idx_x[i, j]]
            for k in range(ny):
                yk = y[idx_y[i, k]]
                gt += xj > yk
                lt += xj < yk
        deltas[i] = (gt - lt) / (nx * ny)
    return deltas


def bootstrap_cliffs_delta_ci(x, y, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)

    idx_x = rng.integers(0, nx, size=(n_boot, nx))
    idx_y = rng.integers(0, ny, size=(n_boot, ny))

    deltas = _bootstrap_deltas(x, y, idx_x, idx_y)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return np.percentile(deltas, [lo, hi])


def find_grader_disagreements_cliff(
    df: pd.DataFrame, reference_type: str, n_boot=10000, alpha=0.05
):
    df = df.copy()
    df["Type"] = df["Type"].astype(str)
    df["Grader"] = df["Grader"].astype(str)

    graders = sorted(df["Grader"].unique())
    assert len(graders) == 2, f"Expected exactly 2 graders, got {graders}"

    types = [t for t in df["Type"].unique() if t != reference_type]

    # Warm up numba JIT on tiny arrays so compilation doesn't count in the loop
    _warmup_x = np.array([1.0, 2.0])
    _warmup_y = np.array([3.0, 4.0])
    cliffs_delta(_warmup_x, _warmup_y)
    _bootstrap_deltas(
        _warmup_x,
        _warmup_y,
        np.zeros((2, 2), dtype=np.int64),
        np.zeros((2, 2), dtype=np.int64),
    )

    results = []
    for typ in tqdm(types, desc="Processing types"):
        row = {"Type": typ}
        excludes_zero = {}

        for grader in graders:
            ref_ratios = (
                df.loc[
                    (df["Type"] == reference_type) & (df["Grader"] == grader), "Ratio"
                ]
                .dropna()
                .values.astype(np.float64)
            )

            typ_ratios = (
                df.loc[(df["Type"] == typ) & (df["Grader"] == grader), "Ratio"]
                .dropna()
                .values.astype(np.float64)
            )

            row[f"n_ref_{grader}"] = len(ref_ratios)
            row[f"n_typ_{grader}"] = len(typ_ratios)

            if len(ref_ratios) < 2 or len(typ_ratios) < 2:
                row[f"delta_{grader}"] = None
                row[f"ci_low_{grader}"] = None
                row[f"ci_high_{grader}"] = None
                row[f"excludes_zero_{grader}"] = None
                excludes_zero[grader] = None
                continue

            d = cliffs_delta(ref_ratios, typ_ratios)
            ci_low, ci_high = bootstrap_cliffs_delta_ci(
                ref_ratios, typ_ratios, n_boot=n_boot, alpha=alpha
            )

            row[f"delta_{grader}"] = d
            row[f"ci_low_{grader}"] = ci_low
            row[f"ci_high_{grader}"] = ci_high

            ez = (ci_low > 0) or (ci_high < 0)
            row[f"excludes_zero_{grader}"] = ez
            excludes_zero[grader] = ez

        vals = {k: v for k, v in excludes_zero.items() if v is not None}
        if len(vals) == 2 and len(set(vals.values())) == 2:
            sig_grader = [g for g, s in vals.items() if s][0]
            row["disagreement"] = f"{sig_grader}_only"
        else:
            row["disagreement"] = None

        results.append(row)

    return pd.DataFrame(results)


from scipy.stats import mannwhitneyu


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def add_significance_brackets(
    ax,
    df,
    reference_type,
    column="Ratio",
    graders=None,
    grader_colors=None,
    y_start=None,
    bracket_spacing_factor=1.15,
    bracket_height_factor=1.03,
    fontsize=12,
    alternative="two-sided",
):
    """
    Draw brackets from reference_type to each other condition,
    annotated with Mann-Whitney U significance stars per grader.

    Parameters
    ----------
    ax : matplotlib Axes
    df : DataFrame with columns [column, "Type", "Grader"]
    reference_type : str, the reference condition name
    column : str, the column to test
    graders : list of grader names (default: inferred from df)
    grader_colors : dict mapping grader name -> color
    y_start : float, y position for first bracket (default: auto)
    bracket_spacing_factor : multiplicative spacing between brackets
    bracket_height_factor : multiplicative height of bracket tip
    fontsize : int
    alternative : str, passed to mannwhitneyu
    """
    if graders is None:
        graders = (
            df["Grader"].cat.categories.tolist()
            if hasattr(df["Grader"], "cat")
            else df["Grader"].unique().tolist()
        )

    if grader_colors is None:
        grader_colors = {g: "dimgray" for g in graders}

    types = (
        df["Type"].cat.categories.tolist()
        if hasattr(df["Type"], "cat")
        else df["Type"].unique().tolist()
    )

    ref_idx = types.index(reference_type)
    other_types = [t for t in types if t != reference_type]

    # Sort by distance to reference (shorter brackets first)
    other_types = sorted(other_types, key=lambda t: abs(types.index(t) - ref_idx))

    if y_start is None:
        y_start = ax.get_ylim()[1] * 1.1

    for i, typ in enumerate(other_types):
        typ_idx = types.index(typ)

        left = min(ref_idx, typ_idx) + 0.1
        right = max(ref_idx, typ_idx) - 0.1

        y_base = y_start * (bracket_spacing_factor**i)
        y_top = y_base * bracket_height_factor

        # Draw bracket
        ax.plot(
            [left, left, right, right],
            [y_base, y_top, y_top, y_base],
            color="dimgray",
            linewidth=1.0,
            clip_on=False,
        )

        # Compute significance for each grader
        mid_x = (left + right) / 2
        n_graders = len(graders)
        total_width = 0.3 * (n_graders - 1)
        x_positions = np.linspace(
            mid_x - total_width / 2, mid_x + total_width / 2, n_graders
        )

        for x_pos, grader in zip(x_positions, graders):
            ref_vals = df[(df["Type"] == reference_type) & (df["Grader"] == grader)][
                column
            ].dropna()
            exp_vals = df[(df["Type"] == typ) & (df["Grader"] == grader)][
                column
            ].dropna()

            if len(ref_vals) < 2 or len(exp_vals) < 2:
                symbol = "?"
            else:
                _, p = mannwhitneyu(ref_vals, exp_vals, alternative=alternative)
                symbol = significance_stars(p)

            ax.annotate(
                symbol,
                xy=(x_pos, y_top),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold" if symbol != "n.s." else "normal",
                color=grader_colors.get(grader, "dimgray"),
                clip_on=False,
            )

    # Adjust ylim
    top_y = (
        y_start * (bracket_spacing_factor ** len(other_types)) * bracket_spacing_factor
    )
    current_ylim = ax.get_ylim()
    ax.set_ylim(top=max(current_ylim[1], top_y))
