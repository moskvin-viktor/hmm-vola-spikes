import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    from hmmstock.path_manager import PathManager
    from hmmstock.stat_testing import test_returns_anova

    pio.templates.default = "plotly_dark"

    mo.md(
        """
        # Explore trained regime models

        Reads whatever `hatch run python fit_model.py` (or
        `scripts/train_all_models.sh`) has written under
        `artifacts/{model}/version_N/` via `PathManager` -- no re-training,
        no network access, just looking at what's already on disk. Plots
        mirror what the old Dash app (`app/app.py`, now retired) showed,
        rebuilt against the current artifact layout and the volatility-
        relabeling/CV fixes since then.
        """
    )
    return PathManager, go, mo, pd, px, test_returns_anova


@app.cell(hide_code=True)
def _(PathManager, mo):
    pm = PathManager("artifacts")
    available_models = (
        sorted(p.name for p in pm.root.iterdir() if p.is_dir())
        if pm.root.exists()
        else []
    )

    mo.stop(
        not available_models,
        mo.md(
            "No trained artifacts found under `artifacts/`. Run "
            "`hatch run python fit_model.py` first, then reopen this notebook."
        ),
    )
    return available_models, pm


@app.cell(hide_code=True)
def _(available_models, mo):
    model_dropdown = mo.ui.dropdown(
        options=available_models, value=available_models[0], label="Model"
    )
    model_dropdown
    return (model_dropdown,)


@app.cell(hide_code=True)
def _(mo, model_dropdown, pm):
    versions = pm.versions(model_dropdown.value)
    version_dropdown = mo.ui.dropdown(
        options=versions, value=versions[-1], label="Version (run)"
    )
    version_dropdown
    return (version_dropdown,)


@app.cell(hide_code=True)
def _(mo, model_dropdown, pm, version_dropdown):
    tickers = pm.tickers(model_dropdown.value, version_dropdown.value)
    mo.stop(
        not tickers,
        mo.md(f"No fitted tickers in {model_dropdown.value}/{version_dropdown.value}."),
    )
    ticker_dropdown = mo.ui.dropdown(options=tickers, value=tickers[0], label="Ticker")
    ticker_dropdown
    return (ticker_dropdown,)


@app.cell(hide_code=True)
def _(mo, model_dropdown, pm, version_dropdown):
    config_text = pm.config_file(
        model_dropdown.value, version_dropdown.value
    ).read_text()
    metrics = pm.load_metrics(model_dropdown.value, version_dropdown.value)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("**Config used for this run**"),
                    mo.md(f"```yaml\n{config_text}```"),
                ]
            ),
            mo.vstack(
                [mo.md("**Metrics (all tickers)**"), mo.md(f"```json\n{metrics}\n```")]
            ),
        ]
    )
    return (metrics,)


@app.cell(hide_code=True)
def _(mo, model_dropdown, pm, ticker_dropdown, version_dropdown):
    states_df = pm.load_regime_states(
        model_dropdown.value, ticker_dropdown.value, version_dropdown.value
    )
    regime_cols = [
        c
        for c in states_df.columns
        if c.startswith("regime_layer") or c in ("top_level_state", "sub_level_state")
    ]

    mo.vstack(
        [mo.md(f"**Regime states — {ticker_dropdown.value}**"), mo.ui.table(states_df)]
    )
    return regime_cols, states_df


@app.cell(hide_code=True)
def _(mo, regime_cols):
    mo.stop(not regime_cols, mo.md("No regime column found in this run's output."))
    # The primary regime column: regime_layer0 for HMMModel/LayeredHMMModel,
    # top_level_state for HierarchicalHMMModel -- both are relabeled by
    # volatility (RegimeModel._volatility_rank_map: 0 = lowest total
    # variance across all features). sub_level_state (if present) is
    # excluded from "feature" columns below but not analyzed on its own
    # here -- it's only comparable *within* a top_level_state group, see
    # the model docstring.
    regime_col = regime_cols[0]
    return (regime_col,)


@app.cell(hide_code=True)
def _(mo, px, regime_col, states_df, ticker_dropdown):
    plot_df = states_df.reset_index(names="date")
    fig = px.scatter(
        plot_df,
        x="date",
        y="normalized_returns",
        color=plot_df[regime_col].astype(str),
        title=f"{ticker_dropdown.value}: normalized returns by {regime_col}",
        labels={"color": regime_col},
    )
    mo.vstack([mo.md("**Returns over time, colored by regime**"), fig])
    return


@app.cell(hide_code=True)
def _(mo, px, regime_col, states_df, ticker_dropdown):
    mo.stop("vol_20" not in states_df.columns, mo.md("No `vol_20` column in this run."))

    plot_df_vol = states_df.reset_index(names="date")
    fig_vol = px.scatter(
        plot_df_vol,
        x="date",
        y="vol_20",
        color=plot_df_vol[regime_col].astype(str),
        title=f"{ticker_dropdown.value}: vol_20 by {regime_col}",
        labels={"color": regime_col},
    )
    mo.vstack([mo.md("**Vol (20d) over time, colored by regime**"), fig_vol])
    return


@app.cell(hide_code=True)
def _(go, mo, model_dropdown, pm, ticker_dropdown, version_dropdown):
    matrices = pm.load_transition_matrices(
        model_dropdown.value, ticker_dropdown.value, version_dropdown.value
    )

    figs = []
    for layer_idx, matrix in enumerate(matrices):
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale="Blues",
                zmin=0,
                zmax=1,
                text=matrix.values.round(2),
                texttemplate="%{text}",
            )
        )
        heatmap.update_layout(
            title=f"{ticker_dropdown.value} — transition matrix, layer {layer_idx}",
            height=350,
        )
        figs.append(heatmap)

    mo.vstack(figs) if figs else mo.md("No transition matrices for this ticker.")
    return (matrices,)


@app.cell(hide_code=True)
def _(matrices, mo, pd, regime_col, states_df, ticker_dropdown):
    mo.stop(not matrices, mo.md("No transition matrix to compute persistence from."))

    diag = matrices[0].to_numpy().diagonal()
    time_share = states_df[regime_col].value_counts(normalize=True).sort_index()

    persistence = pd.DataFrame({"time_share": time_share})
    persistence["self_transition_prob"] = [
        diag[i] if i < len(diag) else float("nan") for i in persistence.index
    ]
    persistence["expected_duration_days"] = 1 / (
        1 - persistence["self_transition_prob"]
    ).clip(lower=1e-6)

    mo.vstack(
        [
            mo.md(f"**Regime persistence — {ticker_dropdown.value}**"),
            mo.ui.table(persistence.round(3)),
        ]
    )
    return (persistence,)


@app.cell(hide_code=True)
def _(mo, px, regime_col, regime_cols, states_df):
    feature_cols = [
        c
        for c in states_df.select_dtypes(include="number").columns
        if c not in regime_cols
    ]
    grouped_means = states_df.groupby(regime_col)[feature_cols].mean()

    fig_means = px.bar(
        grouped_means.reset_index(),
        x=regime_col,
        y=feature_cols,
        barmode="group",
        title="Feature means by regime",
    )
    mo.vstack([mo.md("**Feature means by regime**"), fig_means])
    return feature_cols, grouped_means


@app.cell(hide_code=True)
def _(feature_cols, go, mo, pd, regime_col, states_df):
    quantile = 0.9
    capture = {}
    for col in feature_cols:
        threshold = states_df[col].quantile(quantile)
        high = states_df[states_df[col] >= threshold]
        capture[col] = high[regime_col].value_counts(normalize=True).sort_index()

    capture_df = pd.DataFrame(capture).fillna(0)
    fig_capture = go.Figure()
    for regime_val in capture_df.index:
        fig_capture.add_bar(
            name=f"regime {regime_val}",
            x=capture_df.columns,
            y=capture_df.loc[regime_val],
        )
    fig_capture.update_layout(
        barmode="stack",
        title=f"Which regime captures the top {int(quantile * 100)}% of each feature?",
    )
    mo.vstack([mo.md("**Regime capture of high-value events**"), fig_capture])
    return


@app.cell(hide_code=True)
def _(feature_cols, go, mo, regime_col, states_df):
    corr = (
        states_df[[*feature_cols, regime_col]]
        .corr()[regime_col]
        .drop(regime_col)
        .sort_values()
    )
    fig_corr = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h"))
    fig_corr.update_layout(title="Correlation of each feature with the regime label")
    mo.vstack([mo.md("**Feature ↔ regime correlation**"), fig_corr])
    return


@app.cell(hide_code=True)
def _(mo, regime_col, states_df, test_returns_anova):
    p_value = test_returns_anova(states_df, state_col=regime_col)
    significant = p_value < 0.05

    mo.md(
        f"""
        **ANOVA — do normalized returns differ significantly across regimes?**

        p-value: `{p_value:.4f}` — {"**statistically significant**" if significant else "*not statistically significant*"} at the 5% level.
        """
    )
    return p_value, significant


@app.cell(hide_code=True)
def _(
    metrics,
    mo,
    model_dropdown,
    p_value,
    persistence,
    regime_col,
    significant,
    states_df,
    ticker_dropdown,
    version_dropdown,
):
    ticker_metrics = metrics.get(ticker_dropdown.value, {})
    n_components = ticker_metrics.get("n_components")
    cv_score = ticker_metrics.get("cv_score")

    dominant_regime = persistence["time_share"].idxmax()
    stickiest_regime = persistence["expected_duration_days"].idxmax()
    stickiest_days = persistence["expected_duration_days"].max()

    monotonic_note = ""
    if "vol_20" in states_df.columns:
        is_monotonic = (
            states_df.groupby(regime_col)["vol_20"].mean().is_monotonic_increasing
        )
        monotonic_note = (
            f"- `vol_20` increases monotonically across regime labels: **{is_monotonic}** "
            f"(labels rank by *total* variance across all features, not `vol_20` alone -- "
            f"they can disagree once there's more than one feature).\n"
        )

    mo.md(
        f"""
        ## Findings — {model_dropdown.value} / {ticker_dropdown.value} / {version_dropdown.value}

        - **{n_components} regimes** selected by walk-forward CV (`cv_score = {cv_score:.1f}`).
        - Regime **{dominant_regime}** is the most common, at {persistence.loc[dominant_regime, "time_share"]:.0%} of days.
        - Regime **{stickiest_regime}** is the most persistent: an expected ~{stickiest_days:.0f} trading days once entered.
        {monotonic_note}- Returns differ across regimes with ANOVA p-value `{p_value:.4f}` ({"statistically significant" if significant else "not statistically significant"} at 5%).
        """
    )
    return


if __name__ == "__main__":
    app.run()
