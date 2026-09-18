import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go

    from hmmstock.path_manager import PathManager

    mo.md(
        """
        # Explore trained regime models

        Reads whatever `hatch run python fit_model.py` has written under
        `artifacts/{model}/version_N/` via `PathManager` -- no re-training,
        no network access, just looking at what's already on disk.
        """
    )
    return PathManager, go, mo, px


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
    return


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
def _(mo, px, regime_cols, states_df, ticker_dropdown):
    mo.stop(not regime_cols, mo.md("No regime column found in this run's output."))

    regime_col = regime_cols[0]
    plot_df = states_df.reset_index(names="date")
    fig = px.scatter(
        plot_df,
        x="date",
        y="normalized_returns",
        color=plot_df[regime_col].astype(str),
        title=f"{ticker_dropdown.value}: normalized returns by {regime_col}",
        labels={"color": regime_col},
    )
    fig
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
    return


if __name__ == "__main__":
    app.run()
