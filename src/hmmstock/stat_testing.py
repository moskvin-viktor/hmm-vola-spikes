from scipy.stats import f_oneway
import pandas as pd

def test_returns_anova(df: pd.DataFrame, state_col='regime_layer0') -> float:
    """
    Perform one-way ANOVA on normalized_returns grouped by regime.
    Returns the p-value.
    https://en.wikipedia.org/wiki/One-way_analysis_of_variance
    """
    grouped_returns = [group['normalized_returns'].values for _, group in df.groupby(state_col)]
    f_stat, p_value = f_oneway(*grouped_returns)
    return p_value