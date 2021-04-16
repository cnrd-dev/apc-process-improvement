"""
Custom library for data analysis.

Functions for descriptive statistics and regression analysis.
"""

from typing import Tuple
from statsmodels.graphics.gofplots import ProbPlot
import math
import sklearn.metrics as sklm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


def regression_metrics(y_true, y_predicted, n_parameters):
    """Calculate regression metrics.

    Args:
        y_true ([type]): [description]
        y_predicted ([type]): [description]
        n_parameters ([type]): [description]
    """

    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1) / (y_true.shape[0] - n_parameters) * (1 - r2)

    print(f"Root Mean Square Error = {math.sqrt(sklm.mean_squared_error(y_true, y_predicted)):0.2f}")
    print(f"Mean Absolute Error    = {sklm.mean_absolute_error(y_true, y_predicted):0.2f}")
    print(f"Median Absolute Error  = {sklm.median_absolute_error(y_true, y_predicted):0.2f}")
    print(f"R^2                    = {r2:0.4f}")
    print(f"Adjusted R^2           = {r2_adj:0.4f}")


def regression_eval_metrics(x_true: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate regression evaluation metrics.

    Args:
        x_true (np.ndarray): Known x-values.
        y_true (np.ndarray): Known y-values.
        y_predicted (np.ndarray): Predicted y-values.

    Returns:
        residuals (np.ndarray): Residuals from y_true and y_predicted.
        studentized_residuals (np.ndarray): Standardized residuals from residuals.
        cooks_distance (np.ndarray): Cooks distance values from residuals and hat_diag.
        hat_diag (np.ndarray): Diagonal array of x_true.
    """
    p = np.size(x_true, 1)
    residuals = np.subtract(y_true, y_predicted)
    X = x_true
    hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    hat_diag = np.diag(hat)
    MSE = sklm.mean_squared_error(y_true, y_predicted)
    studentized_residuals = residuals / np.sqrt(MSE * (1 - hat_diag))
    cooks_distance = (residuals ** 2 / (p * MSE)) * (hat_diag / (1 - hat_diag) ** 2)
    return residuals, studentized_residuals, cooks_distance, hat_diag


def diagnostic_plots(x_true: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray) -> None:
    """Generate diagnostic plots for regression evaluation.

    Source: Emre @ https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/

    Args:
        x_true (np.ndarray): Known x-values.
        y_true (np.ndarray): Known y-values.
        y_predicted (np.ndarray): Predicted y-values.
    """
    residuals, studentized_residuals, cooks_distance, hat_diag = regression_eval_metrics(x_true, y_true, y_predicted)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
    plt.tight_layout(pad=5, w_pad=5, h_pad=5)

    # 1. residual plot
    sns.residplot(x=y_predicted, y=residuals, lowess=True, scatter_kws={"alpha": 0.5}, line_kws={"color": "red", "lw": 1, "alpha": 0.8}, ax=axs[0, 0])
    axs[0, 0].set_title("Residuals vs Fitted")
    axs[0, 0].set_xlabel("Fitted values")
    axs[0, 0].set_ylabel("Residuals")

    # 2. qq plot
    qq = ProbPlot(studentized_residuals)
    qq.qqplot(line="45", alpha=0.5, color="#2578B2", lw=0.5, ax=axs[0, 1])
    axs[0, 1].set_title("Normal Q-Q")
    axs[0, 1].set_xlabel("Theoretical Quantiles")
    axs[0, 1].set_ylabel("Standardized Residuals")

    # 3. scale-location plot
    studentized_residuals_abs_sqrt = np.sqrt(np.abs(studentized_residuals))
    axs[1, 0].scatter(y_predicted, studentized_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(
        y_predicted,
        studentized_residuals_abs_sqrt,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("Scale-Location")
    axs[1, 0].set_xlabel("Fitted values")
    axs[1, 0].set_ylabel("$\sqrt{|Standardised Residuals|}$")

    # 4. leverage plot
    axs[1, 1].scatter(hat_diag, studentized_residuals, alpha=0.5)
    sns.regplot(hat_diag, studentized_residuals, scatter=False, ci=False, lowess=True, line_kws={"color": "red", "lw": 1, "alpha": 0.8}, ax=axs[1, 1])
    axs[1, 1].set_xlim(min(hat_diag), max(hat_diag))
    axs[1, 1].set_ylim(min(studentized_residuals), max(studentized_residuals))
    axs[1, 1].set_title("Residuals vs Leverage")
    axs[1, 1].set_xlabel("Leverage")
    axs[1, 1].set_ylabel("Standardised Residuals")

    # annotations
    leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]
    for i in leverage_top_3:
        axs[1, 1].annotate(i, xy=(hat_diag[i], studentized_residuals[i]))

    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        axs[1, 1].plot(x, y, label=label, lw=1, ls="--", color="red")

    p = np.size(x_true, 1)  # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), np.linspace(0.001, max(hat_diag), 50), "Cook's distance")
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), np.linspace(0.001, max(hat_diag), 50))
    axs[1, 1].legend(loc="upper right")


def hist_bin_width_fd(x: pd.Series) -> float:
    """Create bin widths for histograms based on the Freedman-Diaconis rule.

    Args:
        x (pd.Series): Series of data to use to generate bin widths.

    Returns:
        float: Number that specifies the bin widths.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2.0 * iqr * x.size ** (-1.0 / 3.0)
    if (x.max() - x.min()) / h > 1e8 * x.size:
        warnings.warn("Bin width estimated with the Freedman-Diaconis rule is very small" " (= {})".format(h), RuntimeWarning, stacklevel=2)
    return h


def plot_hist(cols: list, data: pd.DataFrame) -> None:
    """Plot multiple histograms.

    Args:
        cols (list): List of column names to plot.
        data (pd.DataFrame): Dataframe containing the data.
    """
    collen = len(cols)
    rows = math.ceil(collen / 3)
    rownum = 1
    colnum = 1
    fig = make_subplots(rows=rows, cols=3, subplot_titles=(cols))

    for col in cols:
        fig.add_trace(go.Histogram(x=data[col], xbins=dict(size=hist_bin_width_fd(data[col]))), row=rownum, col=colnum)

        colnum = colnum + 1
        if colnum > 3:
            rownum = rownum + 1
            colnum = 1
    fig.update_layout(
        height=300 * rows,
        width=1300,
        bargap=0.05,
        showlegend=False,
    )
    fig.show()


def plot_graphs(x1: pd.Series, x2: pd.Series, df: pd.DataFrame, feature: str, title: str) -> None:
    """Generate histogram and box plots to compare APC off versus APC on.

    Args:
        x1 (pd.Series): Series for APC off data.
        x2 (pd.Series): Series for APC on data.
        df (pd.DataFrame): Dataframe containing the data.
        feature (str): Column name of the feature to use in the dataframe.
        title (str): Title of the plot.
    """
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Histogram(x=x1, name="APC OFF", xbins=dict(size=hist_bin_width_fd(df[feature])), histnorm="probability", marker=dict(color="rgba(198,12,48,0.5)")), row=1, col=1)
    fig.add_trace(go.Histogram(x=x2, name="APC ON", xbins=dict(size=hist_bin_width_fd(df[feature])), histnorm="probability", marker=dict(color="rgba(0,39,118,0.5)")), row=1, col=1)

    fig.add_trace(go.Box(y=x1, name="APC OFF", boxmean="sd", fillcolor="rgba(198,12,48,0.5)", marker=dict(color="rgba(198,12,48,0.5)")), row=1, col=2)
    fig.add_trace(go.Box(y=x2, name="APC ON", boxmean="sd", fillcolor="rgba(0,39,118,0.5)", marker=dict(color="rgba(0,39,118,0.5)")), row=1, col=2)

    fig["layout"].update(
        title="<b>" + title + "</b><br>Date range: " + str(min(x1.index)) + " to " + str(max(x1.index)) + "</i>",
        font=dict(size=9),
        margin=dict(l=60, r=60, t=60, b=60),
        annotations=[
            dict(x=0, y=-0.2, showarrow=False, text="The mean is represented by the dashed horizontal line and the standard deviation by the dashed diamond shape.", xref="paper", yref="paper")
        ],
        barmode="overlay",
        showlegend=False,
    )
    fig.show(renderer="notebook")


def generate_stats(x1: pd.Series, x2: pd.Series) -> pd.DataFrame:
    """Generate summary statistic to compare APC off versus APC on.

    Args:
        x1 (pd.Series): Series for APC off data.
        x2 (pd.Series): Series for APC on data.

    Returns:
        pd.DataFrame: Dataframe of summary statistics to compare the two series.
    """
    data_for_stats = {"APC OFF": x1.describe(), "APC ON": x2.describe()}
    data_stats = pd.DataFrame(data_for_stats)
    data_stats2 = data_stats.transpose()
    data_stats2.insert(loc=1, column="% count", value=data_stats.loc["count"] / sum(data_stats.loc["count"]) * 100)

    data_stats2["low_fence"] = data_stats.loc["25%"] - 1.5 * (data_stats.loc["75%"] - data_stats.loc["25%"])
    data_stats2["high_fence"] = data_stats.loc["75%"] + 1.5 * (data_stats.loc["75%"] - data_stats.loc["25%"])
    data_stats2["data min"] = np.where(data_stats2["low_fence"] > data_stats2["min"], data_stats2["low_fence"], data_stats2["min"])
    data_stats2["data max"] = np.where(data_stats2["high_fence"] < data_stats2["max"], data_stats2["high_fence"], data_stats2["max"])

    data_stats2.insert(
        loc=3,
        column="mean \u0394",
        value=data_stats.loc[
            "mean",
        ].diff(),
    )
    data_stats2.insert(
        loc=4,
        column="% mean \u0394",
        value=data_stats.loc[
            "mean",
        ].pct_change()
        * 100,
    )
    data_stats2.insert(
        loc=6,
        column="% std \u0394",
        value=data_stats.loc[
            "std",
        ].pct_change()
        * 100,
    )

    data_stats2.drop(["low_fence", "high_fence"], axis="columns", inplace=True)
    return data_stats2


def plot_timeseries(df: pd.DataFrame, y_traces: list, title: str, x_trace: str = "", use_index: bool = True) -> None:
    """Plot timeseries data from dataframe using plotly library.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        y_traces (list): List of columns to include for the y-axis.
        title (str): Title for the plot.
        x_trace (str, optional): Name of column to use as x-axis. Defaults to "".
        use_index (bool, optional): Specificy as False to use column in x_trace. Defaults to True.
    """
    fig = go.Figure()

    if use_index == True:
        X_trace = df.index
    else:
        X_trace = df[x_trace]

    for y_trace in y_traces:
        fig.add_trace(
            go.Scatter(
                x=X_trace,
                y=df[y_trace],
                name=y_trace,
            )
        )

    fig["layout"].update(title="<b>" + title + "</b>")
    fig.show(renderer="notebook")