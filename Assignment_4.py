import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets
from sklearn.metrics import confusion_matrix


def main():
    # loading a test dataset to check the code
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df["target"] = pd.Series(boston.target)
    boston_df.head()

    categorical_predictors = []
    continuous_predictors = []
    a = []
    figures_cont_res_cont_pred = []
    figures_cont_res_cat_pred = []
    figures_cat_res_cont_pred = []
    figures_cat_res_cat_pred = []
    figures_lin_reg = []
    figures_log_reg = []
    t_vals_lin_reg = []
    p_vals_lin_reg = []
    t_vals_log_reg = []
    p_vals_log_reg = []
    all_bins = []
    diff_w_mean_of_response_dfs = []
    diff_w_mean_of_response_weighted_dfs = []

    def analysis_tool(df, pred_cols, res_col):

        # segregating the predictors into continuous and categorical predictors

        for col in pred_cols:
            if (df[col].dtype == int or df[col].dtype == float) and (
                (df[col].nunique() / len(df[col])) > 0.01
            ):
                continuous_predictors.append(col)
            else:
                categorical_predictors.append(col)

        # determining whether response variable is categorical or continuous

        if (df[res_col].dtype == int or df[res_col].dtype == float) and (
            (df[res_col].nunique() / len(df[res_col])) > 0.01
        ):
            a.append("cont_res")
        else:
            a.append("cat_res")

        # generating plots

        # creating a directory to store the plots
        if not os.path.exists("~/plots"):
            os.makedirs("~/plots")

        # continuous response with different cases

        if a[0] == "cont_res":
            for i, p in enumerate(continuous_predictors):
                fig_i = px.scatter(x=df[p], y=df[res_col], trendline="ols")
                fig_i.update_layout(
                    title="Continuous Response by "
                    "Continuous Predictor for Variable {}".format(p),
                    xaxis_title=p,
                    yaxis_title="Response",
                )
                figures_cont_res_cont_pred.append(fig_i)

            for i, p in enumerate(categorical_predictors):
                hist_data = [df[df[p] == 0][res_col], df[df[p] == 1][res_col]]
                group_labels = ["Group 1", "Group 2"]
                fig_i = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
                fig_i.update_layout(
                    title="Continuous Response by "
                    "Categorical Predictor for Variable {}".format(p),
                    xaxis_title="Response",
                    yaxis_title="Distribution",
                )
                figures_cont_res_cat_pred.append(fig_i)
                fig_j = go.Figure()
                for curr_hist, curr_group in zip(hist_data, group_labels):
                    fig_j.add_trace(
                        go.Violin(
                            x=np.repeat(curr_group, 506),
                            y=curr_hist,
                            name=curr_group,
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
                    fig_j.update_layout(
                        title="Continuous Response by "
                        "Categorical Predictor for Variable {}".format(p),
                        xaxis_title="Response",
                        yaxis_title="Distribution",
                    )
                    figures_cont_res_cat_pred.append(fig_j)

        # categorical response with different cases

        elif a[0] == "cat_res":
            for i, p in enumerate(continuous_predictors):
                hist_data = [df[df[res_col] == 0][p], df[df[res_col] == 1][p]]
                group_labels = ["Response=0", "Response=1"]
                # distribution plot with custom bin size
                fig_i = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
                fig_i.update_layout(
                    title="Continuous Predictor by "
                    "Categorical Response for Variable {}".format(p),
                    xaxis_title="Predictor",
                    yaxis_title="Distribution",
                )
                figures_cat_res_cont_pred.append(fig_i)
                fig_j = go.Figure()
                for curr_hist, curr_group in zip(hist_data, group_labels):
                    fig_j.add_trace(
                        go.Violin(
                            x=np.repeat(curr_group, 506),
                            y=curr_hist,
                            name=curr_group,
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
                    fig_j.update_layout(
                        title="Categorical Response by "
                        "Continuous Predictor for Variable {}".format(p),
                        xaxis_title="Response",
                        yaxis_title="Distribution",
                    )
                    figures_cat_res_cont_pred.append(fig_j)

            for i, p in enumerate(categorical_predictors):
                conf_matrix = confusion_matrix(df[p], df[res_col])
                fig_i = go.Figure(
                    data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
                )
                fig_i.update_layout(
                    title="Categorical Response by "
                    "Categorical Predictor for Variable {}".format(p),
                    xaxis_title="Response",
                    yaxis_title="Predictor",
                )
                figures_cat_res_cat_pred.append(fig_i)

        # linear regression rankings
        if a[0] == "cont_res":
            all_predictors = continuous_predictors + categorical_predictors
            y = df[res_col]
            for idx, pred in enumerate(all_predictors):
                linear_regression_model = statsmodels.api.OLS(y, df[pred])
                linear_regression_model_fitted = linear_regression_model.fit()
                print("Variable: {}".format(pred))
                print(linear_regression_model_fitted.summary())

                # statistics
                t_value = linear_regression_model_fitted.tvalues
                p_value = linear_regression_model_fitted.pvalues
                t_vals_lin_reg.append(t_value)
                p_vals_lin_reg.append(p_value)

                # creating plots
                fig_idx = px.scatter(x=df[pred], y=y, trendline="ols")
                fig_idx.update_layout(
                    title=f"variable : {pred}: (t-value = {t_value}) "
                    f"(p-value = {p_value}",
                    xaxis_title="Variable:{}".format(pred),
                    yaxis_title="y",
                )
                figures_lin_reg.append(fig_idx)

        # logistic regression rankings

        elif a[0] == "cat_res":
            all_predictors = continuous_predictors + categorical_predictors
            y = df[res_col]
            for idx, pred in enumerate(all_predictors):
                logistic_regression_model = statsmodels.api.OLS(y, df[pred])
                logistic_regression_model_fitted = logistic_regression_model.fit()
                print("Variable: {}".format(pred))
                print(logistic_regression_model_fitted.summary())

                # statistics
                t_value = logistic_regression_model_fitted.tvalues
                p_value = logistic_regression_model_fitted.pvalues
                t_vals_log_reg.append(t_value)
                p_vals_log_reg.append(p_value)

                # creating plots
                fig_idx = px.scatter(x=df[pred], y=y, trendline="ols")
                fig_idx.update_layout(
                    title=f"variable : {pred}: (t-value = {t_value}) "
                    f"(p-value = {p_value}",
                    xaxis_title="Variable:{}".format(pred),
                    yaxis_title="y",
                )
                figures_log_reg.append(fig_idx)

        # difference with mean of response rankings

        for pred in all_predictors:
            all_bins.append(pd.cut(df[pred], 10))

        for pred_bins in all_bins:
            x = pd.DataFrame({"intervals": pred_bins})
            x["target"] = df[res_col]
            x = x.groupby("intervals").agg({"intervals": "count", "target": "mean"})
            x.columns = ["counts", "target_mean"]
            x.reset_index(inplace=True)
            x["pop_mean"] = df[res_col].mean()
            lefts = []
            rights = []
            for interval in pred_bins:
                lefts.append(interval.left)
                rights.append(interval.right)
            # x['left'] = lefts
            # x['right'] = rights
            # x['centre'] = x['left']+x['right']/2
            x["mean_diff"] = x["pop_mean"] - x["target_mean"]
            x["mean_sq_diff"] = x["mean_diff"] ** 2
            x["rank_val"] = x["mean_sq_diff"].sum() / len(x)
            diff_w_mean_of_response_dfs.append(x)

        # weighted difference with mean of response
        for pred in all_predictors:
            all_bins.append(pd.cut(df[pred], 10))

        for pred_bins in all_bins:
            x = pd.DataFrame({"intervals": pred_bins})
            x["target"] = df[res_col]
            x = x.groupby("intervals").agg({"intervals": "count", "target": "mean"})
            x.columns = ["counts", "target_mean"]
            x.reset_index(inplace=True)
            x["pop_mean"] = df[res_col].mean()
            lefts = []
            rights = []
            for interval in pred_bins:
                lefts.append(interval.left)
                rights.append(interval.right)
            # x['left'] = lefts
            # x['right'] = rights
            # x['centre'] = x['left']+x['right']/2
            x["mean_diff"] = x["pop_mean"] - x["target_mean"]
            x["mean_sq_diff"] = x["mean_diff"] ** 2
            x["population_proportion"] = x["counts"] / len(df)
            x["mean_square_diff_weighted"] = (
                x["population_proportion"] * x["mean_sq_diff"]
            )
            diff_w_mean_of_response_weighted_dfs.append(x)

        # writing all the figures

    analysis_tool(
        boston_df,
        [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ],
        "target",
    )


if __name__ == "__main__":
    sys.exit(main())
