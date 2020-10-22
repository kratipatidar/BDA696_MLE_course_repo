import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# lists to store variables
categorical_predictors = []
continuous_predictors = []
response_type = []
predictor_type = []
figures = []
p_values = []
t_values = []
lr_plots = []
logr_plots = []
predictors_bins = []
diff_w_mean_of_response = []
difference_w_mean_of_response_ranks = []
diff_w_mean_of_response_weighted = []
diff_w_mean_of_response_weighted_ranks = []
diff_w_mean_of_response_plots = []
random_for_imp = []


def main(input_file, response):
    df = pd.read_csv(input_file)
    df_preds = df.loc[:, df.columns != response]
    predictors = df_preds.columns

    # segregating the predictors into continuous and categorical predictors
    for col in predictors:
        if (df[col].dtype == object) or (
                (df[col].nunique() / len(df[col])) < 0.05
        ):
            categorical_predictors.append(col)
        else:
            continuous_predictors.append(col)

    # making a list of predictor types
    for pred in predictors:
        if pred in continuous_predictors:
            predictor_type.append('continuous')
        else:
            predictor_type.append('categorical')

    # determining whether response variable is categorical or continuous

    if (df[response].dtype == object) or (
            (df[response].nunique() / len(df[response])) < 0.05
    ):
        response_type.append('categorical_response')
    else:
        response_type.append('continuous_response')

    # if the response is categorical (boolean), converting the response into codes
    if response_type[0] == 'categorical_response':
        y1 = df[response].astype('category').cat.codes

    # generating plots

    # creating a directory to store the plots
    if not os.path.exists("~/plots"):
        os.makedirs("~/plots")

    # continuous response with different cases

    if response_type[0] == "continuous_response":
        for i, p in enumerate(continuous_predictors):  # first we consider continuous predictor
            fig_i = px.scatter(x=df[p], y=df[response], trendline="ols")
            fig_i.update_layout(
                title="Continuous Response by "
                      "Continuous Predictor for Variable {}".format(p),
                xaxis_title=p,
                yaxis_title="Response",
            )
            fig_i.write_html(
                file=f'~/plots/cont_res_cont_pred_scatter_var_{p}.html',
                include_plotlyjs='cdn',
            )
            # adding the link to the figures list
            figures.append(
                '<a href =" ~/plots/cont_res_cont_pred_scatter_var_{}.html"> '
                'plot for {}'
                ' </a>'.format(p, p)
            )

        for i, p in enumerate(categorical_predictors):
            # violin plot
            hist_data = [df[df[p] == cat][response] for cat in df[p].unique()]
            group_labels = ['group_{}'.format(n) for n, cat in enumerate(df[p].unique())]
            fig_i = go.Figure()
            for curr_hist, curr_group in zip(hist_data, group_labels):
                fig_i.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(df)),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_i.update_layout(
                title="Continuous Response by "
                      "Categorical Predictor for Variable {}".format(p),
                xaxis_title=p,
                yaxis_title="Response",
            )
            fig_i.write_html(
                file=f'~/plots/cont_res_cat_pred_dist_var_{p}.html',
                include_plotlyjs='cdn',
            )
            # adding the link to the figures list
            figures.append(
                '<a href = "~/plots/cont_res_cat_pred_dist_var_{}.html">'
                'plot for {}'
                '</a>'.format(p, p)
            )

    # categorical response with different cases

    elif response_type[0] == "categorical_response":
        for i, p in enumerate(continuous_predictors):
            hist_data = [df[df[response].astype('category').cat.codes == 0][p],
                         df[df[response].astype('category').cat.codes == 1][p]]
            group_labels = ["Response=0", "Response=1"]

            # violin plot

            fig_i = go.Figure()
            for curr_hist, curr_group in zip(hist_data, group_labels):
                fig_i.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(df)),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_i.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor_{}".format(p),
            )
            fig_i.write_html(
                file=f'~/plots/cat_res_cont_pred_dist_var_{p}.html',
                include_plotlyjs='cdn',
            )
            # adding the link to the figures list
            figures.append(
                '<a href =" ~/plots/cat_res_cont_pred_dist_var_{}.html">'
                'plot for {}'
                '</a>'.format(p, p)
            )

        for i, p in enumerate(categorical_predictors):
            conf_matrix = confusion_matrix(df[p], df[response])
            fig_i = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_i.update_layout(
                title="Categorical Response by "
                      "Categorical Predictor for Variable {}".format(p),
                xaxis_title="Response",
                yaxis_title="Predictor_{}".format(p),
            )
            fig_i.write_html(
                file=f'~/plots/cat_res_cat_pred_heat_var_{p}.html',
                include_plotlyjs='cdn',
            )
            # adding the link to the figures list
            figures.append(
                "<a href = ~/plots/cat_res_cat_pred_heat_var_{}.html>"
                'plot for {}'
                "</a>".format(p, p)
            )

    # linear regression rankings
    if response_type[0] == "continuous_response":
        y = df[response].values
        for idx, pred in enumerate(predictors):
            linear_regression_model = statsmodels.api.OLS(y, df[pred])
            linear_regression_model_fitted = linear_regression_model.fit()
            print("Variable: {}".format(pred))
            print(linear_regression_model_fitted.summary())

            # statistics
            t_value = linear_regression_model_fitted.tvalues
            p_value = linear_regression_model_fitted.pvalues
            t_values.append(t_value)
            p_values.append(p_value)

            # creating plots
            fig_idx = px.scatter(x=df[pred], y=y, trendline="ols")
            fig_idx.update_layout(
                title=f"variable : {pred}: (t-value = {t_value}) "
                      f"(p-value = {p_value}",
                xaxis_title="Variable:{}".format(pred),
                yaxis_title="y",
            )
            fig_idx.write_html(
                file=f'~/plots/scatter_plot_var_{pred}.html',
                include_plotlyjs='cdn',
            )
            # appending the links to the list
            lr_plots.append(
                '<a href = "~/plots/scatter_plot_lr_var_{}.html">'
                'plot for {}'
                '</a>'.format(pred, pred)
            )

    # logistic regression rankings

    else:
        y = df[response].astype('category').cat.codes

        for i, predictor in enumerate(predictors):
            logistic_regression_model = statsmodels.api.Logit(y, df[predictor].values)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            print("Variable: {}".format(predictor))
            print(logistic_regression_model_fitted.summary())

            # statistics
            t_value = logistic_regression_model_fitted.tvalues
            p_value = logistic_regression_model_fitted.pvalues
            t_values.append(t_value)
            p_values.append(p_value)

            # creating plots
            fig_i = px.scatter(x=df[predictor].values, y=y, trendline="ols")
            fig_i.update_layout(
                title=f"variable : {predictor}: (t-value = {t_value}) "
                      f"(p-value = {p_value}",
                xaxis_title="Variable:{}".format(predictor),
                yaxis_title="y",
            )
            fig_i.write_html(
                file=f'~/plots/scatter_plot_logr_var_{predictor}.html',
                include_plotlyjs='cdn',
            )
            # appending the links to the list
            logr_plots.append(
                '<a href =" ~/plots/scatter_plot_logr_var_{}.html">'
                'plot for {}'
                '</a>'.format(predictor, predictor)
            )

    # difference with mean of response rankings

    for predictor in continuous_predictors:
        predictors_bins.append(pd.cut(df[predictor], 10))
    for predictor_bins in predictors_bins:
        x = pd.DataFrame({"intervals": predictor_bins})
        if response_type[0] == 'continuous_response':
            x["target"] = df[response]
        else:
            x["target"] = df[response].astype('category').cat.codes
        x = x.groupby("intervals").agg({"intervals": "count", "target": "mean"})
        x.columns = ["bin_counts", "bin_mean"]
        x.reset_index(inplace=True)
        if response_type[0] == 'continuous_response':
            x["pop_mean"] = df[response].mean()
        else:
            x["pop_mean"] = df[response].astype('category').cat.codes.mean()
        lefts = []
        rights = []
        for interval in x.intervals:
            lefts.append(interval.left)
            rights.append(interval.right)
        x['left'] = lefts
        x['right'] = rights
        x['bins'] = x['left'] + x['right'] / 2
        x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
        x["mean_sq_diff"] = x["mean_diff"] ** 2
        x["rank_val"] = x["mean_sq_diff"].sum() / len(x)
        diff_w_mean_of_response.append(x)

    for predictor in categorical_predictors:
        x = pd.DataFrame({"bins": df[predictor].unique()})
        if response_type[0] == 'continuous_response':
            x["target"] = df[response]
        else:
            x["target"] = df[response].astype('category').cat.codes
        x = x.groupby("bins").agg({"bins": "count", "target": "mean"})
        x.columns = ["bin_counts", "bin_mean"]
        x.reset_index(inplace=True)
        if response_type[0] == 'continuous_response':
            x["pop_mean"] = df[response].mean()
        else:
            x["pop_mean"] = df[response].astype('category').cat.codes.mean()
        x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
        x["mean_sq_diff"] = x["mean_diff"] ** 2
        x["rank_val"] = x["mean_sq_diff"].sum() / len(x)
        diff_w_mean_of_response.append(x)

    for x in diff_w_mean_of_response:
        difference_w_mean_of_response_ranks.append(x.loc[0, 'rank_val'])

    # weighted difference with mean of response

    for predictor_bins in predictors_bins:
            x = pd.DataFrame({"intervals": predictor_bins})
            if response_type[0] == 'continuous_response':
                x["target"] = df[response]
            else:
                x["target"] = df[response].astype('category').cat.codes
            x = x.groupby("intervals").agg({"intervals": "count", "target": "mean"})
            x.columns = ["bin_counts", "bin_mean"]
            x.reset_index(inplace=True)
            if response_type[0] == 'continuous_response':
                x["pop_mean"] = df[response].mean()
            else:
                x["pop_mean"] = df[response].astype('category').cat.codes.mean()
            lefts = []
            rights = []
            for interval in x.intervals:
                lefts.append(interval.left)
                rights.append(interval.right)
            x['left'] = lefts
            x['right'] = rights
            x['bins'] = x['left'] + x['right'] / 2
            x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
            x["mean_sq_diff"] = x["mean_diff"] ** 2
            x["population_proportion"] = x["bin_counts"] / len(df)
            x["mean_square_diff_weighted"] = (
                    x["population_proportion"] * x["mean_sq_diff"]
                )
            x['rank_val'] = x['mean_square_diff_weighted'].sum()
            diff_w_mean_of_response_weighted.append(x)

    for predictor in categorical_predictors:
        x = pd.DataFrame({"bins": df[predictor].unique()})
        if response_type[0] == 'continuous_response':
            x["target"] = df[response]
        else:
            x["target"] = df[response].astype('category').cat.codes
        x = x.groupby("bins").agg({"bins": "count", "target": "mean"})
        x.columns = ["bin_counts", "bin_mean"]
        x.reset_index(inplace=True)
        if response_type[0] == 'continuous_response':
            x["pop_mean"] = df[response].mean()
        else:
            x["pop_mean"] = df[response].astype('category').cat.codes.mean()
        x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
        x["mean_sq_diff"] = x["mean_diff"] ** 2
        x["population_proportion"] = x["bin_counts"] / len(df)
        x["mean_square_diff_weighted"] = (
                x["population_proportion"] * x["mean_sq_diff"]
        )
        x['rank_val'] = x['mean_square_diff_weighted'].sum()
        diff_w_mean_of_response_weighted.append(x)

    for x in diff_w_mean_of_response_weighted:
        diff_w_mean_of_response_weighted_ranks.append(x.loc[0, 'rank_val'])

    # generating plots for difference with mean of response

    for i, tdf in enumerate(diff_w_mean_of_response):
        diff_with_mean_plot = make_subplots(specs=[[{"secondary_y": True}]])

        # adding traces
        diff_with_mean_plot.add_trace(
                go.Bar(
                    x=tdf["bins"],
                    y=tdf["bin_counts"],
                    name=" Histogram",
                ),
                secondary_y=False,
        )

        diff_with_mean_plot.add_trace(
                go.Scatter(
                    x=tdf["bins"],
                    y=tdf["bin_mean"],
                    name="Bin Mean",
                    line=dict(color="red"),
                ),
                secondary_y=True,
        )

        diff_with_mean_plot.add_trace(
                go.Scatter(
                    x=tdf["bins"],
                    y=tdf["pop_mean"],
                    name="Population Mean",
                    line=dict(color="green"),
                ),
                secondary_y=True,
        )
        diff_with_mean_plot.write_html(
                file=f'~/plots/difference_with_mean_for_var_{i}.html',
                include_plotlyjs='cdn',
            )
        diff_w_mean_of_response_plots.append(
                '<a href = "~/plots/difference_with_mean_var_{}.html">'
                'plot for predictor {}'
                '</a>'.format(i, i)
        )

    # random forest variable importance ranking
    if response_type[0] == "continuous_response":
        # RF regressor for continuous response
        rand_imp = RandomForestRegressor(
            n_estimators=65, oob_score=True, random_state=4
        )
        rand_imp.fit(df_preds, df[response].values)
        importance = rand_imp.feature_importances_
        random_for_imp.append(importance)
    else:
        # using RF classifier and coding the categorical response
        df[response] = df[response].astype("category")
        df[response] = df[response].cat.codes
        rand_imp = RandomForestClassifier(
            n_estimators=65, oob_score=True, random_state=4
        )
        rand_imp.fit(df_preds, df[response])
        importance = rand_imp.feature_importances_
        random_for_imp.append(importance)

    # creating a final output dataframe
    output_df = pd.DataFrame(columns=[
        "predictor",
        "predictor_type",
        "link_to_plot",
        "p_value",
        "t_value",
        "diff_w_mean",
        "diff_w_mean_weighted",
        "diff_w_mean_plot",
        "random_for_imp"])

    # assigning values to each column
    output_df['predictor'] = predictors
    output_df['predictor_type'] = predictor_type
    output_df['link_to_plot'] = figures[0:len(predictors)]
    output_df['p_value'] = p_values
    output_df['t_value'] = t_values
    output_df['random_for_imp'] = random_for_imp[0].tolist()
    output_df['diff_w_mean'] = difference_w_mean_of_response_ranks
    output_df['diff_w_mean_weighted'] = diff_w_mean_of_response_weighted_ranks
    output_df['diff_w_mean_plot'] = diff_w_mean_of_response_plots

    # saving the output to a HTML file
    output_df.to_html("assignment_4_kratipatidar.html", render_links=True, escape=False)


if __name__ == "__main__":
    input_file = 'D:/archive/data.csv'
    response = 'diagnosis'
    sys.exit(main(input_file, response))

