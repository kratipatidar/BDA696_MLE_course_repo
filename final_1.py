import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import sqlalchemy
import statsmodels.api
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    # creating a directory to store the plots
    if not os.path.exists("~/results/plots"):
        os.makedirs("~/results/plots")

    # connecting to the database to fetch the table
    db_user = "root"
    db_pass = "secret"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = f"mysql://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
    SELECT * FROM final_baseball_features;
    """
    baseball_df = pd.read_sql_query(query, sql_engine)
    print(baseball_df.head())

    # dropping empty columns from the dataframe
    baseball_df.dropna(axis=1)

    # defining response
    response = 'home_team_wins'

    # making list and dataframe just for predictors
    baseball_df_preds = baseball_df.iloc[:, :-1]
    predictors = baseball_df_preds.columns

    # defining a list for the plots
    cat_res_cont_pred_plots = []

    # defining lists for logistic regression
    p_values = []
    t_values = []
    log_reg_plots = []

    # defining lists for difference with mean of response
    diff_w_mean_of_response_weighted = []
    diff_w_mean_ranks = []
    diff_w_mean_weighted_ranks = []
    diff_w_mean_plots = []

    # making categorical response/ continuous predictor distribution plots
    for i, p in enumerate(predictors):
        hist_data = [baseball_df[baseball_df[response] == 0][p],
                     baseball_df[baseball_df[response] == 1][p]]
        group_labels = ["Response=0", "Response=1"]

        fig_i = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_i.update_layout(
            title=f"Categorical Response by Continuous Predictor {p}",
            xaxis_title="Response",
            yaxis_title="Distribution",
        )
        fig_i.write_html(
            file=f"~/results/plots/dist_plot_for_pred_{p}.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the figures list
        cat_res_cont_pred_plots.append(
            "<a href = ~/results/plots/dist_plot_for_pred_{}.html>"
            'Plot For {}'
            "</a>".format(p, p)
        )

        # Logistic Regression
        # ## p-value and t-value rankings

        logistic_regression_model = statsmodels.api.Logit(baseball_df[response], baseball_df[p].values)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        # print("Variable: {}".format(p))
        # print(logistic_regression_model_fitted.summary())

        # statistics
        t_value = logistic_regression_model_fitted.tvalues
        p_value = logistic_regression_model_fitted.pvalues
        t_values.append(t_value)
        p_values.append(p_value)

        # creating plots
        fig_i = px.scatter(x=baseball_df[p].values, y=baseball_df[response].values, trendline="ols")
        fig_i.update_layout(
            title=f"variable : {p}: (t-value = {t_value}) "
                  f"(p-value = {p_value}",
            xaxis_title="Variable:{}".format(p),
            yaxis_title="y",
        )
        fig_i.write_html(
            file=f'~/results/plots/scatter_plot_LogReg_var_{p}.html',
            include_plotlyjs='cdn',
        )
        # appending the links to the list
        log_reg_plots.append(
            '<a href ="~/results/plots/scatter_plot_LogReg_var_{}.html">'
            'Plot For {}'
            '</a>'.format(p, p)
        )

        # difference with mean of response
        ## rankings

        x = pd.DataFrame({"intervals": pd.cut(baseball_df[p], 10)})
        x["target"] = baseball_df[response]
        x = x.groupby("intervals").agg({"target": ["count", "mean"]})
        x.columns = ["bin_counts", "bin_mean"]
        x.reset_index(inplace=True)
        x["pop_mean"] = baseball_df[response].mean()
        lefts = []
        rights = []
        for interval in x.intervals:
            lefts.append(interval.left)
            rights.append(interval.right)
        x['left'] = lefts
        x['right'] = rights
        x['bins'] = x['left'] + x['right'] / 2
        x["mean_diff"] = x["bin_mean"] - x["pop_mean"]
        x["mean_sq_diff"] = x["mean_diff"] ** 2
        x["population_proportion"] = x["bin_counts"] / len(baseball_df)
        x["mean_square_diff_weighted"] = (
                x["population_proportion"] * x["mean_sq_diff"]
        )
        x['rank_val'] = x['mean_square_diff_weighted'].sum()
        diff_w_mean_of_response_weighted.append(x)
        diff_w_mean_ranks.append(x['mean_sq_diff'].sum())
        diff_w_mean_weighted_ranks.append(x['mean_square_diff_weighted'].sum())

    ## plots
    # generating plots for difference with mean of response
    for tdf, predictor in zip(diff_w_mean_of_response_weighted, predictors):
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
                y=tdf["mean_diff"],
                name="Difference in Means",
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
        diff_with_mean_plot.update_layout(
            title=f"Plot for Variable {predictor}",
            xaxis_title="Predictor Bins",
            yaxis_title="Response",
        )
        diff_with_mean_plot.write_html(
            file=f'~/results/plots/difference_with_mean_for_var_{predictor}.html',
            include_plotlyjs='cdn',
        )
        diff_w_mean_plots.append(
            '<a href = "~/results/plots/difference_with_mean_for_var_{}.html">'
            'Plot For Predictor {}'
            '</a>'.format(predictor, predictor)
        )

    # defining lists for random forest
    rf_imp = []
    # random forest importance rankings
    rand_imp = RandomForestClassifier(
        n_estimators=65, oob_score=True, random_state=0
    )
    rand_imp.fit(baseball_df_preds, baseball_df[response])
    importance = rand_imp.feature_importances_
    rf_imp.append(importance)

    # making a single dataframe for all the rankings coded above
    ranking_table_1 = pd.DataFrame()
    ranking_table_1['Predictor'] = predictors
    ranking_table_1['Pred_w_Response_Plots'] = cat_res_cont_pred_plots
    ranking_table_1['Log_Reg_p_values'] = p_values
    ranking_table_1['Log_Reg_t_values'] = t_values
    ranking_table_1['Log_Reg_Plots'] = log_reg_plots
    ranking_table_1['Diff_w_Mean'] = diff_w_mean_ranks
    ranking_table_1['Diff_w_Mean_weighted'] = diff_w_mean_weighted_ranks
    ranking_table_1['DWM_plots'] = diff_w_mean_plots
    ranking_table_1['RF_imp_ranks'] = rf_imp[0].tolist()

    ranking_table_1 = ranking_table_1.sort_values('Diff_w_Mean_weighted', ascending=False)
    ranking_table_1.to_html("krati_patidar_FP1_table1.html", render_links=True, escape=False)

    # defining lists for correlation
    cont_cont_corr = []
    x_cont_cont_corr = []
    y_cont_cont_corr = []

    # Cont-Cont Correlation
    for x in predictors:
        for y in predictors:
            a, b = stats.pearsonr(baseball_df[x], baseball_df[y])
            cont_cont_corr.append(a)
            x_cont_cont_corr.append(x)
            y_cont_cont_corr.append(y)

    # creating a cont-cont correlation table
    cont_cont_corr_table = pd.DataFrame(
        columns=["predictor_1", "predictor_2", "corr_metric"]
    )
    cont_cont_corr_table["predictor_1"] = x_cont_cont_corr
    cont_cont_corr_table["predictor_2"] = y_cont_cont_corr
    cont_cont_corr_table["corr_metric"] = cont_cont_corr

    # writing to html
    cont_cont_corr_table.to_html("correlation_table.html", render_links=True, escape=False)

    # creating a correlation plot
    cont_cont_corr_plot = go.Figure(
        data=go.Heatmap(
            x=cont_cont_corr_table["predictor_1"],
            y=cont_cont_corr_table["predictor_2"],
            z=cont_cont_corr_table["corr_metric"],
        )
    )
    cont_cont_corr_plot.update_layout(
        title="cont_cont_corr_plot",
        xaxis_title="predictor_1",
        yaxis_title="predictor_2",
    )
    cont_cont_corr_plot.write_html(
        file=f"~/results/plots/cont_cont_corr_plot1.html",
        include_plotlyjs="cdn",
    )

    # brute force variable combinations
    ## cont-cont difference with mean of response weighted tables
    brute_force_cont_cont = pd.DataFrame(
        columns=[
            "pred_1",
            "pred_2",
            "diff_w_mean",
            "diff_w_mean_weighted",
            "bf_heatmap",
        ]
    )
    pred1_BF_cont_cont = []
    pred2_BF_cont_cont = []
    diff_w_mean_cont_cont = []
    diff_w_mean_weighted_cont_cont = []
    bf_heatmap_cont_cont = []
    bf_tables = []

    for x in predictors:
        for y in predictors:
            if x == y:
                continue
            else:
                pred1_BF_cont_cont.append(x)
                pred2_BF_cont_cont.append(y)
                binned_df = pd.DataFrame(
                    {

                        "pred1": baseball_df[x],
                        "pred2": baseball_df[y],
                        "resp": baseball_df[response],
                        "bin1": pd.qcut(baseball_df[x], 10, duplicates="drop"),
                        "bin2": pd.qcut(baseball_df[y], 10, duplicates="drop"),
                    }
                )

                bin_df_grouped = binned_df.groupby(["bin1", "bin2"]).agg(
                    {"resp": ["count", "mean"]}
                )
                bin_df_grouped = bin_df_grouped.reset_index()
                bin_df_grouped.columns = [x, y, "BinCounts", "BinMeans"]
                PopulationMean = baseball_df[response].mean()
                PopulationProportion = bin_df_grouped["BinCounts"] / sum(
                    bin_df_grouped["BinCounts"]
                )
                bin_df_grouped["MeanSqDiff"] = (
                                                       bin_df_grouped["BinMeans"] - PopulationMean
                                               ) ** 2
                bin_df_grouped["MeanSqDiffWeight"] = (
                        bin_df_grouped["MeanSqDiff"] * PopulationProportion
                )
                bin_df_grouped = bin_df_grouped.dropna(axis=0, how='any')
                diff_w_mean_cont_cont.append(sum(bin_df_grouped["MeanSqDiff"]))
                diff_w_mean_weighted_cont_cont.append(sum(bin_df_grouped["MeanSqDiffWeight"]))
                bf_tables.append(bin_df_grouped)
                # cont-cont weighted mean of response correlation plot
                heatmap_data = pd.pivot_table(
                    bin_df_grouped,
                    index=x,
                    columns=y,
                    values="MeanSqDiffWeight",
                )
                heatplot = sns.heatmap(heatmap_data, annot=False, cmap="RdBu", xticklabels=True, yticklabels=True)
                fig = heatplot.get_figure()
                fig.savefig("~/results/plots/BF" + x + "_and_" + y + ".png")
                plt.clf()
                # adding the link to the corr_plots list
                bf_heatmap_cont_cont.append(
                    "<a href = "
                    + "results/figures/BF"
                    + x
                    + "_and_"
                    + y
                    + ".png"
                    + ">"
                    + "heatmap_" + x + "_and_" + y
                    + "</a>"
                )

    # creating ranking table 2
    ranking_table_2 = pd.DataFrame()
    ranking_table_2['Predictor_1'] = x_cont_cont_corr
    ranking_table_2['Predictor_2'] = y_cont_cont_corr
    ranking_table_2['Correlation'] = cont_cont_corr
    ranking_table_2['Corr_Plot'] = '<a href = "~/results/plots/cont_cont_corr_plot.html">' \
                                   'corr_plot' \
                                   '</a>'

    # sorting values
    ranking_table_2 = ranking_table_2.sort_values('Correlation', ascending=False)

    # writing to html
    ranking_table_2.to_html("krati_patidar_FP1_table2.html", render_links=True, escape=False)

    # creating rankin_table_3
    ranking_table_3 = pd.DataFrame()
    ranking_table_3['Predictor_1'] = pred1_BF_cont_cont
    ranking_table_3['Predictor_2'] = pred2_BF_cont_cont
    ranking_table_3['Diff_w_Mean'] = diff_w_mean_cont_cont
    ranking_table_3['Diff_w_Mean_Weighted'] = diff_w_mean_weighted_cont_cont
    ranking_table_3['BF_plots'] = bf_heatmap_cont_cont

    # sorting values
    ranking_table_3 = ranking_table_3.sort_values('Diff_w_Mean_Weighted', ascending=False)
    ranking_table_3.to_html("krati_patidar_FP1_table3.html", render_links=True, escape=False)

    # Model Building and Training
    ## First we split our data into train and test sets

    X = baseball_df_preds.values
    y = baseball_df[response].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    # Now we create lists to store models and their respective scores
    models = ["Logistic Regression", "SVM Classifier",
              "Random Forest Classifier", "Naive Bayes Classifier"]
    scores = []

    # Training a Logistic Regression Model
    train_LogReg = LogisticRegression(random_state=0).fit(X_train, y_train)

    # Getting Test Score
    te_LogReg = train_LogReg.score(X_test, y_test)
    scores.append(te_LogReg)

    # Training a SVM classifier
    train_SVM = svm.SVC().fit(X_train, y_train)

    # Getting Test Score
    te_SVM = train_SVM.score(X_test, y_test)
    scores.append(te_SVM)

    # Training a Random Forest Classifier
    X_train, y_train = make_classification(n_features=33,
                                           random_state=0)
    train_RF = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    # Getting Score
    te_RF = train_RF.score(X_test, y_test)
    scores.append(te_RF)

    # Training a Naive Bayes Classifier
    train_NB = GaussianNB().fit(X_train, y_train)

    # Getting the Score
    te_NB = train_NB.score(X_test, y_test)
    scores.append(te_NB)

    # Making a dataframe to record final performances
    model_perf = pd.DataFrame()

    # Appending values to this dataframe
    model_perf['model_name'] = models
    model_perf['perf_metric'] = scores

    # writing this dataframe to a csv
    model_perf.to_html('Final_Model_Performance.html', render_links=True, escape=False)


if __name__ == "__main__":
    sys.exit(main())
