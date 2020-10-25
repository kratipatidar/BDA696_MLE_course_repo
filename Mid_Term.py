import sys
import warnings

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy import stats

# lists to store variables

categorical_predictors = []
continuous_predictors = []
response_type = []
predictor_type = []
x_cont_cont_corr = []
y_cont_cont_corr = []
cont_cont_corr = []
x_cat_cont_corr = []
y_cat_cont_corr = []
cat_cont_corr = []
x_cat_cat_corr = []
y_cat_cat_corr = []
cat_cat_corr = []
corr_plots = []
all_buckets = []
diff_w_mean_of_response = []
corr_tables = []
brute_force_tables = []
brute_force_plots = []
difference_w_mean_of_response_ranks = []
diff_w_mean_of_response_weighted = []
diff_w_mean_of_response_weighted_ranks = []
diff_w_mean_of_response_plots = []


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


# function for cat-cat correlation
def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff

        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff

    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


# function for cat-cont correlation
def cat_cont_correlation_ratio(categories, values):
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def main(input_file, response):
    df = pd.read_csv(input_file)
    df.dropna(axis=1)
    df_preds = df.loc[:, df.columns != response]
    predictors = df_preds.columns

    # segregating the predictors into continuous and categorical predictors
    for col in predictors:
        if (df[col].dtype == object) or ((df[col].nunique() / len(df[col])) < 0.05):
            categorical_predictors.append(col)
        else:
            continuous_predictors.append(col)

    predictors = continuous_predictors + categorical_predictors

    # making a list of predictor types
    for pred in predictors:
        if pred in continuous_predictors:
            predictor_type.append("continuous")
        else:
            predictor_type.append("categorical")

    # determining whether response variable is categorical or continuous

    if (df[response].dtype == object) or (
            (df[response].nunique() / len(df[response])) < 0.05
    ):
        response_type.append("categorical_response")
    else:
        response_type.append("continuous_response")

    # code for cont-cont correlation
    if len(continuous_predictors) > 0:
        for x in continuous_predictors:
            for y in continuous_predictors:
                a, b = stats.pearsonr(df[x], df[y])
                cont_cont_corr.append(a)
                x_cont_cont_corr.append(x)
                y_cont_cont_corr.append(y)

        # creating a cont-cont correlation table

        cont_cont_corr_table = pd.DataFrame(
            columns=["predictor_1", "predictor_2", "corr_metric", "link_to_plots"]
        )
        cont_cont_corr_table["predictor_1"] = x_cont_cont_corr
        cont_cont_corr_table["predictor_2"] = y_cont_cont_corr
        cont_cont_corr_table["corr_metric"] = cont_cont_corr
        cont_cont_corr_table["link_to_plots"] = (
            '<a href = "assignment_4_kratipatidar.html">' "link_to_predictor_plots" "</a>"
        )
        cont_cont_corr_table.to_html("contcont.html", render_links=True, escape=False)
        corr_tables.append('<a href = "contcont.html"> contcont </a>')
        # creating cont-cont correlation heatmap

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
            file=f"~/plots/cont_cont_corr_plot.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the corr_plots list
        corr_plots.append(
            "<a href = ~/plots/cont_cont_corr_plot_for.html>" "cont_cont_corr" "</a>"
        )
    else:
        corr_plots.append('NA')
        corr_tables.append('NA')

    # code for cat_cat correlation
    if len(categorical_predictors) > 0:
        for x in categorical_predictors:
            for y in categorical_predictors:
                corr = cat_correlation(df[x], df[y], bias_correction=True, tschuprow=False)
                cat_cat_corr.append(corr)
                x_cat_cat_corr.append(x)
                y_cat_cat_corr.append(y)

        # creating a cat-cat correlation table

        cat_cat_corr_table = pd.DataFrame(
            columns=["predictor_1", "predictor_2", "corr_metric", "link_to_plots"]
        )
        cat_cat_corr_table["predictor_1"] = x_cat_cat_corr
        cat_cat_corr_table["predictor_2"] = y_cat_cat_corr
        cat_cat_corr_table["corr_metric"] = cat_cat_corr
        cat_cat_corr_table["link_to_plots"] = (
            '<a href = "assignment_4_kratipatidar.html">' "link_to_predictor_plots" "</a>"
        )
        cat_cat_corr_table.to_html("catcat.html", render_links=True, escape=False)
        corr_tables.append('<a href = "catcat.html"> catcat </a>')

        # creating cat-cat correlation heatmap

        cat_cat_corr_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cat_corr_table["predictor_1"],
                y=cat_cat_corr_table["predictor_2"],
                z=cat_cat_corr_table["corr_metric"],
            )
        )
        cat_cat_corr_plot.update_layout(
            title="cat_cat_corr_plot",
            xaxis_title="predictor_1",
            yaxis_title="predictor_2",
        )
        cat_cat_corr_plot.write_html(
            file=f"~/plots/cat_cat_corr_plot.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the corr_plots list
        corr_plots.append("<a href = ~/plots/cat_cat_corr_plot.html>" "cat_cat_corr" "</a>")
    else:
        corr_plots.append('NA')
        corr_tables.append('NA')

    # code for cat_cont correlation

    if len(categorical_predictors) > 0 and len(continuous_predictors) > 0:
        for x in continuous_predictors:
            for y in categorical_predictors:
                corr = cat_cont_correlation_ratio(
                    np.asarray(df[y].unique()), np.asarray(df[x])
                )
                cat_cont_corr.append(corr)
                x_cat_cont_corr.append(x)
                y_cat_cont_corr.append(y)

            # creating a cat-cont correlation table

        cat_cont_corr_table = pd.DataFrame(
            columns=["predictor_1", "predictor_2", "corr_metric", "link_to_plots"]
        )
        cat_cont_corr_table["predictor_1"] = x_cat_cont_corr
        cat_cont_corr_table["predictor_2"] = y_cat_cont_corr
        cat_cont_corr_table["corr_metric"] = cat_cont_corr
        cat_cont_corr_table["link_to_plots"] = (
            '<a href = "assignment_4_kratipatidar.html">' "link_to_predictor_plots" "</a>"
        )
        cat_cont_corr_table.to_html("catcont.html", render_links=True, escape=False)
        corr_tables.append('<a href = "catcont.html"> catcont </a>')
        # creating cat-cont correlation heatmap

        cat_cont_corr_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cont_corr_table["predictor_1"],
                y=cat_cont_corr_table["predictor_2"],
                z=cat_cont_corr_table["corr_metric"],
            )
        )
        cat_cont_corr_plot.update_layout(
            title="cat_cont_corr_plot",
            xaxis_title="predictor_1",
            yaxis_title="predictor_2",
        )
        cat_cont_corr_plot.write_html(
            file=f"~/plots/cat_cont_corr_plot.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the corr_plots list
        corr_plots.append(
            "<a href = ~/plots/cat_cont_corr_plot.html>" "cat_cont_corr" "</a>"
        )
    else:
        corr_plots.append('NA')
        corr_tables.append("NA")

    # Correlation Tables
    output_df_1 = pd.DataFrame(columns=["tables", "matrices"])
    output_df_1["tables"] = corr_tables
    output_df_1["matrices"] = corr_plots

    # Brute Force Tables
    # binning continuous predictors
    if len(continuous_predictors) > 0:
        for predictor in continuous_predictors:
            all_buckets.append(pd.cut(df[predictor], 10))

        # cont-cont weighted mean of response table
        for bin1 in all_buckets:
            for bin2 in all_buckets:
                x = pd.DataFrame({"interval_1": bin1, "interval_2": bin2})
                if response_type[0] == "continuous_response":
                    x["target"] = df[response]
                else:
                    x["target"] = df[response].astype("category").cat.codes
                x = x.groupby(["interval_1", "interval_2"]).agg(
                    {"interval_2": "count", "target": "mean"}
                )
                x.columns = ["bin_counts", "bin_mean"]
                x.reset_index(inplace=True)
                if response_type[0] == "continuous_response":
                    x["pop_mean"] = df[response].mean()
                else:
                    x["pop_mean"] = df[response].astype("category").cat.codes.mean()
                lefts1 = []
                rights1 = []
                for interval in x.interval_1:
                    lefts1.append(interval.left)
                    rights1.append(interval.right)
                x["left_1"] = lefts1
                x["right_1"] = rights1
                x["bins_1"] = x["left_1"] + x["right_1"] / 2
                lefts2 = []
                rights2 = []
                for interval in x.interval_2:
                    lefts2.append(interval.left)
                    rights2.append(interval.right)
                x["left_2"] = lefts2
                x["right_2"] = rights2
                x["bins_2"] = x["left_2"] + x["right_2"] / 2
                x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
                x["mean_sq_diff"] = x["mean_diff"] ** 2
                x["population_proportion"] = x["bin_counts"] / len(df)
                x["mean_square_diff_weighted"] = (
                        x["population_proportion"] * x["mean_sq_diff"]
                )
                x["rank_val"] = x["mean_square_diff_weighted"].sum()
        cont_cont_diff_w_mean_weighted = pd.DataFrame()
        cont_cont_diff_w_mean_weighted = x
        cont_cont_diff_w_mean_weighted = cont_cont_diff_w_mean_weighted.sort_values(
            by="mean_square_diff_weighted", ascending=False
        )

        # writing to html
        cont_cont_diff_w_mean_weighted.to_html(
            "brute_force_cont_cont.html", render_links=True, escape=False
        )
        brute_force_tables.append('<a href = "brute_force_cont_cont.html"> cont_bf </a>')

        # cont-cont weighted mean of response correlation plot
        cont_cont_diff_w_mean_weighted_plot = go.Figure(
            data=go.Heatmap(
                x=cont_cont_diff_w_mean_weighted["bins_1"],
                y=cont_cont_diff_w_mean_weighted["bins_2"],
                z=cont_cont_diff_w_mean_weighted["mean_square_diff_weighted"],
            )
        )
        cont_cont_diff_w_mean_weighted_plot.update_layout(
            title="cont_cont_dwm_weighted",
            xaxis_title="predictor_1",
            yaxis_title="predictor_2",
        )
        cont_cont_diff_w_mean_weighted_plot.write_html(
            file=f"~/plots/cont_cont_dwm_weighted.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the corr_plots list
        diff_w_mean_of_response_plots.append(
            "<a href = ~/plots/cont_cont_dwm_weighted.html>" "cont_cont_dwm" "</a>"
        )
    else:
        diff_w_mean_of_response_plots.append('NA')
        brute_force_tables.append('NA')

    # cat-cont weighted mean of response table
    if len(categorical_predictors) > 0 and len(continuous_predictors) > 0:
        for predictor in categorical_predictors:
            for bn in all_buckets:
                x = pd.DataFrame({"interval_1": df[predictor], "interval_2": bn})
                x["interval_2"] = bn
                if response_type[0] == "continuous_response":
                    x["target"] = df[response]
                else:
                    x["target"] = df[response].astype("category").cat.codes
                x = x.groupby(["interval_1", "interval_2"]).agg(
                    {"interval_2": "count", "target": "mean"}
                )
                x.columns = ["bin_counts", "bin_mean"]
                x.reset_index(inplace=True)
                if response_type[0] == "continuous_response":
                    x["pop_mean"] = df[response].mean()
                else:
                    x["pop_mean"] = df[response].astype("category").cat.codes.mean()
                lefts2 = []
                rights2 = []
                for interval in x.interval_2:
                    lefts2.append(interval.left)
                    rights2.append(interval.right)
                x["left_2"] = lefts2
                x["right_2"] = rights2
                x["bins"] = x["left_2"] + x["right_2"] / 2
                x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
                x["mean_sq_diff"] = x["mean_diff"] ** 2
                x["population_proportion"] = x["bin_counts"] / len(df)
                x["mean_square_diff_weighted"] = (
                        x["population_proportion"] * x["mean_sq_diff"]
                )
                x["rank_val"] = x["mean_square_diff_weighted"].sum()
        cat_cont_diff_w_mean_weighted = pd.DataFrame()
        cat_cont_diff_w_mean_weighted = x
        cat_cont_diff_w_mean_weighted = cat_cont_diff_w_mean_weighted.sort_values(
            by="mean_square_diff_weighted", ascending=False
        )
        # writing to html
        cat_cont_diff_w_mean_weighted.to_html(
            "brute_force_cat_cont.html", render_links=True, escape=False
        )
        brute_force_tables.append('<a href = "brute_force_cat_cont.html"> catcont_bf </a>')

        # cat-cont weighted mean of response correlation plot
        cat_cont_diff_w_mean_weighted_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cont_diff_w_mean_weighted["interval_1"],
                y=cat_cont_diff_w_mean_weighted["bins"],
                z=cat_cont_diff_w_mean_weighted["mean_square_diff_weighted"],
            )
        )
        cat_cont_diff_w_mean_weighted_plot.update_layout(
            title="cat_cont_dwm_weighted",
            xaxis_title="predictor_1",
            yaxis_title="predictor_2",
        )
        cat_cont_diff_w_mean_weighted_plot.write_html(
            file=f"~/plots/cat_cont_dwm_weighted.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the list
        diff_w_mean_of_response_plots.append(
            "<a href = ~/plots/cat_cont_dwm_weighted.html>" "cat_cont_dwm" "</a>"
        )
    else:
        diff_w_mean_of_response_plots.append('NA')
        brute_force_tables.append('NA')

    # cat-cat difference with mean of response weighted table
    if len(categorical_predictors) > 0:
        for x in categorical_predictors:
            x = pd.DataFrame({"interval_1": df[x]})
            for y in categorical_predictors:
                x["interval_2"] = df[y]
                if response_type[0] == "continuous_response":
                    x["target"] = df[response]
                else:
                    x["target"] = df[response].astype("category").cat.codes
                x = x.groupby(["interval_1", "interval_2"]).agg(
                    {"interval_2": "count", "target": "mean"}
                )
                x.columns = ["bin_counts", "bin_mean"]
                x.reset_index(inplace=True)
                if response_type[0] == "continuous_response":
                    x["pop_mean"] = df[response].mean()
                else:
                    x["pop_mean"] = df[response].astype("category").cat.codes.mean()
                x["mean_diff"] = x["pop_mean"] - x["bin_mean"]
                x["mean_sq_diff"] = x["mean_diff"] ** 2
                x["population_proportion"] = x["bin_counts"] / len(df)
                x["mean_square_diff_weighted"] = (
                        x["population_proportion"] * x["mean_sq_diff"]
                )
                x["rank_val"] = x["mean_square_diff_weighted"].sum()
        cat_cat_diff_w_mean_weighted = pd.DataFrame()
        cat_cat_diff_w_mean_weighted = x
        cat_cat_diff_w_mean_weighted = cat_cat_diff_w_mean_weighted.sort_values(
            by="mean_square_diff_weighted", ascending=False
        )
        # writing to html
        cat_cat_diff_w_mean_weighted.to_html(
            "brute_force_cat_cat.html", render_links=True, escape=False
        )
        brute_force_tables.append('<a href = "brute_force_cat_cat.html"> catcat_bf </a>')

        # cat-cat difference with mean weighted correlation plot
        cat_cat_diff_w_mean_weighted_plot = go.Figure(
            data=go.Heatmap(
                x=cat_cat_diff_w_mean_weighted["interval_1"],
                y=cat_cat_diff_w_mean_weighted["interval_2"],
                z=cat_cat_diff_w_mean_weighted["mean_square_diff_weighted"],
            )
        )
        cat_cat_diff_w_mean_weighted_plot.update_layout(
            title="cat_cat_dwm_weighted",
            xaxis_title="predictor_1",
            yaxis_title="predictor_2",
        )
        cat_cat_diff_w_mean_weighted_plot.write_html(
            file=f"~/plots/cat_cat_dwm_weighted.html",
            include_plotlyjs="cdn",
        )
        # adding the link to the corr_plots list
        diff_w_mean_of_response_plots.append(
            "<a href = ~/plots/cat_cat_dwm_weighted.html>" "cat_cat_dwm" "</a>"
        )
    else:
        diff_w_mean_of_response_plots.append('NA')
        brute_force_tables.append('NA')
    # outputs of brute force

    output_df_2 = pd.DataFrame(columns=["bf_tables", "bf_plots"])
    output_df_2["bf_tables"] = brute_force_tables
    output_df_2["bf_plots"] = diff_w_mean_of_response_plots

    # writing to html
    output_df_1.to_html("Correlation.html", render_links=True, escape=False)
    output_df_2.to_html("Brute_Force.html", render_links=True, escape=False)


if __name__ == "__main__":
    input_file = "D:/archive/data.csv"
    response = "diagnosis"
    sys.exit(main(input_file, response))

