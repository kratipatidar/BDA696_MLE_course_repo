import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def main():
    # reading in the data
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    iris_df = pd.read_csv(
        "C:/Users/KRATI PATIDAR/Desktop/BDA696_MLE/iris.data", names=column_names
    )
    print(iris_df.head())

    # summary statistics
    iris_arr = iris_df.to_numpy()
    print(iris_arr)

    print("Mean = ", np.mean(iris_df))
    print("Minimum = ", np.min(iris_df))
    print("Maximum = ", np.max(iris_df))

    print("First quantile = ", np.quantile(iris_arr[:, :-1], q=0.25, axis=0))
    print("Second quantile = ", np.quantile(iris_arr[:, :-1], q=0.50, axis=0))
    print("Third quantile = ", np.quantile(iris_arr[:, :-1], q=0.75, axis=0))
    print("Fourth quantile = ", np.quantile(iris_arr[:, :-1], q=1, axis=0))

    print(iris_df["class"].unique())

    # making plots

    plot_1 = px.scatter(
        iris_df,
        x="sepal_width",
        y="sepal_length",
        size="petal_length",
        hover_data=["petal_width"],
        color="class",
    )

    plot_1.show()

    plot_2 = px.line(iris_df, x="petal_width", y="petal_length", color="class")
    plot_2.show()

    plot_3 = px.violin(iris_df, x="sepal_width", y="sepal_length", color="class")
    plot_3.show()

    plot_4 = px.scatter_3d(
        iris_df, x="sepal_length", y="sepal_width", z="petal_length", color="class"
    )
    plot_4.show()

    plot_5 = px.line_3d(
        iris_df,
        x="petal_width",
        y="petal_length",
        z="sepal_width",
        hover_data=["sepal_length"],
        color="class",
    )

    plot_5.show()

    # normalization and random forest

    x = iris_arr[:, 0:-1]
    y = iris_df["class"].values

    pipeline = Pipeline(
        [
            ("normalize", Normalizer()),
            ("randomforest", RandomForestClassifier(random_state=1234)),
        ]
    )

    print(pipeline.fit(x, y))

    if __name__ == "__main__":
        sys.exit(main())
