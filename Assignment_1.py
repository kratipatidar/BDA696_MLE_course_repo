import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier


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
        title="Scatter Plot for all variables for different classes",
    )

    plot_1.show()

    plot_2 = px.line(
        iris_df,
        x="petal_width",
        y="petal_length",
        color="class",
        title="Line Plot for Petal Width and Petal Length for all classes",
    )
    plot_2.show()

    plot_3 = px.violin(
        iris_df,
        x="sepal_width",
        y="sepal_length",
        color="class",
        title="Violin Plot for sepal length and sepal width for all classes",
    )
    plot_3.show()

    plot_4 = px.scatter_3d(
        iris_df,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="class",
        title="3-D Scatter Plot for sepal length, sepal width and petal length",
    )
    plot_4.show()

    plot_5 = px.line_3d(
        iris_df,
        x="petal_width",
        y="petal_length",
        z="sepal_width",
        hover_data=["sepal_length"],
        color="class",
        title="3-D Line Plot for all variables of all classes ",
    )

    plot_5.show()

    # normalization, random forest and decision tree classifiers

    x = iris_arr[:, 0:-1]
    y = iris_df["class"].values

    # pipeline_1 for random forest classifier

    pipeline_1 = Pipeline(
        [
            ("normalize", Normalizer()),
            ("randomforest", RandomForestClassifier(random_state=1234)),
        ]
    )

    print(pipeline_1.fit(x, y))

    # pipeline_2 for decision tree classifier

    pipeline_2 = Pipeline(
        [
            ("normalize", Normalizer()),
            ("decisiontree", DecisionTreeClassifier()),
        ]
    )

    print(pipeline_2.fit(x, y))

    if __name__ == "__main__":
        sys.exit(main())
