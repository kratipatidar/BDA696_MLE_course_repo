import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from Rolling_Average import MovingAverageTransform


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    port = "3306"
    user = "root"
    password = "krati"  # pragma: allowlist secret

    # creating spark dataframes for queries

    df_batter_counts = (
        spark.read.format("jdbc")
            .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="batter_counts",
            user=user,
            password=password,  # pragma: allowlist secret
        )
            .load()
    )

    df_batter_counts.show()

    df_game = (
        spark.read.format("jdbc")
            .options(
            url=f"jdbc:mysql"
                f"://localhost:{port}/{database}?zeroDateTimeBehavior=convertToNull",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="game",
            user=user,
            password=password,  # pragma: allowlist secret
        )
            .load()
    )

    df_game.show()

    # creating views for queries
    df_batter_counts.createOrReplaceTempView("batter_counts")
    df_game.createOrReplaceTempView("game")

    # persisting the dataframes in memory
    df_batter_counts.persist(StorageLevel.DISK_ONLY)
    df_game.persist(StorageLevel.DISK_ONLY)

    batters_in_game_df = spark.sql(
        """select g.game_id, bc.batter, bc.Hit, bc.atBat, g.local_date \
        from batter_counts bc \
        join game          g \
        on   bc.game_id = g.game_id ;"""
    )
    batters_in_game_df.show()

    # creating temporary view and persisting
    batters_in_game_df.createOrReplaceTempView("batters_in_game")
    batters_in_game_df.persist(StorageLevel.DISK_ONLY)

    rolling_average_df = spark.sql(
        """
        SELECT curr.game_id, curr.batter, SUM(hist.Hit) as Hits, \
               SUM(hist.atBat) as atBats, curr.local_date \
        FROM   batters_in_game AS curr \
        JOIN   batters_in_game AS hist \
        ON     curr.batter = hist.batter \
        AND    curr.local_date > hist.local_date \
        AND date_sub(curr.local_date, 100) < hist.local_date \
        GROUP BY curr.game_id, curr.batter, curr.local_date;
        """
    )


    rolling_average_transform = MovingAverageTransform(
        inputCols=["Hits", "atBats"], outputCol="rolling_batting_average"
    )
    rolling_average_df = rolling_average_transform.transform(rolling_average_df)
    rolling_average_df.show()


if __name__ == "__main__":
    sys.exit(main())
