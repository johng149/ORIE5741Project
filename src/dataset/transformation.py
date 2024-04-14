from pyspark.sql import DataFrame
from pyspark.sql.functions import col, collect_list, struct, udf
from pyspark.sql.types import ArrayType


def sort_group(structure, sort_col):
    return sorted(structure, key=lambda x: x[sort_col])


def unique_prediction_targets(df: DataFrame, prediction_col: str) -> dict:
    """
    Get unique prediction targets from a DataFrame
    @param df: DataFrame
    @param prediction_col: Prediction column name
    @return: Unique prediction targets
    """
    uniques = df.select(prediction_col).distinct().collect()
    uniques_dict = {}
    for i, row in enumerate(uniques):
        uniques_dict[row[prediction_col]] = i
    return uniques_dict


def gather_and_sort(
    df: DataFrame, gather_col: str, sort_col: str, alias: str
) -> DataFrame:
    """
    Gather and sort a DataFrame
    @param df: DataFrame
    @param gather_col: Column to gather
    @param sort_col: Column to sort
    @param alias: New column name
    @return: Gathered and sorted DataFrame

    Input dataframe will first by grouped by gather_col and placed as list
    of rows in the alias column. Then it will be sorted by the sort_col.

    This will produce a DataFrame with 2 columns. One column has the identifier
    for the gather_col and the other column as a list of rows grouped by the
    identifier and sorted by the sort_col.
    """
    table_cols = df.columns
    s = struct([col(x) for x in table_cols])
    collect = collect_list(s).alias(alias)
    grouped = df.groupBy(gather_col).agg(collect)
    udf_schema = ArrayType(df.schema)
    sort_udf = udf(lambda x: sort_group(x, sort_col), udf_schema)
    sorted_df = grouped.withColumn(alias, sort_udf(col(alias)))
    return sorted_df
