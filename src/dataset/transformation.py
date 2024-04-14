from pyspark.sql import DataFrame
from pyspark.sql.functions import col, collect_list, struct, udf, explode
from pyspark.sql.types import ArrayType, StructType


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
    return sorted_df, udf_schema


def ungroup(
    df: DataFrame, alias: str, original_schema: ArrayType, exploded_col: str = "exloded"
) -> DataFrame:
    """
    For testing purposes, we also have an ungroup function that will
    undo the combining done by gather_and_sort, though it will not
    restore the original order of the rows.
    @param df: DataFrame
    @param alias: Column to ungroup
    @param original_schema: Original schema used for grouping
    @param exploded_col: Column name for exploded column, it is used
            temporarily during the ungrouping process, if `df` already has
            a column named `exploded`, it will be replaced
    @return: Ungrouped DataFrame
    """
    struct_schema = original_schema.elementType  # Get the StructType from the ArrayType
    ungrouped_df = df.select(df.columns + [explode(df[alias]).alias(f"{exploded_col}")])
    for field in struct_schema.fields:
        ungrouped_df = ungrouped_df.withColumn(
            field.name, col(f"{exploded_col}." + field.name)
        )
    ungrouped_df = ungrouped_df.drop(f"{exploded_col}", alias)
    return ungrouped_df
