import os
import sys
from pyspark.sql import SparkSession
from datasets import Dataset
from src.dataset.transformation import gather_and_sort, ungroup
from src.dataset.embeddings import Embeddings
import json


def process_data(
    json_file: str,
    gather_col: str,
    sort_col: str,
    alias: str,
    target_col: str,
    exploded_col: str,
    emb_out_col: str,
    target_out_col: str,
    app_name: str = "Data Processing",
    memory: str = "1g",
    output_dir: str = "./data",
    train_split: float = 0.8,
    model: str = "all-MiniLM-L6-v2",
    train_ds_name: str = "train_dataset",
    test_ds_name: str = "test_dataset",
    unique_targets_name: str = "unique_targets.json",
    train_target_dist_name: str = "train_target_dist.json",
    test_target_dist_name: str = "test_target_dist.json",
):
    """
    Loads a JSON file into Spark, gathers and sorts the data, and then
    splits the resulting DataFrame into training and testing sets.

    Also ungroups the training and testing sets to check the distribution
    of the target column, saving the results to the output directory as a JSON
    file

    @param json_file: The path to the JSON file
    @param gather_col: The column to gather by, for example, "userId"
    @param sort_col: The column to sort by, for example, "timestamp" or "ts"
    @param alias: The alias to use for the gathered column, it should be
        a column that does not exist in the original DataFrame otherwise
        it will be overwritten. It will contain tuple of embeddings
        and target vector for each row
    @param target_col: The target column to check the distribution of
    @param exploded_col: The column which will be used during the
        checking of distribution, note that if a column with the same
        name exists in the original DataFrame it will be overwritten
    @param emb_out_col: The column to save the embeddings to, if it exists
        in the original DataFrame it will be overwritten
    @param target_out_col: The column to save the target vector to, if it exists
        in the original DataFrame it will be overwritten
    @param app_name: The name of the Spark application
    @param memory: The amount of memory to allocate to the Spark driver
    @param output_dir: The directory to save the output to
    @param train_split: The proportion of the data to use for training
    @param model: The name of the SentenceTransformer model to use
    """
    # pyspark setup
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .appName(app_name)
        .getOrCreate()
    )

    # initial data processing
    df = spark.read.json(json_file)
    unique_targets = (
        df.select(target_col).distinct().rdd.map(lambda row: row[target_col]).collect()
    )
    unique_targets = {t: i for i, t in enumerate(unique_targets)}
    target_groups = df.groupBy(target_col).count().collect()
    target_groups = {row[target_col]: row["count"] for row in target_groups}
    df_processed, udf_schema = gather_and_sort(
        df=df,
        gather_col=gather_col,
        sort_col=sort_col,
        alias=alias,
    )

    # checking distribution of target column
    train, test = df_processed.randomSplit([train_split, 1 - train_split])
    train_ungrouped = ungroup(
        df=train, alias=alias, original_schema=udf_schema, exploded_col=exploded_col
    )
    test_ungrouped = ungroup(
        df=test, alias=alias, original_schema=udf_schema, exploded_col=exploded_col
    )
    train_target_groups = train_ungrouped.groupBy(target_col).count().collect()
    train_target_groups = {row[target_col]: row["count"] for row in train_target_groups}
    test_target_groups = test_ungrouped.groupBy(target_col).count().collect()
    test_target_groups = {row[target_col]: row["count"] for row in test_target_groups}

    # computing embeddings
    all_cols = df.columns
    filter_cols = set([gather_col, sort_col])
    cols = [c for c in all_cols if c not in filter_cols]
    embedding_processor = Embeddings(model)
    train_emb_targ = embedding_processor.process_df(
        df=train,
        cols=cols,
        target_col=target_col,
        unique_targets=unique_targets,
        data_col=alias,
        emb_out_col=emb_out_col,
        target_out_col=target_out_col,
    )
    test_emb_targ = embedding_processor.process_df(
        df=test,
        cols=cols,
        target_col=target_col,
        unique_targets=unique_targets,
        data_col=alias,
        emb_out_col=emb_out_col,
        target_out_col=target_out_col,
    )

    # turn into huggingface dataset
    train_dataset = Dataset.from_spark(train_emb_targ)
    test_dataset = Dataset.from_spark(test_emb_targ)

    # define output paths
    os.makedirs(output_dir, exist_ok=True)
    train_ds_path = os.path.join(output_dir, "train_dataset")
    test_ds_path = os.path.join(output_dir, "test_dataset")
    unique_targets_path = os.path.join(output_dir, "unique_targets.json")
    train_target_dist_path = os.path.join(output_dir, "train_target_dist.json")
    test_target_dist_path = os.path.join(output_dir, "test_target_dist.json")
    target_dist_path = os.path.join(output_dir, "target_dist.json")

    # save output
    train_dataset.save_to_disk(train_ds_path)
    test_dataset.save_to_disk(test_ds_path)
    with open(unique_targets_path, "w") as f:
        json.dump(unique_targets, f)
    with open(train_target_dist_path, "w") as f:
        json.dump(train_target_groups, f)
    with open(test_target_dist_path, "w") as f:
        json.dump(test_target_groups, f)
    with open(target_dist_path, "w") as f:
        json.dump(target_groups, f)
