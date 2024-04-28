from sentence_transformers import SentenceTransformer
from pyspark.ml.linalg import MatrixUDT, DenseMatrix, VectorUDT, DenseVector
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, struct
from typing import List
from pyspark.sql import DataFrame
import torch
from tqdm.auto import tqdm
from src.common.cache import LimitedSizeDict
from functools import lru_cache


class Embeddings:
    def __init__(self, model_name, device="cpu", cache_size=1000):
        self.model = SentenceTransformer(model_name, device=device)
        self.col_emb = LimitedSizeDict(size_limit=cache_size)
        self.body_emb = LimitedSizeDict(size_limit=cache_size)
        self.cache_size = cache_size

        @lru_cache(maxsize=cache_size)
        def call_model_encode(text):
            return self.model.encode(text, convert_to_tensor=True)

        self.call_model_encode = call_model_encode

    def get_emb_dim_size(self):
        return self.model.get_sentence_embedding_dimension()

    def cache_col_emb(self, cols):
        if set(cols) == set(self.col_emb.keys()):
            return
        for c in cols:
            self.col_emb[c] = self.model.encode(c)

    def encode_row(self, row, cols, group_col, sort_col, target_col):
        # to make sure that target_col is encoded last, we remove
        # it from the cols and add it back at the end
        cols = [c for c in cols if c != target_col]
        cols.append(target_col)
        result = []
        for d in row[0]:
            for c in cols:
                if c not in [group_col, sort_col]:
                    k = c
                    v = d[c]
                    if k not in self.col_emb:
                        self.col_emb[k] = self.model.encode(k, convert_to_tensor=True)
                    v = str(v)
                    # if v not in self.body_emb:
                    #     self.body_emb[v] = self.model.encode(v, convert_to_tensor=True)
                    k_emb = self.col_emb[k]
                    v_emb = self.call_model_encode(v)
                    summed = k_emb + v_emb
                    result.append(summed)
        return torch.stack(result)

    def encode_row_targets(
        self, row, cols, group_col, sort_col, target_col, unique_targets, target_dist
    ):
        cols = [c for c in cols if c not in [group_col, sort_col, target_col]]
        cols.append(target_col)
        result = []
        for d in row[0]:
            for c in cols:
                if c == target_col:
                    v = d[c]
                    idx = unique_targets[v]
                    if v not in target_dist:
                        target_dist[v] = 0
                    target_dist[v] += 1
                else:
                    idx = -1
                result.append(idx)
        return torch.tensor(result)

    def encode_df(self, df, cols, group_col, sort_col, target_col, unique_targets):
        tensors = []
        targets = []
        target_dist = {}
        for index, row in tqdm(df.iterrows(), total=len(df)):
            tensor = self.encode_row(row, cols, group_col, sort_col, target_col)
            target = self.encode_row_targets(
                row, cols, group_col, sort_col, target_col, unique_targets, target_dist
            )
            tensors.append(tensor)
            targets.append(target)
        return tensors, targets, target_dist

    @staticmethod
    def compute_embeddings(data, cols, column_embeddings, model) -> DenseMatrix:
        embeddings = []
        for row in data:
            for c in cols:
                body = row[c]
                body = str(body)
                name_embedding = column_embeddings[c]
                body_embedding = model.encode(body)
                summed_embedding = name_embedding + body_embedding
                embeddings.append(summed_embedding)
        embeddings = np.stack(embeddings)
        return DenseMatrix(
            embeddings.shape[0],
            embeddings.shape[1],
            embeddings.flatten().tolist(),
            isTransposed=True,
        )

    @staticmethod
    def compute_target_vector(data, cols, unique_targets, target_col):
        targets = []
        for row in data:
            for c in cols:
                body = row[c]
                if body in unique_targets and c == target_col:
                    targets.append(unique_targets[body])
                else:
                    targets.append(-1)
        return DenseVector(targets)

    def process_df(
        self,
        df: DataFrame,
        cols: List[str],
        target_col: str,
        unique_targets: dict,
        data_col: str,
        emb_out_col: str,
        target_out_col: str,
    ) -> DataFrame:
        """
        Process a DataFrame to add embeddings and target vectors
        with the data column containing a tuple of embeddings and target vectors
        for each row

        @param df: DataFrame whose columns are equal to or a subset of `cols`
        @param cols: Columns to compute embeddings for
        @param target_col: Target column to create the target vector
        @param unique_targets: Mapping from unique values on target column to integers
            greater than or equal to 0
        @param data_col: Column name containing the data to compute embeddings for
        @param emb_out_col: Column name for the embeddings, used to store the embeddings
            before it is combined with the target vector and stored in `data_col` column
        @param target_out_col: Column name for the target vector, used to store the target
            vector the embeddings and it are combined and stored in `data_col` column
        @return: DataFrame with the `data_col` column containing the tuple of embeddings
            and target vectors for each row
        """
        self.cache_col_emb(cols)
        compute_emb_udf = udf(
            lambda x: Embeddings.compute_embeddings(x, cols, self.col_emb, self.model),
            MatrixUDT(),
        )
        compute_tar_udf = udf(
            lambda x: Embeddings.compute_target_vector(
                x, cols, unique_targets, target_col
            ),
            VectorUDT(),
        )
        df_emb = df.withColumn(emb_out_col, compute_emb_udf(col(data_col)))
        df_emb_target = df_emb.withColumn(
            target_out_col, compute_tar_udf(col(data_col))
        )
        df_emb_target = df_emb_target.drop(data_col)
        df_emb_target = df_emb_target.withColumn(
            data_col, struct(df_emb_target[emb_out_col], df_emb_target[target_out_col])
        )
        return df_emb_target
