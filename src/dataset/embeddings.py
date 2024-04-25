from sentence_transformers import SentenceTransformer
from pyspark.ml.linalg import MatrixUDT, DenseMatrix, VectorUDT, DenseVector
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, struct
from typing import List
from pyspark.sql import DataFrame


class Embeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.col_emb = {}

    def get_emb_dim_size(self):
        return self.model.get_sentence_embedding_dimension()

    def cache_col_emb(self, cols):
        if set(cols) == set(self.col_emb.keys()):
            return
        for c in cols:
            self.col_emb[c] = self.model.encode(c)

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
