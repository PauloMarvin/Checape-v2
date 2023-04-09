import pandas as pd
from sklearn.utils import resample


class DataframeUtils:
    @staticmethod
    def undersampling_df(unbalanced_dataframe: pd.DataFrame, y_target: str):
        groups = unbalanced_dataframe.groupby(y_target)
        minority_group = groups.size().min()

        balanced_df = pd.DataFrame()
        for group_name, group in groups:
            balanced_group = resample(group, replace=False, n_samples=minority_group, random_state=42)
            balanced_df = pd.concat([balanced_df, balanced_group])

        return balanced_df.reset_index(drop=True)

    @staticmethod
    def remove_by_text_len(dataframe: pd.DataFrame, text_column: str, min_len: int = 5):
        dataframe = dataframe.assign(number_words=dataframe[text_column].apply(lambda x: len(x.split(" "))))
        dataframe = dataframe.drop(dataframe[dataframe.number_words < min_len].index)

        return dataframe.reset_index(drop=True)
