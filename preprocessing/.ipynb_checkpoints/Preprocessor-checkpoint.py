from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from analysis.config import *

class Preprocessor(ABC):
    def __init__(self):
        self.name = None
        self.label = None
        self.file_path = None
    def prepare_df(self, drop_columns = ["Gender"], reference_class = "Control", sample_id_col_name = "Sample ID"):
        df = pd.read_csv(self.file_path, sep=",")
        df = self.exclude_columns_all_zero(df)
        
        drop_columns = [column for column in drop_columns if column in df.columns]
        df = df.drop(columns = drop_columns)
        
        df = df.dropna()
        df['Diagnosis'] = (df['Diagnosis'] != reference_class).astype(int)

        sample_ids = df[sample_id_col_name]
        df = df.drop(columns=[sample_id_col_name])
        df["order_index"] = range(len(df))
        order_index = df['order_index']
        return df, sample_ids, order_index
        
    @abstractmethod
    def transform_custom_file(self):
        pass

    def z_standardize(self, x_train, x_test):
        scaler = StandardScaler()
        numeric_cols = x_train.select_dtypes(include=[np.number]).columns

        x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
        x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

        return x_train, x_test
    def exclude_columns_all_zero(self, df):
        df = df.loc[:, (df != 0).any(axis=0)]
        return df


class MetaboliteLevelsPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path):
        super().__init__()
        self.name = "Metabolite Levels"
        self.label = "metabolite_level"
        self.file_path = metabolite_levels_file_path
    def transform_custom_file(self):
        pass

class PathwayPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, pathway_file_path, x_pathway_path):
        super().__init__()
        self.name = "Pathway Diff Scores"
        self.label = "pathway"
        self.file_path = pathway_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_pathway_path = x_pathway_path
    def transform_custom_file(self):
        pathway_df = pd.read_csv(self.x_pathway_path)
        metabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)

        first_cols = metabolite_levels_df.iloc[:, :6]
        pathway_df = pathway_df.iloc[:, 1:]
        pathway_df = pd.concat([first_cols, pathway_df], axis=1)
        pathway_df.to_csv(self.file_path, index=False, sep=",")

class ReactionPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, reaction_file_path, x_reaction_path):
        super().__init__()
        self.name = "Reaction Diff Scores"
        self.label = "reaction"
        self.file_path = reaction_file_path
        self.x_reaction_path = x_reaction_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
    def transform_custom_file(self):
        reaction_df = pd.read_csv(self.x_reaction_path)
        metabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)

        first_cols = metabolite_levels_df.iloc[:, :6]
        reaction_df = reaction_df.iloc[:, 1:]
        reaction_df = pd.concat([first_cols, reaction_df], axis=1)

        reaction_df.to_csv(self.file_path, index=False, sep=",")

class MetabolitePathwayCombinedPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, metabolite_level_pathway_combined_file_path, x_pathway_path):
        super().__init__()
        self.name = "Metabolite Levels Pathway Combined"
        self.label = "m_level_pathway_combined"
        self.file_path = metabolite_level_pathway_combined_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_pathway_path = x_pathway_path
    def transform_custom_file(self):
        pathway_df = pd.read_csv(self.x_pathway_path)
        metabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)

        pathway_df = pathway_df.iloc[:, 1:]
        combined_df = pd.concat([metabolite_levels_df, pathway_df], axis=1)
        combined_df.to_csv(self.file_path, index=False, sep=",")

class MetaboliteReactionCombinedPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, metabolite_level_reaction_combined_file_path, x_reaction_path):
        super().__init__()
        self.name = "Metabolite Levels Reaction Combined"
        self.label = "m_level_reaction_combined"
        self.file_path = metabolite_level_reaction_combined_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_reaction_path = x_reaction_path
    def transform_custom_file(self):
        reaction_df = pd.read_csv(self.x_reaction_path)
        metabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)

        reaction_df = reaction_df.iloc[:, 1:]
        combined_df = pd.concat([metabolite_levels_df, reaction_df], axis=1)
        combined_df.to_csv(self.file_path, index=False, sep=",")

class TranscriptomicsPreprocessor(Preprocessor):
    def __init__(self, transcriptomics_path):
        super().__init__()
        self.name = "Transcriptomics"
        self.label = "transcriptomics"
        self.file_path = transcriptomics_path
    def transform_custom_file(self):
        pass

class TranscriptomicsMetabolitePreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, transcriptomics_metabolites_combined_path, transcriptomics_path):
        super().__init__()
        self.name = "Transcriptomics Metabolite Levels Combined"
        self.label = "transcriptomics_metabolite_combined"
        self.file_path = transcriptomics_metabolites_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
    def transform_custom_file(self):
        transcriptomics_df = pd.read_csv(self.transcriptomics_path)
        metabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)
        metabolite_levels_columns_to_drop = metabolite_levels_df.columns[1:6]
        metabolite_levels_df = metabolite_levels_df.drop(columns=metabolite_levels_columns_to_drop)

        transcriptomics_df["Sample ID"] = transcriptomics_df["Sample ID"].astype(str)
        metabolite_levels_df["Sample ID"] = metabolite_levels_df["Sample ID"].astype(str)
        
        combined_df = pd.merge(transcriptomics_df, metabolite_levels_df, on="Sample ID", how="inner")

        combined_df.to_csv(self.file_path, index=False, sep=",")

class TranscriptomicsPathwayPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, transcriptomics_pathway_combined_path, transcriptomics_path, x_pathway_path):
        super().__init__()
        self.name = "Transcriptomics Pathway Diff Combined"
        self.label = "transcriptomics_pathway_combined"
        self.file_path = transcriptomics_pathway_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        self.x_pathway_path = x_pathway_path
    def transform_custom_file(self):
        transcriptomics_df = pd.read_csv(self.transcriptomics_path)
        pathway_df = pd.read_csv(self.x_pathway_path)
        pathway_df = pathway_df.iloc[:, 1:]

        mettabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)
        sample_ids = mettabolite_levels_df.iloc[:, 0]
        pathway_df = pd.concat([sample_ids, pathway_df], axis=1)

        transcriptomics_df["Sample ID"] = transcriptomics_df["Sample ID"].astype(str)
        pathway_df["Sample ID"] = pathway_df["Sample ID"].astype(str)
        
        combined_df = pd.merge(transcriptomics_df, pathway_df, on="Sample ID", how="inner")

        combined_df.to_csv(self.file_path, index=False, sep=",")

class TranscriptomicsReactionPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, transcriptomics_reaction_combined_path, transcriptomics_path, x_reaction_path):
        super().__init__()
        self.name = "Transcriptomics Reaction Diff Combined"
        self.label = "transcriptomics_reaction_combined"
        self.file_path = transcriptomics_reaction_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        self.x_reaction_path = x_reaction_path
    def transform_custom_file(self):
        transcriptomics_df = pd.read_csv(self.transcriptomics_path)
        reaction_df = pd.read_csv(self.x_reaction_path)
        reaction_df = reaction_df.iloc[:, 1:]

        mettabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)
        sample_ids = mettabolite_levels_df.iloc[:, 0]
        pathway_df = pd.concat([sample_ids, reaction_df], axis=1)

        transcriptomics_df["Sample ID"] = transcriptomics_df["Sample ID"].astype(str)
        pathway_df["Sample ID"] = pathway_df["Sample ID"].astype(str)
        combined_df = pd.merge(transcriptomics_df, pathway_df, on="Sample ID", how="inner")

        combined_df.to_csv(self.file_path, index=False, sep=",")
class Prs_metabolite_combinedPreprocessor(Preprocessor):
    def __init__(self, metabolite_levels_file_path, transcriptomics_reaction_combined_path, transcriptomics_path, x_reaction_path):
        super().__init__()
        self.name = "PRS Metabolite Combined"
        self.label = "prs_metabolite_combined"
        self.file_path = transcriptomics_reaction_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        self.x_reaction_path = x_reaction_path
    def transform_custom_file(self):
        transcriptomics_df = pd.read_csv(self.transcriptomics_path)
        reaction_df = pd.read_csv(self.x_reaction_path)
        reaction_df = reaction_df.iloc[:, 1:]

        mettabolite_levels_df = pd.read_csv(self.metabolite_levels_file_path)
        sample_ids = mettabolite_levels_df.iloc[:, 0]
        pathway_df = pd.concat([sample_ids, reaction_df], axis=1)

        transcriptomics_df["Sample ID"] = transcriptomics_df["Sample ID"].astype(str)
        pathway_df["Sample ID"] = pathway_df["Sample ID"].astype(str)
        
        combined_df = pd.merge(transcriptomics_df, pathway_df, on="Sample ID", how="inner")

        combined_df.to_csv(self.file_path, index=False, sep=",")