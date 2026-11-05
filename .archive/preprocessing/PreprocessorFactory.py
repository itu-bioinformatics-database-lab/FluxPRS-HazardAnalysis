from abc import ABC, abstractmethod
from .Preprocessor import *

class PreprocessorFactory(ABC):

    def __init__(self, metabolite_levels_file_path):
        self.metabolite_levels_file_path = metabolite_levels_file_path
    @abstractmethod
    def create_preprocessor(self):
        pass

class MetaboliteLevelsPreprocessorFactory(PreprocessorFactory):
    def create_preprocessor(self):
        return MetaboliteLevelsPreprocessor(self.metabolite_levels_file_path)


class PathwayPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, pathway_file_path, x_pathway_path):
        self.pathway_file_path = pathway_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_pathway_path = x_pathway_path
        
    def create_preprocessor(self):
        return PathwayPreprocessor(self.metabolite_levels_file_path, self.pathway_file_path, self.x_pathway_path)


class ReactionPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, reaction_file_path, x_reaction_path):
        self.reaction_file_path = reaction_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_reaction_path = x_reaction_path
        
    def create_preprocessor(self):
        return ReactionPreprocessor(self.metabolite_levels_file_path, self.reaction_file_path, self.x_reaction_path)


class MetabolitePathwayCombinedPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, metabolite_level_pathway_combined_file_path, x_pathway_path):
        self.metabolite_level_pathway_combined_file_path = metabolite_level_pathway_combined_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_pathway_path = x_pathway_path
        
    def create_preprocessor(self):
        return MetabolitePathwayCombinedPreprocessor(self.metabolite_levels_file_path, self.metabolite_level_pathway_combined_file_path, self.x_pathway_path)


class MetaboliteReactionCombinedPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, metabolite_level_reaction_combined_file_path, x_reaction_path):
        self.metabolite_level_reaction_combined_file_path = metabolite_level_reaction_combined_file_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.x_reaction_path = x_reaction_path
        
    def create_preprocessor(self):
        return MetaboliteReactionCombinedPreprocessor(self.metabolite_levels_file_path, self.metabolite_level_reaction_combined_file_path, self.x_reaction_path)


class TranscriptomicsPreprocessorFactory(PreprocessorFactory):

    def __init__(self, transcriptomics_path):
        self.transcriptomics_path = transcriptomics_path
        
    def create_preprocessor(self):
        return TranscriptomicsPreprocessor(self.transcriptomics_path)  # no metabolite path needed


class TranscriptomicsMetabolitePreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, transcriptomics_metabolites_combined_path, transcriptomics_path):
        self.transcriptomics_metabolites_combined_path = transcriptomics_metabolites_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        
    def create_preprocessor(self):
        return TranscriptomicsMetabolitePreprocessor(self.metabolite_levels_file_path, self.transcriptomics_metabolites_combined_path, self.transcriptomics_path)


class TranscriptomicsPathwayPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, transcriptomics_pathway_combined_path, transcriptomics_path, x_pathway_path):
        self.transcriptomics_pathway_combined_path = transcriptomics_pathway_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        self.x_pathway_path = x_pathway_path
        
    def create_preprocessor(self):
        return TranscriptomicsPathwayPreprocessor(self.metabolite_levels_file_path, self.transcriptomics_pathway_combined_path, self.transcriptomics_path, self.x_pathway_path)


class TranscriptomicsReactionPreprocessorFactory(PreprocessorFactory):

    def __init__(self, metabolite_levels_file_path, transcriptomics_reaction_combined_path, transcriptomics_path, x_reaction_path):
        self.transcriptomics_reaction_combined_path = transcriptomics_reaction_combined_path
        self.metabolite_levels_file_path = metabolite_levels_file_path
        self.transcriptomics_path = transcriptomics_path
        self.x_reaction_path = x_reaction_path
        
    def create_preprocessor(self):
        return TranscriptomicsReactionPreprocessor(self.metabolite_levels_file_path, self.transcriptomics_reaction_combined_path, self.transcriptomics_path, self.x_reaction_path)