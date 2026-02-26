import pandas as pd
import time 

from sklearn_utils.utils import SkUtilsIO

from preprocessing.metabolitics_pipeline import MetaboliticsPipeline

from utils import load_metabolite_mapping
from sklearn_utils.preprocessing import FeatureRenaming

from sklearn_utils.utils import feature_importance_report

from sklearn_utils.visualization import plot_heatmap
import matplotlib.pyplot as plt


def trasnform_and_write(X_alzheimer, y_alzheimer):
    MetaboliticsPipeline.steps['metabolite-name-mapping'] = FeatureRenaming(load_metabolite_mapping("new-synonym", X_alzheimer, "MyRIAD Metabolomics"))

    transformer_pipe = MetaboliticsPipeline([
        "metabolite-name-mapping",
        "fold-change-scaler",
        "metabolitics-transformer",
        #'metabolitics-transformer-with-pfba',
    ])

    X_alzheimer_transformed = transformer_pipe.fit_transform(X=X_alzheimer, y=y_alzheimer)


    df = pd.DataFrame(X_alzheimer_transformed)
    df.to_csv("outputs/X_alzheimer_transformed.csv", index=False)

    print("Data has been written to X_alzheimer_transformed")
    return X_alzheimer_transformed

def reaction_dif_write(X_alzheimer_transformed, y_alzheimer):
    diff_score_pipe = MetaboliticsPipeline([       
        'reaction-diff' 
    ])

    X_alzheimer_reaction_diff = diff_score_pipe.fit_transform(X=X_alzheimer_transformed, y=y_alzheimer)

    # Create a DataFrame from the dictionary
    df_reaction_diff = pd.DataFrame(X_alzheimer_reaction_diff)

    df_reaction_diff.to_csv('outputs/X_alzheimer_reaction_diff.csv')

    print("Reaction diff file has been created!")

    return X_alzheimer_reaction_diff

def pathway_score_write(X_alzheimer_reaction_diff, y_alzheimer):
    pathway_pipe = MetaboliticsPipeline([       
        'pathway-transformer', 
        'transport-pathway-elimination'
    ])
    X_alzheimer_pathways = pathway_pipe.fit_transform(X=X_alzheimer_reaction_diff, y=y_alzheimer)

    df_pathway = pd.DataFrame(X_alzheimer_pathways)
    df_pathway.to_csv('outputs/X_alzheimer_pathway.csv')

    print("Pathway file has been created.")
    return X_alzheimer_pathways

def feature_importance_write():
    # # Check for NaN or infinite values
    if df_pathway.isnull().values.any():
        print("Cleaning NaN values...")
        # Fill NaN values with 0 (or another value)
        df_pathway_cleaned = df_pathway.fillna(0)
    else:
        df_pathway_cleaned = df_pathway
        
    X_alzheimer_pathways_cleaned = df_pathway_cleaned.values

    df_feature = feature_importance_report(X_alzheimer_pathways_cleaned, y_alzheimer)

    df_feature.to_csv('outputs/feature_importance_report.csv')

    print("Feature importance report has been created with all available columns!") 


def plot_heatmap_(X_alzheimer_pathways, y_alzheimer):

    plot_heatmap(X_alzheimer_pathways, y_alzheimer)

    plt.savefig('outputs/heatmap_plot.png', format='png')  # Saves the plot as a PNG file

    plt.clf()

    print("Plot has been saved to 'outputs/heatmap_plot.png'!")


def main():
    start_time = time.time()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.2g}'.format)


    X_alzheimer, y_alzheimer = SkUtilsIO("preprocessed_metabolomics_data.csv").from_csv(label_column='Diagnosis')
    X_alzheimer = X_alzheimer[:1]
    y_alzheimer = y_alzheimer[:1]
    X_alzheimer_transformed = trasnform_and_write(X_alzheimer=X_alzheimer, y_alzheimer=y_alzheimer)
    X_alzheimer_reaction_diff = reaction_dif_write(X_alzheimer_transformed, y_alzheimer)
    X_alzheimer_pathways = pathway_score_write(X_alzheimer_reaction_diff, y_alzheimer)
    plot_heatmap_(X_alzheimer_pathways, y_alzheimer)

# Entry point
if __name__ == "__main__":
    main()
