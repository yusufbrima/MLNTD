from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, re
pd.options.display.max_colwidth = 1000
from pathlib import Path
import re
from dotenv import load_dotenv
import os

# Load .env file from the current directory
load_dotenv()

# Access environment variables
api_key = os.getenv("OPEN_API_KEY")
debug_mode = os.getenv("DEBUG")

output_path = "./data"

#####################################################
def classify_ntd_ml(abstract_text, model_name='gpt-4o-mini'):
    prompt = f"""
    Your task is to analyze the following research abstract '{abstract_text}' and determine if it explicitly IMPLEMENTS (not just mentions) any computational/statistical methods:
        - Machine learning algorithms (e.g., random forests, SVMs, neural networks, deep learning)
        - AI methods and models (e.g., NLP, computer vision, explainable AI)
        - Statistical modeling and analysis (e.g., regression models, Bayesian methods, time series analysis)
        - Computational methods specifically designed for predictive modeling and simulations (not just mathematical descriptions)
        - Data mining or preprocessing approaches or pattern recognition algorithms
    The abstract must specifically mention USING these methods in the research, not just referencing them.
    Literature reviews or surveys should be classified as 'NO'.
    Format your response as:
    ml_class: [YES or NO]
    Provide ONLY the required output format with no additional text, explanations, or justifications.
    """
    client = OpenAI(
        api_key = api_key
    )
    system_prompt = "You are an expert computational epidemiologist specializing in neglected tropical diseases and machine learning applications in healthcare."
    input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=input,
        model=model_name,
        temperature=0.1,
        max_tokens=300  # to ensure not too much verbosity
    )
    output = chat_completion.choices[0].message.content.strip()
    ml_class =  None

    if output.startswith('ml_class:'):
      ml_class = output.split(':', 1)[1].strip()

    if not ml_class:
      return None
    else:
      return {
          'ml_class': ml_class
      }

########################################

def classify_ntd_location(affiliation_text, model_name='gpt-4o-mini'):
    prompt = f"""Analyze this following affiliation text '{affiliation_text}' and extract the country, and region, state or city (whichever is most clearly identifiable) of the researcher.
    format your ouput as:
    country: [Country name]
    region: [Region, state, or city name]
    If multiple locations are listed, choose the primary listed."""
    client = OpenAI(
        #add you OpenAI API Key
          api_key = api_key
      )
    system_prompt = "You are an expert in geographic metadata extraction."
    input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=input,
        model=model_name,
        temperature=0.1, #low temperature to ensure reproducability, as We don't need the LLM to be too creative
        max_tokens=300  # to ensure the model outputs content < 300 words
    )
    output = chat_completion.choices[0].message.content.strip()
    country, region = None, None  #default values
    lines = output.split('\n')
    for line in lines:
      if line.startswith('country:'):
        country = line.split(':', 1)[1].strip()
      elif line.startswith('region:'):
        region = line.split(':', 1)[1].strip()
    if not country or not region:
      return None
    else:
      return{
          'country': country,
          'region': region
      }

###################################################################
# add funding information

def classify_ntd_funding(funding_text, funding_details, model_name='gpt-4o-mini'):
    prompt = f"""Analyze the following funding information and extract all funding sources for this research.
    primary funding text: '{funding_text}'
    Additional funding details: '{funding_details if funding_details else "None"}'

    Follow these step by step instructions:
    1. First check the primary funding text for funding information
    2. If no funding information is found in the primary funding text, check the additional funding details
    3. If no funding information exists in either source, return "None"
    4. Otherwise, extract all organizations, grant numbers, and other funding entities.

    Format your output exactly as:
    funding_source: List all funding sources/organizations separated by commas. If no funding information found, write "None"."""

    client = OpenAI(
        #add you OpenAI API Key
          api_key = api_key
      )
    system_prompt = "You are an expert in funding metadata extraction."
    input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=input,
        model=model_name,
        temperature=0.1, #low temperature to ensure reproducability, as We don't need the LLM to be too creative
        max_tokens=300  # to ensure not too much verbosity, the model outputs content < 300 words
    )
    output = chat_completion.choices[0].message.content.strip()
    funding_source = None  #default values
    if output.startswith('funding_source:'):
        funding_source = output.split(':', 1)[1].strip()
    if not funding_source:
      return None
    else:
      return funding_source


###################################################################

# collect the domain categories, the type of neglected tropical diseases, the type of data being analysed as well as the type of ML approach and model used
def classify_ntd_abstract(abstract_text, model_name='gpt-4o-mini'):
    prompt = f"""
    Your task is to analyze the following research abstract '{abstract_text}' and extract key information about the primary domain category(see list below), the type of tropical disease being analyzed, the type of datasets and machine learning model used.
    Follow these instructions:
    1. Determine the Domain Categories of the neglected tropical diseases described:
      - diagnosis: Methods for detecting disease presence, including biomarker identification, screening tools, point-of-care diagnostics, or laboratory techniques.
      - prognosis: Techniques to predict disease progression, patient outcomes, severity assessment, or complication risk.
      - surveillance: Systems for monitoring disease prevalence, geographic distribution, vector tracking, or pathogen dynamics.
      - treatment monitoring: Approaches to evaluate therapeutic efficacy, patient response, or recovery trajectory assessment.
      - drug discovery: Methods for compound identification, virtual screening, drug repurposing, target identification, or pharmacological modeling.
      - policy planning: Frameworks for resource allocation, intervention strategy development, cost-effectiveness modeling, or healthcare systems optimization.
      - outbreak prediction: Models forecasting disease emergence, transmission patterns, epidemic potential, or spatiotemporal risk analysis.

    2. Determine the specific NTD(s) being analyzed in the text (select from the WHO list):
       Buruli ulcer; Chagas disease; dengue and chikungunya; dracunculiasis;
       echinococcosis; foodborne trematodiases; human African trypanosomiasis; leishmaniasis;
       leprosy; lymphatic filariasis; mycetoma, chromoblastomycosis and other deep mycoses;
       noma; onchocerciasis; rabies; scabies and other ectoparasitoses; schistosomiasis;
       soil-transmitted helminthiases; snakebite envenoming; taeniasis/cysticercosis;
       trachoma; and yaws.

    3. Identify the data type(s) used in the research (e.g. Electronic health records, Medical imaging data, Clinical trial data, social media data, surveillance data).

    4. If machine learning methods are used, classify the learning approach as one or more of:
        - SUPERVISED: Uses labeled data for training (e.g., classification, regression)
        - UNSUPERVISED: Discovers patterns without labeled data (e.g., clustering, dimensionality reduction)
        - SEMI-SUPERVISED: Uses both labeled and unlabeled data
        - REINFORCEMENT: Learns through interaction with environment and feedback
        - TRADITIONAL: Statistical methods without machine learning

    Format your response as:
    domain_category: [Choose the SINGLE PRIMARY category most central to the abstract's objective]
    ntd_type: [List all tropical diseases being analysed. If none, write "None".]
    ml_methods: [List all specific computational/statistical methods implemented, separated by commas. If none, write "None".]
    ml_type:[SUPERVISED, UNSUPERVISED, SEMI-SUPERVISED, REINFORCEMENT, TRADITIONAL, or a comma-separated combination if multiple approaches are used. If no methods implemented, write "None".]
    dataset_type: [list all data type(s) used, separated by commas. If not specified, write "Not specified".]

    Provide ONLY the required output format with no additional text, explanations, or justifications.
    """
    client = OpenAI(

        api_key = api_key
    )
    system_prompt = "You are an expert computational epidemiologist specializing in neglected tropical diseases and machine learning applications in healthcare."
    input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=input,
        model=model_name,
        temperature=0.1,
        max_tokens=300  # to ensure not too much verbosity
    )
    output = chat_completion.choices[0].message.content.strip()# return output

    domain_category, ntd_type, ml_method, ml_type, dataset_type = None, None, None, None, None

    lines = output.split('\n') #the LLM outcomes will come newline separated so we disassemble that so we can add the output on each column of our output dataset
    for line in lines:
      if line.startswith('domain_category:'):
        domain_category = line.split(':', 1)[1].strip()
      elif line.startswith('ntd_type:'):
        ntd_type = line.split(':', 1)[1].strip()
      elif line.startswith('ml_methods:'):
        ml_method = line.split(':', 1)[1].strip()
      elif line.startswith('ml_type:'):
        ml_type = line.split(':', 1)[1].strip()
      elif line.startswith('dataset_type:'):
        dataset_type = line.split(':', 1)[1].strip()

    if not domain_category or not ntd_type or not ml_method or not ml_type or not dataset_type:
      return None
    else:
      return {
          'domain_category' : domain_category,
          'ntd_type': ntd_type,
          'methods_used': ml_method,
          'ml_type': ml_type,
          'dataset_type': dataset_type
      }

##############################################



def main():
    tqdm.pandas()
    #step 1: Check if the abstract uses ML algorithm and model
    file_path = 'data/References.xlsx'

    # Access each sheet by its name
    # Read both sheets into separate DataFrames
    # Ensure the first row is treated as the header
    sheet1_df = pd.read_excel(file_path, sheet_name='WoS_to_Review')  # Replace 'WoS' with the actual sheet name
    sheet2_df = pd.read_excel(file_path, sheet_name='Scopus_to_Review')  # Treat the first row as the header
    # Step 2: Select specific columns to merge
    # Define the columns to keep
    columns_to_merge = ['Year', 'Class', 'Affiliations', 'Abstract', 'Keywords', 'Funding','Title']

    # Select the columns from each sheet
    sheet1_selected = sheet1_df[columns_to_merge]
    sheet2_selected = sheet2_df[columns_to_merge]

    # Step 3: Merge records from Sheet 1 and Sheet 2
    # Concatenate the two DataFrames
    dataset = pd.concat([sheet1_selected, sheet2_selected], ignore_index=True)

    dataset_ntd =dataset[columns_to_merge]

    # dataset_ntd['Abstract'] = dataset_ntd['Abstract'].astype(str)
    # dataset_ntd = dataset_ntd[(dataset_ntd['Abstract'].str.strip() != '[No abstract available]') & (dataset_ntd['Class'].notna())].reset_index(drop=True)
    # dataset_ntd = dataset_ntd.sample(200, random_state= 44)

    #processing and saving the outcomes from step 1
    dataset_ntd[['llm_ml_classify']] = dataset_ntd['Abstract'].progress_apply(lambda x: pd.Series(classify_ntd_ml(x)))
    #saving
    file_name = f'ml_llm_human_labelled_ntd.csv'
    file_path = os.path.join(output_path , file_name)
    dataset_ntd.to_csv(file_path, index=False)
    print(f"Saved  dataset to : {file_path}\n")

    #############################################

    #step 2: Checke affliation, funding, ML method, algorithms and everthing else from the content extracted from step 1 above
    dataset_ntd_label = dataset[columns_to_merge]
    dataset_ntd_label = dataset_ntd_label[(dataset_ntd_label['Abstract'].str.strip() != '[No abstract available]') & (dataset_ntd_label['Class'].notna())].reset_index(drop=True)
    # dataset_ntd_label = dataset_ntd_label.sample(200, random_state= 44)

    #processing the last prompt
    dataset_ntd_label[['llm_label', 'tropical_diseases', 'ml_model', 'ml_technique', 'dataset_type']] = dataset_ntd_label['Abstract'].progress_apply(lambda x: pd.Series(classify_ntd_abstract(x)))

    #processing the researcher affiliation by country and regions
    dataset_ntd_label[['country', 'region']] = dataset_ntd_label['Affiliations'].progress_apply(lambda x: pd.Series(classify_ntd_location(x)))

    #processing the funding
    dataset_ntd_label['funding_source'] = dataset_ntd_label.progress_apply(lambda x: classify_ntd_funding(funding_text=x['Funding'],
                                                                                                           funding_details=x['Funding']
                                                                                                           if pd.notna(x['Funding']) else None,), axis = 1)
    #saving outcomes of step 2 as a .csv file
    file_name = f'llm_human_everything_else_labelled_ntd.csv'
    file_path = os.path.join(output_path , file_name)
    dataset_ntd_label.to_csv(file_path, index=False)
    print(f"\nSaved  dataset to : {file_path}\n")

    return dataset_ntd, dataset_ntd_label

if __name__ == "__main__":
  dataset_ntd, dataset_ntd_label = main()