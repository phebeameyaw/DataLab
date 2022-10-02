# DataLab Project

Code from https://github.com/gayaninan/LegalPythia-V2/tree/legal-pythia-v2.1

## Changes made to add in further explanations for users:

New feature function - Explanation Project

- Currently does not accept a .csv, but does accept .txt, .pdf and .docx files

- Checks two documents and produces a similarity percentage (from original code line 91)

- Adds in a similarity percentage to each sentence comparison

- Adds in a graph to show the softmax of the prediction (entailment, contradiction or neutral)

Potential future work:
- add in attention visualisations


## Changes made to add in duplicate values function for "Cleansing and Validation of Priority Services Register Data"

New feature function - Duplication Project

- Currently only accepts .csv files

- Initially checks two files and removes any exact duplicates

- Second process identifies duplicate names and allows the user to pick which ones are not duplicated customers

- Allows the user to download a new .csv file with no duplicates


## Usage

```python
streamlit run codes\scripts\main.py
```

