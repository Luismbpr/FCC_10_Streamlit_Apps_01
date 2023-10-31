### Working Correctly
################################################################################################################
## Streamlit App 010 - 01 - 02/02 Web App
################################################################################################################
## App 010 - Solubility Prediction
################################################################################################################
## Streamlit Web App
## Creating the Streamlit Web App using the model created on prior python file
### Using Train Test Validation Split
################################################################################################################

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle
from PIL import Image
import streamlit as st

######################
### Custom Function
######################

### Calculate Molecular Descriptors
def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i==True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom/HeavyAtom
    return AR

def generate(smiles, verbose=False):

    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if (i==0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i+1
    columnNames = ["MolLogP",
                   "MolWt",
                   "NumRotatableBonds",
                   "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors

######################
## Page Title
######################
image_01 = Image.open('./Projects/App_010_001/App_010_001_Exported/Data/Images/app_10_01_002.jpeg')

st.image(image_01)

######################
st.write("""
### App that predicts the Solubility (LogS) values of molecules.
""")
######################


######################
### Input Molecules (Side Panel)
######################
st.sidebar.header('User Input Features')
st.sidebar.write('Please input the features on a SMILES format.')

## Read SMILES Input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES# Adding C as a dummy, first item
SMILES = SMILES.split('\n')

st.markdown('### Input SMILES')
SMILES[1:]# Skips the dummy first item

st.markdown('')

######################
### Calculate molecular descriptors
######################

st.markdown('### Computed molecular descriptors')
X = generate(SMILES)
X[1:]# Skips the dummy first item

st.markdown('')
######################
### Open and use model for prediction
######################

## Loading Model
load_model = pickle.load(open('./Projects/App_010_001/App_010_001_Exported/Data/Saved_Models/App_010_model_lr_01.pkl', 'rb'))

prediction = load_model.predict(X)

st.markdown('### Predicted LogS values')
#prediction[1:]# Skips the dummy first item
#st.markdown(prediction[1:])

## DataFrame Input String, Prediction
df_03_info = {"Input":SMILES[1:],
              "Predicted LogS Values":prediction[1:]}
df_03 = pd.DataFrame(df_03_info)
st.write(df_03)
###########


st.write('---')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')


st.write('### Resources:')
st.write("""
         [Data Source John S. Delaney. ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure ***J. Chem. #Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005](https://pubs.acs.org/doi/10.1021/ci034243x)\n
         [Streamlit](https://streamlit.io/)
         """)
st.write('')
st.write('')
st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')
st.write('##### Thank you to all of you who make information and knowledge available for free.')



##Results after running python file
##<class 'pandas.core.frame.DataFrame'>
##RangeIndex: 1144 entries, 0 to 1143
##Data columns (total 5 columns):
## #   Column              Non-Null Count  Dtype  
##---  ------              --------------  -----  
## 0   MolLogP             1144 non-null   float64
## 1   MolWt               1144 non-null   float64
## 2   NumRotatableBonds   1144 non-null   float64
## 3   AromaticProportion  1144 non-null   float64
## 4   logS                1144 non-null   float64
##dtypes: float64(5)
##memory usage: 44.8 KB
##Model: LinearRegression()
##Model Coefficients:
##[-0.72521491 -0.00663091  0.00502635 -0.50455953]
##
##
##Model Intercept:
##0.25272417963154803
##
##
##MAE: 0.81
##MSE: 1.06
##RMSE: 1.03
##Coefficient of Determination R2: 0.75
##LogS = 0.25 -0.73 MolLogP -0.0066 MolWt + 0.0050 NumRotatableBonds -0.50 AromaticProportion
##/Users/luis/Documents/Programming/Courses_Programming/FCC_Build_12_DS_Apps_Python_Streamlit/venv_FCC_Build_12_DS_Apps_Python_Streamlit_310/Projects/App_010_001/App_010_001_Export_requirements/App_010_002_002_model_creation_clean.py:148: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
##  plt.show()
##[ 0.68691444 -0.82291026]
## 
##0.6869 x - 0.8229

