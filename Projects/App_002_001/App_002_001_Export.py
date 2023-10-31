################################################################################################################
## Streamlit App 002 - 01 - 01/01
################################################################################################################
## Notes:
## DNA Nucleotide Count Web App
################################################################################################################
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
#import PIL as PIL
from PIL import Image


image_01 = Image.open(fp='./Projects/App_002_001/App_002_001_Exported/Data/Images/app_002_image_01.jpeg', mode='r', formats=None)
st.image(image_01, use_column_width=True)

st.write('')
st.write('')

## Input Text Box
st.markdown('## Enter DNA Sequence')

sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

sequence = st.text_area("###### Input Sequence:", sequence_input, height=250)
sequence = sequence.splitlines()## Creating a list for each line
sequence = sequence[1:]## Skips the sequence name (first line) -> '>DNA Query 2'
sequence = ''.join(sequence)## joining list to string

st.write("""
***
""")

## Printing input DNA Sequence
st.markdown('## Input (DNA Query)')
sequence

st.write("---")

## DNA Nucleotide Count
st.markdown('## Output (DNA Nucleotide Count)')

## 1. Printing Dictionary
#st.subheader('1. Output dictionary')
st.markdown('#### 1. Output dictionary')
def DNA_nucleotide_count(seq):
    d = dict([
        ('A',seq.count('A')),
        ('T',seq.count('T')),
        ('G',seq.count('G')),
        ('C',seq.count('C'))
    ])
    return d

X = DNA_nucleotide_count(sequence)

X_label = list(X)
X_values = list(X.values())
X## Output Dictionary

st.write('')

## 2. Print text
#st.subheader('2. Output Result')
st.markdown('#### 2. Output Result')
st.write('There are ' + str(X['A']) + ' (A) Adenine')
st.write('There are ' + str(X['T']) + ' (T) Thymine')
st.write('There are ' + str(X['G']) + ' (G) Guanine')
st.write('There are ' + str(X['C']) + ' (C) Cytosine')

st.write('')
st.write('')

## 3. Display DataFrame
#st.subheader('3. Display DataFrame')
st.markdown('#### 3. Display DataFrame')
st.write('')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0:'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'nucleotide'})
st.write(df)

st.write('')
st.write('')

## 4. Display Bar Chart using Altair
#st.subheader('4. Display Bar Chart')
st.markdown('#### 4. Display Bar Chart')
st.write('')
p = alt.Chart(data=df).mark_bar().encode(x='nucleotide', y='count')
p = p.properties(
    width=alt.Step(80)## Control width of bar
)
st.write(p)

############################################################### Test 01 Start
## 5. Display Bar Chart using Altair
#import matplotlib.pyplot as plt
#
#st.subheader('5. Display Bar Chart - Test')
#fig04, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
##fig04 = plt.bar(data=df, x='nucleotide', y='count')
#ax = plt.bar(data=df, x='nucleotide', height='count')
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
#st.pyplot(fig04)

############################################################### Test 01 End

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
        [Streamlit](https://streamlit.io/)\n
        [Original image by Warren Umoh on Unsplash](https://unsplash.com/)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')

st.write('##### Thank you kindly to all who make information and knowledge available for free.')