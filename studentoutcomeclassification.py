import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# Import Transformer,Encoder,moddel
catModel=pkl.load(open("catModel.pkl",'rb'))
Transformer=pkl.load(open("Transformer.pkl",'rb'))
encoder=pkl.load(open("encoder.pkl","rb"))


file=st.file_uploader("Upload File",type="csv")
button=st.button("Predict Outcome")

if button:
    if not file:
        st.error("Upload File")
    else:
        data=pd.read_csv(file)
        data.columns = data.columns.str.replace('\t', '', regex=False)
        TransformData=Transformer.transform(data)
        PredictOutcome=catModel.predict(TransformData)

        data["Outcome"]=encoder.inverse_transform(PredictOutcome)
        st.write(data)