import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Transportlarni klassifikatsiya qiluvchi model')

#Rasmni yuklash
file = st.file_uploader('Rasmni yuklash', type=['png', 'jpg', 'jpeg', 'svg', 'gif'])
if file:
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    #UZB function
    def nom(name):
        if name=='Airplane':
            return 'Samolyot'
        elif name=='Car':
            return 'Avtomobil'
        else:
            return 'Qayiq'

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Aniqlangan obyekt: {nom(pred)}")
    st.info(f"Aniqlik darajasi: {probs[pred_id]*100:.1f}%")
    st.image(file)

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)