import streamlit as st
import pickle
from PIL import Image
from sklearn.metrics import euclidean_distances as ed

def main():
    st.title("Heart Failure Prediction")
    image=Image.open("heart.jpg")
    st.image(image,width=500)
    age=st.text_input("Age","Type Here")
    gender=st.radio("Sex",['Male','Female'])
    if gender=='Male':
        sex=1
    else:
        sex=0
    cp = st.text_input("cp", "Type Here")
    trestbps=st.text_input("Trest bps","Type Here")
    chol=st.text_input("chol","Type Here")
    fbs = st.text_input("fbs","Type Here")
    restecg = st.text_input("restecg","Type Here")
    thalach = st.text_input("thalach","Type Here")
    exang = st.text_input("exang","Type Here")
    oldpeak = st.text_input("oldpeak","Type Here")
    slope = st.text_input("slope","Type Here")
    ca = st.text_input("ca","Type Here")
    thal = st.text_input("thal","Type Here")
    features=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

    model=pickle.load(open('model.sav','rb'))
    scaler=pickle.load(open('scaler.sav','rb'))
    pred=st.button('PREDICT')

    if pred:
        prediction=model.predict(scaler.transform([features]))
        if prediction==0:
            st.write("Not Suffering Heart Disease")
        else:
            st.write("Suffering Heart Disease")
