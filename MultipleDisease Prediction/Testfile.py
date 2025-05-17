import streamlit as st
import joblib
import numpy as np
from streamlit_option_menu import option_menu

# Load models
parkinsons_model = joblib.load("ParkinsonsNotebook.pkl")
liver_model = joblib.load("IndianLiverDiseaseNotebook.pkl")
kidney_model = joblib.load("KidneyDiseaseNotebook.pkl")

st.title("Multiple Disease Prediction Interface")

with st.sidebar:
    disease_option = option_menu("Disease Prediction",
                                 ["Parkinson's",
                                  "Liver Disease Prediction",
                                  "Kidney Prediction"],
                                 menu_icon='hospital-fill',
                                 icons=['activity', 'heart', 'person'],
                                 default_index=0)

# Parkinson's Disease Prediction
if disease_option == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    f1 = st.number_input('MDVP:Fo(Hz)')
    f2 = st.number_input('MDVP:Fhi(Hz)')
    f3 = st.number_input('MDVP:Flo(Hz)')
    f4 = st.number_input('MDVP:Jitter(%)')
    f5 = st.number_input('MDVP:Jitter(Abs)')
    f6 = st.number_input('MDVP:RAP')
    f7 = st.number_input('MDVP:PPQ')
    f8 = st.number_input('Jitter:DDP')
    f9 = st.number_input('MDVP:Shimmer')
    f10 = st.number_input('MDVP:Shimmer(dB)')
    f11 = st.number_input('Shimmer:APQ3')
    f12 = st.number_input('Shimmer:APQ5')
    f13 = st.number_input('MDVP:APQ')
    f14 = st.number_input('Shimmer:DDA')
    f15 = st.number_input('NHR')
    f16 = st.number_input('HNR')
    f17 = st.number_input('RPDE')
    f18 = st.number_input('DFA')
    f19 = st.number_input('spread1')
    f20 = st.number_input('spread2')
    f21 = st.number_input('D2')
    f22 = st.number_input('PPE')

    if st.button('Predict Parkinson\'s'):
        features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
                              f13, f14, f15, f16, f17, f18, f19, f20, f21, f22]])
        result = parkinsons_model.predict(features)
        st.success("Parkinson's Detected!" if result[0] == 1 else "No Parkinson's")

# Liver Disease Prediction
elif disease_option == "Liver Disease Prediction":
    st.header("Liver Disease Prediction")
    age = st.number_input('Age')
    gender = st.selectbox("Gender", ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin')
    direct_bilirubin = st.number_input('Direct Bilirubin')
    alk_phosphotase = st.number_input('Alkaline Phosphotase')
    alanine_aminotransferase = st.number_input('Alamine Aminotransferase')
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase')
    total_proteins = st.number_input('Total Proteins')
    albumin = st.number_input('Albumin')
    ag_ratio = st.number_input('Albumin and Globulin Ratio')

    gender_numeric = 1 if gender == 'Male' else 0

    if st.button('Predict Liver Disease'):
        features = np.array([[age, total_bilirubin, direct_bilirubin, alk_phosphotase,
                              alanine_aminotransferase, aspartate_aminotransferase,
                              total_proteins, albumin, ag_ratio, gender_numeric]])
        result = liver_model.predict(features)
        st.success("Liver Disease Detected!" if result[0] == 1 else "No Liver Disease")

# Kidney Disease Prediction
elif disease_option == "Kidney Prediction":
    st.header("Kidney Disease Prediction")

    # Example inputs for key kidney features
    age = st.number_input("Age")
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")
    bgr = st.number_input("Blood Glucose Random")
    bu = st.number_input("Blood Urea")
    sc = st.number_input("Serum Creatinine")
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    hemo = st.number_input("Hemoglobin")

    # Categorical inputs (map to numeric as needed)
    rbc = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
    pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
    pcc = st.selectbox("Pus Cell clumps", ['present', 'notpresent'])
    ba = st.selectbox("Bacteria", ['present', 'notpresent'])
    pcv = st.number_input("Packed Cell Volume")
    wc = st.number_input("White Blood Cell Count")
    rc = st.number_input("Red Blood Cell Count")
    htn = st.selectbox("Hypertension", ['yes', 'no'])
    dm = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
    appet = st.selectbox("Appetite", ['good', 'poor'])
    pe = st.selectbox("Pedal Edema", ['yes', 'no'])
    ane = st.selectbox("Anemia", ['yes', 'no'])

    # Map categorical features
    mapper = {'normal': 1, 'abnormal': 0, 'present': 1, 'notpresent': 0, 'yes': 1, 'no': 0, 'good': 1, 'poor': 0}
    features = np.array([[age, bp, sg, bgr, bu, sc, sod, pot, hemo,
                          mapper[rbc], mapper[pc], mapper[pcc], mapper[ba],
                          pcv, wc, rc, mapper[htn], mapper[dm], mapper[cad],
                          mapper[appet], mapper[pe], mapper[ane]]])

    if st.button('Predict Kidney Disease'):
        result = kidney_model.predict(features)
        st.success("Kidney Disease Detected!" if result[0] == 1 else "No Kidney Disease")
