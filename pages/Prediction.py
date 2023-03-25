import streamlit as st
import numpy as np
import joblib

st.title("DETECTION DES CHURNERS")
st.subheader("Entrer les données pour la prédiction")

# Chargement du modèle
model = joblib.load(filename = "rf_streamlit.joblib")


# Fonction d'inférence
def inference(total_ic_mou_8,total_og_mou_8,loc_ic_mou_8,loc_og_mou_8,total_rech_amt_8,last_day_rch_amt_8,loc_og_t2m_mou_8,loc_ic_t2m_mou_8,loc_og_t2t_mou_8,arpu_8,roam_og_mou_8,max_rech_amt_8,offnet_mou_8,roam_ic_mou_8,onnet_mou_8,loc_ic_t2t_mou_8,loc_og_t2f_mou_8,loc_ic_t2f_mou_8,std_ic_t2m_mou_8,std_ic_mou_8,std_og_t2m_mou_8):
    new_data = np.array([
        total_ic_mou_8,total_og_mou_8,loc_ic_mou_8,loc_og_mou_8,total_rech_amt_8,
        last_day_rch_amt_8,loc_og_t2m_mou_8,loc_ic_t2m_mou_8,loc_og_t2t_mou_8,
        arpu_8,roam_og_mou_8,max_rech_amt_8,offnet_mou_8,roam_ic_mou_8,onnet_mou_8,
        loc_ic_t2t_mou_8,loc_og_t2f_mou_8,loc_ic_t2f_mou_8,std_ic_t2m_mou_8,std_ic_mou_8,std_og_t2m_mou_8  
    ])
    pred = model.predict(new_data.reshape(1,-1))
    return pred


# Saisie des données

total_ic_mou_8 = st.slider('Total minutes appels entrants Aout', 0,6500,1000)
total_og_mou_8 = st.slider('Total minutes appels sortants en Aout', 0,15000,8000)
loc_ic_mou_8 = st.slider('Minutes appels locaux entrant en Aout:', 0,6500,4000)
total_rech_amt_8 = st.slider('Montant total recharge en Aout',0, 50000,30000)
loc_og_mou_8 = st.slider('Minutes appels locaux sortants en Aout',0,12000,5000)
last_day_rch_amt_8 = st.slider('Montant recharge dernier jour mois Aout', 0, 4500,2300)
loc_og_t2m_mou_8 = st.slider('Minutes appels locaux sortants Mobile Aout',0, 5000,2000)
loc_ic_t2m_mou_8 = st.slider('Minutes appels locaux entrants Aout', 0, 6000,500)
loc_og_t2t_mou_8 = st.slider('Minutes appels locaux sortant TELCO à TELCO en Aout',0, 11000,2000)
arpu_8 = st.slider('Revenu moyen abonné Aout', -1000, 35000,20000)
roam_og_mou_8 = st.slider('Roaming sortant Aout', 0, 5500,2500)
max_rech_amt_8 = st.slider('Montant maximum recharges en Aout', 0, 4500,2200)
offnet_mou_8 = st.slider('Appels hors TELCO Aout', 0, 15000,11200)
roam_ic_mou_8 = st.slider('Minutes appels entrants Roaming Aout', 0, 4500,1200)
onnet_mou_8 = st.slider('Minutes tous appels TELCO Aout', 0, 11000,6200)
loc_ic_t2t_mou_8 = st.slider('Minutes Appels locaux entrants vers fixe Aout', 0, 4000,1200)
loc_og_t2f_mou_8 = st.slider('Minutes Appels locaux sortant vers fixe Aout', 0, 1000,200)
loc_ic_t2f_mou_8 = st.slider('Minutes Appels locaux entrants vers fixe TELCO Aout', 0, 2000,800)
std_ic_t2m_mou_8 = st.slider('Appels entrants hors zone vers mobile Aout', 0, 6000,2800)
std_ic_mou_8 = st.slider('Minutes Appels entrants hors zone Aout', 0, 6000,4200)
std_og_t2m_mou_8 = st.slider('Appels sortants hors zone vers mobile Aout', 0, 14000,2900)



# Gestion de la prédiction du modèle
if st.button("Predict"):
    prediction = inference(
        total_ic_mou_8,total_og_mou_8,loc_ic_mou_8,loc_og_mou_8,total_rech_amt_8,
        last_day_rch_amt_8,loc_og_t2m_mou_8,loc_ic_t2m_mou_8,loc_og_t2t_mou_8,
        arpu_8,roam_og_mou_8,max_rech_amt_8,offnet_mou_8,roam_ic_mou_8,onnet_mou_8,
        loc_ic_t2t_mou_8,loc_og_t2f_mou_8,loc_ic_t2f_mou_8,std_ic_t2m_mou_8,std_ic_mou_8,std_og_t2m_mou_8
    )
    if str(prediction[0]):
        result = "Cet(te) abonné(e) est susceptible de résilier son abonnement" 
    else:
        result = "C'est un abonné fidèle"
    st.success(result)