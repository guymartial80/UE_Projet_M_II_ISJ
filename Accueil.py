import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("DETECTION DES CHURNERS")
st.subheader("Par : Guy Martial ATANGANA et Georges MBOCK MBOCK")
st.image("img//doigt-toucher-telephone-wifi.jpg")
st.markdown(
    ("***Cette application permet de détecter, pour le compte d'une entreprise de Télécom basée"
    " en Asie du Sud-Est, les abonnés susceptibles de résilier leur abonnement pour des raisons"
    " liées à la qualité de service, cout des services, etc.***")
    )

# Importation du dataset
data_churn = pd.read_csv('true_df_balanced.csv')

st.sidebar.header('Visualiser le dataset')
if st.sidebar.checkbox("Afficher le dataset", False):
    st.subheader("Dataset TELCO")
    st.write(data_churn)



#Visualisation
st.sidebar.write('-----')
chart_select = st.sidebar.selectbox(
    label = "Sélectionner le type de graphique",
    options = ['Nuage de points', 'Lineplots', 'Histogramme', 'Boxplot']
)

y = data_churn['Churn']
X = data_churn.drop('Churn', axis = 1)

numeric_column = list(X.select_dtypes(['float', 'int']).columns)

if chart_select == 'Nuage de points':
    st.sidebar.subheader('Paramètres Nuage de points')
    try:
        x_values = st.sidebar.selectbox('Axe des X', options = numeric_column)
        y_values = st.sidebar.selectbox('Axe des Y', options = numeric_column)
        plot = px.scatter(data_frame = X, x = x_values, y = y_values)
        st.write(plot)
    except Exception as e:
        print(e)


if chart_select == 'Histogramme':
    st.sidebar.subheader('Paramètres Histogramme')
    try:
        x_values = st.sidebar.selectbox('Axe des X', options = numeric_column)
        plot = px.histogram(data_frame = X, x = x_values)
        st.write(plot)
    except Exception as e:
        print(e)


if chart_select == 'Lineplots':
    st.sidebar.subheader('Paramètres Lineplots')
    try:
        x_values = st.sidebar.selectbox('Axe des X', options = numeric_column)
        y_values = st.sidebar.selectbox('Axe des Y', options = numeric_column)
        plot = px.line(data_frame = X, x = x_values, y = y_values)
        st.write(plot)
    except Exception as e:
        print(e)


if chart_select == 'Boxplot':
    st.sidebar.subheader('Paramètres Boxplot')
    try:
        x_values = st.sidebar.selectbox('Axe des X', options = numeric_column)
        y_values = st.sidebar.selectbox('Axe des Y', options = numeric_column)
        plot = px.box(data_frame = X, x = x_values, y = y_values)
        st.write(plot)
    except Exception as e:
        print(e)

st.write('----')

# Evaluation
plot_perf = st.sidebar.multiselect(
    "Choisir un critère d'évaluation",
    ["Confusion matrix", "ROC Curve", "Precision-Recall Curve"]
)

def perf(all_plot):
    if "Confusion matrix" in all_plot:
        st.subheader("Matrice de Confusion")
        plot_confusion_matrix(
            model,
            X_test_scaled_df,
            y_test
        )
        st.pyplot()

    if "ROC Curve" in all_plot:
        st.subheader("Courbe ROC")
        plot_roc_curve(
            model,
            X_test_scaled_df,
            y_test
        )
        st.pyplot()

    if "Precision-Recall Curve" in all_plot:
        st.subheader("Courbe recall")
        plot_precision_recall_curve(
            model,
            X_test_scaled_df,
            y_test
        )
        st.pyplot()

st.write('----')

# Modélisation
classifier = st.sidebar.selectbox(
    "Classificateur",
    ("Random Forest", "SVM", "Regression Logistique")
)

# Random Forest
if classifier == "Random Forest":
    if st.sidebar.button("Exécution", key = "classify"):
        st.header("Random Forest Results")

        # Division du dataset
        seed = 42

        @st.cache_data(persist=True)
        def split(data_churn):
            y = data_churn['Churn']
            X = data_churn.drop('Churn', axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state= seed) 
            return X_train, X_test, y_train, y_test 

        X_train, X_test, y_train, y_test = split(data_churn)
        
        # Normalisation dataset
        scaler = MinMaxScaler()
        data_scaler  = scaler.fit(X_train)
        X_train_scaled = data_scaler.transform(X_train)
        X_test_scaled = data_scaler.transform(X_test)

        
        # Transformation en DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        X_test_scaled_df  = pd.DataFrame(X_test_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        st.write(X_train_scaled_df.head())
        

        # Initialisation objet
        model = RandomForestClassifier()
        model.fit(X_train_scaled_df, y_train)

        # Prédiction
        y_pred = model.predict(X_test_scaled_df)

        # Métriques
        accuracy = model.score(X_test_scaled_df, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        F1 = (2*precision *recall)/(precision+recall)

        # Affichage des métriques
        st.write(f"Accuracy Score : {accuracy:.3f}%")
        st.write(f"Precision Score : {precision:.3f}%")
        st.write(f"Recall Score : {recall:.3f}%")
        st.write(f"F1 - Score : {F1:.3f}%")

        # Afficher les performances
        perf(plot_perf)


# SVM
if classifier == "SVM":
    if st.sidebar.button("Exécution", key = "classify"):
        st.header("Support Vector Machine Results")

        # Division du dataset
        seed = 42

        @st.cache_data(persist=True)
        def split(data_churn):
            y = data_churn['Churn']
            X = data_churn.drop('Churn', axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state= seed) 
            return X_train, X_test, y_train, y_test 

        X_train, X_test, y_train, y_test = split(data_churn)
        
        # Normalisation dataset
        scaler = MinMaxScaler()
        data_scaler  = scaler.fit(X_train)
        X_train_scaled = data_scaler.transform(X_train)
        X_test_scaled = data_scaler.transform(X_test)

        
        # Transformation en DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        X_test_scaled_df  = pd.DataFrame(X_test_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        st.write(X_train_scaled_df.head())
        

        # Initialisation objet
        model = SVC()
        model.fit(X_train_scaled_df, y_train)

        # Prédiction
        y_pred = model.predict(X_test_scaled_df)

        # Métriques
        accuracy = model.score(X_test_scaled_df, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        F1 = (2*precision *recall)/(precision+recall)

        # Affichage des métriques
        st.write(f"Accuracy Score : {accuracy:.3f}%")
        st.write(f"Precision Score : {precision:.3f}%")
        st.write(f"Recall Score : {recall:.3f}%")
        st.write(f"F1 - Score : {F1:.3f}%")

        # Afficher les performances
        perf(plot_perf)



# Regression Logistique
if classifier == "Regression Logistique":
    if st.sidebar.button("Exécution", key = "classify"):
        st.header("Logistic Regression Results")

        # Division du dataset
        seed = 42

        @st.cache_data(persist=True)
        def split(data_churn):
            y = data_churn['Churn']
            X = data_churn.drop('Churn', axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state= seed) 
            return X_train, X_test, y_train, y_test 

        X_train, X_test, y_train, y_test = split(data_churn)
        
        # Normalisation dataset
        scaler = MinMaxScaler()
        data_scaler  = scaler.fit(X_train)
        X_train_scaled = data_scaler.transform(X_train)
        X_test_scaled = data_scaler.transform(X_test)

        
        # Transformation en DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        X_test_scaled_df  = pd.DataFrame(X_test_scaled, columns=(data_churn.drop('Churn', axis = 1)).columns)
        st.write(X_train_scaled_df.head())
        

        # Initialisation objet
        model = LogisticRegression(max_iter=500)
        model.fit(X_train_scaled_df, y_train)

        # Prédiction
        y_pred = model.predict(X_test_scaled_df)

        # Métriques
        accuracy = model.score(X_test_scaled_df, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        F1 = (2*precision *recall)/(precision+recall)

        # Affichage des métriques
        st.write(f"Accuracy Score : {accuracy:.3f}%")
        st.write(f"Precision Score : {precision:.3f}%")
        st.write(f"Recall Score : {recall:.3f}%")
        st.write(f"F1 - Score : {F1:.3f}%")

        # Afficher les performances
        perf(plot_perf)