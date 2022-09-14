import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("train.csv", index_col=0)


st.sidebar.title("Sommaire")

pages = ["Introduction", "Dataviz", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.title("Mon premier Streamlit")

    st.write("## Introduction")
    st.markdown(
        "Prédiction de la survie d'un passager du [Titanic](https://www.kaggle.com/c/titanic)"
    )
    st.image("titanic.jpg")

    st.write("Voici un aperçu du Dataframe")

    st.dataframe(df.head())

    if st.checkbox("Afficher les variables manquantes "):
        st.dataframe(df.isna().sum())
elif page == pages[1]:
    st.write("## Dataviz")

    import matplotlib.pyplot as plt

    fig = plt.figure()
    sns.countplot(x="Survived", data=df)
    st.pyplot(fig)

    fig_pclass = plt.figure()
    sns.countplot(x="Pclass", data=df)
    st.pyplot(fig_pclass)

    import plotly.express as px

    hist_surv = px.histogram(df, x="Survived", color="Sex", barmode="group")
    st.plotly_chart(hist_surv)

    hist_sun = px.sunburst(df, path=["Sex", "Pclass", "Survived"])
    st.plotly_chart(hist_sun)
elif page == pages[2]:
    st.write("## Modélisation")

    df = df.dropna()
    df = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    def train_model(estimator, name, size=0.8):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, train_size=size
        )
        # estimator.fit(X_train,y_train)
        score_test = estimator.score(X_test, y_test)
        score_train = estimator.score(X_train, y_train)
        print(
            "Le modèle {} a une accuracy sur l'échantillon d'entraînement {}".format(
                name, score_train
            )
        )
        print(
            "Le modèle {} a une accuracy sur l'échantillon de test {}".format(
                name, score_test
            )
        )
        return score_test, score_train

    from joblib import load

    model = {
        "Régression Logistique": load("lr.joblib"),
        "Arbre de Décision": load("tr.joblib"),
        "K-Voisins": load("knn.joblib"),
    }

    mod = st.selectbox("Choisissez votre modèle", model.keys())

    score_test, score_train = train_model(model[mod], mod)

    st.write(
        "Le modèle {} a une accuracy sur l'échantillon d'entraînement {}".format(
            mod, score_train
        )
    )
    st.write(
        "Le modèle {} a une accuracy sur l'échantillon de test {}".format(
            mod, score_test
        )
    )
