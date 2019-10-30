# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:45:29 2019

@author:raf
"""

# Instructions : 
# une séparation en training / testing set
# 2 propositions d'amélioration du modèle qui obtiennent de meilleures performances que la "baseline" (la régression linéaire avec une seule feature)
#Une sélection d'un modèle final à partir des performances

# Ce que j'ai fait  
# Analyse simple du data set pour visualiser les séries 
# Suppression des outliers 
# Sur modèle 1 feature : le truc c'est qu'il n'y a que 2 variables en plus de la target
# l'arrondissement, bien que quanti est une variable quali qui ne rentrerait pas
# dans une regression linéaire
# Après analyse : choix de faire 2 modèles 
# Un modèle pour le 10ème arrondissement et 1 modèle pour les autres 
# Qualité du R² train and test : ok
# RMSE un peu élevé pour le modèle hors 10ème arrondissement 



# Importation du fichiers, à savoir des loyers, par surface, par arrondissement
# Fichier texte / séparateur virgule, décimal '.' 

# Importation des librairies "classique" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importation du fichier 
df = pd.read_csv("C:/01- ITR/04 - Data/Formation Python datascience/2019 10 Intro ML/house_data.csv", sep="," ,
                 decimal ='.',  
                    )
df.info()


#voir les arrondissments et quelques stats sur les autres variables 
# Freq simple 
df["arrondissement"].value_counts(dropna = False).sort_index()
df["price"].describe()
df["surface"].describe()


# Distribution via graph des séries 
# On affiche le nuage de points dont on dispose
plt.plot(df["surface"], df["price"], 'bx', markersize=4)
plt.show()

df["surface"].value_counts(dropna = False).sort_index()
df["price"].value_counts(dropna = False).sort_index()
df["arrondissement"].value_counts(dropna = False).sort_index()


# on constate quelques valeurs aberrantes => nettoyage
df2=df[df["price"]<12000]

plt.plot(df2["surface"], df2["price"], 'rx', markersize=4)
plt.show()

# What about les valeurs manquantes ? 
# => 5 Nan dans "price" et "arrondissement"? A supprimer, comme des outliers 

df3=df2[(df2["surface"]>0) & (df2["arrondissement"]>0)] 

df3["surface"].value_counts(dropna = False).sort_index()
df3["arrondissement"].value_counts(dropna = False).sort_index()

plt.plot(df3["surface"], df3["price"], 'bx', markersize=4)
plt.show()

# Table, à peu près propre 
# Hypothèse que l'arrondissement joue un rôle 
# Le describe des variables par arrondissement 

df3.groupby(df3["arrondissement"]).sum() 
test = df3.groupby(df3["arrondissement"]).mean()   
plt.plot(test["surface"], test["price"], 'rx', markersize=4)
plt.show()

# Il y a un arrondissement qui semble en décalage 

df3.groupby(df3["arrondissement"]).size()
prchoisir=df3.groupby(df3["arrondissement"]).describe()  
prchoisir["price"] / prchoisir["surface"]
# => sur la moyenne, le ration du 10ème, voir du 3eme sont en décalé 
# d'autre ratio sur le 3ème sont plus dans la masse mais le 10ème reste à part
# modéliser au moins le 10ème à part



# Ne prendre que les données du 10ème pour voir ce que donnerait un modèle 
df3_10=df3[(df3["arrondissement"]==10)]
df3_other=df3[(df3["arrondissement"] != 10)]
 
# des volumétries relativement bien réparties entre les arrondissements 
#le 10eme arrondissement : moyenne décalée -> le mettre à part dans la modèlisation
# des écart-types variables d'un arrondissement à l'autre 

# est-ce qu'une modélisation par arrondissement serait pertinente ?
# oui mais les volumes commencent à être faible -> tente 

# Commencer avec le modèle sur le 10ème arrondissement 
# Faire les différentes bases : apprentissage / validation 
# Enlève les variables inutiles 

df4_10=df3_10.drop(["price" , "arrondissement"], axis=1)


from sklearn.model_selection import train_test_split
rc_xtrain, rc_xtest, rc_ytrain, rc_ytest = train_test_split(df4_10, df3_10.price , train_size=0.5)


# Indiquer le modèle souhaité  
from sklearn import linear_model
regr10 = linear_model.LinearRegression()

# Pour calculer le modèle 
regr10.fit(rc_xtrain,rc_ytrain)

# Les coef du modèle 
print(regr10.intercept_)
print(regr10.coef_)
# modèle = Price = 19 * surface + 389 

# Pour avoir le R² ? 
regr10.score(rc_xtrain,rc_ytrain)
regr10.score(rc_xtest,rc_ytest)

#Les R² sont du même ordre de grandeur, une bonne chose


# application du modèle 
regr10.predict(rc_xtest)

# "qualité" de la modélisation 
RMSE=np.sqrt(((rc_ytrain-regr10.predict(rc_xtrain))**2).sum()/len(rc_ytrain))
print(RMSE) 


plt.plot(rc_ytest, regr10.predict(rc_xtest),'.')
plt.show()

plt.plot(rc_ytest, rc_ytest-regr10.predict(rc_xtest),'.')
plt.show()


# Modèle sur les autres arrondissements 
# Faire les différentes bases : apprentissage / validation 

df4_other=df3_other.drop(["price" , "arrondissement"], axis=1)

from sklearn.model_selection import train_test_split
rc_xtrain, rc_xtest, rc_ytrain, rc_ytest = train_test_split(df4_other, df3_other.price , train_size=0.5)


# Indiquer le modèle souhaité  
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Pour calculer le modèle 
regr.fit(rc_xtrain,rc_ytrain)

# Les coef du modèle 
print(regr.intercept_)
print(regr.coef_)
# modèle = Price = 32 * surface + 148


# Pour avoir le R² ? 
regr.score(rc_xtrain,rc_ytrain)
regr.score(rc_xtest,rc_ytest)
#Les R² sont du même ordre de grandeur, une bonne chose


# application du modèle 
regr.predict(rc_xtest)

# "qualité" de la modélisation 
RMSE=np.sqrt(((rc_ytrain-regr.predict(rc_xtrain))**2).sum()/len(rc_ytrain))
print(RMSE) 
# un peu élevé, sur le graph + bas, il y a des points aberrants 

plt.plot(rc_ytest, regr.predict(rc_xtest),'.')
plt.show()

plt.plot(rc_ytest, rc_ytest-regr.predict(rc_xtest),'.')
plt.show()

# THE END 