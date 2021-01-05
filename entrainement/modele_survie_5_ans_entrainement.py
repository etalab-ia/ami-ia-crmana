# Référent : Jean VAN HECKE.
# Entamé le 21/10/20.
# Finalisé le ...



# Imports.

import psycopg2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, make_scorer, confusion_matrix, classification_report, cohen_kappa_score
import json
import pandas as pd
# import shap
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import yaml



# Arguments. # A généraliser. ***

label_metier = sys.argv[1] # Exemple : 'boulangerie'.
id_modele_survie_3_ans = sys.argv[2] # Exemple : 322.



# Paramètres.

def get_project_config():
    return yaml.safe_load(os.getenv("PROJECT_CONFIG"))
db_logins = get_project_config()["project-database"]

# # Identifiants de connexion à la base de données. 
# # A NE PAS PUBLIER. ***
# db_logins = {}



# Fonctions utiles.

def sauver_pickle(data, nom_data, label_metier, id_model):
    nom_fichier = '{}/survie_5_ans/{}_{}'.format(label_metier, label_metier, nom_data)
    nom_fichier_bis = '{}/survie_5_ans/{}_{}'.format(label_metier, id_model, nom_data)
    fichier = open(nom_fichier, 'wb') 
    fichier_bis = open(nom_fichier_bis, 'wb') 
    pickle.dump(data, fichier)
    pickle.dump(data, fichier_bis)



# 1. Récupération des données.


# Connexion à la base de données.

con = psycopg2.connect(
    host = db_logins['hostname'], 
    database = db_logins['name'], 
    user = db_logins['user'], 
    password = db_logins['password']
)
cur = con.cursor()


# Données carroyées pertinentes.

def get_insee_gridded_data_relevant_fields():

    unrelevant_fields = [
        'idinspire', 'id_carr1km', 'i_est_cr', 'id_carr_n', 'groupe', 'depcom', 
        'i_pauv', 'id_car2010', 'i_est_1km', 'log_inc', 'ind_inc'
    ]
    query = """
        select a.column_name
        from information_schema.columns a
        where table_schema = 'public'
        and table_name = 'insee_donnees_carroyees_niveau_200m'
        and a.column_name not in ({})
    """.format(', '.join(list(map(lambda x: "'{}'".format(x), unrelevant_fields))))
    cur.execute(query)

    return list(map(lambda x: x[0], cur.fetchall()))


def get_data_metier_milieu(metier, milieu, relevant_fields):
    
    # Données en France métropolitaine.
    query = """
        select distinct a.siret, b.duree_vie, b.survie_5_ans, c.type_com as type_milieu_2, c.statut_2017 as type_milieu_4
        , e1.indice_densite_emplois
        , case when f1.p16_log != 0 then round((f1.p16_rsecocc/f1.p16_log)::numeric, 2) else 0 end as taux_residences_secondaires
        , f2.taux_motorisation_iris as taux_motorisation
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs1::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_agriculteurs_exploitants
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs2::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_artisans_commercants_entrepreneurs
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs3::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_cadres_professions_intellectuelles_superieures
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs4::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_professions_intermediaires
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs5::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_employes
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs6::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_ouvriers
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs7::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_retraites
        , case when f3.c15_pop15p != 0 then round((f3.c15_pop15p_cs8::double precision/f3.c15_pop15p::double precision)::numeric, 3) else 0 end as taux_autres_sans_activite_professionnelle
        , {}
        , g2.nb_transactions_immobilieres
        , g3.valeur::double precision as indice_population
        , g4.prix_m2 as prix_m2
        , h1.caractere_employeur_etablissement
        , case when h1.activite_principale_registre_metiers_etablissement is not null then 1 else 0 end as inscription_registre_metiers
        , h2.nb_concurrents_debut_{}, h2.nb_habitants_etablissement_debut_{}
        , h2.taux_etablissements_concurrents_debut_{}/*, h2.age_moyen_concurrence_{}*/
        , h3.indice_concurrence_restreinte_urbain::double precision, h3.indice_concurrence_restreinte_rural::double precision
        , h3.indice_concurrence_large_rural::double precision, h3.indice_concurrence_large_rural::double precision
        , h3.indice_artisanat_proximite_{}::double precision, h3.indice_artisanat_construction_{}::double precision
        , h3.indice_artisanat_economie_creative_{}::double precision, h3.indice_artisanat_soutien_{}::double precision
        , h3.indice_densite_naf_niveau_4_urbain, h3.indice_densite_naf_niveau_3_urbain
        , h3.indice_densite_naf_niveau_2_urbain, h3.indice_densite_naf_niveau_1_urbain
        , h3.indice_densite_naf_niveau_4_rural, h3.indice_densite_naf_niveau_3_rural
        , h3.indice_densite_naf_niveau_2_rural, h3.indice_densite_naf_niveau_1_rural
        , h3.indice_equipements_commerces, h3.indice_equipements_loisirs, h3.indice_equipements_transports
        , h3.indice_equipements_enseignement, h3.indice_equipements_sante, h3.indice_equipements_services
        , h3.indice_desserte_train, h3.indice_frequentation_train
        /*, h3.indice_reseau*/
        , h3.nb_stagiaires, h3.nb_permis_locaux, h3.nb_permis_logements, h3.nb_chambres_hotels
        from etablissements_cibles_{} a
        inner join taux_survie_etablissements b on a.siret = b.siret
        inner join unites_urbaines c on a.code_commune = c.codgeo
        inner join categories_communes_aires_urbaines e1 on a.code_commune = e1.codgeo
        inner join insee_logements_iris f1 on a.code_iris = f1.iris
        inner join datagouv_taux_motorisation f2 on a.code_iris = f2.code_iris
        inner join insee_csp_iris f3 on a.code_iris = f3.code_iris
        inner join insee_donnees_carroyees_niveau_200m g1 on a.idinspire = g1.idinspire
        inner join donnees_agregees_carreau g2 on a.idinspire = g2.idinspire
        inner join donnees_ponderees_carreau g3 on a.idinspire = g3.idinspire and g3.fk_donnees_ponderees = 4
        inner join data_immobiliere_final g4 on a.idinspire = g4.idinspire
        inner join sirene_etablissements h1 on a.siret = h1.siret
        inner join indices_concurrence h2 on a.siret = h2.siret
        inner join indices_synergie_territoriale h3 on a.siret = h3.siret
    """.format(
        ', '.join(list(map(lambda x: 'g1.{}'.format(x), relevant_fields))), milieu, milieu, milieu, milieu, 
        milieu, milieu, milieu, milieu, metier
    )
    cur.execute(query)
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=[
        'siret', 
        'duree_vie', 
        'survie_5_ans', 
        'type_milieu_2', 
        'type_milieu_4', 
        'indice_densite_emplois', 
        'taux_residences_secondaires', 
        'taux_motorisation',
        'taux_agriculteurs_exploitants',
        'taux_artisans_commercants_entrepreneurs',
        'taux_cadres_professions_intellectuelles_superieures',
        'taux_professions_intermediaires',
        'taux_employes',
        'taux_ouvriers',        
        'taux_retraites',
        'taux_autres_sans_activite_professionnelle'        
    ] + relevant_fields + [
        'nb_transactions_immobilieres', 
        'indice_population', 
        'prix_m2',
        'caractere_employeur', 
        'inscription_registre_metiers', 
        'nb_concurrents_debut', 
        'nb_habitants_etablissement_debut', 
        'taux_etablissements_concurrents_debut',
        'indice_concurrence_restreinte_urbain', 
        'indice_concurrence_restreinte_rural', 
        'indice_concurrence_large_urbain', 
        'indice_concurrence_large_rural', 
        'indice_artisanat_proximite', 
        'indice_artisanat_construction', 
        'indice_artisanat_economie_creative', 
        'indice_artisanat_soutien', 
        'indice_naf_4_urbain', 
        'indice_naf_3_urbain', 
        'indice_naf_2_urbain', 
        'indice_naf_1_urbain', 
        'indice_naf_4_rural', 
        'indice_naf_3_rural', 
        'indice_naf_2_rural', 
        'indice_naf_1_rural', 
        'indice_equipements_commerces',
        'indice_equipements_loisirs', 
        'indice_equipements_transports',
        'indice_equipements_enseignement', 
        'indice_equipements_sante', 
        'indice_equipements_services', 
        'indice_desserte_train', 
        'indice_frequentation_train', 
        'nb_stagiaires',
        'nb_permis_locaux',
        'nb_permis_logements',
        'nb_chambres_hotels'
    ])
    df = df.loc[df.type_milieu_2 == milieu.upper()]

    return df


def get_data_metier(metier): # A généraliser. ***

    insee_gridded_data_relevant_fields = get_insee_gridded_data_relevant_fields()

    def numerisation_zone(x):
        if x == 'C': y = 4
        if x == 'B': y = 3
        if x == 'I': y = 2
        if x == 'R': y = 1
        return y

    df_urbain = get_data_metier_milieu(metier, 'urbain', insee_gridded_data_relevant_fields)
    df_urbain['milieu'] = 1
    df_urbain['zone'] = df_urbain['type_milieu_4'].map(numerisation_zone)

    df_rural = get_data_metier_milieu(metier, 'rural', insee_gridded_data_relevant_fields)
    df_rural['milieu'] = 0
    df_rural['zone'] = df_rural['type_milieu_4'].map(numerisation_zone)

    return {
        'insee_gridded_data_relevant_fields': insee_gridded_data_relevant_fields,
        'urbain': df_urbain,
        'rural': df_rural
    }



# 2. Formatage des variables.


def transform_data(df, relevant_fields):

    # Construction de variables intensives.
    
    def build_age_average(ind_0_3, ind_4_5, ind_6_10, ind_11_17, ind_18_24, ind_25_39, ind_40_54, ind_55_64, ind_65_79, ind_80p):
        return (ind_0_3 * 2 + ind_4_5 * 4.5 + ind_6_10 * 8 + ind_11_17 * 14 + ind_18_24 * 21 + ind_25_39 * 32 + ind_40_54 * 47 + ind_55_64 * 59 + ind_65_79 * 72 + ind_80p * 82) / (ind_0_3 + ind_4_5 + ind_6_10 + ind_11_17 + ind_18_24 + ind_25_39 + ind_40_54 + ind_55_64 + ind_65_79 + ind_80p)
    
    df['age_moyen'] = df.apply(
        lambda row: build_age_average(
            row['ind_0_3'], row['ind_4_5'], row['ind_6_10'], row['ind_11_17'], row['ind_18_24'], row['ind_25_39'], row['ind_40_54'], row['ind_55_64'], row['ind_65_79'], row['ind_80p']
        )
    , axis=1
    )

    def build_intensive_variable(quantity, total):
        return quantity/total
    
    df['taux_pauvrete'] = df.apply(
        lambda row: build_intensive_variable(row['men_pauv'], row['men']), 
        axis=1
    )
    df['taux_menages_1_individu'] = df.apply(
        lambda row: build_intensive_variable(row['men_1ind'], row['men']), 
        axis=1
    )
    df['taux_menages_plus_5_individus'] = df.apply(
        lambda row: build_intensive_variable(row['men_5ind'], row['men']), 
        axis=1
    )
    df['taux_menages_proprietaires'] = df.apply(
        lambda row: build_intensive_variable(row['men_prop'], row['men']), 
        axis=1
    )
    df['taux_menages_monoparentaux'] = df.apply(
        lambda row: build_intensive_variable(row['men_fmp'], row['men']), 
        axis=1
    )
    df['niveau_vie'] = df.apply(
        lambda row: build_intensive_variable(row['ind_snv'], row['ind']), 
        axis=1
    )
    df['surface_logement'] = df.apply(
        lambda row: build_intensive_variable(row['men_surf'], row['men']), 
        axis=1
    )
    df['taux_logements_collectifs'] = df.apply(
        lambda row: build_intensive_variable(row['men_coll'], row['men']), 
        axis=1
    )
    df['taux_maisons'] = df.apply(
        lambda row: build_intensive_variable(row['men_mais'], row['men']), 
        axis=1
    )
    df['taux_logements_avant_1945'] = df.apply(
        lambda row: build_intensive_variable(row['log_av45'], row['men']), 
        axis=1
    )
    df['taux_logements_1945_1970'] = df.apply(
        lambda row: build_intensive_variable(row['log_45_70'], row['men']), 
        axis=1
    )
    df['taux_logements_1970_1990'] = df.apply(
        lambda row: build_intensive_variable(row['log_70_90'], row['men']), 
        axis=1
    )
    df['taux_logements_apres_1990'] = df.apply(
        lambda row: build_intensive_variable(row['log_ap90'], row['men']), 
        axis=1
    )
    df['taux_logements_sociaux'] = df.apply(
        lambda row: build_intensive_variable(row['log_soc'], row['men']), 
        axis=1
    )
    df['taux_population_moins_3_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_0_3'], row['ind']), 
        axis=1
    )
    df['taux_population_4_5_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_4_5'], row['ind']), 
        axis=1
    )
    df['taux_population_6_10_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_6_10'], row['ind']), 
        axis=1
    )
    df['taux_population_11_17_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_11_17'], row['ind']), 
        axis=1
    )
    df['taux_population_18_24_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_18_24'], row['ind']), 
        axis=1
    )
    df['taux_population_25_39_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_25_39'], row['ind']), 
        axis=1
    )
    df['taux_population_40_54_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_40_54'], row['ind']), 
        axis=1
    )
    df['taux_population_55_64_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_55_64'], row['ind']), 
        axis=1
    )
    df['taux_population_65_79_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_65_79'], row['ind']), 
        axis=1
    )
    df['taux_population_plus_80_ans'] = df.apply(
        lambda row: build_intensive_variable(row['ind_80p'], row['ind']), 
        axis=1
    )

    # Numérisation des variables catégorielles.

    df['caractere_employeur'] = df.apply(
        lambda row: 1 if row['caractere_employeur'] == 'O' else 0, 
        axis=1
    )

    # Suppression des champs bruts inutiles.

    df = df.drop(relevant_fields, axis=1)
    
    return df



# 3. Ajout des prédictions de survie à 3 ans parmi les variables explicatives.


def add_predictions_survie_3_ans(df):

    # Chargement du modèle à 3 ans.
    nom_fichier_modele = '{}/survie_3_ans/{}_trained_model'.format(label_metier, id_modele_survie_3_ans)
    fichier_modele = open(nom_fichier_modele, 'rb')
    modele = pickle.load(fichier_modele)
    fichier_modele.close()
    nom_fichier_X_3 = '{}/survie_3_ans/{}_X'.format(label_metier, id_modele_survie_3_ans)
    fichier_X_3 = open(nom_fichier_X_3, 'rb')
    X_3 = pickle.load(fichier_X_3)
    fichier_X_3.close()

    df_entrainement = df[
        df.nb_concurrents_debut.notnull() & 
        df.nb_habitants_etablissement_debut.notnull() &
        df.taux_motorisation.notnull() 
    ]

    # X = df_entrainement.drop(
    #     ['siret', 'duree_vie', 'type_milieu_2', 'type_milieu_4', 'survie_5_ans'], axis=1
    # )
    X = df_entrainement[list(X_3.columns)]

    y_predit = pd.DataFrame(modele.predict_proba(X)[:,1], columns=['prediction_survie_3_ans'])

    # y_predit = pd.DataFrame(modele.predict(X), columns=['prediction_survie_3_ans'])
    # y_predit['prediction_survie_3_ans'] = y_predit.apply(
    #     lambda row: 1 if row['prediction_survie_3_ans'] == True else 0, 
    #     axis=1
    # )

    return pd.concat([df_entrainement.reset_index(drop=True), y_predit.reset_index(drop=True)], axis=1)



# 4. Paramétrage des modèles.


def find_best_rf_model(X, y, scorer, cv, dataset_name):

    parameters = {
        'n_estimators': [100] # [100, 200, 500]
        #, 'criterion': ['gini', 'entropy']
        , 'max_depth': [2, 5, 10, 20, 50, 100]
        #, 'min_samples_split': [2]
        #, 'min_samples_leaf': [1]
        #, 'max_features': ['auto', 'sqrt']
        , 'class_weight': ['balanced']
    } 

    scorer = make_scorer(scorer) 

    grid_search = GridSearchCV(RandomForestClassifier(), parameters, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(X, y)
    #print(grid_search.best_params_)
    #print(grid_search.best_score_)

    # Modèle entraîné résultant du GridSearch.
    #model = grid_search.best_estimator_
    # Modèle neuf paramétré optimalement.
    best_rf_model = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1)

    return {'parameterized_model': best_rf_model, 'parameters': grid_search.best_params_}


def filter_most_signifiant_features(model, X, y):
    
    trained_model = model['parameterized_model'].fit(X, y)

    # Classement des variables explicatives par ordre d'importance.
    dict_feature_importances = {}
    for i in range(X.shape[1]):
        dict_feature_importances[list(X.columns)[i]] = list(trained_model.feature_importances_)[i]
    dict_ordered_feature_importances = {k: v for k, v in sorted(dict_feature_importances.items(), key=lambda item: item[1], reverse=True)}
    list_ordered_importances = list(dict_ordered_feature_importances.values())
    list_ordered_features = list(dict_ordered_feature_importances.keys())

    # Tests de sélection de variables.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    list_nb_features = sorted(list(set(list(range(5, X.shape[1], 5)) + [X.shape[1]])))
    tmp_best_score = 0
    liste_chiffres = []
    for nb_features in list_nb_features:
        threshold = list_ordered_importances[nb_features-1]
        selector = SelectFromModel(trained_model, threshold=threshold, prefit=True)
        X_train_transformed = pd.DataFrame(selector.transform(X_train), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])
        new_model = RandomForestClassifier(**trained_model.get_params())
        new_model.fit(X_train_transformed, y_train)
        X_test_transformed = pd.DataFrame(selector.transform(X_test), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])
        y_pred = new_model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        chiffres = 'En se limitant aux {} variables les plus discriminantes, on obtient les scores suivants : accuracy = {} ; cohen_kappa = {}, f1 = {}.'.format(nb_features, accuracy, cohen_kappa, f1)
        liste_chiffres.append(chiffres)
        if f1 >= tmp_best_score:
            tmp_best_score = f1
            X_transformed = pd.DataFrame(selector.transform(X), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])

    return X_transformed, liste_chiffres, dict_ordered_feature_importances


def find_best_models(data, scorer, cv):

    clean_data_urbain = data['urbain']
    clean_data_rural = data['rural']
    clean_data = pd.concat([clean_data_urbain, clean_data_rural]).sample(frac=1).reset_index(drop=True)

    training_data_survie_5_ans = clean_data.drop(
        ['siret', 'duree_vie', 'type_milieu_2', 'type_milieu_4'], axis=1
    )[
        clean_data.nb_concurrents_debut.notnull() & 
        clean_data.nb_habitants_etablissement_debut.notnull() &
        clean_data.taux_motorisation.notnull() &
        clean_data.prediction_survie_3_ans.notnull()
    ]
    X_5 = training_data_survie_5_ans.drop(['survie_5_ans'], axis=1)
    y_5 = training_data_survie_5_ans['survie_5_ans']
    print('Optimisation du Random Forest sur les données 5...')
    best_rf_model_5 = find_best_rf_model(X_5, y_5, scorer=scorer, cv=cv, dataset_name='5')
    # print('Optimisation du Gradient Boosting sur les données 5...')
    # best_gb_model_5 = find_best_gb_model(X_5, y_5, scorer=scorer, cv=cv, dataset_name='5')

    simplified_X_5, liste_chiffres, dict_ordered_feature_importances = filter_most_signifiant_features(best_rf_model_5, X_5, y_5)

    # return {
    #     '5': {'dataset_name': '5', 'X': X_5, 'y': y_5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_5}, {'model_name': 'gb', 'optimized_model': best_gb_model_5}]}
    # }
    return {
        '5': {'dataset_name': '5', 'X': simplified_X_5, 'y': y_5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_5}], 'liste_chiffres': liste_chiffres, 'dict_ordered_feature_importances': dict_ordered_feature_importances}
    }



# 5. Détermination des meilleurs modèles.


def evaluate_model(X, y, model, cv):

    # Validation croisée.
    #cv_score = cross_val_score(model, X, y, scoring=scorer, cv=cv)
    #print('Cross validated scores:', cv_score)
    #print('Cross validated overall score:', cv_score.mean())
    
    # Accuracy par Cross Validation.
    y_pred = cross_val_predict(model, X, y, cv=cv)
    #print('Cross validated accuracy:', accuracy_score(y, y_pred))

    # Rapport de classification du modèle entraîné par validation croisée.
    #print('Classification report:', classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    
    return {
        'score': accuracy_score(y, y_pred),
        'classification_report': classification_report(y_true=y, y_pred=y_pred, output_dict=True),
        'confusion_matrix': {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    }


def save_in_database(to_be_saved):

    # Sauvegarde des données utiles dans la base de données. ***
    query = """
        insert into entrainements_modeles(fk_metier, date, fk_donnees_entrainement, fk_modele, parametres, accuracy, rapport_classification, confusion_matrix, chiffres_selection_variables, dict_ordered_feature_importances)
        values ('{}', now(), '{}', '{}', '{}', {}, '{}', '{}', '{}', '{}');
    """.format(
        label_metier, to_be_saved['dataset_name'], to_be_saved['model_name'], json.dumps(to_be_saved['parameters']), 
        to_be_saved['score'], json.dumps(to_be_saved['classification_report']), json.dumps(to_be_saved['confusion_matrix']),
        ' ; '.join(to_be_saved['liste_chiffres']), json.dumps(to_be_saved['dict_ordered_feature_importances'])
    )
    cur.execute(query)
    con.commit()

    query = "select max(a.id) from entrainements_modeles a;"
    cur.execute(query)
    id_model = cur.fetchall()[0][0]
    
    # # Export des données utiles sous format json. # Provisoire. ***
    # with open('{}_{}'.format(to_be_saved['dataset_name'], to_be_saved['model_name']), 'w') as f:
    #     json.dump(json.dumps(to_be_saved), f)

    return id_model


def find_best_model(best_models, cv):

    # Evaluation des modèles et sauvegarde des résultats.
    saved_models = {}
    for model in best_models['optimized_models']:
        print('Evaluation du modèle {} sur les données {}...'.format(model['model_name'], best_models['dataset_name']))
        model_evaluation = evaluate_model(X=best_models['X'], y=best_models['y'], model=model['optimized_model']['parameterized_model'], cv=cv)
        to_be_saved = {
            'dataset_name': best_models['dataset_name'],
            'model_name': model['model_name'],
            'parameters': model['optimized_model']['parameters'],
            'score': model_evaluation['score'],
            'classification_report': model_evaluation['classification_report'], 
            'confusion_matrix': model_evaluation['confusion_matrix'],
            'liste_chiffres': best_models['liste_chiffres'],
            'dict_ordered_feature_importances': best_models['dict_ordered_feature_importances']
        }
        id_model = save_in_database(to_be_saved)
        saved_models[id_model] = {
            'id_model': int(id_model),
            'dataset_name': str(best_models['dataset_name']),
            'model_name': str(model['model_name']),                
            'X': best_models['X'],
            'y': best_models['y'],
            'parameterized_model': model['optimized_model']['parameterized_model'],
            'parameters': model['optimized_model']['parameters']
        }

    # Récupération du meilleur modèle.
    query = """
        select a.id
        from entrainements_modeles a 
        inner join (
            select max(a.rapport_classification::json->>'accuracy') as accuracy_max
            from entrainements_modeles a
            where a.id in ({})
        ) b on a.rapport_classification::json->>'accuracy' = b.accuracy_max
        order by a.id desc
		limit 1
    """.format(', '.join(list(map(lambda x: str(x), list(saved_models.keys())))))
    cur.execute(query)
    id_model = cur.fetchall()[0][0]

    query = """
        update entrainements_modeles as a
        set actif = false
        where a.fk_donnees_entrainements = {};
        update entrainements_modeles as a
        set actif = true
        where a.id = {};
    """.format(best_models['dataset_name'], id_model)

    return saved_models[id_model]



# 6. Entraînement des meilleurs modèles.


def train_model(best_model):

    X = best_model['X']
    y = best_model['y']
    model = best_model['parameterized_model']

    print('Entraînement du modèle en cours...')
    trained_model = model.fit(X, y)
    print('Entraînement du modèle terminé.')
       
    # print('Calcul du poids des variables en cours...')
    # shap_values = shap.TreeExplainer(trained_model).shap_values(X)
    # if len(shap_values) == 2: shap_values = shap_values[1]
    # print('Calcul du poids des variables terminé.')

    # shap_values_file_name = '{}/{}_shap_values'.format(label_metier, id_model)
    # shap_values_file = open(shap_values_file_name, 'wb') 
    # pickle.dump(shap_values, shap_values_file)

    # f = plt.figure()
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # summary_plot_file_name = '{}/{}_shap_values_summary_plot_1.png'.format(label_metier, id_model)
    # f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)
    # f = plt.figure()
    # shap.summary_plot(shap_values, X)
    # summary_plot_file_name = '{}/{}_shap_values_summary_plot_2.png'.format(label_metier, id_model)
    # f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)

    # path = "{}/{}_shap_values_dependence_plots".format(label_metier, id_model)
    # os.mkdir(path)
    # feature_importances = dict(zip(list(X.columns), list(trained_model.feature_importances_)))
    # for x in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
    #     f = plt.figure()
    #     shap.dependence_plot(x[0], shap_values, X, show=False)
    #     dependence_plot_file_name = "{}/{}_shap_values_dependence_plots/{}_{}.png".format(label_metier, id_model, x[0], round(x[1], 4))
    #     plt.savefig(dependence_plot_file_name)

    return trained_model



# MAIN.


def main(label_metier):

    print('1. Récupération des données en cours...')
    data = get_data_metier(label_metier)
    print('1. Récupération des données terminée.\n')

    print('2. Transformation des données en cours...')
    clean_data = {
        'urbain': transform_data(data['urbain'], data['insee_gridded_data_relevant_fields']), 
        'rural': transform_data(data['rural'], data['insee_gridded_data_relevant_fields'])
    }
    print('2. Transformation des données terminée.\n')

    print('3. Ajout du score de survie à 3 ans prédit parmi les variables explicatives en cours...')
    complete_data = {
        'urbain': add_predictions_survie_3_ans(clean_data['urbain']),
        'rural': add_predictions_survie_3_ans(clean_data['rural'])
    }
    print('3. Ajout du score de survie à 3 ans prédit parmi les variables explicatives terminé...\n')

    print('4. Optimisation des modèles envisagés en cours...')
    best_models = find_best_models(data=complete_data, scorer=f1_score, cv=5)
    print('4. Optimisation des modèles envisagés terminée.\n')

    print('5. Evaluation des modèles optimaux en cours...')
    best_model_survie_5_ans = find_best_model(best_models['5'], cv=10)
    print('5. Evaluation des modèles optimaux terminée.\n')

    print('6. Entraînement du meilleur modèle en cours...')
    trained_model_survie_5_ans = train_model(best_model_survie_5_ans) 
    print('6. Entraînement du meilleur modèle terminé.\n')

    sauver_pickle(complete_data, 'complete_data', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(best_model_survie_5_ans['X'], 'X', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(list(best_model_survie_5_ans['X'].columns), 'liste_variables_selectionnees', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(best_model_survie_5_ans['y'], 'y', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(trained_model_survie_5_ans, 'trained_model', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(best_model_survie_5_ans['parameters'], 'parameters', label_metier, best_model_survie_5_ans['id_model'])
    sauver_pickle(best_models['5']['dict_ordered_feature_importances'], 'dict_ordered_feature_importances', label_metier, best_model_survie_5_ans['id_model'])


main(label_metier)
