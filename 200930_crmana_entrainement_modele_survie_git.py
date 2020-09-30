# Référent : Jean VAN HECKE.
# Entamé le 16/09/20.
# Finalisé le ...



# Imports.

import psycopg2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, make_scorer, confusion_matrix, classification_report
import json
import pandas as pd
import shap
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os



# Arguments. # A généraliser. ***

label_metier = 'boulangerie' # ***



# Paramètres.

# Identifiants de connexion à la base de données. 
# A NE PAS PUBLIER. ***
db_logins = {}



# 1. Récupération des données.


# Connexion à la base de données.

con = psycopg2.connect(
    host = db_logins['host'], 
    database = db_logins['database'], 
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
    query_france = """
        select a.siret, b.duree_vie, b.survie_3_ans, b.survie_5_ans, c.type_com as type_milieu_2, c.statut_2017 as type_milieu_4
        , e1.indice_densite_emplois
        , case when f1.p16_log != 0 then round((f1.p16_rsecocc/f1.p16_log)::numeric, 2) else 0 end as taux_residences_secondaires
        , {}
        , g2.nb_transactions_immobilieres
        , g3.valeur::double precision as indice_population
        , h1.caractere_employeur_etablissement
        , case when h1.activite_principale_registre_metiers_etablissement is not null then 1 else 0 end as inscription_registre_metiers
        , h2.nb_concurrents_debut_{}, h2.nb_habitants_etablissement_debut_{}
        , h2.taux_etablissements_concurrents_debut_{}/*, h2.age_moyen_concurrence_{}*/
        , h3.indice_concurrence_restreinte_{}::double precision, h3.indice_concurrence_large_{}
        , h3.indice_artisanat_proximite_{}::double precision, h3.indice_artisanat_construction_{}::double precision
        , h3.indice_artisanat_economie_creative_{}::double precision, h3.indice_artisanat_soutien_{}::double precision
        , h3.indice_densite_naf_niveau_4_{}, h3.indice_densite_naf_niveau_3_{}
        , h3.indice_densite_naf_niveau_2_{}, h3.indice_densite_naf_niveau_1_{}
        , h3.indice_equipements_commerces, h3.indice_equipements_loisirs, h3.indice_equipements_transports
        , h3.indice_equipements_enseignement, h3.indice_equipements_sante, h3.indice_equipements_services
        , h3.indice_desserte_train, h3.indice_frequentation_train
        , h3.indice_reseau
        from etablissements_cibles_{} a
        inner join taux_survie_etablissements b on a.siret = b.siret
        inner join unites_urbaines c on a.code_commune = c.codgeo
        inner join categories_communes_aires_urbaines e1 on a.code_commune = e1.codgeo
        inner join insee_logements_iris f1 on a.code_iris = f1.iris
        inner join insee_donnees_carroyees_niveau_200m g1 on a.idinspire = g1.idinspire
        inner join donnees_agregees_carreau g2 on a.idinspire = g2.idinspire
        inner join donnees_ponderees_carreau g3 on a.idinspire = g3.idinspire and g3.fk_donnees_ponderees = 4
        inner join sirene_etablissements h1 on a.siret = h1.siret
        inner join indices_concurrence h2 on a.siret = h2.siret
        inner join indices_synergie_territoriale h3 on a.siret = h3.siret
    """.format(
        ', '.join(list(map(lambda x: 'g1.{}'.format(x), relevant_fields))), milieu, milieu, milieu, milieu, milieu, milieu, 
        milieu, milieu, milieu, milieu, milieu, milieu, milieu, milieu, metier
    )
    cur.execute(query_france)
    data_france = cur.fetchall()
    df_france = pd.DataFrame(data_france, columns=[
        'siret', 'duree_vie', 'survie_3_ans', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4', 
        'indice_densite_emplois', 'taux_residences_secondaires'] + relevant_fields + ['nb_transactions_immobilieres', 'indice_population', 
        'caractere_employeur', 'inscription_registre_metiers', 
        'nb_concurrents_debut', 'nb_habitants_etablissement_debut', 'taux_etablissements_concurrents_debut',
        'indice_concurrence_restreinte', 'indice_concurrence_large', 
        'indice_artisanat_proximite', 'indice_artisanat_construction', 
        'indice_artisanat_economie_creative', 'indice_artisanat_soutien', 
        'indice_naf_4', 'indice_naf_3', 'indice_naf_2', 'indice_naf_1', 
        'indice_equipements_commerces','indice_equipements_loisirs', 'indice_equipements_transports',
        'indice_equipements_enseignement', 'indice_equipements_sante', 'indice_equipements_services', 
        'indice_desserte_train', 'indice_frequentation_train', 'indice_reseau'
    ])
    
    # Données en Nouvelle-Aquitaine.
    query_na = """
        select a.siret, b.duree_vie, b.survie_3_ans, b.survie_5_ans, c.type_com as type_milieu_2, c.statut_2017 as type_milieu_4
        , e1.indice_densite_emplois
        , case when f1.p16_log != 0 then round((f1.p16_rsecocc/f1.p16_log)::numeric, 2) else 0 end as taux_residences_secondaires
        , {}
        , g2.nb_transactions_immobilieres
        , g3.valeur::double precision as indice_population
        , h1.caractere_employeur_etablissement
        , case when h1.activite_principale_registre_metiers_etablissement is not null then 1 else 0 end as inscription_registre_metiers
        , h2.nb_concurrents_debut_{}, h2.nb_habitants_etablissement_debut_{}
        , h2.taux_etablissements_concurrents_debut_{}/*, h2.age_moyen_concurrence_{}*/
        , h3.indice_concurrence_restreinte_{}::double precision, h3.indice_concurrence_large_{}
        , h3.indice_artisanat_proximite_{}::double precision, h3.indice_artisanat_construction_{}::double precision
        , h3.indice_artisanat_economie_creative_{}::double precision, h3.indice_artisanat_soutien_{}::double precision
        , h3.indice_densite_naf_niveau_4_{}, h3.indice_densite_naf_niveau_3_{}
        , h3.indice_densite_naf_niveau_2_{}, h3.indice_densite_naf_niveau_1_{}
        , h3.indice_equipements_commerces, h3.indice_equipements_loisirs, h3.indice_equipements_transports
        , h3.indice_equipements_enseignement, h3.indice_equipements_sante, h3.indice_equipements_services
        , h3.indice_desserte_bus, h3.indice_desserte_train, h3.indice_frequentation_train
        , h3.indice_reseau
        from etablissements_cibles_{}_na a
        inner join taux_survie_etablissements b on a.siret = b.siret
        inner join unites_urbaines c on a.code_commune = c.codgeo
        inner join categories_communes_aires_urbaines e1 on a.code_commune = e1.codgeo
        inner join insee_logements_iris f1 on a.code_iris = f1.iris
        inner join insee_donnees_carroyees_niveau_200m g1 on a.idinspire = g1.idinspire
        inner join donnees_agregees_carreau g2 on a.idinspire = g2.idinspire
        inner join donnees_ponderees_carreau g3 on a.idinspire = g3.idinspire and g3.fk_donnees_ponderees = 4        
        inner join sirene_etablissements h1 on a.siret = h1.siret
        inner join indices_concurrence h2 on a.siret = h2.siret
        inner join indices_synergie_territoriale h3 on a.siret = h3.siret
    """.format(
        ', '.join(list(map(lambda x: 'g1.{}'.format(x), relevant_fields))), milieu, milieu, milieu, milieu, milieu, milieu, 
        milieu, milieu, milieu, milieu, milieu, milieu, milieu, milieu, metier
    )
    cur.execute(query_na)
    data_na = cur.fetchall()
    df_na = pd.DataFrame(data_na, columns=[
        'siret', 'duree_vie', 'survie_3_ans', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4', 
        'indice_densite_emplois', 'taux_residences_secondaires'] + relevant_fields + ['nb_transactions_immobilieres', 'indice_population', 
        'caractere_employeur', 'inscription_registre_metiers', 
        'nb_concurrents_debut', 'nb_habitants_etablissement_debut', 'taux_etablissements_concurrents_debut',
        'indice_concurrence_restreinte', 'indice_concurrence_large', 
        'indice_artisanat_proximite', 'indice_artisanat_construction', 
        'indice_artisanat_economie_creative', 'indice_artisanat_soutien', 
        'indice_naf_4', 'indice_naf_3', 'indice_naf_2', 'indice_naf_1', 
        'indice_equipements_commerces', 'indice_equipements_loisirs', 'indice_equipements_transports',
        'indice_equipements_enseignement', 'indice_equipements_sante', 'indice_equipements_services', 
        'indice_desserte_bus','indice_desserte_train', 'indice_frequentation_train', 
        'indice_reseau'
    ])

    return {'france': df_france, 'na': df_na}


def get_data_metier(metier): # A généraliser. ***

    insee_gridded_data_relevant_fields = get_insee_gridded_data_relevant_fields()

    dfs_urbain = get_data_metier_milieu(metier, 'urbain', insee_gridded_data_relevant_fields)
    dfs_rural = get_data_metier_milieu(metier, 'rural', insee_gridded_data_relevant_fields)

    return {
        'insee_gridded_data_relevant_fields': insee_gridded_data_relevant_fields,
        'urbain': dfs_urbain,
        'rural': dfs_rural
    } # A généraliser. ***



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



# 3. Paramétrage des modèles.


# def find_parameters_optimisation_window(X, y, estimator, parameters, scorer, dataset_name, model_name):
    
#     data = pd.concat([X, y], axis=1)
#     data_sample = data.sample(n=min(len(data), 6000)) # ***
#     X_sample = data_sample.iloc[:, :-1]
#     y_sample = data_sample.iloc[:, -1]

#     parameters_grid = {x: parameters[x]['values'] for x in list(parameters.keys())}
#     print(parameters_grid) # ***

#     window_found = False
#     window_found_dict = {x: False for x in list(parameters.keys())}
#     while window_found == False:
#         grid_search = GridSearchCV(estimator=estimator, param_grid=parameters_grid, scoring=scorer, cv=3, n_jobs=-1, verbose=1)
#         grid_search.fit(X_sample, y_sample)
#         print(grid_search.best_params_) # ***
#         new_parameters_grid = {}
#         for parameter in list(parameters.keys()):
#             if parameters[parameter]['type'] == 'text':
#                 new_parameters_grid[parameter] = parameters[parameter]['values']
#                 window_found_dict[parameter] = True
#             else:
#                 if parameters[parameter]['type'] == 'int':
#                     new_parameters_grid[parameter] = list(max(1, parameters[parameter]['min'], int(x)) for x in [grid_search.best_params_[parameter]*2**n for n in range(-int((len(parameters[parameter]['values'])-1)/2), int((len(parameters[parameter]['values'])-1)/2)+1)])
#                 else:
#                     new_parameters_grid[parameter] = list(max(1, parameters[parameter]['min'], x) for x in [grid_search.best_params_[parameter]*2**n for n in range(-int((len(parameters[parameter]['values'])-1)/2), int((len(parameters[parameter]['values'])-1)/2)+1)])                       
#                 if (grid_search.best_params_[parameter] != parameters_grid[parameter][0] and grid_search.best_params_[parameter] != parameters_grid[parameter][-1]):
#                     window_found_dict[parameter] = True
#                 else:
#                     window_found_dict[parameter] = False
#         parameters_grid = new_parameters_grid
#         print(parameters_grid) # ***
#         if window_found_dict == {x: True for x in list(parameters.keys())}:
#             window_found = True

#     # On prépare la fenêtre d'optimisation des paramètres pour la deuxième étape. 
#     # Pour les paramètres numériques, on crée des listes de cinq éléments, entre la valeur précédent et la valeur suivant le paramètre optimisé à l'étape 1. A revoir ? ***
#     parameters_optimisation_window = {}
#     for parameter in list(parameters.keys()):
#         if parameters[parameter]['type'] == 'text':
#             parameters_optimisation_window[parameter] = parameters[parameter]['values']
#         else:
#             best_parameter_index = int(len(parameters_grid[parameter])/2)
#             if parameters[parameter]['type'] == 'int':
#                 parameters_optimisation_window[parameter] = list(set(list(max(1, parameters[parameter]['min'], int(x)) for x in [parameters_grid[parameter][best_parameter_index-1], (parameters_grid[parameter][best_parameter_index-1]+parameters_grid[parameter][best_parameter_index])/2, parameters_grid[parameter][best_parameter_index], (parameters_grid[parameter][best_parameter_index]+parameters_grid[parameter][best_parameter_index+1])/2, parameters_grid[parameter][best_parameter_index+1]])))
#             else:
#                 parameters_optimisation_window[parameter] = list(set(list(max(1, parameters[parameter]['min'], x) for x in [parameters_grid[parameter][best_parameter_index-1], (parameters_grid[parameter][best_parameter_index-1]+parameters_grid[parameter][best_parameter_index])/2, parameters_grid[parameter][best_parameter_index], (parameters_grid[parameter][best_parameter_index]+parameters_grid[parameter][best_parameter_index+1])/2, parameters_grid[parameter][best_parameter_index+1]])))
#     print('parameters_optimisation_window:', parameters_optimisation_window)
    
#     return parameters_optimisation_window


# def find_best_parameters(X, y, estimator, parameters, scorer, dataset_name, model_name):

#     scorer = make_scorer(scorer)
    
#     print('GridSearch 1...')
#     parameters_optimisation_window = find_parameters_optimisation_window(X, y, estimator, parameters, scorer, dataset_name, model_name)
#     print('parameters_optimisation_window:', parameters_optimisation_window) # ***
#     parameters_optimisation_window_file_name = '{}/{}_{}_parameters_optimisation_window'.format(label_metier, dataset_name, model_name)
#     parameters_optimisation_window_file = open(parameters_optimisation_window_file_name, 'wb') 
#     pickle.dump(parameters_optimisation_window, parameters_optimisation_window_file)
#     with open('{}/{}_{}_parameters_optimisation_window.json'.format(label_metier, dataset_name, model_name), 'w') as json_file:
#         json.dump(parameters_optimisation_window, json_file)

#     print('GridSearch 2...')
#     parameters_grid = parameters_optimisation_window

#     data = pd.concat([X, y], axis=1)
#     data_sample = data.sample(n=min(len(data), 6000)) # ***
#     X_sample = data_sample.iloc[:, :-1]
#     y_sample = data_sample.iloc[:, -1]

#     grid_search = GridSearchCV(estimator=estimator, param_grid=parameters_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=1)
#     grid_search.fit(X_sample, y_sample)
#     best_parameters = grid_search.best_params_
#     print('best parameters:', best_parameters) # ***
#     best_parameters_file_name = '{}/{}_{}_best_parameters'.format(label_metier, dataset_name, model_name)
#     best_parameters_file = open(best_parameters_file_name, 'wb') 
#     pickle.dump(best_parameters, best_parameters_file)
#     with open('{}/{}_{}_best_parameters.json'.format(label_metier, dataset_name, model_name), 'w') as json_file:
#         json.dump(best_parameters, json_file)

#     return best_parameters


def find_best_rf_model(X, y, scorer, cv, dataset_name):

    parameters = {
        'n_estimators': [100, 200] 
        #, 'criterion': ['gini', 'entropy']
        #, 'max_depth': [10, 20, 50, 100]
        #, 'min_samples_split': [2]
        #, 'min_samples_leaf': [1]
        #, 'max_features': ['auto', 'sqrt']
        , 'class_weight': ['balanced']
    }
    with open('{}/{}_rf_parameters_grid'.format(label_metier, dataset_name), 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)  

    scorer = make_scorer(scorer) 

    grid_search = GridSearchCV(RandomForestClassifier(), parameters, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(X, y)
    #print(grid_search.best_params_)
    #print(grid_search.best_score_)

    # Modèle entraîné résultant du GridSearch.
    #model = grid_search.best_estimator_
    # Modèle neuf paramétré optimalement.
    best_rf_model = RandomForestClassifier(**grid_search.best_params_)

    return {'parameterized_model': best_rf_model, 'parameters': grid_search.best_params_}


# def find_best_rf_model(X, y, scorer, cv, dataset_name):

#     best_parameters = {"n_estimators": 1000}
#     best_rf_model = RandomForestClassifier(**best_parameters)

#     return {'parameterized_model': best_rf_model, 'parameters': best_parameters}


# def find_best_rf_model(X, y, scorer, cv):

#     n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
#     max_features = ['auto', 'sqrt']
#     max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
#     max_depth.append(None)
#     min_samples_split = [2, 5, 10]
#     min_samples_leaf = [1, 2, 4]
#     bootstrap = [True, False]
    
#     random_grid = {
#         'n_estimators': n_estimators,
#         'max_features': max_features,
#         'max_depth': max_depth,
#         'min_samples_split': min_samples_split,
#         'min_samples_leaf': min_samples_leaf,
#         'bootstrap': bootstrap
#     }

#     acc_scorer = make_scorer(scorer)

#     rf = RandomForestClassifier(class_weight='balanced')

#     rf_random = RandomizedSearchCV(estimator=rf, scoring=acc_scorer, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
#     print('RandomizedSearch...')
#     rf_random.fit(X, y)

#     param_grid = rf_random.best_params_

#     param_grid['max_depth'] = list(set([max(1, rf_random.best_params_['max_depth'] + 10 * x) for x in range(-2, 3)]))
#     param_grid['max_features'] = [2, 3]
#     param_grid['bootstrap'] = [rf_random.best_params_['bootstrap']]
#     param_grid['n_estimators'] = [max(1, rf_random.best_params_['n_estimators'] + 100 * x) for x in range(-2, 3)]
#     param_grid['min_samples_leaf'] = [max(1, rf_random.best_params_['min_samples_leaf'] + x) for x in range(-1, 2)]
#     param_grid['min_samples_split'] = [max(2, rf_random.best_params_['min_samples_split'] + 2 * x) for x in range(-1, 2)]

#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=acc_scorer, cv=cv, n_jobs=-1)
#     print('GridSearch...')
#     grid_search.fit(X, y)

#     best_parameters = grid_search.best_params_
#     best_rf_model = RandomForestClassifier(**grid_search.best_params_)

#     return {'parameterized_model': best_rf_model, 'parameters': best_parameters}


def find_best_gb_model(X, y, scorer, cv, dataset_name):

    parameters = {
        'n_estimators': [100, 200] #, 200, 500, 1000, 2000] 
        , 'learning_rate': [0.01, 0.1]
        #, 'max_features': []
        , 'max_depth': [3, 4, 5]
    }
    with open('{}/{}_rf_parameters_grid'.format(label_metier, dataset_name), 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)  

    scorer = make_scorer(scorer) 

    grid_search = GridSearchCV(GradientBoostingClassifier(), parameters, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(X, y)
    #print(grid_search.best_params_)
    #print(grid_search.best_score_)

    # Modèle entraîné résultant du GridSearch.
    #model = grid_search.best_estimator_
    # Modèle neuf paramétré optimalement.
    best_gb_model = GradientBoostingClassifier(**grid_search.best_params_)

    return {'parameterized_model': best_gb_model, 'parameters': grid_search.best_params_}


# # Provisoire. ***
# def find_best_gb_model(X, y, scorer, cv, dataset_name):

#     best_parameters = {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 4}
#     best_gb_model = GradientBoostingClassifier(**best_parameters)

#     return {'parameterized_model': best_gb_model, 'parameters': best_parameters}


# def find_best_rf_model(X, y, scorer, cv, dataset_name):

#     rf_parameters = {
#         'n_estimators': {'name': 'n_estimators', 'type': 'int', 'values': [100], 'min': -1}, 
#         'min_samples_split': {'name': 'min_samples_split', 'type': 'int', 'values': [2], 'min': 2},
#         'min_samples_leaf': {'name': 'min_samples_leaf', 'type': 'int', 'values': [1], 'min': -1},
#         'max_depth': {'name': 'max_depth', 'type': 'int', 'values': [80], 'min': -1},
#         'max_features': {'name': 'max_features', 'type': 'text', 'values': ['auto', 'sqrt']},
#     }

#     best_parameters = find_best_parameters(X, y, RandomForestClassifier(), rf_parameters, scorer, dataset_name, 'rf')

#     best_rf_model = RandomForestClassifier(**best_parameters, n_jobs=-1)

#     return {'parameterized_model': best_rf_model, 'parameters': best_parameters}


# def find_best_gb_model(X, y, scorer, cv, dataset_name):

#     gb_parameters = {
#         'n_estimators': {'name': 'n_estimators', 'type': 'int', 'values': [100, 400]},#, 700, 1000]}, 
#         'min_samples_split': {'name': 'min_samples_split', 'type': 'int', 'values': [2]},#, 5, 10]},
#         'min_samples_leaf': {'name': 'min_samples_leaf', 'type': 'int', 'values': [1]},#, 2, 4]},
#         'max_depth': {'name': 'max_depth', 'type': 'int', 'values': [110]},#int(x) for x in np.linspace(10, 110, num=11)]},
#         'max_features': {'name': 'max_features', 'type': 'text', 'values': ['auto', 'sqrt']}
#     }

#     best_parameters = find_best_parameters(X, y, GradientBoostingClassifier(), gb_parameters, scorer, dataset_name, 'gb')

#     best_gb_model = GradientBoostingClassifier(**best_parameters)

#     return {'parameterized_model': best_gb_model, 'parameters': best_parameters}


def find_best_models(data, scorer, cv):

    clean_data_france_urbain = data['urbain']['france']
    clean_data_na_urbain = data['urbain']['na']
    clean_data_france_rural = data['rural']['france']
    clean_data_na_rural = data['rural']['na']

    training_data_france_urbain_survie_3_ans = clean_data_france_urbain.loc[clean_data_france_urbain['type_milieu_2'] == 'URBAIN'].drop(
        ['siret', 'duree_vie', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    )[
        clean_data_france_urbain.nb_concurrents_debut.notnull() & 
        clean_data_france_urbain.nb_habitants_etablissement_debut.notnull() 
    ]
    X_fu3 = training_data_france_urbain_survie_3_ans.drop(['survie_3_ans'], axis=1)
    y_fu3 = training_data_france_urbain_survie_3_ans['survie_3_ans']
    print('Optimisation du Random Forest sur les données fu3...')
    best_rf_model_fu3 = find_best_rf_model(X_fu3, y_fu3, scorer=scorer, cv=cv, dataset_name='fu3')
    print('Optimisation du Gradient Boosting sur les données fu3...')
    best_gb_model_fu3 = find_best_gb_model(X_fu3, y_fu3, scorer=scorer, cv=cv, dataset_name='fu3')

    training_data_na_urbain_survie_3_ans = clean_data_na_urbain.loc[clean_data_na_urbain['type_milieu_2'] == 'URBAIN'].drop(
        ['siret', 'duree_vie', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    )[
        clean_data_na_urbain.nb_concurrents_debut.notnull() & 
        clean_data_na_urbain.nb_habitants_etablissement_debut.notnull() 
    ]
    X_nu3 = training_data_na_urbain_survie_3_ans.drop(['survie_3_ans'], axis=1)
    y_nu3 = training_data_na_urbain_survie_3_ans['survie_3_ans']
    print('Optimisation du Random Forest sur les données nu3...')
    best_rf_model_nu3 = find_best_rf_model(X_nu3, y_nu3, scorer=scorer, cv=cv, dataset_name='nu3')
    print('Optimisation du Gradient Boosting sur les données nu3...')
    best_gb_model_nu3 = find_best_gb_model(X_nu3, y_nu3, scorer=scorer, cv=cv, dataset_name='nu3')

    training_data_france_rural_survie_3_ans = clean_data_france_rural.loc[clean_data_france_rural['type_milieu_2'] == 'RURAL'].drop(
        ['siret', 'duree_vie', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    )[
        clean_data_france_rural.nb_concurrents_debut.notnull() & 
        clean_data_france_rural.nb_habitants_etablissement_debut.notnull() 
    ]
    X_fr3 = training_data_france_rural_survie_3_ans.drop(['survie_3_ans'], axis=1)
    y_fr3 = training_data_france_rural_survie_3_ans['survie_3_ans']
    print('Optimisation du Random Forest sur les données fr3...')
    best_rf_model_fr3 = find_best_rf_model(X_fr3, y_fr3, scorer=scorer, cv=cv, dataset_name='fr3')
    print('Optimisation du Gradient Boosting sur les données fr3...')
    best_gb_model_fr3 = find_best_gb_model(X_fr3, y_fr3, scorer=scorer, cv=cv, dataset_name='fr3')

    training_data_na_rural_survie_3_ans = clean_data_na_rural.loc[clean_data_na_rural['type_milieu_2'] == 'RURAL'].drop(
        ['siret', 'duree_vie', 'survie_5_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    )[
        clean_data_na_rural.nb_concurrents_debut.notnull() & 
        clean_data_na_rural.nb_habitants_etablissement_debut.notnull() 
    ]
    X_nr3 = training_data_na_rural_survie_3_ans.drop(['survie_3_ans'], axis=1)
    y_nr3 = training_data_na_rural_survie_3_ans['survie_3_ans']  
    print('Optimisation du Random Forest sur les données nr3...')
    best_rf_model_nr3 = find_best_rf_model(X_nr3, y_nr3, scorer=scorer, cv=cv, dataset_name='nr3')
    print('Optimisation du Gradient Boosting sur les données nr3...')
    best_gb_model_nr3 = find_best_gb_model(X_nr3, y_nr3, scorer=scorer, cv=cv, dataset_name='nr3')

    # training_data_france_urbain_survie_5_ans = clean_data_france_urbain.loc[clean_data_france_urbain['type_milieu_2'] == 'URBAIN'].drop(
    #     ['siret', 'duree_vie', 'survie_3_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    # )[
    #     clean_data_france_urbain.nb_concurrents_debut.notnull() & 
    #     clean_data_france_urbain.nb_habitants_etablissement_debut.notnull() 
    # ]
    # X_fu5 = training_data_france_urbain_survie_5_ans.drop(['survie_5_ans'], axis=1)
    # y_fu5 = training_data_france_urbain_survie_5_ans['survie_5_ans']
    # print('Optimisation du Random Forest sur les données fu5...')
    # best_rf_model_fu5 = find_best_rf_model(X_fu5, y_fu5, scorer=scorer, cv=cv, dataset_name='fu5')
    # print('Optimisation du Gradient Boosting sur les données fu5...')
    # best_gb_model_fu5 = find_best_gb_model(X_fu5, y_fu5, scorer=scorer, cv=cv, dataset_name='fu5')

    # training_data_na_urbain_survie_5_ans = clean_data_na_urbain.loc[clean_data_na_urbain['type_milieu_2'] == 'URBAIN'].drop(
    #     ['siret', 'duree_vie', 'survie_3_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    # )[
    #     clean_data_na_urbain.nb_concurrents_debut.notnull() & 
    #     clean_data_na_urbain.nb_habitants_etablissement_debut.notnull() 
    # ]
    # X_nu5 = training_data_na_urbain_survie_5_ans.drop(['survie_5_ans'], axis=1)
    # y_nu5 = training_data_na_urbain_survie_5_ans['survie_5_ans']
    # print('Optimisation du Random Forest sur les données nu5...')
    # best_rf_model_nu5 = find_best_rf_model(X_nu5, y_nu5, scorer=scorer, cv=cv, dataset_name='nu5')
    # print('Optimisation du Gradient Boosting sur les données nu5...')
    # best_gb_model_nu5 = find_best_gb_model(X_nu5, y_nu5, scorer=scorer, cv=cv, dataset_name='nu5')

    # training_data_france_rural_survie_5_ans = clean_data_france_rural.loc[clean_data_france_rural['type_milieu_2'] == 'RURAL'].drop(
    #     ['siret', 'duree_vie', 'survie_3_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    # )[
    #     clean_data_france_rural.nb_concurrents_debut.notnull() & 
    #     clean_data_france_rural.nb_habitants_etablissement_debut.notnull() 
    # ]
    # X_fr5 = training_data_france_rural_survie_5_ans.drop(['survie_5_ans'], axis=1)
    # y_fr5 = training_data_france_rural_survie_5_ans['survie_5_ans']
    # print('Optimisation du Random Forest sur les données fr5...')
    # best_rf_model_fr5 = find_best_rf_model(X_fr5, y_fr5, scorer=scorer, cv=cv, dataset_name='fr5')
    # print('Optimisation du Gradient Boosting sur les données fr5...')
    # best_gb_model_fr5 = find_best_gb_model(X_fr5, y_fr5, scorer=scorer, cv=cv, dataset_name='fr5')

    # training_data_na_rural_survie_5_ans = clean_data_na_rural.loc[clean_data_na_rural['type_milieu_2'] == 'RURAL'].drop(
    #     ['siret', 'duree_vie', 'survie_3_ans', 'type_milieu_2', 'type_milieu_4'], axis=1
    # )[
    #     clean_data_na_rural.nb_concurrents_debut.notnull() & 
    #     clean_data_na_rural.nb_habitants_etablissement_debut.notnull() 
    # ]
    # X_nr5 = training_data_na_rural_survie_5_ans.drop(['survie_5_ans'], axis=1)
    # y_nr5 = training_data_na_rural_survie_5_ans['survie_5_ans'] 
    # print('Optimisation du Random Forest sur les données nr3...')
    # best_rf_model_nr5 = find_best_rf_model(X_nr5, y_nr5, scorer=scorer, cv=cv, dataset_name='nr5')
    # print('Optimisation du Gradient Boosting sur les données nr3...')
    # best_gb_model_nr5 = find_best_gb_model(X_nr5, y_nr5, scorer=scorer, cv=cv, dataset_name='nr5')

    return {
        'fu3': {'dataset_name': 'fu3', 'X': X_fu3, 'y': y_fu3, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_fu3}, {'model_name': 'gb', 'optimized_model': best_gb_model_fu3}]},
        'nu3': {'dataset_name': 'nu3', 'X': X_nu3, 'y': y_nu3, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_nu3}, {'model_name': 'gb', 'optimized_model': best_gb_model_nu3}]},
        'fr3': {'dataset_name': 'fr3', 'X': X_fr3, 'y': y_fr3, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_fr3}, {'model_name': 'gb', 'optimized_model': best_gb_model_fr3}]},
        'nr3': {'dataset_name': 'nr3', 'X': X_nr3, 'y': y_nr3, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_nr3}, {'model_name': 'gb', 'optimized_model': best_gb_model_nr3}]},
        # 'fu5': {'dataset_name': 'fu5', 'X': X_fu5, 'y': y_fu5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_fu5}, {'model_name': 'gb', 'optimized_model': best_gb_model_fu5}]},
        # 'nu5': {'dataset_name': 'nu5', 'X': X_nu5, 'y': y_nu5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_nu5}, {'model_name': 'gb', 'optimized_model': best_gb_model_nu5}]},
        # 'fr5': {'dataset_name': 'fr5', 'X': X_fr5, 'y': y_fr5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_fr5}, {'model_name': 'gb', 'optimized_model': best_gb_model_fr5}]},
        # 'nr5': {'dataset_name': 'nr5', 'X': X_nr5, 'y': y_nr5, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model_nr5}, {'model_name': 'gb', 'optimized_model': best_gb_model_nr5}]},
    }



# 4. Détermination des meilleurs modèles.


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
        insert into entrainements_modeles(fk_metier, date, fk_donnees_entrainement, fk_modele, parametres, accuracy, rapport_classification, confusion_matrix)
        values ('{}', now(), '{}', '{}', '{}', {}, '{}', '{}');
    """.format(
        label_metier, to_be_saved['dataset_name'], to_be_saved['model_name'], json.dumps(to_be_saved['parameters']), 
        to_be_saved['score'], json.dumps(to_be_saved['classification_report']), json.dumps(to_be_saved['confusion_matrix'])
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
    for models in best_models:
        for model in models['optimized_models']: # ***
            print('Evaluation du modèle {} sur les données {}...'.format(model['model_name'], models['dataset_name']))
            model_evaluation = evaluate_model(X=models['X'], y=models['y'], model=model['optimized_model']['parameterized_model'], cv=cv)
            to_be_saved = {
                'dataset_name': models['dataset_name'],
                'model_name': model['model_name'],
                'parameters': model['optimized_model']['parameters'],
                'score': model_evaluation['score'],
                'classification_report': model_evaluation['classification_report'], 
                'confusion_matrix': model_evaluation['confusion_matrix']
            }
            id_model = save_in_database(to_be_saved)
            saved_models[id_model] = {
                'id_model': int(id_model),
                'dataset_name': str(models['dataset_name']),
                'model_name': str(model['model_name']),                
                'X': models['X'],
                'y': models['y'],
                'parameterized_model': model['optimized_model']['parameterized_model']
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

    return saved_models[id_model]



# 5. Entraînement des meilleurs modèles.


def train_model(best_model):

    id_model = best_model['id_model']
    dataset_name = best_model['dataset_name']
    model_name = best_model['model_name']
    X = best_model['X']
    y = best_model['y']
    model = best_model['parameterized_model']

    print('Entraînement du modèle en cours...')
    trained_model = model.fit(X, y)
    print('Entraînement du modèle terminé.')

    print('Calcul du poids des variables en cours...')
    shap_values = shap.TreeExplainer(trained_model).shap_values(X)
    if len(shap_values) == 2: shap_values = shap_values[1]
    print('Calcul du poids des variables terminé.')

    X_file_name = '{}/{}_{}_{}_X'.format(label_metier, id_model, dataset_name, model_name)
    X_file = open(X_file_name, 'wb') 
    pickle.dump(X, X_file)  
    y_file_name = '{}/{}_{}_{}_y'.format(label_metier, id_model, dataset_name, model_name)  
    y_file = open(y_file_name, 'wb') 
    pickle.dump(y, y_file)    
    trained_model_file_name = '{}/{}_{}_{}_trained_model'.format(label_metier, id_model, dataset_name, model_name)
    trained_model_file = open(trained_model_file_name, 'wb') 
    pickle.dump(trained_model, trained_model_file)  
    shap_values_file_name = '{}/{}_{}_{}_shap_values'.format(label_metier, id_model, dataset_name, model_name)
    shap_values_file = open(shap_values_file_name, 'wb') 
    pickle.dump(shap_values, shap_values_file)

    f = plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar")
    summary_plot_file_name = '{}/{}_{}_{}_shap_values_summary_plot_1.png'.format(label_metier, id_model, dataset_name, model_name)
    f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)
    f = plt.figure()
    shap.summary_plot(shap_values, X)
    summary_plot_file_name = '{}/{}_{}_{}_shap_values_summary_plot_2.png'.format(label_metier, id_model, dataset_name, model_name)
    f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)

    path = "{}/{}_{}_{}_shap_values_dependence_plots".format(label_metier, id_model, dataset_name, model_name)
    os.mkdir(path)
    feature_importances = dict(zip(list(X.columns), list(trained_model.feature_importances_)))
    for x in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
        f = plt.figure()
        shap.dependence_plot(x[0], shap_values, X, show=False)
        dependence_plot_file_name = "{}/{}_{}_{}_shap_values_dependence_plots/{}_{}.png".format(label_metier, id_model, dataset_name, model_name, x[0], round(x[1], 4))
        plt.savefig(dependence_plot_file_name)

    return trained_model



# MAIN.


def main(label_metier):

    print('1. Récupération des données en cours...')
    data = get_data_metier(label_metier)
    print('1. Récupération des données terminée.\n')

    print('2. Transformation des données en cours...')
    clean_data = {
        'urbain': {
            'france': transform_data(data['urbain']['france'], data['insee_gridded_data_relevant_fields']), 
            'na': transform_data(data['urbain']['na'], data['insee_gridded_data_relevant_fields'])
        },
        'rural': {
            'france': transform_data(data['rural']['france'], data['insee_gridded_data_relevant_fields']), 
            'na': transform_data(data['rural']['na'], data['insee_gridded_data_relevant_fields'])
        }
    }
    print('2. Transformation des données terminée.\n')

    print('3. Optimisation des modèles envisagés en cours...')
    best_models = find_best_models(data=clean_data, scorer=f1_score, cv=10)
    print('3. Optimisation des modèles envisagés terminée.\n')

    print('4. Evaluation des modèles optimaux en cours...')
    best_model_urbain_survie_3_ans = find_best_model([best_models['fu3'], best_models['nu3']], cv=5) # cv à revoir. ***
    best_model_rural_survie_3_ans = find_best_model([best_models['fr3'], best_models['nr3']], cv=5)
    # best_model_urbain_survie_5_ans = find_best_model([best_models['fu5'], best_models['nu5']], cv=5)
    # best_model_rural_survie_5_ans = find_best_model([best_models['fr5'], best_models['nr5']], cv=5)
    print('4. Evaluation des modèles optimaux terminée.\n')

    print('5. Entraînement des meilleurs modèles en cours...')
    trained_model_urbain_survie_3_ans = train_model(best_model_urbain_survie_3_ans) 
    trained_model_rural_survie_3_ans = train_model(best_model_rural_survie_3_ans) 
    # trained_model_urbain_survie_5_ans = train_model(best_model_urbain_survie_5_ans) 
    # trained_model_rural_survie_5_ans = train_model(best_model_rural_survie_5_ans) 
    print('5. Entraînement des meilleurs modèles terminé.')


main(label_metier)
