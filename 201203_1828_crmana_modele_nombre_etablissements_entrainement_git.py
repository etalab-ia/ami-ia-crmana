# Référent : Jean VAN HECKE.
# Entamé le 20/10/20.
# Finalisé le 10/11/20.



# Imports.

import psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import json
import pandas as pd
# import shap
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import psutil
import kaleido
import plotly.express as px



# Arguments. # A généraliser. ***

label_metier = 'boulangerie' # ***



# Paramètres.

# Identifiants de connexion à la base de données. 
# A NE PAS PUBLIER. ***
db_logins = {}



# Fonctions utiles.

def sauver_pickle(data, nom_data, label_metier, id_model):
    nom_fichier = '{}/{}_{}'.format(label_metier, id_model, nom_data)
    fichier = open(nom_fichier, 'wb') 
    pickle.dump(data, fichier)



# 1. Récupération des données.


# Connexion à la base de données.

con = psycopg2.connect(
    host = db_logins['host'], 
    database = db_logins['database'], 
    user = db_logins['user'], 
    password = db_logins['password']
)
cur = con.cursor(cursor_factory = extras.DictCursor)


def get_data(metier):
    
    query = """
    select a.code_iris, d.type_com as milieu, d.statut_2017 as zone
    , a.population, a.age_moyen, a.taux_pauvrete, a.taux_menages_1_individu
    , a.taux_menages_plus_5_individus, a.taux_menages_proprietaires, a.taux_menages_monoparentaux
    , a.niveau_vie, a.surface_logement, a.taux_menages_collectifs, a.taux_maisons
    , a.taux_logements_avant_1945, a.taux_logements_1945_1970, a.taux_logements_1970_1990
    , a.taux_logements_apres_1990, a.taux_logements_sociaux, a.taux_population_moins_3_ans
    , a.taux_population_4_5_ans, a.taux_population_6_10_ans, a.taux_population_11_17_ans
    , a.taux_population_18_24_ans, a.taux_population_25_39_ans, a.taux_population_40_54_ans
    , a.taux_population_55_64_ans, a.taux_population_65_79_ans, a.taux_population_plus_80_ans
    , a.nb_equipements_commerces, a.nb_equipements_loisirs, a.nb_equipements_transports
    , a.nb_equipements_enseignement, a.nb_equipements_sante, a.nb_equipements_services
    , a.indice_artisanat_proximite, a.indice_artisanat_economie_creative
    , a.indice_artisanat_construction, a.indice_artisanat_soutien
    , a.indice_naf_niveau_1_{}
    , a.indice_densite_emplois, a.taux_residences_secondaires, a.nb_transactions_immobilieres
    , a.taux_motorisation, a.nb_chambres_hotels
    , a.taux_agriculteurs_exploitants
    , a.taux_artisans_commercants_entrepreneurs
    , a.taux_cadres_professions_intellectuelles_superieures
    , a.taux_professions_intermediaires
    , a.taux_employes
    , a.taux_ouvriers       
    , a.taux_retraites
    , a.taux_autres_sans_activite_professionnelle
    , b.nb_etablissements_{} as nb_etablissements
    from donnees_agregees_iris a
    inner join nb_etablissements_iris b on a.code_iris = b.code_iris
    inner join iris_communes c on a.code_iris = c.code_iris
    left join unites_urbaines d on c.code_commune = d.codgeo
    """.format(
        label_metier, label_metier
    )
    cur.execute(query)
    df = pd.DataFrame([dict(x) for x in cur.fetchall()])

    df['milieu'] = df['milieu'].map(lambda x: (1 if x == 'URBAIN' else 0))
    def numerisation_zone(x):
        if x == 'C': y = 4
        elif x == 'B': y = 3
        elif x == 'I': y = 2
        elif x == 'R': y = 1
        else: y = None
        return y
    df['zone'] = df['zone'].map(lambda x: numerisation_zone(x))
    
    return df



# 2. Traitement des données.


def clean_data(data):

    data_cleaned = data.fillna(data.mean())
    # [
    #     data.population.notnull() &
    #     data.indice_densite_emplois.notnull() &
    #     data.taux_residences_secondaires.notnull()
    # ]

    return data_cleaned



# 3. Paramétrage des modèles.


def find_best_rf_model(X, y, scorer, cv, dataset_name):

    parameters = {
        'n_estimators': [100, 200, 500] # ***
        #, 'criterion': ['gini', 'entropy']
        , 'max_depth': [2, 5, 10, 20, 50, 100] # ***
        #, 'min_samples_split': [2]
        #, 'min_samples_leaf': [1]
        #, 'max_features': ['auto', 'sqrt']
    }

    scorer = make_scorer(scorer) 

    grid_search = GridSearchCV(RandomForestRegressor(), parameters, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(X, y)
    #print(grid_search.best_params_)
    #print(grid_search.best_score_)

    # Modèle entraîné résultant du GridSearch.
    #model = grid_search.best_estimator_
    # Modèle neuf paramétré optimalement.
    best_rf_model = RandomForestRegressor(**grid_search.best_params_)

    return {'parameterized_model': best_rf_model, 'parameters': grid_search.best_params_}


def find_best_gb_model(X, y, scorer, cv, dataset_name):

    parameters = {
        'n_estimators': [100, 200] #, 200, 500, 1000, 2000] # ***
        , 'learning_rate': [0.01, 0.1] # ***
        #, 'max_features': []
        , 'max_depth': [3, 4, 5] # ***
    }  

    scorer = make_scorer(scorer) 

    grid_search = GridSearchCV(GradientBoostingRegressor(), parameters, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(X, y)
    #print(grid_search.best_params_)
    #print(grid_search.best_score_)

    # Modèle entraîné résultant du GridSearch.
    #model = grid_search.best_estimator_
    # Modèle neuf paramétré optimalement.
    best_gb_model = GradientBoostingRegressor(**grid_search.best_params_)

    return {'parameterized_model': best_gb_model, 'parameters': grid_search.best_params_}


def find_best_models(data, scorer, cv):

    training_data = data.drop(
        ['code_iris'], axis=1
    )
    X = training_data.drop(['nb_etablissements'], axis=1)
    y = training_data['nb_etablissements']
    print('Optimisation du Random Forest...')
    best_rf_model = find_best_rf_model(X, y, scorer=scorer, cv=cv, dataset_name='')
    # print('Optimisation du Gradient Boosting...')
    # best_gb_model = find_best_gb_model(X, y, scorer=scorer, cv=cv, dataset_name='')

    # return {
    #     '': {'dataset_name': '', 'X': X, 'y': y, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model}, {'model_name': 'gb', 'optimized_model': best_gb_model}]},
    # }
    return {
        '': {'dataset_name': '', 'X': X, 'y': y, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model}]},
    }



# 4. Détermination des meilleurs modèles.


def evaluate_model(X, y, model, cv):

    # # MAPE non implémenté dans la dernière version stable de sklearn.
    # from sklearn.utils import check_array
    # def mean_absolute_percentage_error(y_true, y_pred): 
    #     y_true, y_pred = check_arrays(y_true, y_pred)
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Evaluation par Cross Validation.
    y_pred = cross_val_predict(model, X, y, cv=cv)
    mse_cv = mean_squared_error(y, y_pred)
    mae_cv = mean_absolute_error(y, y_pred)
    # mape_cv = mean_absolute_percentage_error(y, y_pred)

    # Evaluation du modèle entraîné sur l'ensemble des données.
    trained_model = model.fit(X, y)
    y_pred = trained_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    # mape = mean_absolute_percentage_error(y, y_pred)
    
    return {
        'mean_squared_error_cv': mse_cv,
        'mean_absolute_error_cv': mae_cv,        
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        # 'mean_absolute_percentage_error': mape
    }


def save_in_database(to_be_saved):

    # Sauvegarde des données utiles dans la base de données. ***
    query = """
        insert into entrainements_modeles_nombre_etablissements(fk_metier, date, fk_donnees_entrainement, fk_modele, parametres, mean_squared_error_cv, mean_absolute_error_cv, mean_squared_error, mean_absolute_error)
        values ('{}', now(), '{}', '{}', '{}', {}, {}, {}, {});
    """.format(
        label_metier, to_be_saved['dataset_name'], to_be_saved['model_name'], json.dumps(to_be_saved['parameters']), 
        to_be_saved['mean_squared_error_cv'], to_be_saved['mean_absolute_error_cv'], to_be_saved['mean_squared_error'], to_be_saved['mean_absolute_error']
    )
    cur.execute(query)
    con.commit()

    query = "select max(a.id) from entrainements_modeles_nombre_etablissements a;"
    cur.execute(query)
    id_model = cur.fetchall()[0][0]
    
    # # Export des données utiles sous format json. # Provisoire. ***
    # with open('{}_{}'.format(to_be_saved['dataset_name'], to_be_saved['model_name']), 'w') as f:
    #     json.dump(json.dumps(to_be_saved), f)

    return id_model


def find_best_model(best_models, cv):

    # Evaluation des modèles et sauvegarde des résultats.
    saved_models = {}
    for model in best_models['optimized_models']: # ***
        print('Evaluation du modèle {}...'.format(model['model_name']))
        model_evaluation = evaluate_model(X=best_models['X'], y=best_models['y'], model=model['optimized_model']['parameterized_model'], cv=cv)
        to_be_saved = {
            'dataset_name': best_models['dataset_name'],
            'model_name': model['model_name'],
            'parameters': model['optimized_model']['parameters'],
            'mean_squared_error_cv': model_evaluation['mean_squared_error_cv'],
            'mean_absolute_error_cv': model_evaluation['mean_absolute_error_cv'],            
            'mean_squared_error': model_evaluation['mean_squared_error'],
            'mean_absolute_error': model_evaluation['mean_absolute_error'],
            # 'mean_absolute_percentage_error': model_evaluation['mean_absolute_percentage_error']
        }
        id_model = save_in_database(to_be_saved)
        saved_models[id_model] = {
            'id_model': int(id_model),
            # 'dataset_name': str(best_models['dataset_name']),
            # 'model_name': str(model['model_name']),                
            # 'X': best_models['X'],
            # 'y': best_models['y'],
            'parameterized_model': model['optimized_model']['parameterized_model'],
            'parameters': model['optimized_model']['parameters']
        }

    # Récupération du meilleur modèle.
    query = """
        select a.id
        from entrainements_modeles_nombre_etablissements a 
        inner join (
            select min(a.mean_squared_error) as error_min
            from entrainements_modeles_nombre_etablissements a
            where a.id in ({})
        ) b on a.mean_squared_error = b.error_min
        order by a.id desc
		limit 1;
    """.format(', '.join(list(map(lambda x: str(x), list(saved_models.keys())))))
    cur.execute(query)
    id_model = cur.fetchall()[0][0]

    query = """
        update entrainements_modeles_nombre_etablissements as a
        set actif = false;
        update entrainements_modeles_nombre_etablissements as a
        set actif = true
        where a.id = {};
    """.format(id_model)

    return saved_models[id_model]



# 5. Entraînement des meilleurs modèles.


def fit_and_predict(best_model, data_cleaned):

    training_data = data_cleaned.drop(
        ['code_iris'], axis=1
    )
    model = best_model['parameterized_model']
    X = training_data.drop(['nb_etablissements'], axis=1)
    y = training_data['nb_etablissements']
    cv = 10

    print('Entraînement du modèle en cours...')
    trained_model = model.fit(X, y)
    print('Entraînement du modèle terminé.')

    # print('Calcul du poids des variables en cours...')
    # shap_values = shap.TreeExplainer(trained_model).shap_values(X)
    # if len(shap_values) == 2: shap_values = shap_values[1]
    # print('Calcul du poids des variables terminé.')

    # shap_values_file_name = '{}/{}_{}_{}_shap_values'.format(label_metier, id_model, dataset_name, model_name)
    # shap_values_file = open(shap_values_file_name, 'wb') 
    # pickle.dump(shap_values, shap_values_file)

    # f = plt.figure()
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # summary_plot_file_name = '{}/{}_{}_{}_shap_values_summary_plot_1.png'.format(label_metier, id_model, dataset_name, model_name)
    # f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)
    # f = plt.figure()
    # shap.summary_plot(shap_values, X)
    # summary_plot_file_name = '{}/{}_{}_{}_shap_values_summary_plot_2.png'.format(label_metier, id_model, dataset_name, model_name)
    # f.savefig(summary_plot_file_name, bbox_inches='tight', dpi=600)

    # path = "{}/{}_{}_{}_shap_values_dependence_plots".format(label_metier, id_model, dataset_name, model_name)
    # os.mkdir(path)
    # feature_importances = dict(zip(list(X.columns), list(trained_model.feature_importances_)))
    # for x in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
    #     f = plt.figure()
    #     shap.dependence_plot(x[0], shap_values, X, show=False)
    #     dependence_plot_file_name = "{}/{}_{}_{}_shap_values_dependence_plots/{}_{}.png".format(label_metier, id_model, dataset_name, model_name, x[0], round(x[1], 4))
    #     plt.savefig(dependence_plot_file_name)

    print('Prédictions du modèle entraîné sur toutes les données en cours...')
    y_pred = trained_model.predict(X)
    print('Prédictions du modèle entraîné sur toutes les données terminées.')

    print('Prédictions du modèle entraîné par validation croisée en cours...')
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    print('Prédictions du modèle entraîné par validation croisée terminées.')

    print('Sauvegarde des prédictions en cours...')

    df_y_pred = pd.concat([
        data_cleaned['code_iris'].reset_index(drop=True), 
        pd.DataFrame(y_pred, columns=['nb_etablissements_predit']).reset_index(drop=True),
        pd.DataFrame(y_pred_cv, columns=['nb_etablissements_predit_cv']).reset_index(drop=True)
    ], axis=1)

    engine = create_engine(
        'postgresql://{}:{}@{}:{}/{}'.format(
            db_logins['user'], 
            db_logins['password'], 
            db_logins['host'], 
            db_logins['port'], 
            db_logins['database']
        )
    )
    df_y_pred.to_sql('tmp', engine, if_exists='replace')

    query = """
        alter table predictions_nombre_etablissements add column if not exists {} double precision;
        alter table predictions_nombre_etablissements add column if not exists {}_cv double precision;

        update predictions_nombre_etablissements as a 
        set {} = b.nb_etablissements_predit
        , {}_cv = b.nb_etablissements_predit_cv
        from tmp b
        where a.code_iris = b.code_iris;

        drop table tmp;

        /*
        delete from res_carreaux_scores where score = 'opportunite' and fk_metier = '{}';

        insert into res_carreaux_scores(idinspire, score, valeur, fk_metier, date_prediction, reel)
        select a.idinspire
        , 'opportunite'
        , case 
            when a.valeur < -0.5 then 0
            when a.valeur between -0.5 and 0 then a.valeur + 0.5
            when a.valeur between 0 and 1 then a.valeur * 0.3 + 0.5
            when a.valeur between 1 and 2 then a.valeur * 0.2 - 0.6
            else 1
            end
        , '{}'
        , now()
        , true
        from (
            select distinct (a.idinspire)
            , case 
                when c.nb_etablissements_{} != 0 
                then round(((d.{} - c.nb_etablissements_{})/c.nb_etablissements_{})::numeric, 3) 
                else d.{} 
                end as valeur
            from carreaux_iris a
            inner join mailles_geographiques b on a.code_iris = b.geo_value and b.fk_geo_1 = 51010
            inner join nb_etablissements_iris c on a.code_iris = c.code_iris
            inner join predictions_nombre_etablissements d on a.code_iris = d.code_iris
        ) a;
        */
    """.format(label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier)

    with engine.begin() as con:
        con.execute(query)
    print('Sauvegarde des prédictions terminée.')

    return trained_model, X, y



# 6. Construction du graphe de distribution des écarts.


def construire_graphes_ecarts(label_metier):

    query = """
        select a.code_iris
        , a.nb_etablissements_{} as nb_effectif
        , round(b.{}::numeric, 3) as nb_predit
        , round(abs(b.{} - a.nb_etablissements_{})::numeric, 1) as ecart_absolu
        , round(((b.{} - a.nb_etablissements_{})^2)::numeric, 1) as ecart_quadratique
        , case 
            when a.nb_etablissements_{} != 0 
            then round(((b.{} - a.nb_etablissements_{})/a.nb_etablissements_{})::numeric, 1) 
            else round(b.{}::numeric, 1) 
            end as ecart_relatif
        , case when c.fk_geo_1 = 51010 then true else false end as na
        from nb_etablissements_iris a 
        inner join predictions_nombre_etablissements b on a.code_iris = b.code_iris and b.{} is not null
        inner join mailles_geographiques c on a.code_iris = c.geo_value
        inner join iris_communes d on a.code_iris = d.code_iris
        order by a.nb_etablissements_{} desc
    """.format(
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
    )
    cur.execute(query)
    df = pd.DataFrame([dict(x) for x in cur.fetchall()])
    fig = px.histogram(df, x="ecart_relatif")
    fig.write_image("{}/graphe_distribution_ecarts.png".format(label_metier))

    query = """
        select a.code_iris
        , a.nb_etablissements_{} as nb_effectif
        , round(b.{}_cv::numeric, 3) as nb_predit
        , round(abs(b.{}_cv - a.nb_etablissements_{})::numeric, 1) as ecart_absolu
        , round(((b.{}_cv - a.nb_etablissements_{})^2)::numeric, 1) as ecart_quadratique
        , case 
            when a.nb_etablissements_{} != 0 
            then round(((b.{}_cv - a.nb_etablissements_{})/a.nb_etablissements_{})::numeric, 1) 
            else round(b.{}_cv::numeric, 1) 
            end as ecart_relatif
        , case when c.fk_geo_1 = 51010 then true else false end as na
        from nb_etablissements_iris a 
        inner join predictions_nombre_etablissements b on a.code_iris = b.code_iris and b.{} is not null
        inner join mailles_geographiques c on a.code_iris = c.geo_value
        inner join iris_communes d on a.code_iris = d.code_iris
        order by a.nb_etablissements_{} desc
    """.format(
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
    )
    cur.execute(query)
    df = pd.DataFrame([dict(x) for x in cur.fetchall()])
    fig = px.histogram(df, x="ecart_relatif")
    fig.write_image("{}/graphe_distribution_ecarts_cv.png".format(label_metier))
    


# MAIN.


def main(label_metier):

    print('1. Récupération des données en cours...')
    data = get_data(label_metier)
    print('1. Récupération des données terminée.\n')

    print('2. Traitement des données en cours...')
    data_cleaned = clean_data(data)
    print('2. Traitement des données terminé.\n')

    print('3. Optimisation des modèles envisagés en cours...')
    best_models = find_best_models(data=data_cleaned, scorer=mean_squared_error, cv=5)
    print('3. Optimisation des modèles envisagés terminée.\n')

    print('4. Evaluation des modèles optimaux en cours...')
    best_model = find_best_model(best_models[''], cv=10) # cv à revoir. ***
    print('4. Evaluation des modèles optimaux terminée.\n')

    print('5. Entraînement des meilleurs modèles en cours...')
    trained_model, X, y = fit_and_predict(best_model, data_cleaned)
    print('5. Entraînement des meilleurs modèles terminé.')

    print('6. Construction du graphe de la distribution des écarts en cours...')
    construire_graphes_ecarts(label_metier)
    print('6. Construction du graphe de la distribution des écarts terminée.')

    sauver_pickle(data, 'data', label_metier, best_model['id_model'])
    sauver_pickle(data_cleaned, 'data_cleaned', label_metier, best_model['id_model'])
    sauver_pickle(X, 'X', label_metier, best_model['id_model'])
    sauver_pickle(y, 'y', label_metier, best_model['id_model'])
    sauver_pickle(trained_model, 'trained_model', label_metier, best_model['id_model'])
    sauver_pickle(best_model['parameters'], 'parameters', label_metier, best_model['id_model'])


main(label_metier)