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
import sys
import yaml



# Argument.

label_metier = sys.argv[1] # Exemple : 'boulangerie'.



# Paramètres.

def get_project_config():
    return yaml.safe_load(os.getenv("PROJECT_CONFIG"))
db_logins = get_project_config()["project-database"]

# # Identifiants de connexion à la base de données. 
# # A NE PAS PUBLIER. ***
# db_logins = {}



# Fonctions utiles.

def sauver_pickle(data, nom_data, label_metier, id_model):
    nom_fichier = '{}/opportunite/{}_{}'.format(label_metier, label_metier, nom_data)
    nom_fichier_bis = '{}/opportunite/{}_{}'.format(label_metier, id_model, nom_data)
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
    , b.nb_etablissements_actifs_{} as nb_etablissements
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
        'n_estimators': [200] # ***
        #, 'criterion': ['gini', 'entropy']
        , 'max_depth': [2, 5, 10] # ***
        # , 'min_samples_split': [2, 3, 5]
        # , 'min_samples_leaf': [1, 2, 5]
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
    best_rf_model = RandomForestRegressor(**grid_search.best_params_, n_jobs=-1)

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


def filter_most_signifiant_features(model, X, y):
    
    trained_model = model['parameterized_model'].fit(X, y)

    # Classement des variables explicatives par ordre d'importance.
    dict_feature_importances = {}
    for i in range(X.shape[1]):
        dict_feature_importances[list(X.columns)[i]] = list(trained_model.feature_importances_)[i]
    dict_ordered_feature_importances = {k: v for k, v in sorted(dict_feature_importances.items(), key=lambda item: item[1], reverse=True)}
    list_ordered_importances = list(dict_ordered_feature_importances.values())
    list_ordered_features = list(dict_ordered_feature_importances.keys())

    # # Tests de sélection de variables.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # list_nb_features = sorted(list(set(list(range(5, X.shape[1], 5)) + [X.shape[1]])))
    # tmp_best_score = 0
    # liste_chiffres = []
    # for nb_features in list_nb_features:
    #     threshold = list_ordered_importances[nb_features-1]
    #     selector = SelectFromModel(trained_model, threshold=threshold, prefit=True)
    #     X_train_transformed = pd.DataFrame(selector.transform(X_train), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])
    #     new_model = RandomForestClassifier(**trained_model.get_params())
    #     new_model.fit(X_train_transformed, y_train)
    #     X_test_transformed = pd.DataFrame(selector.transform(X_test), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])
    #     y_pred = new_model.predict(X_test_transformed)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     cohen_kappa = cohen_kappa_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #     chiffres = 'En se limitant aux {} variables les plus discriminantes, on obtient les scores suivants : accuracy = {} ; cohen_kappa = {}, f1 = {}.'.format(nb_features, accuracy, cohen_kappa, f1)
    #     print(chiffres)
    #     liste_chiffres.append(chiffres)
    #     if f1 >= tmp_best_score:
    #         tmp_best_score = f1
    #         X_transformed = pd.DataFrame(selector.transform(X), columns=[list(X.columns)[i] for i in list(selector.get_support(indices=True))])

    # return X_transformed, liste_chiffres, dict_ordered_feature_importances
    return dict_ordered_feature_importances


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

    dict_ordered_feature_importances = filter_most_signifiant_features(best_rf_model, X, y)

    # return {
    #     '': {'dataset_name': '', 'X': X, 'y': y, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model}, {'model_name': 'gb', 'optimized_model': best_gb_model}]},
    # }
    return {
        '': {'dataset_name': '', 'X': X, 'y': y, 'optimized_models': [{'model_name': 'rf', 'optimized_model': best_rf_model}], 'dict_ordered_feature_importances': dict_ordered_feature_importances},
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
        insert into entrainements_modeles_nombre_etablissements(fk_metier, date, fk_donnees_entrainement, fk_modele, parametres, mean_squared_error_cv, mean_absolute_error_cv, mean_squared_error, mean_absolute_error, dict_ordered_feature_importances)
        values ('{}', now(), '{}', '{}', '{}', {}, {}, {}, {}, '{}');
    """.format(
        label_metier, to_be_saved['dataset_name'], to_be_saved['model_name'], json.dumps(to_be_saved['parameters']), 
        to_be_saved['mean_squared_error_cv'], to_be_saved['mean_absolute_error_cv'], to_be_saved['mean_squared_error'], to_be_saved['mean_absolute_error'],
        json.dumps(to_be_saved['dict_ordered_feature_importances'])
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
            'dict_ordered_feature_importances': best_models['dict_ordered_feature_importances']
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
            db_logins['hostname'], 
            db_logins['port'], 
            db_logins['name']
        )
    )
    df_y_pred.to_sql('tmp_{}'.format(label_metier), engine, if_exists='replace')

    query = """
        alter table predictions_nombre_etablissements add column if not exists {} double precision;
        alter table predictions_nombre_etablissements add column if not exists {}_cv double precision;

        update predictions_nombre_etablissements as a 
        set {} = b.nb_etablissements_predit
        , {}_cv = b.nb_etablissements_predit_cv
        from tmp_{} b
        where a.code_iris = b.code_iris;

        drop table tmp_{};

        delete from res_carreaux_scores where score = 'opportunite' and fk_metier = '{}';

        with ecarts_relatifs as (
            with ecarts_relatifs as (
                with ecarts_relatifs as (
                    select a.code_iris
                    , case 
                    when b.nb_etablissements_actifs_{} = 0 
                    then a.{}
                    else (a.{} - b.nb_etablissements_actifs_{})/b.nb_etablissements_actifs_{}
                    end as ecart_relatif
                    from predictions_nombre_etablissements a
                    inner join nb_etablissements_iris b on a.code_iris = b.code_iris
                )
                select a.code_iris
                , a.ecart_relatif
                , min(a.ecart_relatif) over (partition by 1)
                , avg(a.ecart_relatif) over (partition by 1)
                , max(a.ecart_relatif) over (partition by 1)
                from ecarts_relatifs a
            )
            select a.code_iris
            , a.ecart_relatif
            , avg(a.min) over (partition by 1) as min
            , avg(a.ecart_relatif) filter (where a.ecart_relatif >= a.avg) over (partition by 1) as avg_first_half
            , avg(a.avg) over (partition by 1) as avg
            , avg(a.ecart_relatif) filter (where a.ecart_relatif <= a.avg) over (partition by 1) as avg_last_half
            , avg(a.max) over (partition by 1) as max
            from ecarts_relatifs a
        )   

        insert into res_carreaux_scores(idinspire, score, valeur, fk_metier, date_prediction, reel)
        select a.idinspire
        , 'opportunite'
        , case 
            when c.ecart_relatif between c.min and c.avg_last_half then round((c.ecart_relatif * 0.25/(c.avg_last_half - c.min) - 0.25/(c.avg_last_half - c.min) * c.min)::numeric, 2)
            when c.ecart_relatif between c.avg_last_half and c.avg then round((c.ecart_relatif * 0.25/(c.avg - c.avg_last_half) - 0.25/(c.avg - c.avg_last_half) * c.avg_last_half + 0.25)::numeric, 2)
            when c.ecart_relatif between c.avg and c.avg_first_half then round((c.ecart_relatif * 0.25/(c.avg_first_half - c.avg) - 0.25/(c.avg_first_half - c.avg) * c.avg + 0.5)::numeric, 2)
            when c.ecart_relatif between c.avg_first_half and c.max then round((c.ecart_relatif * 0.25/(c.max - c.avg_first_half) - 0.25/(c.max - c.avg_first_half) * c.avg_first_half + 0.75)::numeric, 2)
            else 1
            end
        , '{}'
        , now()
        , true
        from carreaux_iris a
        inner join mailles_geographiques b on a.code_iris = b.geo_value and b.fk_geo_1 = 51010
        inner join ecarts_relatifs c on a.code_iris = c.code_iris;
    """.format(label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier)

    with engine.begin() as con:
        con.execute(query)
    print('Sauvegarde des prédictions terminée.')

    return trained_model, X, y



# 6. Construction du graphe de distribution des écarts.


def construire_graphes_ecarts(label_metier, id_model):

    query = """
        select a.code_iris
        , a.nb_etablissements_actifs_{} as nb_effectif
        , round(b.{}::numeric, 3) as nb_predit
        , round(abs(b.{} - a.nb_etablissements_actifs_{})::numeric, 1) as ecart_absolu
        , round(((b.{} - a.nb_etablissements_actifs_{})^2)::numeric, 1) as ecart_quadratique
        , case 
            when a.nb_etablissements_actifs_{} != 0 
            then round(((b.{} - a.nb_etablissements_actifs_{})/a.nb_etablissements_actifs_{})::numeric, 1) 
            else round(b.{}::numeric, 1) 
            end as ecart_relatif
        , case when c.fk_geo_1 = 51010 then true else false end as na
        from nb_etablissements_iris a 
        inner join predictions_nombre_etablissements b on a.code_iris = b.code_iris and b.{} is not null
        inner join mailles_geographiques c on a.code_iris = c.geo_value
        inner join iris_communes d on a.code_iris = d.code_iris
        order by a.nb_etablissements_actifs_{} desc
    """.format(
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
    )
    cur.execute(query)
    df = pd.DataFrame([dict(x) for x in cur.fetchall()])
    fig = px.histogram(df, x="ecart_relatif")
    fig.write_image("{}/opportunite/{}_graphe_distribution_ecarts.png".format(label_metier, label_metier))
    fig.write_image("{}/opportunite/{}_graphe_distribution_ecarts.png".format(label_metier, id_model))

    query = """
        select a.code_iris
        , a.nb_etablissements_actifs_{} as nb_effectif
        , round(b.{}_cv::numeric, 3) as nb_predit
        , round(abs(b.{}_cv - a.nb_etablissements_actifs_{})::numeric, 1) as ecart_absolu
        , round(((b.{}_cv - a.nb_etablissements_actifs_{})^2)::numeric, 1) as ecart_quadratique
        , case 
            when a.nb_etablissements_actifs_{} != 0 
            then round(((b.{}_cv - a.nb_etablissements_actifs_{})/a.nb_etablissements_actifs_{})::numeric, 1) 
            else round(b.{}_cv::numeric, 1) 
            end as ecart_relatif
        , case when c.fk_geo_1 = 51010 then true else false end as na
        from nb_etablissements_iris a 
        inner join predictions_nombre_etablissements b on a.code_iris = b.code_iris and b.{} is not null
        inner join mailles_geographiques c on a.code_iris = c.geo_value
        inner join iris_communes d on a.code_iris = d.code_iris
        order by a.nb_etablissements_actifs_{} desc
    """.format(
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
        label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
    )
    cur.execute(query)
    df = pd.DataFrame([dict(x) for x in cur.fetchall()])
    fig = px.histogram(df, x="ecart_relatif")
    fig.write_image("{}/opportunite/{}_graphe_distribution_ecarts_cv.png".format(label_metier, label_metier))
    fig.write_image("{}/opportunite/{}_graphe_distribution_ecarts_cv.png".format(label_metier, id_model))
    


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
    print('5. Entraînement des meilleurs modèles terminé.\n')

    print('6. Construction du graphe de la distribution des écarts en cours...')
    construire_graphes_ecarts(label_metier, best_model['id_model'])
    print('6. Construction du graphe de la distribution des écarts terminée.\n')

    sauver_pickle(data, 'data', label_metier, best_model['id_model'])
    sauver_pickle(data_cleaned, 'data_cleaned', label_metier, best_model['id_model'])
    sauver_pickle(X, 'X', label_metier, best_model['id_model'])
    sauver_pickle(y, 'y', label_metier, best_model['id_model'])
    sauver_pickle(trained_model, 'trained_model', label_metier, best_model['id_model'])
    sauver_pickle(best_model['parameters'], 'parameters', label_metier, best_model['id_model'])
    sauver_pickle(best_models['']['dict_ordered_feature_importances'], 'dict_ordered_feature_importances', label_metier, best_model['id_model'])


main(label_metier)