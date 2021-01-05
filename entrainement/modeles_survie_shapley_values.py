import pickle
import sys
import shap
from matplotlib import pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
import os
import numpy



# Argument.

# id_model = sys.argv[1]
metier = sys.argv[1]
id_model_survie_3_ans = sys.argv[2]
id_model_survie_5_ans = sys.argv[3]



# Paramètres.

nb_features_max = 15
nb_samples_max = 1000



# MAIN.


def main(id_model, destination_folder):


    print("Version de numpy :", numpy.__version__) # ***
    print("Version de shap :", shap.__version__) # ***


    # Chargement des fichiers utiles.

    nom_fichier = '{}/{}_ans/{}_trained_model'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    trained_model = pickle.load(fichier)
    fichier.close()
    nom_fichier = '{}/{}_ans/{}_X'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    X = pickle.load(fichier)
    fichier.close()
    nom_fichier = '{}/{}_ans/{}_dict_ordered_feature_importances'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    dict_ordered_feature_importances = pickle.load(fichier)
    fichier.close()
    nom_fichier = '{}/{}_ans/{}_y'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    y = pickle.load(fichier)
    fichier.close()
    nom_fichier = '{}/{}_ans/{}_parameters'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    parameters = pickle.load(fichier)
    fichier.close()
    nom_fichier = '{}/{}_ans/{}_liste_variables_selectionnees'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'rb')
    liste_variables_selectionnees = pickle.load(fichier)
    fichier.close()


    # Entraînement d'un modèle simplifié.

    print('Entraînement du modèle simplifié en cours...')
    new_model = RandomForestClassifier(**parameters)
    selected_features = list(dict_ordered_feature_importances.keys())[:min(len(liste_variables_selectionnees), nb_features_max)]
    new_X = X[selected_features]
    new_model.fit(new_X, y)
    print('Entraînement du modèle simplifié terminé.\n')


    # Calcul des Shapley values.

    print('Calcul du poids des variables en cours...\n')
    new_X = new_X.sample(n=min(new_X.shape[0], nb_samples_max))
    print('new_X.shape:', new_X.shape) # ***
    shap_values = shap.TreeExplainer(new_model).shap_values(new_X)
    if len(shap_values) == 2: shap_values = shap_values[1]
    print('Calcul du poids des variables terminé.\n')

    nom_fichier = '{}/{}_ans/{}_shap_values'.format(metier, destination_folder, metier)
    nom_fichier_bis = '{}/{}_ans/{}_shap_values'.format(metier, destination_folder, id_model)
    fichier = open(nom_fichier, 'wb') 
    fichier_bis = open(nom_fichier_bis, 'wb') 
    pickle.dump(shap_values, fichier)
    pickle.dump(shap_values, fichier_bis)


    # Enregistrement des graphes

    f = plt.figure()
    shap.summary_plot(shap_values, new_X, plot_type="bar")
    nom_fichier = '{}/{}_ans/{}_shap_values_summary_plot_1.png'.format(metier, destination_folder, metier)
    nom_fichier_bis = '{}/{}_ans/{}_shap_values_summary_plot_1.png'.format(metier, destination_folder, id_model)
    f.savefig(nom_fichier, bbox_inches='tight', dpi=600)
    f.savefig(nom_fichier_bis, bbox_inches='tight', dpi=600)
    f = plt.figure()
    shap.summary_plot(shap_values, new_X)
    nom_fichier = '{}/{}_ans/{}_shap_values_summary_plot_2.png'.format(metier, destination_folder, metier)
    nom_fichier_bis = '{}/{}_ans/{}_shap_values_summary_plot_2.png'.format(metier, destination_folder, id_model)
    f.savefig(nom_fichier, bbox_inches='tight', dpi=600)
    f.savefig(nom_fichier_bis, bbox_inches='tight', dpi=600)

    # path = "{}_shap_values_dependence_plots".format(id_model)
    # os.mkdir(path)
    # feature_importances = dict(zip(list(new_X.columns), list(new_model.feature_importances_)))
    # for x in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
    #     f = plt.figure()
    #     shap.dependence_plot(x[0], shap_values, new_X, show=False)
    #     dependence_plot_file_name = "{}_shap_values_dependence_plots/{}_{}.png".format(id_model, x[0], round(x[1], 4))
    #     plt.savefig(dependence_plot_file_name)

    os.system('gsutil cp {}/{}_ans/{}_shap* gs://heka-dev-crmana-storage/modeles/{}/{}/'.format(metier, destination_folder, metier, metier, destination_folder))
    os.system('gsutil cp {}/{}_ans/{}_shap* gs://heka-dev-crmana-storage/modeles/{}/{}/'.format(metier, destination_folder, id_model, metier, destination_folder))



print('Calcul des Shapley values du modèle de survie à 3 ans en cours...')
main(id_model_survie_3_ans, 'survie_3')
print('Calcul des Shapley values du modèle de survie à 3 ans terminé.\n')
print('Calcul des Shapley values du modèle de survie à 5 ans en cours...')
main(id_model_survie_5_ans, 'survie_5')
print('Calcul des Shapley values du modèle de survie à 5 ans terminé.\n')