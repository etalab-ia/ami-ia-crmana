# Imports.

import sys
import os
import psycopg2
import psycopg2.extras as extras
import yaml
import json
import subprocess
import numpy


# Arguments.

id_metier = sys.argv[1] # Exemple : id_metier = 'boulangerie'


# Connexion à la base de données.


def get_project_config():
    return yaml.safe_load(os.getenv("PROJECT_CONFIG"))


db_logins = get_project_config()["project-database"]

# # Identifiants de connexion à la base de données.
# # A NE PAS PUBLIER. ***
# db_logins = {}

con = psycopg2.connect(
    host=db_logins["hostname"],
    database=db_logins["name"],
    user=db_logins["user"],
    password=db_logins["password"],
)
cur = con.cursor(cursor_factory=extras.DictCursor)


# MAIN.

print("Version de numpy :", numpy.__version__) # ***

provider_credentials = json.loads(
    os.environ.get("PROVIDER_CREDENTIALS") or "null"
)
with open("provider_credentials.json", "w") as f:
    json.dump(provider_credentials, f)
gcloud_auth = subprocess.Popen(
    [
        "gcloud",
        "auth",
        "activate-service-account",
        "--key-file=provider_credentials.json",
    ],
)
gcloud_auth.wait()

# os.system("printf '%s' '$PROVIDER_CREDENTIALS' > provider_credentials.json")
# os.system("cat provider_credentials.json")
# os.system("gcloud auth activate-service-account --key-file=provider_credentials.json")

os.system("rm -r {}".format(id_metier))
os.system("mkdir {}".format(id_metier))

print("\nEntraînement du modèle de survie à 3 ans en cours...\n")
os.system("mkdir {}/survie_3_ans".format(id_metier))
os.system("python3 modele_survie_3_ans_entrainement.py {}".format(id_metier))
os.system("gsutil cp {}/survie_3_ans/* gs://heka-dev-crmana-storage/modeles/{}/survie_3/".format(id_metier, id_metier))
print("\nEntraînement du modèle de survie à 3 ans terminé.\n\n")

cur.execute("select a.id from entrainements_modeles a where a.fk_metier = '{}' order by a.id desc limit 1".format(id_metier))
id_model_survie_3_ans = cur.fetchone()["id"]

print("\nEntraînement du modèle de survie à 5 ans en cours...\n")
os.system("mkdir {}/survie_5_ans".format(id_metier))
os.system("python3 modele_survie_5_ans_entrainement.py {} {}".format(id_metier, id_model_survie_3_ans))
os.system("gsutil cp {}/survie_5_ans/* gs://heka-dev-crmana-storage/modeles/{}/survie_5/".format(id_metier, id_metier))
print("\nEntraînement du modèle de survie à 5 ans terminé.\n\n")

cur.execute("select a.id from entrainements_modeles a where a.fk_metier = '{}' order by a.id desc limit 1".format(id_metier))
id_model_survie_5_ans = cur.fetchone()["id"]

print("\nCalcul des Shapley values des modèles de survie en cours...\n")
os.system("python3 modeles_survie_shapley_values.py {} {} {}".format(id_metier, id_model_survie_3_ans, id_model_survie_5_ans))
print("\nCalcul des Shapley values des modèles de survie terminé.\n\n")

print("\nEntraînement du modèle d'opportunité d'implantation en cours...\n")
os.system("mkdir {}/opportunite".format(id_metier))
os.system("python3 modele_opportunite_entrainement.py {}".format(id_metier))
os.system("gsutil cp {}/opportunite/* gs://heka-dev-crmana-storage/modeles/{}/opportunite/".format(id_metier, id_metier))
print("\nEntraînement du modèle d'opportunité d'implantation terminé.\n\n")

cur.execute("select a.id from entrainements_modeles_nombre_etablissements a where a.fk_metier = '{}' order by a.id desc limit 1".format(id_metier))
id_model_opportunite = cur.fetchone()["id"]

print("\nCalcul des Shapley values du modèle d'opportunité en cours...\n")
os.system("python3 modele_opportunite_shapley_values.py {} {}".format(id_metier, id_model_opportunite))
print("\nCalcul des Shapley values du modèle d'opportunité terminé.\n\n")

os.system("rm -r {}".format(id_metier))
