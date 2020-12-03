# Référent : Jean VAN HECKE.
# Entamé le 27/10/20.
# Finalisé le ...



import pickle
import sklearn
import psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import json
import io
import sys
from heka.tools import BucketManager
import yaml
import os



# Connexion à la base de données.

def get_project_config():
    return yaml.safe_load(os.getenv("PROJECT_CONFIG"))
db_logins = get_project_config()["project-database"]

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(
    db_logins['user'], 
    db_logins['password'], 
    db_logins['hostname'], 
    db_logins['port'], 
    db_logins['name']
))
con = psycopg2.connect(
    host = db_logins['hostname'], 
    database = db_logins['name'], 
    user = db_logins['user'], 
    password = db_logins['password']
)
cur = con.cursor(cursor_factory = extras.DictCursor)



# Arguments.

id_projet = sys.argv[1]
caractere_employeur = sys.argv[2]
inscription_registre_metiers = 1
# id_projet = 74 # ***
# caractere_employeur = 1 # ***



# Variables globales.

variables_explicatives = { # A généraliser. ***
    'indice_equipements_commerces': 'ind_equipements_commerce', 
    'indice_equipements_loisirs': 'ind_equipements_loisirs', 
    'indice_equipements_transports': 'ind_equipements_transports', 
    'indice_equipements_enseignement': 'ind_equipements_enseignement', 
    'indice_equipements_sante': 'ind_equipements_sante', 
    'indice_equipements_services': 'ind_equipements_services',
    'indice_artisanat_economie_creative': 'indice_artisanat_economie_creative', 
    'indice_artisanat_construction': 'indice_artisanat_construction', 
    'indice_artisanat_soutien': 'indice_artisanat_soutien',
    'indice_artisanat_proximite': 'indice_artisanat_proximite', 
    'indice_naf_4_urbain': 'indice_densite_naf_niveau_4_urbain', 
    'indice_naf_3_urbain': 'indice_densite_naf_niveau_3_urbain', 
    'indice_naf_2_urbain': 'indice_densite_naf_niveau_2_urbain', 
    'indice_naf_1_urbain': 'indice_densite_naf_niveau_1_urbain',
    'indice_naf_4_rural': 'indice_densite_naf_niveau_4_rural', 
    'indice_naf_3_rural': 'indice_densite_naf_niveau_3_rural', 
    'indice_naf_2_rural': 'indice_densite_naf_niveau_2_rural', 
    'indice_naf_1_rural': 'indice_densite_naf_niveau_1_rural',
    'indice_population': 'densite_pop',
    'indice_concurrence_large_urbain': 'ind_concurrence_large_urbain', 
    'indice_concurrence_restreinte_urbain': 'ind_concurrence_restreinte_urbain',
    'indice_concurrence_large_rural': 'ind_concurrence_large_rural', 
    'indice_concurrence_restreinte_rural': 'ind_concurrence_restreinte_rural',
    'indice_desserte_train': 'ind_desserte_train',
    'indice_frequentation_train': 'ind_frequentation_train',
    'nb_concurrents_debut': 'nb_concurrents',
    'nb_habitants_etablissement_debut': 'nb_habitants_etablissement',
    'taux_etablissements_concurrents_debut': 'taux_concurrents_',
    'indice_densite_emplois': 'densite_emplois',
    'taux_residences_secondaires': 'taux_res_sec',
    'age_moyen': 'age_moyen',
    'taux_pauvrete': 'taux_pauvrete',
    'taux_menages_1_individu': 'taux_menages_1_individu',
    'taux_menages_plus_5_individus': 'taux_menages_plus_5_individus',
    'taux_menages_proprietaires': 'taux_menages_proprietaires',
    'taux_menages_monoparentaux': 'taux_menages_monoparentaux',
    'niveau_vie': 'niveau_vie',
    'surface_logement': 'surface_logement',
    'taux_logements_collectifs': 'taux_logements_collectifs',
    'taux_maisons': 'taux_maisons',
    'taux_logements_avant_1945': 'taux_logements_avant_1945',
    'taux_logements_1945_1970': 'taux_logements_1945_1970',
    'taux_logements_1970_1990': 'taux_logements_1970_1990',
    'taux_logements_apres_1990': 'taux_logements_apres_1990',
    'taux_logements_sociaux': 'taux_logements_sociaux',
    'taux_population_moins_3_ans': 'taux_population_moins_3_ans',
    'taux_population_4_5_ans': 'taux_population_4_5_ans',
    'taux_population_6_10_ans': 'taux_population_6_10_ans',
    'taux_population_11_17_ans': 'taux_population_11_17_ans',
    'taux_population_18_24_ans': 'taux_population_18_24_ans',
    'taux_population_25_39_ans': 'taux_population_25_39_ans',
    'taux_population_40_54_ans': 'taux_population_40_54_ans',
    'taux_population_55_64_ans': 'taux_population_55_64_ans',
    'taux_population_65_79_ans': 'taux_population_65_79_ans',
    'taux_population_plus_80_ans': 'taux_population_plus_80_ans',
    'nb_transactions_immobilieres': 'nb_transactions_immobilieres',
    'indice_reseau': 'indice_reseau',
    'nb_stagiaires': 'nb_stagiaires',
    'taux_motorisation': 'taux_motorisation',
    'taux_agriculteurs_exploitants': 'taux_agriculteurs_exploitants',
    'taux_artisans_commercants_entrepreneurs': 'taux_artisans_commercants_entrepreneurs',
    'taux_cadres_professions_intellectuelles_superieures': 'taux_cadres_professions_intellectuelles_superieures',
    'taux_professions_intermediaires': 'taux_professions_intermediaires',
    'taux_employes': 'taux_employes',
    'taux_ouvriers': 'taux_ouvriers',      
    'taux_retraites': 'taux_retraites',       
    'taux_autres_sans_activite_professionnelle': 'taux_autres_sans_activite_professionnelle',
    'nb_permis_locaux': 'nb_permis_locaux',
    'nb_permis_logements': 'nb_permis_logements',
    'nb_chambres_hotels': 'nb_chambres_hotels',
    'prix_m2': 'prix_m2',
}

variables_explicatives_globales = { # A généraliser. ***
    'indice_equipements_commerces': 'ind_equipements_commerce', 
    'indice_equipements_loisirs': 'ind_equipements_loisirs', 
    'indice_equipements_transports': 'ind_equipements_transports', 
    'indice_equipements_enseignement': 'ind_equipements_enseignement', 
    'indice_equipements_sante': 'ind_equipements_sante', 
    'indice_equipements_services': 'ind_equipements_services',
    'indice_artisanat_economie_creative': 'indice_artisanat_economie_creative', 
    'indice_artisanat_construction': 'indice_artisanat_construction', 
    'indice_artisanat_soutien': 'indice_artisanat_soutien',
    'indice_artisanat_proximite': 'indice_artisanat_proximite', 
    'indice_population': 'densite_pop',
    'indice_desserte_train': 'ind_desserte_train',
    'indice_frequentation_train': 'ind_frequentation_train',
    'indice_densite_emplois': 'densite_emplois',
    'taux_residences_secondaires': 'taux_res_sec',
    'age_moyen': 'age_moyen',
    'taux_pauvrete': 'taux_pauvrete',
    'taux_menages_1_individu': 'taux_menages_1_individu',
    'taux_menages_plus_5_individus': 'taux_menages_plus_5_individus',
    'taux_menages_proprietaires': 'taux_menages_proprietaires',
    'taux_menages_monoparentaux': 'taux_menages_monoparentaux',
    'niveau_vie': 'niveau_vie',
    'surface_logement': 'surface_logement',
    'taux_logements_collectifs': 'taux_logements_collectifs',
    'taux_maisons': 'taux_maisons',
    'taux_logements_avant_1945': 'taux_logements_avant_1945',
    'taux_logements_1945_1970': 'taux_logements_1945_1970',
    'taux_logements_1970_1990': 'taux_logements_1970_1990',
    'taux_logements_apres_1990': 'taux_logements_apres_1990',
    'taux_logements_sociaux': 'taux_logements_sociaux',
    'taux_population_moins_3_ans': 'taux_population_moins_3_ans',
    'taux_population_4_5_ans': 'taux_population_4_5_ans',
    'taux_population_6_10_ans': 'taux_population_6_10_ans',
    'taux_population_11_17_ans': 'taux_population_11_17_ans',
    'taux_population_18_24_ans': 'taux_population_18_24_ans',
    'taux_population_25_39_ans': 'taux_population_25_39_ans',
    'taux_population_40_54_ans': 'taux_population_40_54_ans',
    'taux_population_55_64_ans': 'taux_population_55_64_ans',
    'taux_population_65_79_ans': 'taux_population_65_79_ans',
    'taux_population_plus_80_ans': 'taux_population_plus_80_ans',
    'nb_transactions_immobilieres': 'nb_transactions_immobilieres',
    'indice_reseau': 'indice_reseau',
    'taux_motorisation': 'taux_motorisation',
    'taux_agriculteurs_exploitants': 'taux_agriculteurs_exploitants',
    'taux_artisans_commercants_entrepreneurs': 'taux_artisans_commercants_entrepreneurs',
    'taux_cadres_professions_intellectuelles_superieures': 'taux_cadres_professions_intellectuelles_superieures',
    'taux_professions_intermediaires': 'taux_professions_intermediaires',
    'taux_employes': 'taux_employes',
    'taux_ouvriers': 'taux_ouvriers',      
    'taux_retraites': 'taux_retraites',       
    'taux_autres_sans_activite_professionnelle': 'taux_autres_sans_activite_professionnelle',
    'nb_permis_locaux': 'nb_permis_locaux',
    'nb_permis_logements': 'nb_permis_logements',
    'nb_chambres_hotels': 'nb_chambres_hotels',
    'prix_m2': 'prix_m2'
}

variables_explicatives_metier = { # A généraliser. ***
    'indice_naf_4_urbain': 'indice_densite_naf_niveau_4_urbain', 
    'indice_naf_3_urbain': 'indice_densite_naf_niveau_3_urbain', 
    'indice_naf_2_urbain': 'indice_densite_naf_niveau_2_urbain', 
    'indice_naf_1_urbain': 'indice_densite_naf_niveau_1_urbain',
    'indice_naf_4_rural': 'indice_densite_naf_niveau_4_rural', 
    'indice_naf_3_rural': 'indice_densite_naf_niveau_3_rural', 
    'indice_naf_2_rural': 'indice_densite_naf_niveau_2_rural', 
    'indice_naf_1_rural': 'indice_densite_naf_niveau_1_rural',
    'indice_concurrence_large_urbain': 'ind_concurrence_large_urbain', 
    'indice_concurrence_restreinte_urbain': 'ind_concurrence_restreinte_urbain',
    'indice_concurrence_large_rural': 'ind_concurrence_large_rural', 
    'indice_concurrence_restreinte_rural': 'ind_concurrence_restreinte_rural',
    'nb_concurrents_debut': 'nb_concurrents',
    'nb_habitants_etablissement_debut': 'nb_habitants_etablissement',
    'taux_etablissements_concurrents_debut': 'taux_concurrents_',
    'nb_stagiaires': 'nb_stagiaires'
}

# dictionnaire_entrainement_inference = {
#     'indice_equipements_commerces': 'ind_equipements_commerces'
# }



# Récupération des métadonnées d'un projet.

def get_data_projet(id_projet):
    query = """
        select a.*, b.idinspire, c.milieu, d.statut_2017 as zone
        from (
            select a.longitude, a.latitude, a.activite as label_metier, b.libelle_metier as libelle_activite, a.localisation_type
            , case 
                when a.localisation_type = 'dessin sur une zone' then st_centroid(st_convexhull(st_collect(st_setsrid(st_makepoint(c.longitude, c.latitude), 4326))))
                else st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) 
            end as centre
            from projets a 
            inner join ref_metier b on a.activite = b.id_metier
            left join projets_dessin c on a.id_projet = c.id_projet
            where a.id_projet = {}
            group by a.longitude, a.latitude, a.activite, b.libelle_metier, a.localisation_type
        ) a
        left join insee_carreaux b on st_intersects(a.centre, b.wkb_geometry)
        inner join carreaux_communes c on b.idinspire = c.idinspire
        inner join unites_urbaines d on c.code_commune = d.codgeo
    """.format(id_projet)
    cur.execute(query)
    data_projet = cur.fetchone()
    return data_projet

query_carreaux_alentours = """
WITH perimetre_communes_voisines AS (
       WITH commune_projet AS (
           SELECT geom
           FROM projets p, formes_mailles_geographiques fmg, mailles_geographiques mg
           WHERE fmg.fk_geo = mg.id_geo
           AND mg.geo_level = 3
           AND id_projet = {}
           AND ST_Within(ST_SetSRID(ST_MakePoint(longitude, latitude),4326), geom) IS true
       )
       SELECT fmg.geom
       FROM formes_mailles_geographiques fmg, mailles_geographiques mg, commune_projet ep
       WHERE fmg.fk_geo = mg.id_geo
       AND mg.geo_level = 3
       AND ST_Touches(ep.geom,  fmg.geom)
   ), union_commune_dessin AS (
       SELECT ST_Union(
           
            (   SELECT COALESCE(ST_ConcaveHull(ST_Collect(array_agg(ST_SetSRID(ST_MakePoint(longitude, latitude), 4326))), 0.8), ST_GeomFromText('GEOMETRYCOLLECTION EMPTY'))

               FROM projets_dessin
               WHERE id_projet = {}
           ),
           (
               SELECT DISTINCT geom
               FROM projets p, formes_mailles_geographiques fmg, mailles_geographiques mg
               WHERE fmg.fk_geo = mg.id_geo
               AND mg.geo_level = 3
               AND ST_Within(ST_SetSRID(ST_MakePoint(p.longitude, p.latitude),4326), geom) IS true
               AND p.id_projet = {}
           )
       ) as geom
   )
   SELECT ic.idinspire, wkb_geometry
   FROM insee_carreaux ic
   inner join insee_donnees_carroyees_niveau_200m a on ic.idinspire = a.idinspire
   , union_commune_dessin ucd
   WHERE (ST_Overlaps(ucd.geom,ic.wkb_geometry) OR ST_Within(ic.wkb_geometry, ucd.geom))
   UNION
   SELECT ic.idinspire, wkb_geometry
   FROM insee_carreaux ic, perimetre_communes_voisines pev
   WHERE ST_Overlaps(pev.geom, ic.wkb_geometry) OR ST_Within(ic.wkb_geometry, pev.geom)
   limit 10000 -- ***
""".format(id_projet, id_projet, id_projet) # A revoir. ***



# Calcul des variables explicatives manquantes.

def calcul_nb_concurrents(label_metier, milieu, rayon_chalandise):
    query = """
        insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)
        select a.idinspire, 'nb_concurrents', count(d.siret) as indice, true, now()
        from (
            select a.idinspire
            from ({}) a
            left join res_carreaux_varexp_{} b on a.idinspire = b.idinspire and b.varexp = 'nb_concurrents'
	        where b.valeur is null
        ) a
        inner join carreaux_communes b on a.idinspire = b.idinspire and b.milieu = '{}'
        inner join insee_carreaux c on b.idinspire = c.idinspire
        left join etablissements_cibles_{} d on (
            st_dwithin(st_transform(st_centroid(c.wkb_geometry), 2154), d.point_2154, {}) and
            d.date_fin is null
        )
        group by a.idinspire;
    """.format(label_metier, query_carreaux_alentours, label_metier, milieu, label_metier, rayon_chalandise)
    print(query)
    cur.execute(query)
    con.commit()

def calcul_nb_habitants_etablissement(label_metier):
    query = """
        insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)
        select a.idinspire, 'nb_habitants_etablissement', (b.valeur / (c.valeur + 1)  * 3.14 * 25/4), true, now()
        from (
            select a.idinspire
            from ({}) a
            left join res_carreaux_varexp_{} b on a.idinspire = b.idinspire and b.varexp = 'nb_habitants_etablissement'
	        where b.valeur is null
        ) a
        inner join res_carreaux_varexp b on a.idinspire = b.idinspire and b.varexp = 'densite_pop'
        inner join res_carreaux_varexp_{} c on a.idinspire = c.idinspire and c.varexp = 'nb_concurrents';
    """.format(label_metier, query_carreaux_alentours, label_metier, label_metier)
    print(query)
    cur.execute(query)
    con.commit()



# Chargement des données pour l'inférence.

def get_data_inference_survie_etablissements_3_ans(id_projet, query_carreaux, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        select a.idinspire, b.milieu, e.statut_2017 as zone, {}, {}
        from ({}) a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {}
        inner join unites_urbaines e on b.code_commune = e.codgeo
    """.format(query_variables_1, query_variables_2, query_carreaux, query_jointures_1, query_jointures_2)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data

def get_data_zone_envisagee_inference_survie_etablissements_3_ans(id_projet, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("round(avg({}.valeur)::numeric, 3) as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("round(avg({}.valeur)::numeric, 3) as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        with carreaux_pertinents as (
            select a.idinspire
            from insee_carreaux a
            inner join (
                select st_convexhull(st_collect(a.point)) as zone_implantation
                from (
                    select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
                    from projets_dessin a 
                    where a.id_projet = {}
                ) a
            ) b on st_intersects(a.wkb_geometry, b.zone_implantation)
        )
        select {}, {}
        from carreaux_pertinents a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {};
    """.format(id_projet, query_variables_1, query_variables_2, query_jointures_1, query_jointures_2)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data

def get_data_adresse_envisagee_inference_survie_etablissements_3_ans(id_projet, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        with carreau_adresse as (
            select b.idinspire
            from (
                select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
                from projets a 
                where a.id_projet = {}
            ) a
            inner join insee_carreaux b on st_intersects(a.point, b.wkb_geometry)
            limit 1
        )
        select {}, {}
        from carreau_adresse a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {};
    """.format(id_projet, query_variables_1, query_variables_2, query_jointures_1, query_jointures_2)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data

def get_data_inference_survie_etablissements_5_ans(id_projet, query_carreaux, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        select a.idinspire, b.milieu, f.statut_2017 as zone, {}, {}, e.valeur as prediction_survie_3_ans
        from ({}) a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {}
        inner join res_carreaux_scores e on a.idinspire = e.idinspire and e.score = 'survie_3' and e.fk_projet = {}
        inner join unites_urbaines f on b.code_commune = f.codgeo
    """.format(query_variables_1, query_variables_2, query_carreaux, query_jointures_1, query_jointures_2, id_projet)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data

def get_data_zone_envisagee_inference_survie_etablissements_5_ans(id_projet, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("round(avg({}.valeur)::numeric, 3) as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("round(avg({}.valeur)::numeric, 3) as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        with carreaux_pertinents as (
            select a.idinspire
            from insee_carreaux a
            inner join (
                select st_convexhull(st_collect(a.point)) as zone_implantation
                from (
                    select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
                    from projets_dessin a 
                    where a.id_projet = {}
                ) a
            ) b on st_intersects(a.wkb_geometry, b.zone_implantation)
        )
        select {}, {}, e.valeur as prediction_survie_3_ans
        from carreaux_pertinents a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {}
        inner join res_carreaux_scores e on a.idinspire = e.idinspire and e.score = 'survie_3' and e.fk_projet = {};
    """.format(id_projet, query_variables_1, query_variables_2, query_jointures_1, query_jointures_2, id_projet)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data

def get_data_adresse_envisagee_inference_survie_etablissements_5_ans(id_projet, label_metier):
    liste_alias_tables_1 = ["c{}".format(i) for i in range(1, len(variables_explicatives_globales) + 1)]
    liste_alias_tables_2 = ["d{}".format(i) for i in range(1, len(variables_explicatives_metier) + 1)]
    liste_variables_explicatives_cles = list(variables_explicatives.keys())
    liste_variables_explicatives_cles_1 = list(variables_explicatives_globales.keys())
    liste_variables_explicatives_cles_2 = list(variables_explicatives_metier.keys())
    liste_variables_explicatives_valeurs = list(variables_explicatives.values())
    liste_variables_explicatives_valeurs_1 = list(variables_explicatives_globales.values())
    liste_variables_explicatives_valeurs_2 = list(variables_explicatives_metier.values())
    query_variables_1 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_cles_1))
    query_variables_2 = ", ".join("{}.valeur as {}".format(x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_cles_2))
    query_jointures_1 = " inner join ".join("res_carreaux_varexp {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(x, x, x, y) for x, y in zip(liste_alias_tables_1, liste_variables_explicatives_valeurs_1))
    query_jointures_2 = " inner join ".join("res_carreaux_varexp_{} {} on a.idinspire = {}.idinspire and {}.varexp = '{}'".format(label_metier, x, x, x, y) for x, y in zip(liste_alias_tables_2, liste_variables_explicatives_valeurs_2))
    query = """
        with carreau_adresse as (
            select b.idinspire
            from (
                select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
                from projets a 
                where a.id_projet = {}
            ) a
            inner join insee_carreaux b on st_intersects(a.point, b.wkb_geometry)
            limit 1
        )
        select {}, {}, e.valeur as prediction_survie_3_ans
        from carreau_adresse a
        inner join carreaux_communes b on a.idinspire = b.idinspire
        inner join {}
        inner join {}
        inner join res_carreaux_scores e on a.idinspire = e.idinspire and e.score = 'survie_3' and e.fk_projet = {};
    """.format(id_projet, query_variables_1, query_variables_2, query_jointures_1, query_jointures_2, id_projet)
    cur.execute(query)
    data = pd.DataFrame([dict(x) for x in cur.fetchall()])
    return data



# Chargement des modèles.

# A généraliser. ***
all_modeles = {
    'boulangerie': {
        'survie_3': 362,
        'survie_5': 365
    }
}

def get_modele(label_metier, score):

    bucket_sdq = BucketManager()
    id_modele = all_modeles[label_metier][score]

    nom_fichier_bucket = "modeles/{}/{}/{}_trained_model".format(label_metier, score, id_modele)
    nom_fichier_modele = 'fichier_modele'
    bucket_sdq.download_blob(nom_fichier_bucket, file_name=nom_fichier_modele)
    fichier_modele = open(nom_fichier_modele, 'rb')
    modele = pickle.load(fichier_modele)
    fichier_modele.close()

    nom_fichier_bucket = "modeles/{}/{}/{}_liste_variables_selectionnees".format(label_metier, score, id_modele)
    nom_fichier_liste_variables_selectionnees = 'fichier_liste_variables_selectionnees'
    bucket_sdq.download_blob(nom_fichier_bucket, file_name=nom_fichier_liste_variables_selectionnees)
    fichier_liste_variables_selectionnees = open(nom_fichier_liste_variables_selectionnees, 'rb')
    liste_variables_selectionnees = pickle.load(fichier_liste_variables_selectionnees)
    fichier_liste_variables_selectionnees.close()

    return {'modele': modele, 'liste_variables_selectionnees': liste_variables_selectionnees}



# Calcul du score d'opportunité d'un projet.

def calcul_score_opportunite_zone_envisagee(id_projet):
    query = """
    with carreaux_pertinents as (
        select a.idinspire
        from insee_carreaux a
        inner join (
            select st_convexhull(st_collect(a.point)) as zone_implantation
            from (
                select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
                from projets_dessin a 
                where a.id_projet = {}
            ) a
        ) b on st_intersects(a.wkb_geometry, b.zone_implantation)
    )
    insert into res_projets_scores(id_projet, score, valeur, date_prediction, reel)
    select {}, 'opportunite', round(avg(b.valeur)::numeric, 3), now(), true
    from carreaux_pertinents a
    inner join res_carreaux_scores b on a.idinspire = b.idinspire and b.score = 'opportunite'
    """.format(id_projet, id_projet)
    cur.execute(query)
    con.commit()

def calcul_score_opportunite_adresse_envisagee(id_projet):
    query = """
    with carreaux_pertinents as (
        select b.idinspire
        from (
            select st_setsrid(st_makepoint(a.longitude, a.latitude), 4326) as point
            from projets a 
            where a.id_projet = {}
        ) a
        inner join insee_carreaux b on st_intersects(a.point, b.wkb_geometry)
        limit 1
    )
    insert into res_projets_scores(id_projet, score, valeur, date_prediction, reel)
    select {}, 'opportunite', round(b.valeur::numeric, 3), now(), true
    from carreaux_pertinents a
    inner join res_carreaux_scores b on a.idinspire = b.idinspire and b.score = 'opportunite'
    """.format(id_projet, id_projet)
    cur.execute(query)
    con.commit()



# MAIN.


def main(id_projet):

    print("Inférence pour le projet {}".format(id_projet))


    query = "UPDATE projets SET statut_etude = 'calculs en cours' WHERE id_projet = {}".format(id_projet)
    cur.execute(query)
    con.commit()


    # 1. Récupération des données du projet.

    data_projet = get_data_projet(id_projet)


    # 2. Calcul des variables explicatives manquantes.

    print("Calcul des variables explicatives manquantes en cours...")
    calcul_nb_concurrents(label_metier=data_projet['label_metier'], milieu='URBAIN', rayon_chalandise=500) # A généraliser. ***
    calcul_nb_concurrents(label_metier=data_projet['label_metier'], milieu='RURAL', rayon_chalandise=10000) # A généraliser. ***
    calcul_nb_habitants_etablissement(label_metier=data_projet['label_metier'])
    print("Calcul des variables explicatives manquantes terminé.")


    # 3. Modèle de prédiction de la survie à 3 ans.

    print("Prédiction de la survie des établissements à 3 ans...")

    # Chargement du modèle.
    print("Chargement du modèle en cours...")
    modele_et_variables = get_modele(data_projet['label_metier'], 'survie_3')
    modele = modele_et_variables['modele']
    liste_variables_selectionnees = modele_et_variables['liste_variables_selectionnees']
    print("Chargement du modèle terminé.")

    # 3.1. Inférence sur la zone d'implantation envisagée.

    # Chargement des données.
    if data_projet['localisation_type'] == 'dessin sur une zone':
        data_inference = get_data_zone_envisagee_inference_survie_etablissements_3_ans(id_projet, data_projet['label_metier'])
    if data_projet['localisation_type'] == 'adresse':
        data_inference = get_data_adresse_envisagee_inference_survie_etablissements_3_ans(id_projet, data_projet['label_metier'])
    data_inference['indice_frequentation_train'] = data_inference['indice_frequentation_train'].map(lambda x: min(x, 2140000000)) # Pansement, à corriger en amont. ***
    data_inference['caractere_employeur'] = caractere_employeur
    data_inference['inscription_registre_metiers'] = inscription_registre_metiers
    if data_projet['milieu'] == 'URBAIN': 
        data_inference['milieu'] = 1 
    if data_projet['milieu'] == 'RURAL': 
        data_inference['milieu'] = 0
    if data_projet['zone'] == 'C': 
        data_inference['zone'] = 4
    if data_projet['zone'] == 'B': 
        data_inference['zone'] = 3
    if data_projet['zone'] == 'I': 
        data_inference['zone'] = 2
    if data_projet['zone'] == 'R': 
        data_inference['zone'] = 1

    # Enregistrement des valeurs des variables explicatives.
    variables_explicatives_projet = pd.DataFrame(data_inference.T.index, columns=['varexp'])
    variables_explicatives_projet['valeur'] = list(data_inference.T[0])
    variables_explicatives_projet['id_projet'] = id_projet
    variables_explicatives_projet['date_calcul'] = datetime.now()
    variables_explicatives_projet['reel'] = True
    variables_explicatives_projet.to_sql('res_projets_varexp', con=engine, if_exists="append", index=False) # ***

    # Inférence.
    print(liste_variables_selectionnees) # ***
    print(data_inference[liste_variables_selectionnees]) # ***
    resultats = pd.DataFrame(modele.predict_proba(data_inference[liste_variables_selectionnees]), columns=['False', 'valeur']).drop(['False'], axis=1)  
    print(resultats.shape) # ***

    # Enregistrement du résultat.
    resultats['score'] = 'survie_3' 
    resultats['id_projet'] = id_projet
    resultats['date_prediction'] = datetime.now()
    resultats['reel'] = True 
    print('resultats_survie_3_ans_projet:', resultats) # ***
    resultats.to_sql('res_projets_scores', con=engine, if_exists="append", index=False) # ***

    # 3.2. Inférence sur les carreaux alentours

    # 3.2.1. Chargement des données.
    data_inference = get_data_inference_survie_etablissements_3_ans(id_projet, query_carreaux_alentours, data_projet['label_metier'])
    data_inference['indice_frequentation_train'] = data_inference['indice_frequentation_train'].map(lambda x: min(x, 2140000000)) # Pansement, à corriger en amont. ***
    data_inference['caractere_employeur'] = caractere_employeur
    data_inference['inscription_registre_metiers'] = inscription_registre_metiers
    data_inference['milieu'] = data_inference['milieu'].map(lambda x: (1 if x == 'URBAIN' else 0))
    def numerisation_zone(x):
        if x == 'C': y = 4
        if x == 'B': y = 3
        if x == 'I': y = 2
        if x == 'R': y = 1
        return y
    data_inference['zone'] = data_inference['zone'].map(numerisation_zone)

    # 3.2.2. Inférence.
    resultats = pd.concat(
        [
            pd.DataFrame(data_inference['idinspire']).reset_index(drop=True), 
            pd.DataFrame(modele.predict_proba(data_inference[liste_variables_selectionnees]), columns=['False', 'valeur'])
        ],
        axis=1
    ).drop(['False'], axis=1)    

    # 3.2.3. Enregistrement des résultats.
    resultats['fk_metier'] = data_projet['label_metier']
    resultats['score'] = 'survie_3' 
    resultats['fk_projet'] = id_projet
    resultats['date_prediction'] = datetime.now()
    resultats['reel'] = True 
    resultats.to_sql('res_carreaux_scores', con=engine, if_exists="append", index=False) # ***


    # 4. Modèle de prédiction de la survie à 5 ans.

    print("Prédiction de la survie des établissements à 5 ans...")

    # Chargement du modèle.
    print("Chargement du modèle en cours...")
    modele_et_variables = get_modele(data_projet['label_metier'], 'survie_5')
    modele = modele_et_variables['modele']
    liste_variables_selectionnees = modele_et_variables['liste_variables_selectionnees']
    print("Chargement du modèle terminé.")

    # 4.1. Inférence sur la zone d'implantation envisagée.

    # Chargement des données.
    if data_projet['localisation_type'] == 'dessin sur une zone':
        data_inference = get_data_zone_envisagee_inference_survie_etablissements_5_ans(id_projet, data_projet['label_metier'])
    if data_projet['localisation_type'] == 'adresse':
        data_inference = get_data_adresse_envisagee_inference_survie_etablissements_5_ans(id_projet, data_projet['label_metier'])
    data_inference['indice_frequentation_train'] = data_inference['indice_frequentation_train'].map(lambda x: min(x, 2140000000)) # Pansement, à corriger en amont. ***
    data_inference['caractere_employeur'] = caractere_employeur
    data_inference['inscription_registre_metiers'] = inscription_registre_metiers
    if data_projet['milieu'] == 'URBAIN': 
        data_inference['milieu'] = 1 
    if data_projet['milieu'] == 'RURAL': 
        data_inference['milieu'] = 0
    if data_projet['zone'] == 'C': 
        data_inference['zone'] = 4
    if data_projet['zone'] == 'B': 
        data_inference['zone'] = 3
    if data_projet['zone'] == 'I': 
        data_inference['zone'] = 2
    if data_projet['zone'] == 'R': 
        data_inference['zone'] = 1

    # Enregistrement des valeurs des variables explicatives. 
    # Seulement si différentes de celles pour le modèle de survie à 3 ans. ***

    # Inférence.
    resultats = pd.DataFrame(modele.predict_proba(data_inference[liste_variables_selectionnees]), columns=['False', 'valeur']).drop(['False'], axis=1)  

    # Enregistrement du résultat.
    resultats['score'] = 'survie_5'
    resultats['id_projet'] = id_projet
    resultats['date_prediction'] = datetime.now()
    resultats['reel'] = True
    print('resultats_survie_5_ans_projet:', resultats) # ***
    resultats.to_sql('res_projets_scores', con=engine, if_exists="append", index=False) # ***

    # 4.2. Inférence sur les carreaux alentours

    # 4.2.1. Chargement des données.
    data_inference = get_data_inference_survie_etablissements_5_ans(id_projet, query_carreaux_alentours, data_projet['label_metier'])
    data_inference['indice_frequentation_train'] = data_inference['indice_frequentation_train'].map(lambda x: min(x, 2140000000)) # Pansement, à corriger en amont. ***
    data_inference['caractere_employeur'] = caractere_employeur
    data_inference['inscription_registre_metiers'] = inscription_registre_metiers
    data_inference['milieu'] = data_inference['milieu'].map(lambda x: (1 if x == 'URBAIN' else 0))
    def numerisation_zone(x):
        if x == 'C': y = 4
        if x == 'B': y = 3
        if x == 'I': y = 2
        if x == 'R': y = 1
        return y
    data_inference['zone'] = data_inference['zone'].map(numerisation_zone)

    # 4.2.2. Inférence.
    resultats = pd.concat(
        [
            pd.DataFrame(data_inference['idinspire']).reset_index(drop=True), 
            pd.DataFrame(modele.predict_proba(data_inference[liste_variables_selectionnees]), columns=['False', 'valeur']),
        ],
        axis=1
    ).drop(['False'], axis=1)   

    # 4.2.3. Enregistrement des résultats.
    resultats['fk_metier'] = data_projet['label_metier']
    resultats['score'] = 'survie_5' 
    resultats['fk_projet'] = id_projet
    resultats['date_prediction'] = datetime.now()
    resultats['reel'] = True
    resultats.to_sql('res_carreaux_scores', con=engine, if_exists="append", index=False) # ***


    # 5. Score d'opprtunité d'implantation.

    if data_projet['localisation_type'] == 'dessin sur une zone':
        calcul_score_opportunite_zone_envisagee(id_projet)
    if data_projet['localisation_type'] == 'adresse':
        calcul_score_opportunite_adresse_envisagee(id_projet)
    
    
    query = "UPDATE projets SET statut_etude = 'disponible' WHERE id_projet = {}".format(id_projet)
    cur.execute(query)
    con.commit()


main(id_projet)
# try:
#     main(id_projet)
# except Exception as e: 
#     sys.stdout(e)
#     query = "UPDATE projets SET statut_etude = 'echec' WHERE id_projet = {}".format(id_projet)