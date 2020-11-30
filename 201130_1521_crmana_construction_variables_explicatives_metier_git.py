import psycopg2
import pandas as pd
from datetime import datetime



# Paramètres.
libelle_metier = 'Boulangerie, pâtisserie (NAR80)'
label_metier = 'boulangerie'
rayon_zone_chalandise_max = 10000
rayon_zone_chalandise_rural = 10000
rayon_zone_chalandise_urbain = 500
codes_metiers_concurrence_large = [
    ['niv_4', '10.71'],
    ['niv_4', '47.11']
]
codes_metiers_concurrence_restreinte = [
    ['niv_4', '10.71']
]



# Identifiants de connexion à la base de données. A NE PAS PUBLIER. ***
db_logins = {}

# Connexion à la base de données.
con = psycopg2.connect(
    host = db_logins['host'], 
    database = db_logins['database'], 
    user = db_logins['user'], 
    password = db_logins['password']
)
cur = con.cursor()



# 1. etablissements_cibles_{label_metier}(_na).


query = """
    create materialized view etablissements_cibles_{} as
        select *
        from etablissements_cibles a
        where a.intitule_crmana = '{}';
    create index on etablissements_cibles_{}(siret);
    create index on etablissements_cibles_{}(date_debut);
    create index on etablissements_cibles_{}(date_fin);
    create index on etablissements_cibles_{} using gist(point_4326);
    create index on etablissements_cibles_{} using gist(point_2154);
    create index on etablissements_cibles_{}(code_postal);
    create index on etablissements_cibles_{}(code_commune);
    create index on etablissements_cibles_{}(code_iris);
    create index on etablissements_cibles_{}(idinspire);
    create index on etablissements_cibles_{}(geohash_entier);
    create index on etablissements_cibles_{}(geohash_7);
    create index on etablissements_cibles_{}(geohash_6);
    create index on etablissements_cibles_{}(geohash_5);
    create index on etablissements_cibles_{}(geohash_4);
    create index on etablissements_cibles_{}(premiere_date_debut);
""".format(label_metier, libelle_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
           label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, 
           label_metier, label_metier, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue etablissements_cibles_{} : {} secondes.'.format(label_metier, (date_fin_requete - date_debut_requete).seconds))

# query = """
#     create materialized view etablissements_cibles_{}_na as
#     select a.*
#     from etablissements_cibles_{} a 
#     inner join localisation_etablissements b on (
#         a.siret = b.siret and 
#         b.code_commune in (
#             select a.geo_value
#             from mailles_geographiques a
#             where a.fk_geo_1 = 51010
#             and a.geo_level = 3
#         )
#     )
# """.format(label_metier, label_metier)
# date_debut_requete = datetime.now()
# cur.execute(query)
# con.commit()
# date_fin_requete = datetime.now()
# print('Création de la vue etablissements_cibles_{}_na : {} secondes.'.format(label_metier, (date_fin_requete - date_debut_requete).seconds))



# 2. Pour l'entraînement.


# Construction des indices de synergie territoriale.

# Insertion dans la table indices_synergie_territoriale des siret du métier considéré.
query = """
    insert into indices_synergie_territoriale(siret)
    select distinct a.siret
    from etablissements_cibles a 
    where a.intitule_crmana = '{}'
""".format(libelle_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Insertion des SIRET dans la table indices_synergie_territoriale : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

# Indices de densité d'artisanat sectoriel.
secteurs = {
    'indice_artisanat_proximite': 'Artisanat de proximité',
    'indice_artisanat_economie_creative': 'Artisanat de l\'économie créative', 
    'indice_artisanat_construction': 'Artisanat de la construction', 
    'indice_artisanat_soutien': 'Artisanat de soutien'
}
milieu = 'rural'
niveau_geohash = 5
for indice in list(secteurs.keys()):
    query = """
        update indices_synergie_territoriale as a
        set {}_{} = b.indice
        from (
            select a.siret
            , case when indice is null then 0 else indice end as indice
            from (
                select a.siret, count(b.siret) as indice
                from etablissements_cibles_{} a 
                left join etablissements_pertinents b on (
                    b.categorie = '{}' and 
                    a.siret != b.siret and
                    a.geohash_{} = b.geohash_{} and (
                        (a.premiere_date_debut between b.date_debut and b.date_fin) or
                        (a.premiere_date_debut > b.date_debut and b.date_fin is null)
                    )
                )
                group by a.siret
            ) a
        ) b
        where a.siret = b.siret
    """.format(indice, milieu, label_metier, secteurs[indice].replace("'", "''"), niveau_geohash, 
               niveau_geohash)
    date_debut_requete = datetime.now()
    cur.execute(query)
    con.commit()
    date_fin_requete = datetime.now()
    print("Remplissage des indices de densité d'artisanat sectoriel en milieu rural : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))
milieu = 'urbain'
niveau_geohash = 6
for indice in list(secteurs.keys()):
    query = """
        update indices_synergie_territoriale as a
        set {}_{} = b.indice
        from (
            select a.siret
            , case when indice is null then 0 else indice end as indice
            from (
                select a.siret, count(b.siret) as indice
                from etablissements_cibles_{} a 
                left join etablissements_pertinents b on (
                    b.categorie = '{}' and 
                    a.siret != b.siret and
                    a.geohash_{} = b.geohash_{} and (
                        (a.premiere_date_debut between b.date_debut and b.date_fin) or
                        (a.premiere_date_debut > b.date_debut and b.date_fin is null)
                    )
                )
                group by a.siret
            ) a
        ) b
        where a.siret = b.siret
    """.format(indice, milieu, label_metier, secteurs[indice].replace("'", "''"), niveau_geohash, 
               niveau_geohash)
    date_debut_requete = datetime.now()
    cur.execute(query)
    con.commit()
    date_fin_requete = datetime.now()
    print("Remplissage des indices de densité d'artisanat sectoriel en milieu urbain : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Indices de densité d'artisans par niveau NAF englobant l'activité concernée.
for niveau_naf in [1, 2, 3, 4]:
    subquery = """
        select a.niv_5 
        from  naf_rev2_niveaux a
        where a.niv_{} in (
            select distinct c.niv_{}
            from nomenclature_metiers_crmana a
            left join naf_rev2_intitules b on a.code_naf = b.code_clean
            left join naf_rev2_niveaux c on b.code = c.niv_5
            where a.metier = '{}'
        )
    """.format(niveau_naf, niveau_naf, libelle_metier)
    cur.execute(subquery)
    codes_naf = ', '.join(map(lambda x: "'" + str(x[0]) + "'", cur.fetchall()))
    milieu = 'rural'
    niveau_geohash = 5
    query = """
        update indices_synergie_territoriale as a
        set indice_densite_naf_niveau_{}_{} = b.indice
        from (
            select a.siret
            , case when indice is null then 0 else indice end as indice
            from (
                select a.siret, count(b.siret) as indice
                from etablissements_cibles_{} a 
                left join etablissements_pertinents b on (
                    b.code_naf in ({}) and
                    a.siret != b.siret and
                    a.geohash_{} = b.geohash_{} and (
                        (a.premiere_date_debut between b.date_debut and b.date_fin) or
                        (a.premiere_date_debut > b.date_debut and b.date_fin is null)
                    )
                )
                group by a.siret
            ) a
        ) b
        where a.siret = b.siret
    """.format(niveau_naf, milieu, label_metier, codes_naf, niveau_geohash, niveau_geohash)
    date_debut_requete = datetime.now()
    cur.execute(query)
    con.commit()
    date_fin_requete = datetime.now()
    print("Remplissage des indices de densité d'artisans par niveau NAF en milieu rural : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))
    milieu = 'urbain'
    niveau_geohash = 6
    query = """
        update indices_synergie_territoriale as a
        set indice_densite_naf_niveau_{}_{} = b.indice
        from (
            select a.siret
            , case when indice is null then 0 else indice end as indice
            from (
                select a.siret, count(b.siret) as indice
                from etablissements_cibles_{} a 
                left join etablissements_pertinents b on (
                    b.code_naf in ({}) and
                    a.siret != b.siret and
                    a.geohash_{} = b.geohash_{} and (
                        (a.premiere_date_debut between b.date_debut and b.date_fin) or
                        (a.premiere_date_debut > b.date_debut and b.date_fin is null)
                    )
                )
                group by a.siret
            ) a
        ) b
        where a.siret = b.siret
    """.format(niveau_naf, milieu, label_metier, codes_naf, niveau_geohash, niveau_geohash)
    date_debut_requete = datetime.now()
    cur.execute(query)
    con.commit()
    date_fin_requete = datetime.now()
    print("Remplissage des indices de densité d'artisans par niveau NAF en milieu urbain : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Construction de l'indice de concurrence large. 
subquery_filter = 'where ' + ' or '.join(list(map(
    lambda x: "a.{} = '{}'".format(x[0], x[1]), 
    codes_metiers_concurrence_large
))) 
milieu = 'rural'
niveau_geohash = 5
query = """
    update indices_synergie_territoriale as a
    set indice_concurrence_large_{} = b.indice
    from (
        select a.siret
        , case when indice is null then 0 else indice end as indice
        from (
            select a.siret, count(b.siret) as indice
            from etablissements_cibles_{} a 
            left join etablissements_pertinents b on (
                b.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and
                a.siret != b.siret and
                a.geohash_{} = b.geohash_{} and 
                a.premiere_date_debut between b.date_debut and b.date_fin
            )
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, label_metier, subquery_filter, niveau_geohash, niveau_geohash)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage du champ indice_concurrence_large_rural : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))
milieu = 'urbain'
niveau_geohash = 6
query = """
    update indices_synergie_territoriale as a
    set indice_concurrence_large_{} = b.indice
    from (
        select a.siret
        , case when indice is null then 0 else indice end as indice
        from (
            select a.siret, count(b.siret) as indice
            from etablissements_cibles_{} a 
            left join etablissements_pertinents b on (
                b.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and
                a.siret != b.siret and
                a.geohash_{} = b.geohash_{} and 
                a.premiere_date_debut between b.date_debut and b.date_fin
            )
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, label_metier, subquery_filter, niveau_geohash, niveau_geohash)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage du champ indice_concurrence_large_urbain : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Construction de l'indice de concurrence restreinte. 
subquery_filter = 'where ' + ' or '.join(list(map(
    lambda x: "a.{} = '{}'".format(x[0], x[1]), 
    codes_metiers_concurrence_restreinte
))) 
milieu = 'rural'
niveau_geohash = 5
query = """
    update indices_synergie_territoriale as a
    set indice_concurrence_restreinte_{} = b.indice
    from (
        select a.siret
        , case when indice is null then 0 else indice end as indice
        from (
            select a.siret, count(b.siret) as indice
            from etablissements_cibles_{} a 
            left join etablissements_pertinents b on (
                b.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and
                a.siret != b.siret and
                a.geohash_{} = b.geohash_{} and 
                a.premiere_date_debut between b.date_debut and b.date_fin
            )
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, label_metier, subquery_filter, niveau_geohash, niveau_geohash)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage du champ indice_concurrence_restreinte_rural : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))
milieu = 'urbain'
niveau_geohash = 6
query = """
    update indices_synergie_territoriale as a
    set indice_concurrence_restreinte_{} = b.indice
    from (
        select a.siret
        , case when indice is null then 0 else indice end as indice
        from (
            select a.siret, count(b.siret) as indice
            from etablissements_cibles_{} a 
            left join etablissements_pertinents b on (
                b.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and
                a.siret != b.siret and
                a.geohash_{} = b.geohash_{} and 
                a.premiere_date_debut between b.date_debut and b.date_fin
            )
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, label_metier, subquery_filter, niveau_geohash, niveau_geohash)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage du champ indice_concurrence_restreinte_urbain : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Indices de densité d'équipement public.
categories_equipements = {
    'commerces': 'Commerces', 
    'loisirs': 'Sports, loisirs et culture', 
    'transports': 'Transports et déplacements', 
    'enseignement': 'Enseignement', 
    'sante': 'Santé', 
    'services': 'Services aux particuliers'
}
for categorie in list(categories_equipements.keys()):
    subquery = """
        select a.code_equipement
        from bpe_gamme_equipements a
        where a.domaine_libelle = '{}'
    """.format(categories_equipements[categorie])
    cur.execute(subquery)
    types_equipements = ', '.join(map(lambda x: "'" + str(x[0]) + "'", cur.fetchall()))
    query = """
        update indices_synergie_territoriale as a
        set indice_equipements_{} = b.indice
        from (
            select a.siret
            , case when indice is null then 0 else indice end as indice
            from (
                select a.siret, count(b.*) as indice
                from etablissements_cibles_{} a 
                left join bpe_2018 b on (
                    a.geohash_6 = b.geohash_6 and 
                    b.typequ in ({})
                ) 
                group by a.siret
            ) a
        ) b
        where a.siret = b.siret;
    """.format(categorie, label_metier, types_equipements)
    date_debut_requete = datetime.now()
    cur.execute(query)
    con.commit()
    date_fin_requete = datetime.now()
    print("Remplissage des indices de densité d'équipements : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Indice de desserte par bus.
query = """
    update indices_synergie_territoriale as a
    set indice_desserte_bus = b.indice
    from (
        select a.siret
        , case when indice is null then 0 else indice end as indice
        from (
            select a.siret, sum((500 - a.distance) * a.frequence) as indice
            from (
                -- On calcule les distances des entreprises pertinentes à l'entreprise cible, après avoir préalablement filtré celles
                -- de même geohash, puis situées dans la zone de chalandise.
                select a.siret, st_distance(a.point_2154, b.point_2154) as distance, frequence
                from etablissements_cibles_{} a 
                left join (
                    select ps.stop_id, ps.stop_lat, ps.stop_lon, ps.geohash_4, ps.point_2154
                    , count(distinct(trip_id, departure_time)) as frequence
                    from pigma_stop_times pt, pigma_stops ps
                    where ps.stop_id = pt.stop_id
                    group by ps.stop_id, ps.stop_lat, ps.stop_lon, ps.geohash_4, ps.point_2154
                ) b on (
                    a.geohash_4 = b.geohash_4 and 
                    st_dwithin(a.point_2154, b.point_2154, 500) 
                )
            ) a
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage de l'indice de desserte par bus : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Indice de desserte par train.
query = """
    update indices_synergie_territoriale as a
    set indice_desserte_train = b.indice
    from (
        select a.siret
        , case when distance_min is null then 0 else distance_min end as indice
        from (
            select a.siret, min(a.distance) as distance_min
            from (
                select a.siret, st_distance(a.point_2154, b.point_2154) as distance
                from etablissements_cibles_{} a 
                left join sncf_gares b on (
                    st_dwithin(a.point_2154, b.point_2154, 100000) and
                    voyageurs = 'O'
                ) 
            ) a
            group by a.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage de l'indice de desserte par train : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Indice de fréquentation par train.
query = """
    update indices_synergie_territoriale as a
    set indice_frequentation_train = b.indice
    from (
        select a.siret
        , case when a.indice is null then 0 else a.indice end as indice
        from (
            select distinct on (a.siret) a.siret, (52000 - a.distance) * a.frequentation as indice
            from (
                select a.siret, b.code_uic, st_distance(a.point_2154, b.point_2154) as distance, c.tot_voyageurs_2018 as frequentation
                from etablissements_cibles_{} a 
                left join sncf_gares b on (
                    st_dwithin(a.point_2154, b.point_2154, 100000) and
                    voyageurs = 'O'
                ) 
                left join sncf_frequentation_gares c on b.code_uic = c.code_uic
            ) a
            order by a.siret, a.distance, a.frequentation
        ) a
    ) b
    where a.siret = b.siret;
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage de l'indice de fréquentation par train : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))

# Couverture réseau.
query = """
    update indices_synergie_territoriale as a
    set indice_reseau = 4 where indice_reseau is null -- Provisoire. ***
    /*
    set indice_reseau = b.indice
    from (
        select a.siret, a.bouygues + a.free + a.orange + a.sfr as indice
        from (
            select a.siret
            , case when b.ogc_fid is not null then 1 else 0 end as bouygues
            , case when c.ogc_fid is not null then 1 else 0 end as free
            , case when d.ogc_fid is not null then 1 else 0 end as orange
            , case when e.ogc_fid is not null then 1 else 0 end as sfr
            from etablissements_cibles_{} a 
            left join couverture_4g_bouygues b on st_intersects(a.point_4326, b.wkb_geometry)
            left join couverture_4g_free c on st_intersects(a.point_4326, c.wkb_geometry)
            left join couverture_4g_orange d on st_intersects(a.point_4326, d.wkb_geometry)
            left join couverture_4g_sfr e on st_intersects(a.point_4326, e.wkb_geometry)
        ) a 
    ) b
    where a.siret = b.siret
    */
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print("Remplissage de l'indice de couverture réseau : {} secondes.".format((date_fin_requete - date_debut_requete).seconds))



# Construction des indices de concurrence.

# Insertion dans la table indices_concurrence des siret du métier considéré.
query = """
    insert into indices_concurrence(siret)
    select distinct a.siret
    from etablissements_cibles_{} a 
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Insertion des SIRET dans la table indices_concurrence : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

# Vue intérmédiaire consolidant les établissements à l'intérieur de la zone de chalandise maximale, sans considération pour la 
# date de présence.
# On n'utilise pas de clause 'WITH' car les index sont nécessaires.
query = """
    create materialized view etablissements_voisins_{}_{}m as 
        select a.siret, a.intitule_crmana as activite, b.siret as siret_concurrent, st_distance(a.point_2154, b.point_2154) as distance
        from etablissements_cibles_{} a
        left join etablissements_cibles_{} b on (
            st_dwithin(a.point_2154, b.point_2154, {}) and
            a.siret != b.siret -- Pour écarter l'établissement lui-même.
        );
    create index on etablissements_voisins_{}_{}m(siret);
    create index on etablissements_voisins_{}_{}m(siret_concurrent);
    create index on etablissements_voisins_{}_{}m(distance);
""".format(label_metier, rayon_zone_chalandise_max, label_metier, label_metier, rayon_zone_chalandise_max, 
           label_metier, rayon_zone_chalandise_max, label_metier, rayon_zone_chalandise_max, label_metier, 
           rayon_zone_chalandise_max)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue etablissements_voisins_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_max, (date_fin_requete - date_debut_requete).seconds))

# Etablissements concurrents (à l'intérieur de la zone de chalandise durant les mêmes années d'activité)
# La fonction extract(year from ...) permet d'éliminer les SIRET pour lesquels la durée de vie est négative, absurdité due aux
# erreurs dans la table sirene_etablissements_liens_succession, quand un établissement "successeur" est en réalité antérieur à
# son établissement prédécesseur. (12 établissements concernés sur les 29 798 boulangeries relevées)
# Milieu rural.
query = """
    create materialized view etablissements_concurrents_{}_{}m as
        select a.siret, a.date, b.siret_concurrent, b.distance
        from (
            select a.siret, extract(year from generate_series(a.date_debut_activite, a.date_fin_activite, '1 year'::interval)) as date
            from (
                select distinct a.siret, b.date_debut_activite
                , case when b.date_fin_activite is not null then b.date_fin_activite else now() end as date_fin_activite
                from etablissements_voisins_{}_{}m a
                left join sirene_4_dates_activites b on a.siret = b.siret
            ) a
        ) a
        left join (
            select a.siret, a.date, a.siret_concurrent, a.distance
            from (
                select a.siret, extract(year from generate_series(a.date_debut_activite, a.date_fin_activite, '1 year'::interval)) as date
                , a.siret_concurrent, a.annee_debut_activite_concurrent, a.annee_fin_activite_concurrent, a.distance
                from (
                    select a.siret, b.date_debut_activite
                    , case when b.date_fin_activite is not null then b.date_fin_activite else now() end as date_fin_activite
                    , a.siret_concurrent, extract(year from c.date_debut_activite) as annee_debut_activite_concurrent
                    , case when c.date_fin_activite is not null then extract(year from c.date_fin_activite) else extract(year from now()) end as annee_fin_activite_concurrent
                    , a.distance
                    from etablissements_voisins_{}_{}m a
                    left join sirene_4_dates_activites b on a.siret = b.siret
                    left join sirene_4_dates_activites c on a.siret_concurrent = c.siret
                    -- where a.distance <= {} -- ***
                ) a
            ) a
            where a.date between a.annee_debut_activite_concurrent and a.annee_fin_activite_concurrent
        ) b on a.siret = b.siret and a.date = b.date;
    create index on etablissements_concurrents_{}_{}m(siret);
    create index on etablissements_concurrents_{}_{}m(siret_concurrent);
""".format(label_metier, rayon_zone_chalandise_rural, label_metier, rayon_zone_chalandise_max, label_metier, 
           rayon_zone_chalandise_max, rayon_zone_chalandise_rural, label_metier, rayon_zone_chalandise_rural, label_metier, 
           rayon_zone_chalandise_rural)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue etablissements_concurrents_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_rural, (date_fin_requete - date_debut_requete).seconds))

# Etablissements concurrents (à l'intérieur de la zone de chalandise durant les mêmes années d'activité)
# La fonction extract(year from ...) permet d'éliminer les SIRET pour lesquels la durée de vie est négative, absurdité due aux
# erreurs dans la table sirene_etablissements_liens_succession, quand un établissement "successeur" est en réalité antérieur à
# son établissement prédécesseur. (12 établissements concernés sur les 29 798 boulangeries relevées)
# Milieu urbain.
# A optimiser ? ***
query = """
    create materialized view etablissements_concurrents_{}_{}m as
        select a.siret, a.date, b.siret_concurrent, b.distance
        from (
            select a.siret, extract(year from generate_series(a.date_debut_activite, a.date_fin_activite, '1 year'::interval)) as date
            from (
                select distinct a.siret, b.date_debut_activite
                , case when b.date_fin_activite is not null then b.date_fin_activite else now() end as date_fin_activite
                from etablissements_voisins_{}_{}m a
                left join sirene_4_dates_activites b on a.siret = b.siret
            ) a
        ) a
        left join (
            select a.siret, a.date, a.siret_concurrent, a.distance
            from (
                select a.siret, extract(year from generate_series(a.date_debut_activite, a.date_fin_activite, '1 year'::interval)) as date
                , a.siret_concurrent, a.annee_debut_activite_concurrent, a.annee_fin_activite_concurrent, a.distance
                from (
                    select a.siret, b.date_debut_activite
                    , case when b.date_fin_activite is not null then b.date_fin_activite else now() end as date_fin_activite
                    , a.siret_concurrent, extract(year from c.date_debut_activite) as annee_debut_activite_concurrent
                    , case when c.date_fin_activite is not null then extract(year from c.date_fin_activite) else extract(year from now()) end as annee_fin_activite_concurrent
                    , a.distance
                    from etablissements_voisins_{}_{}m a
                    left join sirene_4_dates_activites b on a.siret = b.siret
                    left join sirene_4_dates_activites c on a.siret_concurrent = c.siret
                    where a.distance <= {} 
                ) a
            ) a
            where a.date between a.annee_debut_activite_concurrent and a.annee_fin_activite_concurrent
        ) b on a.siret = b.siret and a.date = b.date;
    create index on etablissements_concurrents_{}_{}m(siret);
    create index on etablissements_concurrents_{}_{}m(siret_concurrent);
""".format(label_metier, rayon_zone_chalandise_urbain, label_metier, rayon_zone_chalandise_max, label_metier, 
           rayon_zone_chalandise_max, rayon_zone_chalandise_urbain, label_metier, rayon_zone_chalandise_urbain, label_metier, 
           rayon_zone_chalandise_urbain)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue etablissements_concurrents_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_urbain, (date_fin_requete - date_debut_requete).seconds))

# Nombre de concurrents annuels (à l'intérieur de la zone de chalandise définie).
# Milieu rural.
query = """
    create materialized view tmp_nb_concurrents_annuels_{}_{}m as
        select a.siret, a.date, count(siret_concurrent) as nb_concurrents
        from etablissements_concurrents_{}_{}m a
        group by a.siret, a.date;
    create index on tmp_nb_concurrents_annuels_{}_{}m(siret);
""".format(label_metier, rayon_zone_chalandise_rural, label_metier, rayon_zone_chalandise_rural, label_metier,
        rayon_zone_chalandise_rural)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue tmp_nb_concurrents_annuels_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_rural, (date_fin_requete - date_debut_requete).seconds))

# nb_concurrents_debut_rural.
milieu = 'rural'
query = """
    update indices_concurrence as a
    set nb_concurrents_debut_{} = c.nb_concurrents
    from (
        select a.siret, min(a.date) as date
        from tmp_nb_concurrents_annuels_{}_{}m a
        group by a.siret
    ) b
    left join tmp_nb_concurrents_annuels_{}_{}m c on b.siret = c.siret and b.date = c.date
    where a.siret = b.siret;
""".format(milieu, label_metier, rayon_zone_chalandise_rural, label_metier, rayon_zone_chalandise_rural)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_concurrents_debut_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

# nb_habitants_etablissement_rural.
milieu = 'rural'
query = """
    update indices_concurrence as a
    set nb_habitants_etablissement_debut_{} = (d.valeur / (c.nb_concurrents_debut_{} + 1) * 3.14 * 25/4)::double precision
    from etablissements_cibles_{} b
    inner join indices_concurrence c on b.siret = c.siret
    inner join donnees_ponderees_carreau d on (b.idinspire = d.idinspire and d.fk_donnees_ponderees = 4)
    where a.siret = b.siret;
""".format(milieu, milieu, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_habitants_etablissement_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

# taux_etablissements_concurrents_debut_rural.
milieu = 'rural'
query = """
    update indices_concurrence as a
    set taux_etablissements_concurrents_debut_{} = b.indice
    from (
        select a.siret, case when a.indice is not null then a.indice else 0 end as indice 
        from (
            select a.siret
            , case 
                when indice_densite_naf_niveau_1_{} != 0 
                then round((b.indice_concurrence_restreinte_{} / b.indice_densite_naf_niveau_1_{})::numeric, 3)
                else null
            end as indice
            from etablissements_cibles_{} a
            inner join indices_synergie_territoriale b on a.siret = b.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, milieu, milieu, milieu, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ taux_etablissements_concurrents_debut_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

query = "drop materialized view tmp_nb_concurrents_annuels_{}_{}m".format(label_metier, rayon_zone_chalandise_rural)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Suppression de la vue tmp_nb_concurrents_annuels_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_rural, (date_fin_requete - date_debut_requete).seconds))

# Nombre de concurrents annuels (à l'intérieur de la zone de chalandise définie).
# Milieu urbain.
query = """
    create materialized view tmp_nb_concurrents_annuels_{}_{}m as
        select a.siret, a.date, count(siret_concurrent) as nb_concurrents
        from etablissements_concurrents_{}_{}m a
        group by a.siret, a.date;
    create index on tmp_nb_concurrents_annuels_{}_{}m(siret);
""".format(label_metier, rayon_zone_chalandise_urbain, label_metier, rayon_zone_chalandise_urbain, label_metier,
        rayon_zone_chalandise_urbain)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Création de la vue tmp_nb_concurrents_annuels_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_urbain, (date_fin_requete - date_debut_requete).seconds))

# nb_concurrents_debut_urbain.
milieu = 'urbain'
query = """
    update indices_concurrence as a
    set nb_concurrents_debut_{} = c.nb_concurrents
    from (
        select a.siret, min(a.date) as date
        from tmp_nb_concurrents_annuels_{}_{}m a
        group by a.siret
    ) b
    left join tmp_nb_concurrents_annuels_{}_{}m c on b.siret = c.siret and b.date = c.date
    where a.siret = b.siret;
""".format(milieu, label_metier, rayon_zone_chalandise_urbain, label_metier, rayon_zone_chalandise_urbain)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_concurrents_debut_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

# nb_habitants_etablissement_urbain.
milieu = 'urbain'
query = """
    update indices_concurrence as a
    set nb_habitants_etablissement_debut_{} = (d.valeur / (c.nb_concurrents_debut_{} + 1) * 3.14 * 25/4)::double precision
    from etablissements_cibles_{} b
    inner join indices_concurrence c on b.siret = c.siret
    inner join donnees_ponderees_carreau d on (b.idinspire = d.idinspire and d.fk_donnees_ponderees = 4)
    where a.siret = b.siret;
""".format(milieu, milieu, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_habitants_etablissement_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

# taux_etablissements_concurrents_debut_urbain.
milieu = 'urbain'
query = """
    update indices_concurrence as a
    set taux_etablissements_concurrents_debut_{} = b.indice
    from (
        select a.siret, case when a.indice is not null then a.indice else 0 end as indice 
        from (
            select a.siret
            , case 
                when indice_densite_naf_niveau_1_{} != 0 
                then round((b.indice_concurrence_restreinte_{} / b.indice_densite_naf_niveau_1_{})::numeric, 3)
                else null
            end as indice
            from etablissements_cibles_{} a
            inner join indices_synergie_territoriale b on a.siret = b.siret
        ) a
    ) b
    where a.siret = b.siret
""".format(milieu, milieu, milieu, milieu, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ taux_etablissements_concurrents_debut_{} : {} secondes.'.format(milieu, (date_fin_requete - date_debut_requete).seconds))

query = "drop materialized view tmp_nb_concurrents_annuels_{}_{}m".format(label_metier, rayon_zone_chalandise_urbain)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Supprrssion de la vue tmp_nb_concurrents_annuels_{}_{}m : {} secondes.'.format(label_metier, rayon_zone_chalandise_urbain, (date_fin_requete - date_debut_requete).seconds))

# nb_stagiaires.
query = """
    alter table indices_synergie_territoriale add column if not exists nb_stagiaires integer;
    update indices_synergie_territoriale as a 
    set nb_stagiaires = b.nb_stagiaires
    from (
        select a.siret
        , case when d1.nb_stagiaires is not null then d1.nb_stagiaires else 0 end as nb_stagiaires
        from etablissements_cibles_{} a
        left join (
            select a.code_postal, a.nb_stagiaires
            from analyse_ofs a
            inner join correspondances_metiers_ofs b on a.id_formation = b.id_formation and b.metier = '{}'
        ) d1 on a.code_postal = d1.code_postal::integer
    ) b 
    where a.siret = b.siret;
""".format(label_metier, libelle_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_stagiaires : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

# nb_permis_locaux.
query = """
    alter table indices_synergie_territoriale add column if not exists nb_permis_locaux integer;
    update indices_synergie_territoriale as a 
    set nb_permis_locaux = b.nb_permis_locaux
    from (
        select a.siret
        , case when e2.nb_permis_locaux is not null then e2.nb_permis_locaux else 0 end as nb_permis_locaux
        from etablissements_cibles_{} a
        left join (
            select a.comm as code_commune, count(*) as nb_permis_locaux
            from sitadel_locaux a 
            group by a.comm
        ) e2 on a.code_commune = e2.code_commune
    ) b 
    where a.siret = b.siret;
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_permis_locaux : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

# nb_permis_logements.
query = """
    alter table indices_synergie_territoriale add column if not exists nb_permis_logements integer;
    update indices_synergie_territoriale as a 
    set nb_permis_logements = b.nb_permis_logements
    from (
        select a.siret
        , case when e3.nb_permis_logements is not null then e3.nb_permis_logements else 0 end as nb_permis_logements
        from etablissements_cibles_{} a
        left join (
            select a.comm as code_commune, count(*) as nb_permis_logements
            from sitadel_logements a 
            group by a.comm
        ) e3 on a.code_commune = e3.code_commune
    ) b 
    where a.siret = b.siret;
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_permis_logements : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

# nb_chambres_hotels.
query = """
    alter table indices_synergie_territoriale add column if not exists nb_chambres_hotels integer;
    update indices_synergie_territoriale as a 
    set nb_chambres_hotels = b.nb_chambres_hotels
    from (
        select a.siret
        , case when e4.nb_chambres_hotels is not null then e4.nb_chambres_hotels else 0 end as nb_chambres_hotels
        from etablissements_cibles_{} a
        left join (
            select a.codgeo as code_commune, sum(a.htch19) as nb_chambres_hotels
            from datagouv_tourisme a 
            group by a.codgeo
        ) e4 on a.code_commune = e4.code_commune
    ) b 
    where a.siret = b.siret;
""".format(label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Remplissage du champ nb_chambres_hotels : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))



# 3. Pour l'inférence.

query = """
CREATE TABLE public.res_carreaux_varexp_{}
(
    id serial NOT NULL primary key,
    idinspire character varying COLLATE pg_catalog."default" NOT NULL,
    varexp character varying COLLATE pg_catalog."default" NOT NULL,
    valeur double precision,
    centile double precision,
    reel boolean,
    date_calcul timestamp without time zone
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;
ALTER TABLE public.res_carreaux_varexp_{}
    OWNER to crmana;
CREATE INDEX res_carreaux_varexp_{}_idinspire_idx
    ON public.res_carreaux_varexp_{} USING btree
    (idinspire COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
CREATE INDEX res_carreaux_varexp_{}_idinspire_idx1
    ON public.res_carreaux_varexp_{} USING btree
    (idinspire COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
CREATE INDEX res_carreaux_varexp_{}_varexp_idx
    ON public.res_carreaux_varexp_{} USING btree
    (varexp COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
""".format(label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier, label_metier)
cur.execute(query)
con.commit()


# Indices de densité d'artisans par niveau NAF englobant l'activité.

niveaux_naf = [4, 3, 2, 1]
codes_naf = {}
for niveau_naf in niveaux_naf:
    subquery = """
        select a.niv_5 
        from  naf_rev2_niveaux a
        where a.niv_{} in (
            select distinct c.niv_{}
            from nomenclature_metiers_crmana a
            left join naf_rev2_intitules b on a.code_naf = b.code_clean
            left join naf_rev2_niveaux c on b.code = c.niv_5
            where a.metier = '{}'
        )
    """.format(niveau_naf, niveau_naf, libelle_metier)
    cur.execute(subquery)
    codes_naf[niveau_naf] = ', '.join(map(lambda x: "'" + str(x[0]) + "'", cur.fetchall()))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'indice_densite_naf_niveau_4'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_4
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire

union all

select a.idinspire, 'indice_densite_naf_niveau_3'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_3
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire

union all

select a.idinspire, 'indice_densite_naf_niveau_2'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_2
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire

union all

select a.idinspire, 'indice_densite_naf_niveau_1'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_1
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire;
""".format(label_metier, codes_naf[4], codes_naf[3], codes_naf[2], codes_naf[1])
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indices de densité d\'artisans par niveau NAF englobant en milieu urbain : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'indice_densite_naf_niveau_4'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_4
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, codes_naf[4])
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indices de densité d\'artisans par niveau NAF englobant en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'indice_densite_naf_niveau_3'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_3
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, codes_naf[3])
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indices de densité d\'artisans par niveau NAF englobant en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'indice_densite_naf_niveau_2'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_2
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, codes_naf[2])
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indices de densité d\'artisans par niveau NAF englobant en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)
select a.idinspire, 'indice_densite_naf_niveau_1'
, count(distinct d.siret) filter (where d.code_naf in ({}) and d.date_fin is null) as indice_densite_naf_niveau_1
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010    
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, codes_naf[1])
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indices de densité d\'artisans par niveau NAF englobant en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))


# Indices de concurrence large et restreinte.

subquery_concurrence_large = 'where ' + ' or '.join(list(map(
    lambda x: "a.{} = '{}'".format(x[0], x[1]), 
    codes_metiers_concurrence_large
)))
subquery_concurrence_restreinte = 'where ' + ' or '.join(list(map(
    lambda x: "a.{} = '{}'".format(x[0], x[1]), 
    codes_metiers_concurrence_restreinte
)))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'ind_concurrence_large'
, count(distinct d.siret) filter (where d.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and d.date_fin is null) as indice_concurrence_large
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010        
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire;
""".format(label_metier, subquery_concurrence_large)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indice de concurrene large en milieu urbain : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'ind_concurrence_restreinte'
, count(distinct d.siret) filter (where d.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and d.date_fin is null) as indice_concurrence_restreinte
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010        
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_6 = d.geohash_6
where a.milieu = 'URBAIN'
group by a.idinspire;
""".format(label_metier, subquery_concurrence_restreinte)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indice de concurrence restreinte en milieu urbain : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'ind_concurrence_large'
, count(distinct d.siret) filter (where d.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and d.date_fin is null) as indice_concurrence_large
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010        
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, subquery_concurrence_large)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indice de concurrence large en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))

query = """
insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)

select a.idinspire, 'ind_concurrence_restreinte'
, count(distinct d.siret) filter (where d.code_naf in (select a.niv_5 from naf_rev2_niveaux a {}) and d.date_fin is null) as indice_concurrence_restreinte
, true, now()
from carreaux_communes a 
inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010        
inner join carreaux_geohashing c on a.idinspire = c.idinspire 
left join etablissements_pertinents d on c.geohash_5 = d.geohash_5
where a.milieu = 'RURAL'
group by a.idinspire;
""".format(label_metier, subquery_concurrence_restreinte)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Indice de concurrence restreinte en milieu rural : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))


# Taux d'établissements concurrents.

query = """
    insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)
    select a.idinspire, 'taux_concurrents_'
    , case when d.valeur = 0 then 0 else (c.valeur / d.valeur) end
    , true, now()
    from carreaux_communes a 
    inner join mailles_geographiques b on a.code_commune = b.geo_value and b.fk_geo_1 = 51010
    inner join res_carreaux_varexp_{} c on a.idinspire = c.idinspire and c.varexp = 'ind_concurrence_restreinte'
    inner join res_carreaux_varexp_{} d on a.idinspire = d.idinspire and d.varexp = 'indice_densite_naf_niveau_1';
""".format(label_metier, label_metier, label_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Taux d\'établissements concurrents : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))


# nb_stagiaires.

query = """
    insert into res_carreaux_varexp_{}(idinspire, varexp, valeur, reel, date_calcul)
    select a.idinspire, 'nb_stagiaires'
    , case when d.nb_stagiaires is not null then d.nb_stagiaires else 0 end 
    , true, now()
    from carreaux_codes_postaux a
    inner join carreaux_communes b on a.idinspire = b.idinspire     
    inner join mailles_geographiques c on b.code_commune = c.geo_value and c.fk_geo_1 = 51010
    left join (
        select a.code_postal, a.nb_stagiaires
        from analyse_ofs a
        inner join correspondances_metiers_ofs b on a.id_formation = b.id_formation and b.metier = '{}'
    ) d on a.code_postal = d.code_postal;
""".format(label_metier, libelle_metier)
date_debut_requete = datetime.now()
cur.execute(query)
con.commit()
date_fin_requete = datetime.now()
print('Nombre de stagiaires : {} secondes.'.format((date_fin_requete - date_debut_requete).seconds))