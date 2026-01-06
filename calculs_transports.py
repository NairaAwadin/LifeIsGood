import pandas as pd
import numpy as np

df = pd.read_csv('ressources/data/transport/Datasets/tarifs_transports.csv')

def cout_transports_publics(ville, profil, salaire):
    """
    Calcule le coût des transports publics selon la ville et le profil
    """
    ville_data = df[df['ville'].str.lower() == ville.lower()]

    if ville_data.empty:
        raise ValueError(f"Ville '{ville}' non trouvée dans le dataset")

    if profil.lower() == 'adulte':
        cout = ville_data['abonnement_mensuel_adulte'].values[0]
    elif profil.lower() == 'etudiant':
        cout = ville_data['abonnement_mensuel_etudiant'].values[0]
    elif profil.lower() == 'senior':
        cout = ville_data['abonnement_mensuel_senior'].values[0]
    elif profil.lower() == 'scolaire':
        cout = ville_data['abonnement_mensuel_scolaire'].values[0]
    elif profil.lower() == 'junior':
        cout = ville_data['abonnement_mensuel_junior'].values[0]
    else:
        cout = 0

    pourcentage = (cout / salaire) * 100 if salaire > 0 else 0
    return round(cout, 2), round(pourcentage, 2)

def cout_voiture(ville, type_voiture, km_mensuel, salaire, consommation=6):
    '''
    Calcule le cout mensuel d'une voiture basé sur le type de carburant ou si electrique
    Consommation par défaut (essence ou diesel) = 6L/100km par défaut, si on ne connait pas la conso de sa voiture
    Consommation voiture electrique : autour de 17kWh/100 km
    '''
    ville_data = df[df['ville'].str.lower() == ville.lower()]

    if ville_data.empty:
        raise ValueError(f"Ville '{ville}' non trouvée dans le dataset")

    if type_voiture.lower() == 'essence':
        prix_carburant = ville_data['prix_essence_l'].values[0]
        carburant_necessaire = (km_mensuel / 100) * consommation
        cout_total = carburant_necessaire * prix_carburant

    elif type_voiture.lower() == 'diesel':
        prix_carburant = ville_data['prix_diesel_l'].values[0]
        carburant_necessaire = (km_mensuel / 100) * consommation
        cout_total = carburant_necessaire * prix_carburant

    elif type_voiture.lower() == 'electrique':
        prix_elec = ville_data['prix_elec_kwh'].values[0]
        kwh_necessaire = (km_mensuel/100) * 17 # 17 kWh par 100km
        cout_total = kwh_necessaire * prix_elec

    else:
        cout_total = 0

    pourcentage = (cout_total / salaire) * 100 if salaire > 0 else 0
    return round(cout_total, 2), round(pourcentage, 2)

def budget_logement(salaire, pourcentage_transport):
    '''
    Calcule le budget logement recommandé
    '''
    budget = (salaire * (40 - pourcentage_transport)) / 100
    return round(budget, 2)


#print(df)