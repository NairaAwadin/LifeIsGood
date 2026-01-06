from datetime import datetime
from calculs_transports import budget_logement

def create_dictionnaire_recherche(params):
    """
    Crée un dictionnaire structuré pour la recherche de logement avec toutes les informations
    """
    donnees = {
        "ville": params["ville"],
        "search_zone" : params["search_zone"],
        "surface_min": params["surface_min"],
        "surface_max": params["surface_max"],
        "min_chambres": params.get("min_chambres"),
        "max_chambres": params.get("max_chambres"),
        "min_pieces": params.get("min_pieces"),
        "max_pieces": params.get("max_pieces"),
        "preference": params.get("preference", ""),
        "start_page": params.get("start_page", 1),
        "end_page": params.get("end_page", 1),
        "use_filters" : params.get("use_filters"),
        "timestamp": datetime.now().isoformat(),
    }

    if "salaire" in params:
        donnees["mode"] = "salaire"
        donnees["salaire"] = params["salaire"]

        if "type_transport" in params:
            donnees["type_transport"] = params["type_transport"]
            if params["type_transport"] == "Transport public":
                donnees["profil"] = params.get("profil")
            else:
                donnees["type_voiture"] = params.get("type_voiture")
                donnees["km_mensuel"] = params.get("km_mensuel")
                donnees["consommation"] = params.get("consommation")

        donnees["cout_transport"] = params.get("cout_transport", 0)
        pourcentage = params.get("pourcentage_transport", 0)
        donnees["pourcentage_transport"] = pourcentage

        donnees["budget_recommande"] = budget_logement(params["salaire"], pourcentage)
        donnees["pourcentage_logement"] = round(40 - pourcentage, 2)

    else:
        donnees["mode"] = "budget_direct"
        donnees["budget_logement"] = params.get("budget_logement", 800)
    return donnees