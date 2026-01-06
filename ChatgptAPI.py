"""
============================================================
"Orchestrateur" principal (prompt + boucle outils)
============================================================

Objectif conceptuel:
- Construire un prompt structuré à partir de:
  (1) recherche utilisateur (message courant)
  (2) mémoire (contexte historique)
  (3) instruction système (règles / comportement)
- Appeler le modèle (LocalLLM.chat_raw) pour générer une réponse.
- Gérer la boucle d'outils (tool-calling):
  - si le modèle demande un outil => exécuter la fonction Python correspondante
  - renvoyer le résultat au modèle via un message role="tool"
  - relancer le modèle pour obtenir la réponse finale

Fonctions utilisées dans ce fichier:
- call_chatgpt_real_estate(...) :
  Fonction centrale: prompt -> appel modèle -> (optionnel) outils -> réponse texte.
"""

from __future__ import annotations

import json
import os
import requests
from getpass import getpass
# from openai import OpenAI
from ChatgptAPI_helper import *  # contient types + LocalLLM + autres helpers (selon ton projet)


def call_chatgpt_real_estate(
    model,  # instance du modèle local (ou objet compatible)
    recherche: str,
    memoire: str,
    instruction: str = "Tu es un assistant specialise en analyse d'annonces immobilieres. Reponds en francais, de facon concise.",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_map: Optional[Dict[str, Any]] = None,
    tool_choice: Optional[str] = "auto",
    max_tokens: int = 320,
    temperature: float = 0.4,
    max_tool_rounds: int = 2,
) -> str:
    """
    Concept:
    - On envoie au LLM un "message system" (instruction) + un "message user" (payload).
    - Le payload contient la recherche ET la mémoire historique (texte).
    - Si le LLM renvoie tool_calls, on exécute les fonctions Python correspondantes (tool_map)
      et on renvoie leurs résultats au LLM. Puis le LLM finalise la réponse.

    Paramètres:
    - model: instance LocalLLM (ou compat) qui expose chat_raw(...)
    - recherche: demande courante de l'utilisateur (ce qu'il veut maintenant)
    - memoire: transcript / résumé des tours précédents (contexte)
    - instruction: règles + style + garde-fous (message système)
    - tools: schémas des outils que le modèle peut appeler (format OpenAI-like)
    - tool_map: mapping {"nom_outil": fonction_python_callable}
    - tool_choice: "auto" => le modèle décide, None => pas de tool-calling
    - max_tool_rounds: nombre max d'itérations outil->modèle pour éviter boucles infinies
    """

    # --------------------------------------------------------
    # 1) Construction d'un payload textuel structuré
    # --------------------------------------------------------
    # Concept: on délimite clairement les sections pour aider le modèle à:
    # - distinguer "ce qui est demandé maintenant" (recherche) vs "contexte passé" (memoire)
    # - suivre la checklist de la tâche
    user_payload = f"""\
[RECHERCHE UTILISATEUR]
{recherche}

[MEMOIRE / HISTORIQUE]
{memoire}

Tache:
- Utilise l'outil uniquement si l'utilisateur demande explicitement les annonces.
- Si use_filters == true, applique les filtres: surface_min 0, surface_max 100, min_chambres 0, max_chambres 2, min_pieces 0, max_pieces 3.
- Si use_filters == false, n'evalue pas ces filtres.
- Le reste des criteres doit respecter exactement les preferences de l'utilisateur.
- Recommande les annonces les plus pertinentes.
- Justifie en 1-2 phrases par annonce.
- Si infos manquantes, pose 2 a 4 questions.
- Si possible, renvoie un petit TOP 3.
""".strip()

    # --------------------------------------------------------
    # 2) Messages envoyés au modèle (chat format)
    # --------------------------------------------------------
    # Concept: pour un serveur "OpenAI-compatible", on utilise:
    # - system: règles + comportement
    # - user: contenu à traiter maintenant
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_payload},
    ]

    # --------------------------------------------------------
    # 3) Fonction interne: appel brut au modèle
    # --------------------------------------------------------
    # Concept: on centralise l'appel au modèle au même endroit, avec ou sans tools.
    # Si tools est None => on n'envoie pas tool_choice (évite comportement imprévisible côté serveur).
    def _chat() -> Dict[str, Any]:
        return model.chat_raw(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,  # schéma: ce que le LLM a le droit d'appeler
            tool_choice=tool_choice if tools else None,
        )

    # --------------------------------------------------------
    # 4) Boucle d'outils: outil(s) -> résultat(s) -> réponse finale
    # --------------------------------------------------------
    # Concept: un LLM peut:
    # - répondre directement (pas de tool_calls)
    # - OU demander un/des outil(s) (tool_calls), puis répondre après avoir reçu les résultats.
    response = _chat()

    for _ in range(max_tool_rounds):
        # tool_calls est généralement une liste (si dict OpenAI-like)
        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None

        # Cas 1: pas d'outil demandé => on sort avec le texte final
        if not tool_calls:
            return str(response.get("content", "")).strip()

        # Cas 2: outils demandés mais pas de map => impossible d'exécuter => on sort du loop
        if not tool_map:
            break

        # Important: ajouter le message assistant "tool_calls" à l'historique
        # pour que le modèle "voit" qu'il a demandé un outil.
        messages.append(response)

        # ----------------------------------------------------
        # 5) Exécution des outils demandés
        # ----------------------------------------------------
        for tool_call in tool_calls:
            # tool_call contient une description de la fonction + arguments
            func = tool_call.get("function", {})
            name = func.get("name")

            # arguments est souvent une string JSON (ex: '{"k": 3}')
            args_str = func.get("arguments", "") or "{}"
            try:
                args = json.loads(args_str)  # conversion en dict Python
            except json.JSONDecodeError:
                args = {}  # fallback si le modèle a produit un JSON invalide

            # ------------------------------------------------
            # 6) Choix + exécution de la fonction Python
            # ------------------------------------------------
            if tool_map.get(name) is None:
                tool_output = f"ERREUR: outil '{name}' introuvable."
            else:
                try:
                    tool_output = tool_map[name](**args)  # exécute l'outil avec args
                except Exception as exc:
                    tool_output = f"ERREUR: {exc}"

            # ------------------------------------------------
            # 7) Retour du résultat au modèle (role="tool")
            # ------------------------------------------------
            # Concept: "tool_call_id" relie la réponse au tool_call correspondant.
            # Le LLM utilisera ce contenu dans la prochaine génération.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": str(tool_output),
                }
            )

        # ----------------------------------------------------
        # 8) On relance le modèle après les outils
        # ----------------------------------------------------
        response = _chat()

    # Fallback final si on sort de boucle (ex: pas de tool_map, trop d'itérations, etc.)
    return str(response.get("content", "")).strip()


def call_chatgpt_real_estate_openai(
    api_key: str,
    recherche: str,
    memoire: str,
    instruction: str = "Tu es un assistant specialise en analyse d'annonces immobilieres. Reponds en francais, de facon concise.",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_map: Optional[Dict[str, Any]] = None,
    tool_choice: Optional[str] = "auto",
    max_tokens: int = 320,
    temperature: float = 0.4,
    max_tool_rounds: int = 2,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
) -> str:
    """
    Variante en ligne: utilise une cle API OpenAI compatible.
    Meme logique que la version locale, avec support des outils.
    """
    if not api_key:
        raise ValueError("API key manquante pour le mode en ligne.")

    user_payload = f"""\
[RECHERCHE UTILISATEUR]
{recherche}

[MEMOIRE / HISTORIQUE]
{memoire}

Tache:
- Utilise l'outil uniquement si l'utilisateur demande explicitement les annonces.
- Si use_filters == true, applique les filtres: surface_min 0, surface_max 100, min_chambres 0, max_chambres 2, min_pieces 0, max_pieces 3.
- Si use_filters == false, n'evalue pas ces filtres.
- Le reste des criteres doit respecter exactement les preferences de l'utilisateur.
- Recommande les annonces les plus pertinentes.
- Justifie en 1-2 phrases par annonce.
- Si infos manquantes, pose 2 a 4 questions.
- Si possible, renvoie un petit TOP 3.
""".strip()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_payload},
    ]

    def _chat() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tools and tool_choice is not None:
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    response = _chat()
    for _ in range(max_tool_rounds):
        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
        if not tool_calls:
            return str(response.get("content", "")).strip()
        if not tool_map:
            break

        messages.append(response)
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name")
            args_str = func.get("arguments", "") or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            if tool_map.get(name) is None:
                tool_output = f"ERREUR: outil '{name}' introuvable."
            else:
                try:
                    tool_output = tool_map[name](**args)
                except Exception as exc:
                    tool_output = f"ERREUR: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": str(tool_output),
                }
            )

        response = _chat()

    return str(response.get("content", "")).strip()
