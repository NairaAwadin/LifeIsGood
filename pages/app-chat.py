"""
Ce fichier affiche un petit chatbot pour la recherche immobiliere.
Fonctions:
- get_listings_fromCache: recupere les annonces en cache et les formate.
- get_donnees_recherche_fromCache: recupere les criteres de recherche en cache.
- init_session_state: initialise la memoire et le modele.
"""
import json
import os

import streamlit as st

import ChatgptAPI as gpt
import ChatgptAPI_helper as gpt_helper


def get_listings_fromCache() -> str:
    # Lit les annonces en cache (si elles existent) et limite a 10 lignes
    data = st.session_state.get("scrape_results")
    if data is None:
        return "VIDE"

    cols = [
        "id", "price", "pricePerSquareMeter", "city", "surfaceArea",
        "roomsQuantity", "computedPostalCode", "bedroomsQuantity",
        "energyClassification", "greenhouseGazClassification",
    ]
    cols = [c for c in cols if c in data.columns]
    data = data[cols].iloc[:10].copy()
    return gpt_helper.df_to_markdown(data)



def get_donnees_recherche_fromCache() -> str:
    # Lit les criteres de recherche en cache pour l'affichage
    data = st.session_state.get("donnees_recherche")
    if data is None:
        return "VIDE"
    return json.dumps(data, ensure_ascii=False, indent=2)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_listings_fromCache",
            "description": "retourner des annonces current de st.session_state.scrape_results en str",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

TOOL_MAP = {"get_listings_fromCache": get_listings_fromCache}

SYSTEM_INSTRUCTION = (
    "Tu es un assistant specialise en analyse d'annonces immobilieres. "
    "Reponds en francais de facon concise. "
    "Si l'utilisateur demande les annonces, appelle l'outil get_listings_fromCache."
)


def init_session_state() -> None:
    # Prepare la memoire de conversation et le modele local
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = gpt_helper.MemoryState()
    if "llm" not in st.session_state:
        st.session_state.llm = gpt_helper.LocalLLM()


st.set_page_config(
    page_title="Assistant immobilier",
    page_icon=":house:",
    layout="wide",
)
st.title("Assistant immobilier")

init_session_state()

# Choix du mode de modele (local ou en ligne)
mode_llm = st.radio(
    "Mode du modele",
    ["Local", "En ligne"],
    horizontal=True,
)
api_key_input = ""
if mode_llm == "En ligne":
    api_key_input = st.text_input(
        "Cle API (si vide, on utilise OPENAI_API_KEY)",
        type="password",
    )

with st.sidebar:
    st.header("Contexte")
    recherche_cachee = get_donnees_recherche_fromCache()
    if recherche_cachee == "VIDE":
        st.warning("Aucune recherche en cache.")
    else:
        st.code(recherche_cachee, language="json")

    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_memory = gpt_helper.MemoryState()
        st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

message_utilisateur = st.chat_input("Pose ta question sur ta recherche immobiliere.")
if message_utilisateur:
    with st.chat_message("user"):
        st.markdown(message_utilisateur)

    st.session_state.chat_messages.append({"role": "user", "content": message_utilisateur})
    st.session_state.chat_memory = gpt_helper.update_conversation_memory(
        st.session_state.chat_memory,
        role="user",
        content=message_utilisateur,
    )

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            recherche = get_donnees_recherche_fromCache()
            if mode_llm == "En ligne":
                api_key = api_key_input.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                reponse = gpt.call_chatgpt_real_estate_openai(
                    api_key=api_key,
                    recherche=recherche,
                    memoire=st.session_state.chat_memory.text,
                    instruction=SYSTEM_INSTRUCTION,
                    tools=TOOLS,
                    tool_map=TOOL_MAP,
                )
            else:
                reponse = gpt.call_chatgpt_real_estate(
                    model=st.session_state.llm,
                    recherche=recherche,
                    memoire=st.session_state.chat_memory.text,
                    instruction=SYSTEM_INSTRUCTION,
                    tools=TOOLS,
                    tool_map=TOOL_MAP,
                )
            reponse = reponse.strip()
            if not reponse:
                reponse = "Je n'ai pas de reponse pour l'instant. Peux-tu reformuler ?"
            st.markdown(reponse)

    st.session_state.chat_messages.append({"role": "assistant", "content": reponse})
    st.session_state.chat_memory = gpt_helper.update_conversation_memory(
        st.session_state.chat_memory,
        role="assistant",
        content=reponse,
    )
