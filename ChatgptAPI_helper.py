"""
============================================================
Helpers: mémoire, Markdown, client LLM local
============================================================

Objectif conceptuel:
- Fournir des utilitaires réutilisables pour un chatbot:
  1) MemoryState + update_conversation_memory:
     - stocker un transcript compact de la conversation
     - limiter la taille (nb de tours + nb de caractères)
  2) df_to_markdown:
     - convertir un DataFrame en tableau Markdown lisible (bon pour prompts)
  3) LocalLLM:
     - client HTTP minimal pour parler à un serveur LLM local OpenAI-compatible
     - support optionnel des tools (tools/tool_choice) au niveau request

Fonctions / classes:
- MemoryState: conteneur {text, turns}
- LocalLLM.chat_raw: envoie messages au serveur local (/chat/completions) et renvoie message assistant
- df_to_markdown: transforme DataFrame -> Markdown (avec cellule safe)
- update_conversation_memory: ajoute un tour et tronque intelligemment
"""

import pandas as pd
import json
from pathlib import Path
from typing import Any, List, Union, Dict, Optional
from dataclasses import dataclass
import requests


# ============================================================
# 1) MemoryState: structure simple pour mémoriser le contexte
# ============================================================
@dataclass
class MemoryState:
    """
    Concept:
    - text: transcript compact (ex: "[USER] ... [ASSISTANT] ...")
    - turns: compteur pour savoir combien d'ajouts ont été faits (utile pour stratégies futures)
    """
    text: str = ""
    turns: int = 0


# ============================================================
# 2) LocalLLM: client HTTP pour serveur OpenAI-compatible local
# ============================================================
@dataclass
class LocalLLM:
    # base_url: adresse du serveur local (LM Studio / vLLM / etc.)
    base_url: str = "http://127.0.0.1:1234/v1"
    # model: identifiant du modèle servi par le serveur
    model: str = "dolphin-2.2.1-mistral-7b"
    # timeout: temps max (en secondes) pour éviter que l'app freeze
    timeout: int = 360

    def __post_init__(self) -> None:
        # Nettoie l'URL pour éviter "//" accidentels
        self.base_url = self.base_url.rstrip("/")

    def chat_raw(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 160,
        temperature: float = 0.4,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,  # ex: "auto"
    ) -> Dict[str, Any]:
        """
        Concept:
        - Envoie une requête de chat au serveur local:
          POST {base_url}/chat/completions
        - Renvoie le message assistant "brut" (dict) contenant:
          - content (texte)
          - potentiellement tool_calls (si le serveur supporte tools)
        """

        # ----------------------------------------------------
        # A) Construire le payload OpenAI-like
        # ----------------------------------------------------
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Ajoute outils si fournis: permet au modèle de demander une fonction
        if tools is not None:
            payload["tools"] = tools

        # tool_choice contrôle si le modèle peut/ doit appeler un outil
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # ----------------------------------------------------
        # B) Appel HTTP vers le serveur local
        # ----------------------------------------------------
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        r.raise_for_status()

        # ----------------------------------------------------
        # C) Extraction du message assistant (format OpenAI)
        # ----------------------------------------------------
        # Typiquement: {"choices":[{"message": {...}}]}
        return r.json()["choices"][0]["message"]


# ============================================================
# 3) df_to_markdown: DataFrame -> tableau Markdown
# ============================================================
def df_to_markdown(
    df: "pd.DataFrame",
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    index: bool = False,
) -> str:
    """
    Concept:
    - Les LLM comprennent mieux un tableau structuré (Markdown) qu'un blob texte.
    - On limite lignes/colonnes pour réduire tokens.
    - On "stringifie" chaque cellule pour éviter que dict/list cassent le rendu.
    """
    out = df.copy()

    # --------------------------------------------------------
    # A) Réduire la taille (contrôle token/cout/latence)
    # --------------------------------------------------------
    if max_cols:
        if out.shape[1] > max_cols:
            out = out.iloc[:, :max_cols]
    if max_rows:
        if out.shape[0] > max_rows:
            out = out.iloc[:max_rows, :]

    # --------------------------------------------------------
    # B) Normalisation cellule -> string
    # --------------------------------------------------------
    def _cell(x: Any) -> str:
        # None => cellule vide
        if x is None:
            return ""
        # dict/list => JSON string, lisible, conserve accents (ensure_ascii=False)
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False)
        # fallback: string
        return str(x)

    # map => applique _cell à chaque cellule (élément par élément)
    out = out.map(_cell)  # type: ignore[attr-defined]

    # --------------------------------------------------------
    # C) Conversion finale en Markdown
    # --------------------------------------------------------
    return out.to_markdown(index=index)


# ============================================================
# 4) update_conversation_memory: append + truncate
# ============================================================
def update_conversation_memory(
    memory: Union[str, MemoryState, None, List[Any], Dict[str, Any]],
    *,
    role: str,
    content: str,
    max_chars: int = 20000,
    keep_last_turns: int = 12,
) -> MemoryState:
    """
    Concept:
    - Le "contexte" d'un LLM est limité (tokens).
    - On conserve donc une mémoire courte:
      - on ajoute le nouveau message
      - on garde les derniers "tours"
      - et on applique un plafond max en caractères

    Cette fonction accepte différents formats de memory:
    - None => mémoire vide
    - str => transcript brut
    - MemoryState => déjà structuré
    - list => liste de messages (dicts ou strings)
    - dict => {"text": "...", "turns": ...}
    """

    # --------------------------------------------------------
    # A) Normalisation: convertir n'importe quel input -> MemoryState
    # --------------------------------------------------------
    if memory is None:
        state = MemoryState(text="", turns=0)

    elif isinstance(memory, MemoryState):
        state = memory

    elif isinstance(memory, str):
        state = MemoryState(text=memory.strip(), turns=0)

    elif isinstance(memory, list):
        # Convertit une liste de messages en transcript textuel
        blocks: List[str] = []
        for item in memory:
            if isinstance(item, dict):
                item_role = str(item.get("role", "user")).strip().lower()
                item_content = str(item.get("content", "")).strip()
            else:
                item_role = "user"
                item_content = str(item).strip()

            # Ignore les contenus vides
            if not item_content:
                continue

            # Sécurise le rôle
            if item_role not in {"user", "assistant", "system", "developer"}:
                item_role = "user"

            blocks.append(f"\n\n[{item_role.upper()}]\n{item_content}")

        state = MemoryState(text="".join(blocks).strip(), turns=0)

    elif isinstance(memory, dict):
        # Format dict attendu: {"text": "...", "turns": ...}
        text_val = str(memory.get("text", "")).strip()
        turns_val = memory.get("turns", 0)
        try:
            turns_val = int(turns_val)
        except (TypeError, ValueError):
            turns_val = 0
        state = MemoryState(text=text_val, turns=turns_val)

    elif hasattr(memory, "text"):
        # Support objet "memory-like" (duck typing)
        text_val = str(getattr(memory, "text", "")).strip()
        turns_val = getattr(memory, "turns", 0)
        try:
            turns_val = int(turns_val)
        except (TypeError, ValueError):
            turns_val = 0
        state = MemoryState(text=text_val, turns=turns_val)

    else:
        raise TypeError("memory must be str | MemoryState | None | list | dict")

    # --------------------------------------------------------
    # B) Ajout du nouveau tour au transcript
    # --------------------------------------------------------
    role = role.strip().lower()
    if role not in {"user", "assistant", "system", "developer"}:
        role = "user"

    new_block = f"\n\n[{role.upper()}]\n{content.strip()}"
    combined = (state.text + new_block).strip()

    # --------------------------------------------------------
    # C) Tronquage: garder seulement les derniers "blocs"
    # --------------------------------------------------------
    # On split sur "\n\n[" car notre format est:
    # \n\n[USER]...
    blocks = [b.strip() for b in combined.split("\n\n[") if b.strip()]

    # On reconstruit les blocs en ré-ajoutant le "[" perdu au split
    rebuilt: List[str] = []
    for i, b in enumerate(blocks):
        if i == 0 and b.startswith(("USER]", "ASSISTANT]", "SYSTEM]", "DEVELOPER]")):
            rebuilt.append("[" + b)
        elif i == 0:
            rebuilt.append(b)
        else:
            rebuilt.append("[" + b)

    # Garde les N derniers tours (ex: 12 derniers blocs)
    if len(rebuilt) > keep_last_turns:
        rebuilt = rebuilt[-keep_last_turns:]

    truncated = "\n\n".join(rebuilt).strip()

    # --------------------------------------------------------
    # D) Tronquage final par taille (plafond de caractères)
    # --------------------------------------------------------
    if len(truncated) > max_chars:
        # On coupe par la fin (plus récent) puis on enlève les blancs initiaux
        truncated = truncated[-max_chars:].lstrip()

    # --------------------------------------------------------
    # E) Mise à jour du state
    # --------------------------------------------------------
    state.text = truncated
    state.turns += 1
    return state
