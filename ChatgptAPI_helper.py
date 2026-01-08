import pandas as pd
import json
from pathlib import Path
from typing import Any, List, Union, Dict, Optional
from dataclasses import dataclass
import requests

@dataclass
class MemoryState:
    text: str = ""
    turns: int = 0

@dataclass
class LocalLLM:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "dolphin-2.2.1-mistral-7b"
    timeout: int = 360

    def __post_init__(self) -> None:
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

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

def df_to_markdown(
    df: "pd.DataFrame",
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    index: bool = False,
) -> str:
    out = df.copy()
    if max_cols:
        if out.shape[1] > max_cols:
            out = out.iloc[:, :max_cols]
    if max_rows:
        if out.shape[0] > max_rows:
            out = out.iloc[:max_rows, :]

    def _cell(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)
    out = out.map(_cell)
    return out.to_markdown(index=index)

def update_conversation_memory(
    memory: Union[str, MemoryState, None, List[Any], Dict[str, Any]],
    *,
    role: str,
    content: str,
    max_chars: int = 20000,
    keep_last_turns: int = 12,
) -> MemoryState:

    if memory is None:
        state = MemoryState(text="", turns=0)

    elif isinstance(memory, MemoryState):
        state = memory

    elif isinstance(memory, str):
        state = MemoryState(text=memory.strip(), turns=0)

    elif isinstance(memory, list):
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

    role = role.strip().lower()
    if role not in {"user", "assistant", "system", "developer"}:
        role = "user"

    new_block = f"\n\n[{role.upper()}]\n{content.strip()}"
    combined = (state.text + new_block).strip()

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

    if len(truncated) > max_chars:
        # On coupe par la fin (plus récent) puis on enlève les blancs initiaux
        truncated = truncated[-max_chars:].lstrip()

    state.text = truncated
    state.turns += 1
    return state
