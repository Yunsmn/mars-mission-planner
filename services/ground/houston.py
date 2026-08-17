"""HOUSTON — the ground segment. A cloud IBM Granite model (watsonx.ai) in the role Earth actually
holds: the switchboard with knowledge. It never touches the actuators — it is FARTHER from them than
MARVIN, never closer. Three jobs, in order: (1) route every operator message (answer here, or relay
to the rover), (2) expand a one-sentence intent into a full mission briefing, (3) short general
space Q&A.

Backend is pluggable so the whole flow is testable without cloud credentials:
  - watsonx.ai (default when WATSONX_API_KEY/PROJECT_ID are set) — the real ground model, via REST.
  - local Ollama Granite (dev fallback) — same model family, so the architecture is exercisable
    offline. The Earth/Mars distinction is which *service* and whether the light-time link is paid,
    not which weights answer — that stays true either way.

Voice: terse, procedural, NASA radio-speak. No flourish, no emoji. MARVIN is the star.
"""
from __future__ import annotations

import os
import time

import requests

from services.ground import briefing

_DEGRADED = "Ground segment degraded — retry your last transmission."


class Houston:
    def __init__(self, backend: str | None = None, ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "granite4.1:3b"):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        has_wx = bool(os.getenv("WATSONX_API_KEY") and os.getenv("WATSONX_PROJECT_ID"))
        self.backend = backend or ("watsonx" if has_wx else "ollama")
        self._wx_token = None
        self._wx_token_exp = 0.0

    @property
    def backend_label(self) -> str:
        return "watsonx.ai Granite (cloud)" if self.backend == "watsonx" else \
               "local Granite (dev fallback)"

    # ---- generation backends ----------------------------------------------------
    def _generate(self, prompt: str, max_tokens: int = 700) -> str:
        try:
            return (self._watsonx(prompt, max_tokens) if self.backend == "watsonx"
                    else self._ollama(prompt, max_tokens))
        except Exception:
            try:                                  # one retry, per the failure-mode spec
                time.sleep(0.5)
                return (self._watsonx(prompt, max_tokens) if self.backend == "watsonx"
                        else self._ollama(prompt, max_tokens))
            except Exception:
                return ""                         # caller emits the canned degraded line

    def _ollama(self, prompt: str, max_tokens: int) -> str:
        r = requests.post(f"{self.ollama_host}/api/generate", timeout=120, json={
            "model": self.ollama_model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens}})
        r.raise_for_status()
        return r.json().get("response", "")

    def _watsonx_token(self) -> str:
        if self._wx_token and time.time() < self._wx_token_exp - 60:
            return self._wx_token
        r = requests.post("https://iam.cloud.ibm.com/identity/token", timeout=30,
                          headers={"Content-Type": "application/x-www-form-urlencoded"},
                          data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                                "apikey": os.environ["WATSONX_API_KEY"]})
        r.raise_for_status()
        tok = r.json()
        self._wx_token, self._wx_token_exp = tok["access_token"], time.time() + tok.get("expires_in", 3600)
        return self._wx_token

    def _watsonx(self, prompt: str, max_tokens: int) -> str:
        region = os.getenv("WATSONX_REGION", "us-south")
        url = f"https://{region}.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-01"
        r = requests.post(url, timeout=120,
                          headers={"Authorization": f"Bearer {self._watsonx_token()}",
                                   "Content-Type": "application/json"},
                          json={"model_id": os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"),
                                "project_id": os.environ["WATSONX_PROJECT_ID"],
                                "input": prompt,
                                "parameters": {"decoding_method": "greedy",
                                               "max_new_tokens": max_tokens}})
        r.raise_for_status()
        return r.json()["results"][0]["generated_text"]

    # ---- job 1: routing ---------------------------------------------------------
    def route(self, message: str, delay_s: float = 12.0) -> dict:
        """Decide: answer here (Earth-side) or relay to MARVIN (Mars-side). Returns
        {target: 'houston'|'marvin', reply: str}. Only Mars answers Mars questions."""
        prompt = (
            "You are HOUSTON, NASA ground control for a Mars rover. Terse, procedural radio-speak, "
            "1-3 lines, no emoji.\n"
            "Route this operator message. MARVIN (the rover, answers after a light-time delay) owns "
            "anything that needs being ON the rover: terrain, sensors, battery/power, what it sees, "
            "route options, live status. A mission objective ('get me a sample...', 'go to...', "
            "'collect...') also goes to MARVIN (you will brief it). YOU (instant) handle general "
            "space/Earth questions and recaps you already know.\n\n"
            f'OPERATOR: "{message}"\n\n'
            "Reply with EXACTLY:\n"
            "ROUTE: <marvin|houston>\n"
            "REPLY: <if houston, your answer; if marvin, a short relay acknowledgement>"
        )
        text = self._generate(prompt, 300)
        if not text:
            return {"target": "houston", "reply": _DEGRADED}
        target, reply = "houston", ""
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith("ROUTE:"):
                target = "marvin" if "marvin" in s.lower() else "houston"
            elif s.upper().startswith("REPLY:"):
                reply = s.split(":", 1)[1].strip()
        if target == "marvin":
            reply = reply or "Copy. Relaying to MARVIN."
            if "uplink" not in reply.lower():
                reply = f"{reply} Uplink window in {int(delay_s)}s."
        return {"target": target, "reply": reply or _DEGRADED}

    # ---- job 2: briefing generation ---------------------------------------------
    def brief(self, operator_intent: str, scenario_context: str, mission_id: str,
              delay_s: float = 12.0) -> dict | None:
        """Expand one sentence into a validated mission briefing. Re-asks once on schema failure.
        Returns a normalized briefing dict, or None if the model can't produce a valid one."""
        base = (
            "You are HOUSTON, NASA ground control. Expand the operator's intent into a structured "
            "mission briefing as JSON ONLY (no prose, no fences).\n"
            "You have ORBITAL data only — a DEM and known mineralogy. You CANNOT see local soft-sand "
            "hazards on the surface.\n"
            f"Known science targets in this area: {scenario_context}\n"
            f'OPERATOR INTENT: "{operator_intent}"\n\n'
            "Match this schema exactly:\n" + briefing.schema_hint() + "\n\n"
            "Rules: objective types are sample|image|drive|observe. route_advisory must suggest the "
            "SHORTEST/most direct approach from the orbital DEM (you can't see soft sand), basis "
            "'orbital', binding false. science_rationale: one real-planetary-science sentence each."
        )
        prompt = base
        for attempt in range(2):
            data = briefing.parse_model_json(self._generate(prompt, 800))
            if data is not None:
                errs = briefing.validate(data)
                if not errs:
                    return briefing.normalize(data, mission_id=mission_id,
                                              operator_intent=operator_intent, delay_s=delay_s)
                prompt = base + f"\n\nYour previous attempt had errors: {errs}. Fix them; JSON only."
        return None
