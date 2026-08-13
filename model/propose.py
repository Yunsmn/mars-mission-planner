"""The proposer — a LOCAL model (IBM Granite 4.1 via Ollama) that ONLY proposes action sequences.

It never executes anything. Malformed or invalid proposals are dropped, never crash the loop.
The surrogate + gate are what make trusting a local model acceptable.

SCALE: Real world is ±5m, targets ~1-2m away. Validate all coordinates to terrain bounds.

Authored by IBM Bob for the MARVIN mission planner.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from common.types import Action, ActionKind, ActionSeq, MissionState, Perception

logger = logging.getLogger(__name__)

# Real-world terrain bounds (match world/sim.py)
TERRAIN_RADIUS = 5.0  # world spans [-5, 5] m

# The planning procedure the model must follow (model/SKILL.md). Loaded once and injected into
# every prompt so the compact model follows a reliable process instead of improvising.
from pathlib import Path as _Path
try:
    _SKILL = (_Path(__file__).resolve().parent / "SKILL.md").read_text(encoding="utf-8")
except Exception:
    _SKILL = ""


class Proposer:
    """IBM Granite proposer via Ollama API (local, offline)."""
    
    def __init__(self, name: str, host: str, temperature: float) -> None:
        self.name = name
        self.host = host
        self.temperature = temperature
        self.endpoint = f"{host}/api/generate"
    
    def propose(self, state: MissionState, perception: Perception, k: int) -> list[ActionSeq]:
        """Query Gemma 4 for k candidate action sequences; parse + validate to schema.
        
        Args:
            state: Current mission state (pose, battery, targets, etc.)
            perception: Current perception (slope, roughness, visible targets)
            k: Number of candidate sequences to generate
        
        Returns:
            List of valid ActionSeq tuples (may be < k if some are invalid)
        """
        prompt = self._build_prompt(state, perception, k)
        
        try:
            response = self._call_ollama(prompt)
            candidates = self._parse_response(response, state, perception)
            
            # Limit to k candidates
            return candidates[:k]
            
        except Exception as e:
            logger.error(f"Proposer failed: {e}")
            # Return safe fallback: HOLD action
            return [self._safe_hold()]
    
    def _build_prompt(self, state: MissionState, perception: Perception, k: int) -> str:
        """Build the prompt for Gemma 4."""
        # Calculate distances to targets and format with distance info
        import math
        targets_with_dist = []
        for t in perception.visible_targets:
            dist = math.hypot(t.xy[0] - state.pose.xy[0], t.xy[1] - state.pose.xy[1])
            in_range = "IN SAMPLING RANGE" if dist < 0.6 else f"{dist:.1f}m away"
            targets_with_dist.append(
                f"  - {t.id}: xy=({t.xy[0]:.1f}, {t.xy[1]:.1f}), "
                f"distance={in_range}, "
                f"science_value={t.science_value:.2f}, mineral={t.mineral_class}"
            )
        
        targets_str = "\n".join(targets_with_dist)
        
        # Format already collected
        collected_str = ", ".join(state.collected) if state.collected else "none"

        # Environment context (weather/dust + local hazard) so Granite plans with the full picture
        _slope = getattr(perception, "slope_deg", None)
        slope_max = float(_slope.max()) if _slope is not None and getattr(_slope, "size", 0) else 0.0
        
        # Check if any target is in sampling range
        sampling_ready = []
        for t in perception.visible_targets:
            dist = math.hypot(t.xy[0] - state.pose.xy[0], t.xy[1] - state.pose.xy[1])
            if dist < 0.6:
                sampling_ready.append(f"*** TARGET {t.id} IS IN SAMPLING RANGE (distance={dist:.2f}m < 0.6m) - SAMPLE IT NOW! ***")
        
        sampling_alert = "\n".join(sampling_ready) if sampling_ready else ""
        
        prompt = f"""{_SKILL}

=== Use the notes above as guidance — you make the call. Propose {k} candidate action sequences for the state below. ===

CURRENT STATE:
- Position: ({state.pose.xy[0]:.1f}, {state.pose.xy[1]:.1f})
- Heading: {state.pose.heading_rad:.2f} rad
- Battery: {state.battery_pct:.1f}%
- Localization uncertainty: {state.localization_sigma:.2f} m
- Sol time: {state.sol_time:.2f}
- Collected samples: {collected_str}
- Samples needed: {2 - len(state.collected)} more to complete mission

ENVIRONMENT (onboard sensors):
- Dust opacity tau: {perception.dust_tau:.2f} (higher dust -> less solar power -> conserve battery)
- Local terrain slope: up to {slope_max:.1f} deg (avoid steep/rough terrain to reduce risk)

{sampling_alert}

VISIBLE TARGETS:
{targets_str}

TERRAIN CONSTRAINTS:
- World bounds: x,y in [-5.0, 5.0] meters
- Targets are typically 1-2 meters away
- Battery reserve: must keep >15%

AVAILABLE ACTIONS:
1. DRIVE to (x, y) - navigate to coordinates (must be within ±5m)
2. SAMPLE target_id - collect sample at current location (use exact target id)
   - IMPORTANT: If rover is within ~0.6m of a target, SAMPLE it immediately
   - Sampling range: approximately 0.6 meters
3. SCAN - refresh perception
4. OBSERVE - take additional observation to reduce uncertainty
5. HOLD - safe default, do nothing

CONSTRAINTS:
- All DRIVE coordinates must be within [-5.0, 5.0] range
- SAMPLE must use exact target ids from visible targets
- Battery reserve: must keep >15% battery
- Risk ceiling: avoid high-slope/rough terrain
- Prioritize high science value targets

SAMPLING STRATEGY:
- Check distance to each visible target
- If within ~0.6m of an uncollected target, include SAMPLE in your sequence
- Drive-then-sample sequences are efficient for nearby targets
- Mission goal: collect 2 samples

Generate {k} diverse candidate sequences. Each sequence should be 1-3 actions.
Consider: 
- If near a target (<0.6m), sample it
- Direct drive-and-sample routes to nearby high-value targets
- Cautious routes with scans for distant targets

OUTPUT FORMAT - JSON array only, no other text:
[
  [{{"action": "DRIVE", "params": {{"xy": [1.6, 0.8]}}}}, {{"action": "SAMPLE", "params": {{"target": "sample_a"}}}}]
]
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and return the response text."""
        payload = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,   # Ollama reads temperature under options
                "num_predict": 512,                 # cap output tokens for speed
            },
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=120  # Reduced timeout with token cap
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    def _parse_response(
        self,
        response: str,
        state: MissionState,
        perception: Perception
    ) -> list[ActionSeq]:
        """Parse and validate the model's response into ActionSeq tuples.
        
        Drops malformed or invalid proposals without crashing.
        Validates coordinates to terrain bounds and target ids.
        """
        candidates = []
        
        try:
            # Extract JSON from response (model might add extra text)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in response")
                return [self._safe_hold()]
            
            json_str = response[json_start:json_end]
            sequences = json.loads(json_str)
            
            if not isinstance(sequences, list):
                logger.warning("Response is not a list")
                return [self._safe_hold()]
            
            # Parse each sequence
            for seq_data in sequences:
                if not isinstance(seq_data, list):
                    continue
                
                actions = []
                valid = True
                
                for action_data in seq_data:
                    action = self._parse_action(action_data, state, perception)
                    if action is None:
                        valid = False
                        break
                    actions.append(action)
                
                if valid and actions:
                    candidates.append(tuple(actions))
            
            # If no valid candidates, return safe hold
            if not candidates:
                logger.warning("No valid candidates after parsing, using HOLD")
                candidates = [self._safe_hold()]
            
            return candidates
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return [self._safe_hold()]
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return [self._safe_hold()]
    
    def _parse_action(
        self,
        action_data: dict[str, Any],
        state: MissionState,
        perception: Perception
    ) -> Action | None:
        """Parse a single action from JSON data.
        
        Validates:
        - DRIVE coordinates are within ±5m bounds
        - SAMPLE targets exist in visible targets
        """
        try:
            action_name = action_data.get("action", "").upper()
            params = action_data.get("params", {})
            
            # Map action name to ActionKind
            if action_name == "DRIVE":
                kind = ActionKind.DRIVE
                # Validate xy coordinates
                xy = params.get("xy")
                if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                    logger.warning(f"Invalid DRIVE coordinates: {xy}")
                    return None
                
                # Clip to terrain bounds [-5, 5]
                x = float(xy[0])
                y = float(xy[1])
                
                if abs(x) > TERRAIN_RADIUS or abs(y) > TERRAIN_RADIUS:
                    logger.warning(f"DRIVE coordinates out of bounds: ({x:.1f}, {y:.1f}), clipping")
                    x = max(-TERRAIN_RADIUS, min(TERRAIN_RADIUS, x))
                    y = max(-TERRAIN_RADIUS, min(TERRAIN_RADIUS, y))
                
                params = {"xy": (x, y)}
                
            elif action_name == "SAMPLE":
                kind = ActionKind.SAMPLE
                # Validate target exists
                target_id = params.get("target")
                if not target_id:
                    logger.warning("SAMPLE action missing target id")
                    return None
                
                # Check if target is visible
                valid_targets = [t.id for t in perception.visible_targets]
                if target_id not in valid_targets:
                    logger.warning(f"SAMPLE target {target_id} not in visible targets: {valid_targets}")
                    return None
                
                params = {"target": target_id}
                
            elif action_name == "SCAN":
                kind = ActionKind.SCAN
                params = {}
                
            elif action_name == "OBSERVE":
                kind = ActionKind.OBSERVE
                params = {}
                
            elif action_name == "HOLD":
                kind = ActionKind.HOLD
                params = {}
                
            else:
                logger.warning(f"Unknown action: {action_name}")
                return None
            
            return Action(kind=kind, params=params)
            
        except Exception as e:
            logger.error(f"Action parsing failed: {e}")
            return None
    
    def _safe_hold(self) -> ActionSeq:
        """Return a safe HOLD action sequence as fallback."""
        return (Action(kind=ActionKind.HOLD, params={}),)
