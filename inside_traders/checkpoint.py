"""Checkpoint utilities: save and restore agent learned state.

Only the *learned* portions of each agent are serialised — preference
vectors, epistemic credibility scores, and the communication Q-table.
Economic state (cash, positions) is intentionally excluded so that the
joint phase starts with a fair, freshly-seeded market.

Typical usage::

    # After solo pretraining:
    save_checkpoint(agents, "checkpoints/pretrained.json")

    # Before joint simulation:
    state_dicts = load_checkpoint("checkpoints/pretrained.json")
    apply_checkpoint(sim.agents, state_dicts, reset_comm=True)
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List

import numpy as np

from .agent import Agent


def save_checkpoint(agents: List[Agent], path: str) -> None:
    """Serialise agent learned state to *path* (JSON)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"agents": [_state_dict(ag) for ag in agents]}
    with open(path, "w") as fh:
        json.dump(payload, fh)


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    """Return a list of agent state dicts loaded from *path*."""
    with open(path) as fh:
        data = json.load(fh)
    return data["agents"]


def apply_checkpoint(
    agents: List[Agent],
    state_dicts: List[Dict[str, Any]],
    reset_comm: bool = True,
) -> None:
    """Apply saved state dicts onto *agents* in-place.

    Parameters
    ----------
    agents:
        Freshly constructed agent list (from Simulation or build_agents).
    state_dicts:
        Output of :func:`load_checkpoint`.
    reset_comm:
        If ``True`` (default), zero the comm Q-table and reset epsilon to
        0.5 so that communication strategy learning starts from a blank
        slate in the joint phase.  Pass ``False`` to resume Q-learning
        from the checkpoint values (e.g. continued pretraining).
    """
    if len(agents) != len(state_dicts):
        raise ValueError(
            f"Agent count mismatch: simulation has {len(agents)} agents "
            f"but checkpoint contains {len(state_dicts)} entries."
        )
    for ag, sd in zip(agents, state_dicts):
        ag.preference_vector = dict(sd["preference_vector"])
        ag.epistemic_model.credibility = {
            src: dict(scores)
            for src, scores in sd["epistemic_credibility"].items()
        }
        if reset_comm:
            ag._comm_q[:] = 0.0
            ag._comm_q_visits[:] = 0
            ag._comm_epsilon = 0.5
            ag._prev_reward_signal = 0.0
            ag._last_comm_state = None
            ag._last_comm_action = None
        else:
            ag._comm_q = np.array(sd["comm_q"])
            ag._comm_q_visits = np.array(sd["comm_q_visits"], dtype=np.int64)
            ag._comm_epsilon = float(sd["comm_epsilon"])


def _state_dict(ag: Agent) -> Dict[str, Any]:
    return {
        "agent_id": ag.agent_id,
        "preference_vector": dict(ag.preference_vector),
        "epistemic_credibility": {
            src: dict(scores)
            for src, scores in ag.epistemic_model.credibility.items()
        },
        "comm_q": ag._comm_q.tolist(),
        "comm_q_visits": ag._comm_q_visits.tolist(),
        "comm_epsilon": float(ag._comm_epsilon),
    }
