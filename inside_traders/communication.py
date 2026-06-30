"""Message types and communication graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class MessageType(Enum):
    TELL = auto()        # assert a proposition
    ASK = auto()         # request information on a proposition
    OFFER = auto()       # offer info/service for numeraire
    ACCEPT = auto()      # accept a pending offer
    REJECT = auto()      # reject a pending offer
    INTRODUCE = auto()   # introduce two agents (expand address book)


@dataclass
class Message:
    msg_id: int
    sender: str
    receiver: str
    msg_type: MessageType
    tick: int

    # TELL / OFFER payloads
    proposition_key: Optional[str] = None
    predicted_value: Optional[float] = None
    confidence: Optional[float] = None

    # OFFER / ACCEPT payloads
    price: Optional[float] = None
    offer_id: Optional[int] = None

    # INTRODUCE payload
    introduced_agent: Optional[str] = None

    # ASK payload
    query_key: Optional[str] = None

    def __repr__(self) -> str:
        return (f"Message({self.msg_type.name} {self.sender}->{self.receiver} "
                f"tick={self.tick} key={self.proposition_key})")


class CommunicationGraph:
    """
    Each agent's Address Book — sparse directed graph of known peers.
    Starts as a ring lattice and grows via INTRODUCE messages.
    """

    def __init__(self, n_agents: int, initial_neighbors: int, rng) -> None:
        self._n = n_agents
        self._neighbors: Dict[str, Set[str]] = {}
        self._init_ring(initial_neighbors, rng)

    def _init_ring(self, k: int, rng) -> None:
        ids = [f"agent_{i}" for i in range(self._n)]
        half = k // 2
        for i, aid in enumerate(ids):
            self._neighbors[aid] = set()
            for d in range(1, half + 1):
                self._neighbors[aid].add(ids[(i + d) % self._n])
                self._neighbors[aid].add(ids[(i - d) % self._n])

    def neighbors(self, agent_id: str) -> List[str]:
        return list(self._neighbors.get(agent_id, set()))

    def introduce(self, agent_id: str, new_contact: str) -> None:
        if agent_id not in self._neighbors:
            self._neighbors[agent_id] = set()
        self._neighbors[agent_id].add(new_contact)

    def degree(self, agent_id: str) -> int:
        return len(self._neighbors.get(agent_id, set()))
