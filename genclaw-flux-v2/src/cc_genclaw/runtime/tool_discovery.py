from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set


@dataclass
class DiscoveredToolsState:

    discovered: Set[str] = field(default_factory=set)

    def is_discovered(self, name: str) -> bool:
        return name in self.discovered

    def mark_discovered(self, names: Iterable[str]) -> None:
        for n in names:
            self.discovered.add(n)

    def clear(self) -> None:
        self.discovered.clear()

    def snapshot(self) -> Set[str]:
        return set(self.discovered)


_GLOBAL_DISCOVERED = DiscoveredToolsState()


def get_discovered_tools_state() -> DiscoveredToolsState:
    return _GLOBAL_DISCOVERED


def reset_discovered_tools() -> None:
    _GLOBAL_DISCOVERED.clear()

