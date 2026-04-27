from collections.abc import Callable
from typing import Any

try:
    from neo4j import Driver
except ImportError:  # pragma: no cover
    Driver = Any


class GraphClient:
    def __init__(self, driver: Driver):
        self._driver = driver

    def execute_write(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not configured")
        with self._driver.session() as session:
            return session.execute_write(func, *args, **kwargs)

    def execute_read(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not configured")
        with self._driver.session() as session:
            return session.execute_read(func, *args, **kwargs)

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not configured")
        with self._driver.session() as session:
            result = session.run(query, **params)
            # Use record keys/values directly — record.data() converts graph types to
            # plain dicts and drops Node/Relationship APIs needed by subgraph builders.
            return [{key: record[key] for key in record.keys()} for record in result]

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

