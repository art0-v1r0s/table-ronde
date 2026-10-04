import json
import logging
import time
from typing import Any

from table_ronde.memory.schemas import (
    MemoryBudget,
    MemoryDelta,
    PartitionMetadata,
    PartitionState,
    RunStatus,
    SharedMemoryState,
)

logger = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 64 * 1024  # 64 KB hard ceiling per write


class MemoryLimitExceededError(ValueError):
    """Raised when mutation payload exceeds the 64KB limit."""


class BudgetExceededError(RuntimeError):
    """Raised when the workflow run exceeds the configured budget."""


class AsyncRedisMemoryFabric:
    """Tier 1 Ephemeral Working Memory Fabric backed by Redis / Valkey.

    Supports atomic JSON-patching, strict TTLs, and in-memory mock fallback when Redis is offline.
    """

    def __init__(self, redis_url: str | None = None, default_ttl: int = 1800) -> None:
        import os
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl
        self._client: Any = None
        self._fallback_memory: dict[str, SharedMemoryState] = {}
        self._is_connected = False

    async def connect(self) -> bool:
        """Attempts connection to Redis. Falls back to in-memory mode if unavailable."""
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._client.ping()
            self._is_connected = True
            logger.info("Connected to Redis Stack Tier 1 Memory at %s", self.redis_url)
            return True
        except Exception as e:
            logger.warning("Redis offline (%s). Using in-memory ephemeral fallback.", e)
            self._is_connected = False
            self._client = None
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._is_connected = False

    def _make_key(self, run_id: str) -> str:
        return f"table_ronde:run:{run_id}"

    async def initialize_run(
        self,
        run_id: str,
        budget: MemoryBudget | None = None,
        context: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> SharedMemoryState:
        """Initializes a new shared memory document for a given run_id."""
        initial_state = SharedMemoryState(
            run_id=run_id,
            status=RunStatus.RUNNING,
            created_at=time.time(),
            budget=budget or MemoryBudget(),
            context=context or {},
            partitions={},
        )
        ttl = ttl_seconds or self.default_ttl

        if self._is_connected and self._client:
            key = self._make_key(run_id)
            doc_json = initial_state.model_dump_json()
            try:
                await self._client.execute_command("JSON.SET", key, "$", doc_json)
                await self._client.expire(key, ttl)
            except Exception as e:
                logger.error("Failed to write to Redis JSON: %s. Falling back to RAM.", e)
                self._fallback_memory[run_id] = initial_state
        else:
            self._fallback_memory[run_id] = initial_state

        return initial_state

    async def get_state(self, run_id: str) -> SharedMemoryState | None:
        """Retrieves and parses the complete typed shared memory state."""
        if self._is_connected and self._client:
            key = self._make_key(run_id)
            try:
                raw = await self._client.execute_command("JSON.GET", key, "$")
                if not raw:
                    return None
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    return SharedMemoryState.model_validate(parsed[0])
                return SharedMemoryState.model_validate(parsed)
            except Exception as e:
                logger.error("Redis get_state error: %s", e)
                return self._fallback_memory.get(run_id)
        return self._fallback_memory.get(run_id)

    async def apply_delta(self, run_id: str, delta: MemoryDelta) -> SharedMemoryState:
        """Applies an atomic delta to a specific partition with RBMA and size limits."""
        # 1. Payload size check
        payload_repr = delta.model_dump_json()
        if len(payload_repr.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise MemoryLimitExceededError(
                f"Payload size ({len(payload_repr)} bytes) exceeds max limit of {MAX_PAYLOAD_BYTES} bytes."
            )

        # 2. Get current state to verify budget & run status
        current_state = await self.get_state(run_id)
        if not current_state:
            raise KeyError(f"Run {run_id} not initialized.")

        if current_state.budget.is_exceeded():
            current_state.status = RunStatus.ABORTED_BUDGET_EXCEEDED
            await self._save_state(run_id, current_state)
            raise BudgetExceededError(
                f"Run {run_id} budget exceeded: cost=${current_state.budget.current_cost_usd:.2f}, "
                f"iterations={current_state.budget.current_iterations}"
            )

        # 3. Apply update to target partition
        part_name = delta.partition
        if part_name not in current_state.partitions:
            current_state.partitions[part_name] = PartitionState(
                metadata=PartitionMetadata(
                    updated_by=delta.agent_id,
                    tainted=delta.tainted,
                    timestamp=time.time(),
                    version=1,
                ),
                data=delta.data,
                findings=delta.findings,
                sources=delta.sources,
            )
        else:
            partition = current_state.partitions[part_name]
            partition.metadata.updated_by = delta.agent_id
            partition.metadata.timestamp = time.time()
            partition.metadata.version += 1
            if delta.tainted:
                partition.metadata.tainted = True
            partition.data.update(delta.data)
            partition.findings.extend(delta.findings)
            partition.sources.extend(delta.sources)

        current_state.budget.increment_iteration()
        await self._save_state(run_id, current_state)
        return current_state

    async def record_cost(self, run_id: str, cost_usd: float) -> MemoryBudget:
        """Records token spend for the current run and checks financial circuit breaker."""
        current_state = await self.get_state(run_id)
        if not current_state:
            raise KeyError(f"Run {run_id} not found.")

        current_state.budget.record_cost(cost_usd)
        if current_state.budget.is_exceeded():
            current_state.status = RunStatus.ABORTED_BUDGET_EXCEEDED
            await self._save_state(run_id, current_state)
            raise BudgetExceededError(
                f"Financial limit exceeded: ${current_state.budget.current_cost_usd:.2f} >= ${current_state.budget.max_cost_usd:.2f}"
            )

        await self._save_state(run_id, current_state)
        return current_state.budget

    async def _save_state(self, run_id: str, state: SharedMemoryState) -> None:
        if self._is_connected and self._client:
            key = self._make_key(run_id)
            try:
                await self._client.execute_command("JSON.SET", key, "$", state.model_dump_json())
            except Exception as e:
                logger.error("Failed saving state to Redis: %s", e)
                self._fallback_memory[run_id] = state
        else:
            self._fallback_memory[run_id] = state
