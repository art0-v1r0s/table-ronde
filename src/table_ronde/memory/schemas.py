import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED_BUDGET_EXCEEDED = "ABORTED_BUDGET_EXCEEDED"
    FAILED = "FAILED"


class MemoryBudget(BaseModel):
    max_cost_usd: float = Field(default=2.50, description="Hard financial ceiling per run in USD")
    current_cost_usd: float = Field(default=0.0, description="Total cost incurred so far")
    max_iterations: int = Field(default=25, description="Maximum allowed loop iterations")
    current_iterations: int = Field(default=0, description="Current iteration count")

    def is_exceeded(self) -> bool:
        return (
            self.current_cost_usd >= self.max_cost_usd
            or self.current_iterations >= self.max_iterations
        )

    def record_cost(self, cost_usd: float) -> None:
        self.current_cost_usd += max(cost_usd, 0.0)

    def increment_iteration(self) -> None:
        self.current_iterations += 1


class PartitionMetadata(BaseModel):
    updated_by: str = Field(description="Identifier of the last agent that updated this partition")
    tainted: bool = Field(default=False, description="True if content contains unverified external data")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of last modification")
    version: int = Field(default=1, description="Monotonically increasing version counter")


class PartitionState(BaseModel):
    metadata: PartitionMetadata
    data: dict[str, Any] = Field(default_factory=dict, description="Arbitrary domain-specific structured data")
    findings: list[str] = Field(default_factory=list, description="Verified or unverified textual findings")
    sources: list[str] = Field(default_factory=list, description="Referenced URLs or file paths")


class SharedMemoryState(BaseModel):
    run_id: str = Field(description="Unique workflow execution identifier")
    status: RunStatus = Field(default=RunStatus.RUNNING, description="Current execution status")
    created_at: float = Field(default_factory=time.time, description="Creation epoch timestamp")
    budget: MemoryBudget = Field(default_factory=MemoryBudget, description="Financial and execution budget")
    context: dict[str, Any] = Field(default_factory=dict, description="Global context (original prompt, trace IDs, etc.)")
    partitions: dict[str, PartitionState] = Field(default_factory=dict, description="Agent domain partitions")


class MemoryDelta(BaseModel):
    partition: str = Field(description="Target partition namespace")
    agent_id: str = Field(description="Agent applying the delta")
    data: dict[str, Any] = Field(default_factory=dict, description="Key-values to merge/update")
    findings: list[str] = Field(default_factory=list, description="New findings to append")
    sources: list[str] = Field(default_factory=list, description="New sources to append")
    tainted: bool = Field(default=False, description="Whether incoming delta data is untrusted/tainted")
