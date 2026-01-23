"""
Query trace models for tracking NL2SQL pipeline execution.

These models provide comprehensive tracing and debugging capabilities
for the entire query processing pipeline.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from .base_enums import QueryStatus, PipelineStepName, PipelineStepStatus


class PipelineStep(BaseModel):
    """Represents a single step in the NL2SQL processing pipeline."""

    step_name: PipelineStepName = Field(..., description="Name of the pipeline step")
    status: PipelineStepStatus = Field(..., description="Step execution status")
    start_time: datetime = Field(..., description="Step start timestamp")
    end_time: Optional[datetime] = Field(default=None, description="Step completion timestamp")
    duration_ms: Optional[float] = Field(default=None, description="Step execution duration in milliseconds")

    # Step data
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input data for this step (sanitized for logging)"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data from this step (sanitized for logging)"
    )

    # Error handling
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if step failed"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error that occurred"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional step-specific metadata"
    )

    def mark_completed(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Mark the step as completed."""
        self.end_time = datetime.now(timezone.utc)
        self.status = PipelineStepStatus.COMPLETED
        if output_data:
            self.output_data = output_data
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def mark_failed(self, error_message: str, error_type: Optional[str] = None) -> None:
        """Mark the step as failed."""
        self.end_time = datetime.now(timezone.utc)
        self.status = PipelineStepStatus.FAILED
        self.error_message = error_message
        self.error_type = error_type
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class QueryTrace(BaseModel):
    """Comprehensive trace of a complete NL2SQL query processing pipeline."""

    # Basic identification
    trace_id: str = Field(..., description="Unique identifier for this trace")
    user_query: str = Field(..., description="Original natural language query")

    # Execution tracking
    status: QueryStatus = Field(..., description="Overall query execution status")
    start_time: datetime = Field(..., description="Query processing start timestamp")
    end_time: Optional[datetime] = Field(default=None, description="Query processing completion timestamp")
    total_duration_ms: Optional[float] = Field(default=None, description="Total processing time in milliseconds")

    # Pipeline steps
    steps: List[PipelineStep] = Field(
        default_factory=list,
        description="Ordered list of pipeline steps executed"
    )

    # Results
    generated_sql: Optional[str] = Field(default=None, description="Final generated SQL query")
    validation_errors: Optional[List[str]] = Field(default=None, description="SQL validation errors")
    execution_results: Optional[Dict[str, Any]] = Field(default=None, description="Query execution results metadata")

    # Schema retrieval info
    retrieved_schema_elements: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Schema elements retrieved for context"
    )
    schema_retrieval_query: Optional[str] = Field(
        default=None,
        description="Query used for schema retrieval"
    )

    # Generation info
    generation_attempts: int = Field(default=0, description="Number of SQL generation attempts")
    final_prompt_tokens: Optional[int] = Field(default=None, description="Token count of final prompt")

    # Error information
    error_message: Optional[str] = Field(default=None, description="Final error message if query failed")
    error_step: Optional[PipelineStepName] = Field(default=None, description="Pipeline step where error occurred")

    # Configuration used
    config_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration values used for this query"
    )

    def add_step(self, step_name: PipelineStepName, input_data: Optional[Dict[str, Any]] = None) -> PipelineStep:
        """Add a new pipeline step and return it for tracking."""
        step = PipelineStep(
            step_name=step_name,
            status=PipelineStepStatus.STARTED,
            start_time=datetime.now(timezone.utc),
            input_data=input_data
        )
        self.steps.append(step)
        return step

    def mark_completed(self, generated_sql: Optional[str] = None) -> None:
        """Mark the entire query trace as completed."""
        self.end_time = datetime.now(timezone.utc)
        self.status = QueryStatus.COMPLETED
        if generated_sql:
            self.generated_sql = generated_sql
        if self.end_time and self.start_time:
            self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def mark_failed(self, error_message: str, error_step: Optional[PipelineStepName] = None) -> None:
        """Mark the entire query trace as failed."""
        self.end_time = datetime.now(timezone.utc)
        self.status = QueryStatus.FAILED
        self.error_message = error_message
        self.error_step = error_step
        if self.end_time and self.start_time:
            self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def get_step_by_name(self, step_name: PipelineStepName) -> Optional[PipelineStep]:
        """Get a pipeline step by name."""
        for step in self.steps:
            if step.step_name == step_name:
                return step
        return None

    def get_total_step_duration_ms(self) -> float:
        """Calculate total duration of all completed steps."""
        total_duration = 0.0
        for step in self.steps:
            if step.duration_ms:
                total_duration += step.duration_ms
        return total_duration

    def get_step_summary(self) -> Dict[str, Any]:
        """Get a summary of all pipeline steps."""
        return {
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s.status == PipelineStepStatus.COMPLETED]),
            "failed_steps": len([s for s in self.steps if s.status == PipelineStepStatus.FAILED]),
            "step_names": [s.step_name for s in self.steps],
            "total_step_duration_ms": self.get_total_step_duration_ms()
        }