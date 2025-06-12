"""
MCP Workflow Engine
Provides advanced workflow orchestration, task chaining, and automation for MCP operations
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from pathlib import Path
import logging

from sources.utility import pretty_print
from sources.tools.mcp_tools import MCPCommunicator, MCPToolCall, MCPToolResult

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskType(Enum):
    """Types of workflow tasks"""
    MCP_TOOL = "mcp_tool"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    DELAY = "delay"
    CUSTOM = "custom"

class ConditionType(Enum):
    """Types of conditional logic"""
    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"

@dataclass
class WorkflowTask:
    """Individual task in a workflow"""
    id: str
    name: str
    task_type: TaskType
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    on_success: Optional[List[str]] = None
    on_failure: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution context"""
    workflow_id: str
    name: str
    tasks: List[WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_task: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get workflow execution duration"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

class MCPWorkflowEngine:
    """Advanced workflow engine for MCP task orchestration"""
    
    def __init__(self, communicator: MCPCommunicator):
        self.communicator = communicator
        self.workflows: Dict[str, WorkflowExecution] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Built-in task handlers
        self.task_handlers = {
            TaskType.MCP_TOOL: self._execute_mcp_tool,
            TaskType.CONDITION: self._execute_condition,
            TaskType.LOOP: self._execute_loop,
            TaskType.PARALLEL: self._execute_parallel,
            TaskType.DELAY: self._execute_delay,
            TaskType.CUSTOM: self._execute_custom
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "task_started": [],
            "task_completed": [],
            "task_failed": []
        }
    
    def register_workflow_template(self, template_name: str, template: Dict[str, Any]):
        """Register a reusable workflow template"""
        self.workflow_templates[template_name] = template
        pretty_print(f"Registered workflow template: {template_name}", color="info")
    
    def create_workflow_from_template(self, template_name: str, 
                                    parameters: Dict[str, Any] = None) -> str:
        """Create workflow from template"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.workflow_templates[template_name].copy()
        parameters = parameters or {}
        
        # Replace template variables
        workflow_def = self._substitute_template_variables(template, parameters)
        
        return self.create_workflow(workflow_def)
    
    def _substitute_template_variables(self, template: Dict[str, Any], 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in workflow template"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Simple variable substitution
                for key, value in parameters.items():
                    obj = obj.replace(f"${{{key}}}", str(value))
                return obj
            else:
                return obj
        
        return substitute_recursive(template)
    
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow from definition"""
        workflow_id = str(uuid.uuid4())
        
        # Parse tasks
        tasks = []
        for task_def in workflow_definition.get("tasks", []):
            task = WorkflowTask(
                id=task_def["id"],
                name=task_def["name"],
                task_type=TaskType(task_def["type"]),
                parameters=task_def.get("parameters", {}),
                dependencies=task_def.get("dependencies", []),
                conditions=task_def.get("conditions", []),
                max_retries=task_def.get("max_retries", 3),
                timeout=task_def.get("timeout"),
                on_success=task_def.get("on_success"),
                on_failure=task_def.get("on_failure"),
                metadata=task_def.get("metadata", {})
            )
            tasks.append(task)
        
        # Create workflow execution
        workflow = WorkflowExecution(
            workflow_id=workflow_id,
            name=workflow_definition.get("name", f"Workflow-{workflow_id[:8]}"),
            tasks=tasks,
            context=workflow_definition.get("context", {})
        )
        
        self.workflows[workflow_id] = workflow
        pretty_print(f"Created workflow: {workflow.name} ({workflow_id})", color="success")
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow asynchronously"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PENDING:
            raise ValueError(f"Workflow {workflow_id} is not in pending state")
        
        # Start workflow execution
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = time.time()
        
        # Trigger workflow started event
        await self._trigger_event("workflow_started", workflow)
        
        try:
            # Create task dependency graph
            task_graph = self._build_task_graph(workflow.tasks)
            
            # Execute tasks in dependency order
            success = await self._execute_task_graph(workflow, task_graph)
            
            if success:
                workflow.status = WorkflowStatus.COMPLETED
                await self._trigger_event("workflow_completed", workflow)
            else:
                workflow.status = WorkflowStatus.FAILED
                await self._trigger_event("workflow_failed", workflow)
            
            workflow.completed_at = time.time()
            return success
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = time.time()
            workflow.errors.append({
                "type": "workflow_error",
                "message": str(e),
                "timestamp": time.time()
            })
            
            await self._trigger_event("workflow_failed", workflow)
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            return False
        
        finally:
            # Cleanup
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        task_map = {task.id: task for task in tasks}
        
        for task in tasks:
            # Validate dependencies exist
            for dep_id in task.dependencies:
                if dep_id not in task_map:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep_id}")
            
            graph[task.id] = task.dependencies
        
        # Check for circular dependencies
        if self._has_circular_dependencies(graph):
            raise ValueError("Circular dependencies detected in workflow")
        
        return graph
    
    def _has_circular_dependencies(self, graph: Dict[str, List[str]]) -> bool:
        """Check for circular dependencies using DFS"""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in graph}
        
        def dfs(node):
            if colors[node] == GRAY:
                return True  # Back edge found
            if colors[node] == BLACK:
                return False
            
            colors[node] = GRAY
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            colors[node] = BLACK
            return False
        
        return any(dfs(node) for node in graph if colors[node] == WHITE)
    
    async def _execute_task_graph(self, workflow: WorkflowExecution, 
                                 graph: Dict[str, List[str]]) -> bool:
        """Execute tasks according to dependency graph"""
        task_map = {task.id: task for task in workflow.tasks}
        completed_tasks = set()
        failed_tasks = set()
        
        while len(completed_tasks) + len(failed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task_id, dependencies in graph.items():
                if (task_id not in completed_tasks and 
                    task_id not in failed_tasks and
                    all(dep in completed_tasks for dep in dependencies)):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # No tasks ready - check if we're stuck
                remaining_tasks = set(graph.keys()) - completed_tasks - failed_tasks
                if remaining_tasks:
                    # Some tasks are stuck due to failed dependencies
                    break
                else:
                    # All tasks processed
                    break
            
            # Execute ready tasks (can be done in parallel)
            task_results = await self._execute_tasks_parallel(
                workflow, [task_map[task_id] for task_id in ready_tasks]
            )
            
            # Process results
            for task_id, success in task_results.items():
                if success:
                    completed_tasks.add(task_id)
                else:
                    failed_tasks.add(task_id)
                    
                    # Check if failure should stop workflow
                    task = task_map[task_id]
                    if not task.metadata.get("continue_on_failure", False):
                        # Mark dependent tasks as failed
                        self._mark_dependent_tasks_failed(task_id, graph, failed_tasks)
        
        # Workflow succeeds if all tasks completed successfully
        return len(failed_tasks) == 0
    
    def _mark_dependent_tasks_failed(self, failed_task_id: str, 
                                   graph: Dict[str, List[str]], 
                                   failed_tasks: set):
        """Mark all dependent tasks as failed"""
        for task_id, dependencies in graph.items():
            if failed_task_id in dependencies:
                failed_tasks.add(task_id)
                # Recursively mark dependents
                self._mark_dependent_tasks_failed(task_id, graph, failed_tasks)
    
    async def _execute_tasks_parallel(self, workflow: WorkflowExecution,
                                    tasks: List[WorkflowTask]) -> Dict[str, bool]:
        """Execute multiple tasks in parallel"""
        if not tasks:
            return {}
        
        # Create task coroutines
        task_coroutines = {
            task.id: self._execute_single_task(workflow, task)
            for task in tasks
        }
        
        # Execute in parallel
        results = await asyncio.gather(
            *task_coroutines.values(),
            return_exceptions=True
        )
        
        # Map results back to task IDs
        task_results = {}
        for task_id, result in zip(task_coroutines.keys(), results):
            if isinstance(result, Exception):
                task_results[task_id] = False
            else:
                task_results[task_id] = result
        
        return task_results
    
    async def _execute_single_task(self, workflow: WorkflowExecution,
                                 task: WorkflowTask) -> bool:
        """Execute a single workflow task"""
        workflow.current_task = task.id
        
        # Trigger task started event
        await self._trigger_event("task_started", workflow, task)
        
        try:
            # Check conditions
            if not self._evaluate_conditions(task.conditions, workflow.context):
                # Task skipped due to conditions
                workflow.results[task.id] = {"skipped": True, "reason": "conditions_not_met"}
                return True
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type {task.task_type}")
            
            # Execute task with retries
            last_exception = None
            for attempt in range(task.max_retries + 1):
                try:
                    # Set timeout if specified
                    if task.timeout:
                        result = await asyncio.wait_for(
                            handler(workflow, task),
                            timeout=task.timeout
                        )
                    else:
                        result = await handler(workflow, task)
                    
                    # Store result
                    workflow.results[task.id] = result
                    
                    # Trigger task completed event
                    await self._trigger_event("task_completed", workflow, task)
                    
                    # Execute on_success tasks if specified
                    if task.on_success:
                        await self._execute_callback_tasks(workflow, task.on_success)
                    
                    return True
                    
                except Exception as e:
                    last_exception = e
                    task.retry_count += 1
                    
                    if attempt < task.max_retries:
                        # Wait before retry (exponential backoff)
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Max retries reached
                        break
            
            # Task failed after all retries
            error_info = {
                "type": "task_error",
                "task_id": task.id,
                "message": str(last_exception),
                "attempts": task.retry_count,
                "timestamp": time.time()
            }
            workflow.errors.append(error_info)
            
            # Trigger task failed event
            await self._trigger_event("task_failed", workflow, task)
            
            # Execute on_failure tasks if specified
            if task.on_failure:
                await self._execute_callback_tasks(workflow, task.on_failure)
            
            return False
            
        except Exception as e:
            # Unexpected error
            error_info = {
                "type": "unexpected_error",
                "task_id": task.id,
                "message": str(e),
                "timestamp": time.time()
            }
            workflow.errors.append(error_info)
            
            await self._trigger_event("task_failed", workflow, task)
            return False
    
    def _evaluate_conditions(self, conditions: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> bool:
        """Evaluate task conditions"""
        if not conditions:
            return True
        
        for condition in conditions:
            condition_type = condition.get("type", "if")
            expression = condition.get("expression", "true")
            
            try:
                # Simple expression evaluation (can be extended)
                if expression == "true":
                    result = True
                elif expression == "false":
                    result = False
                else:
                    # Evaluate expression in context
                    # This is a simplified version - in practice, you'd want
                    # a more sophisticated expression evaluator
                    result = eval(expression, {"__builtins__": {}}, context)
                
                if not result:
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Error evaluating condition {expression}: {e}")
                return False
        
        return True
    
    async def _execute_callback_tasks(self, workflow: WorkflowExecution, 
                                    callback_task_ids: List[str]):
        """Execute callback tasks"""
        # This is a simplified implementation
        # In practice, you might want to create sub-workflows or inline tasks
        for task_id in callback_task_ids:
            pretty_print(f"Executing callback task: {task_id}", color="info")
    
    # Task handlers
    async def _execute_mcp_tool(self, workflow: WorkflowExecution, 
                              task: WorkflowTask) -> Dict[str, Any]:
        """Execute MCP tool task"""
        params = task.parameters
        server_name = params.get("server_name")
        tool_name = params.get("tool_name")
        arguments = params.get("arguments", {})
        
        if not server_name or not tool_name:
            raise ValueError("server_name and tool_name are required for MCP tool tasks")
        
        # Substitute context variables in arguments
        arguments = self._substitute_context_variables(arguments, workflow.context)
        
        # Create and execute tool call
        tool_call = MCPToolCall(tool_name=tool_name, arguments=arguments)
        result = await self.communicator.call_tool(server_name, tool_call)
        
        if not result.success:
            raise Exception(f"MCP tool call failed: {result.error}")
        
        # Update workflow context with result
        if params.get("store_result_as"):
            workflow.context[params["store_result_as"]] = result.result
        
        return {
            "success": True,
            "result": result.result,
            "execution_time": result.execution_time
        }
    
    async def _execute_condition(self, workflow: WorkflowExecution,
                                task: WorkflowTask) -> Dict[str, Any]:
        """Execute conditional task"""
        # Placeholder for conditional logic
        return {"success": True, "type": "condition"}
    
    async def _execute_loop(self, workflow: WorkflowExecution,
                           task: WorkflowTask) -> Dict[str, Any]:
        """Execute loop task"""
        # Placeholder for loop logic
        return {"success": True, "type": "loop"}
    
    async def _execute_parallel(self, workflow: WorkflowExecution,
                               task: WorkflowTask) -> Dict[str, Any]:
        """Execute parallel task"""
        # Placeholder for parallel execution
        return {"success": True, "type": "parallel"}
    
    async def _execute_delay(self, workflow: WorkflowExecution,
                            task: WorkflowTask) -> Dict[str, Any]:
        """Execute delay task"""
        delay_seconds = task.parameters.get("seconds", 1)
        await asyncio.sleep(delay_seconds)
        return {"success": True, "type": "delay", "delayed_seconds": delay_seconds}
    
    async def _execute_custom(self, workflow: WorkflowExecution,
                             task: WorkflowTask) -> Dict[str, Any]:
        """Execute custom task"""
        # Placeholder for custom task execution
        return {"success": True, "type": "custom"}
    
    def _substitute_context_variables(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Substitute context variables in object"""
        if isinstance(obj, dict):
            return {k: self._substitute_context_variables(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_context_variables(item, context) for item in obj]
        elif isinstance(obj, str) and obj.startswith("$"):
            # Variable reference
            var_name = obj[1:]
            return context.get(var_name, obj)
        else:
            return obj
    
    async def _trigger_event(self, event_name: str, workflow: WorkflowExecution,
                           task: WorkflowTask = None):
        """Trigger workflow event callbacks"""
        callbacks = self.event_callbacks.get(event_name, [])
        for callback in callbacks:
            try:
                if task:
                    await callback(workflow, task)
                else:
                    await callback(workflow)
            except Exception as e:
                self.logger.error(f"Error in event callback {event_name}: {e}")
    
    def register_event_callback(self, event_name: str, callback: Callable):
        """Register event callback"""
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []
        self.event_callbacks[event_name].append(callback)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and progress"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        # Calculate progress
        total_tasks = len(workflow.tasks)
        completed_tasks = len([t for t in workflow.tasks if t.id in workflow.results])
        progress_percent = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress_percent": progress_percent,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "current_task": workflow.current_task,
            "duration": workflow.duration,
            "errors": workflow.errors,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [self.get_workflow_status(wf_id) for wf_id in self.workflows]

# Example workflow templates
STOCK_ANALYSIS_WORKFLOW = {
    "name": "Stock Analysis Workflow",
    "context": {},
    "tasks": [
        {
            "id": "search_stock_server",
            "name": "Search for stock MCP server",
            "type": "mcp_tool",
            "parameters": {
                "server_name": "mcp_finder",
                "tool_name": "find_mcp_servers",
                "arguments": {"query": "stock"},
                "store_result_as": "stock_servers"
            }
        },
        {
            "id": "install_stock_server",
            "name": "Install stock server",
            "type": "mcp_tool",
            "dependencies": ["search_stock_server"],
            "parameters": {
                "server_name": "registry",
                "tool_name": "install_server",
                "arguments": {"qualified_name": "${stock_server_name}"}
            }
        },
        {
            "id": "get_stock_data",
            "name": "Get stock market data",
            "type": "mcp_tool",
            "dependencies": ["install_stock_server"],
            "parameters": {
                "server_name": "${stock_server_name}",
                "tool_name": "get_stock_data",
                "arguments": {"symbol": "${stock_symbol}"},
                "store_result_as": "stock_data"
            }
        }
    ]
}

# Example usage
if __name__ == "__main__":
    async def test_workflow_engine():
        """Test workflow engine functionality"""
        from sources.tools.mcp_tools import MCPCommunicator
        
        communicator = MCPCommunicator()
        engine = MCPWorkflowEngine(communicator)
        
        # Register template
        engine.register_workflow_template("stock_analysis", STOCK_ANALYSIS_WORKFLOW)
        
        # Create workflow from template
        workflow_id = engine.create_workflow_from_template(
            "stock_analysis",
            {"stock_symbol": "IBM", "stock_server_name": "test_stock_server"}
        )
        
        print(f"Created workflow: {workflow_id}")
        
        # Get status
        status = engine.get_workflow_status(workflow_id)
        print(f"Workflow status: {json.dumps(status, indent=2)}")
    
    # Run test
    asyncio.run(test_workflow_engine())
