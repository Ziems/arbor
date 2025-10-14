"""
Monitoring and observability endpoints for system health, debugging, and metrics.

Provides comprehensive monitoring capabilities including health checks, error tracking,
job debugging, system metrics, and a real-time dashboard.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, StreamingResponse

from arbor.server.utils.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    error_handler,
)
from arbor.server.utils.logging import RequestContext, get_logger

router = APIRouter(prefix="/monitor", tags=["monitoring"])
logger = get_logger(__name__)


@router.get("/health")
async def health_check(request: Request):
    """Basic health check endpoint."""

    with RequestContext(operation="health_check"):
        try:
            # Basic health checks
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "0.2.4",  # Get from config or package
                "uptime_seconds": time.time()
                - getattr(request.app.state, "start_time", time.time()),
                "checks": {},
            }

            # Check job manager
            try:
                job_manager = request.app.state.job_manager
                all_jobs = job_manager.get_all_jobs()
                health_status["checks"]["job_manager"] = {
                    "status": "healthy",
                    "total_jobs": len(all_jobs),
                }
            except Exception as e:
                health_status["checks"]["job_manager"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["status"] = "degraded"

            # Check file manager
            try:
                _file_manager = request.app.state.file_manager
                health_status["checks"]["file_manager"] = {"status": "healthy"}
            except Exception as e:
                health_status["checks"]["file_manager"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["status"] = "degraded"

            # Check error rate
            error_summary = error_handler.get_error_summary()
            recent_errors = error_summary.get("total", 0)
            if recent_errors > 10:  # More than 10 errors in recent history
                health_status["status"] = "degraded"
                health_status["checks"]["error_rate"] = {
                    "status": "warning",
                    "recent_errors": recent_errors,
                }
            else:
                health_status["checks"]["error_rate"] = {
                    "status": "healthy",
                    "recent_errors": recent_errors,
                }

            logger.debug("Health check completed", status=health_status["status"])

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check with system metrics."""

    with RequestContext(operation="detailed_health_check"):
        try:
            health_info = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": {},
                "services": {},
                "metrics": {},
                "errors": error_handler.get_error_summary(),
            }

            # System metrics
            try:
                import psutil

                # CPU info
                health_info["system"]["cpu"] = {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "load_avg": (
                        psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                    ),
                }

                # Memory info
                memory = psutil.virtual_memory()
                health_info["system"]["memory"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                }

                # Disk info
                disk = psutil.disk_usage("/")
                health_info["system"]["disk"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                }

                # Process info
                health_info["system"]["processes"] = {
                    "total": len(psutil.pids()),
                    "current_process": {
                        "pid": psutil.Process().pid,
                        "memory_percent": psutil.Process().memory_percent(),
                        "cpu_percent": psutil.Process().cpu_percent(),
                    },
                }

            except ImportError:
                health_info["system"]["error"] = "psutil not available"
            except Exception as e:
                health_info["system"]["error"] = str(e)

            # GPU info (if available) - simplified without health_manager
            try:
                gpu_manager = getattr(request.app.state, "gpu_manager", None)
                if gpu_manager:
                    status = gpu_manager.get_status()
                    health_info["system"]["gpu"] = {
                        "total_gpus": len(status["total_gpus"]),
                        "free_gpus": len(status["free_gpus"]),
                        "allocated_gpus": len(status["allocated_gpus"]),
                        "allocations": status["allocations"],
                    }
            except Exception as e:
                health_info["system"]["gpu"] = {"error": str(e)}

            # Service health
            services = [
                "job_manager",
                "file_manager",
                "inference_manager",
                "grpo_manager",
                "file_train_manager",
            ]
            for service_name in services:
                try:
                    service = getattr(request.app.state, service_name, None)
                    if service:
                        health_info["services"][service_name] = {
                            "status": "healthy",
                            "type": type(service).__name__,
                        }

                        # Add service-specific health info
                        if hasattr(service, "get_health_info"):
                            health_info["services"][service_name].update(
                                service.get_health_info()
                            )
                    else:
                        health_info["services"][service_name] = {
                            "status": "not_available"
                        }
                except Exception as e:
                    health_info["services"][service_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }

            # Overall status determination
            unhealthy_services = [
                s
                for s, info in health_info["services"].items()
                if info.get("status") == "unhealthy"
            ]
            if unhealthy_services:
                health_info["status"] = "degraded"
                health_info["unhealthy_services"] = unhealthy_services

            # Critical system checks
            if health_info["system"].get("memory", {}).get("percent", 0) > 90:
                health_info["status"] = "critical"
                health_info["critical_issues"] = health_info.get("critical_issues", [])
                health_info["critical_issues"].append("High memory usage")

            if health_info["system"].get("disk", {}).get("percent", 0) > 95:
                health_info["status"] = "critical"
                health_info["critical_issues"] = health_info.get("critical_issues", [])
                health_info["critical_issues"].append("Disk space critical")

            logger.info("Detailed health check completed", status=health_info["status"])

            return health_info

        except Exception as e:
            logger.error(f"Detailed health check failed: {str(e)}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }


@router.get("/errors")
async def get_recent_errors(
    request: Request,
    count: int = Query(
        10, ge=1, le=100, description="Number of recent errors to return"
    ),
    category: Optional[str] = Query(None, description="Filter by error category"),
    severity: Optional[str] = Query(None, description="Filter by error severity"),
):
    """Get recent errors with filtering options."""

    with RequestContext(operation="get_recent_errors"):
        try:
            errors = error_handler.get_recent_errors(count)

            # Apply filters
            if category:
                try:
                    cat_filter = ErrorCategory(category.lower())
                    errors = [e for e in errors if e.category == cat_filter]
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid category: {category}. Valid categories: {[c.value for c in ErrorCategory]}",
                    )

            if severity:
                try:
                    sev_filter = ErrorSeverity(severity.lower())
                    errors = [e for e in errors if e.severity == sev_filter]
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid severity: {severity}. Valid severities: {[s.value for s in ErrorSeverity]}",
                    )

            # Convert to dict format
            error_dicts = [error.to_dict() for error in errors]

            logger.info(
                f"Retrieved {len(error_dicts)} errors",
                count=count,
                category=category,
                severity=severity,
                total_found=len(error_dicts),
            )

            return {
                "errors": error_dicts,
                "total": len(error_dicts),
                "filters": {"category": category, "severity": severity, "count": count},
            }

        except Exception as e:
            logger.error(f"Failed to retrieve errors: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve errors")


@router.get("/errors/summary")
async def get_error_summary(request: Request):
    """Get summary of recent errors."""

    with RequestContext(operation="get_error_summary"):
        try:
            summary = error_handler.get_error_summary()

            logger.debug("Retrieved error summary", summary=summary)

            return summary

        except Exception as e:
            logger.error(f"Failed to get error summary: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to get error summary")


@router.get("/jobs/{job_id}")
async def get_job_debug_info(request: Request, job_id: str):
    """Get comprehensive debugging information for a specific job."""

    with RequestContext(operation="get_job_debug", user="debug"):
        try:
            # Get job manager from app state
            job_manager = request.app.state.job_manager

            # Check if job exists
            job = job_manager.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            # Build debug information
            debug_info = {
                "job_id": job_id,
                "job_type": type(job).__name__,
                "status": getattr(job, "status", "unknown"),
                "created_at": getattr(job, "created_at", None),
                "updated_at": getattr(job, "updated_at", None),
                "metadata": {},
            }

            # Add job-specific debug info
            if hasattr(job, "get_debug_info"):
                debug_info["job_specific"] = job.get_debug_info()

            # Add process information if available
            if hasattr(job, "process") and job.process:
                debug_info["process"] = {
                    "pid": job.process.pid,
                    "poll": job.process.poll(),
                    "returncode": job.process.returncode,
                }

            # Add training process info if available
            if hasattr(job, "training_process") and job.training_process:
                debug_info["training_process"] = {
                    "pid": job.training_process.pid,
                    "poll": job.training_process.poll(),
                    "returncode": job.training_process.returncode,
                }

            # Get related errors
            job_errors = [
                error.to_dict()
                for error in error_handler.error_history
                if error.context.job_id == job_id
            ]
            debug_info["errors"] = job_errors[-10:]  # Last 10 errors
            debug_info["error_count"] = len(job_errors)

            logger.info(
                f"Retrieved debug info for job {job_id}", job_type=type(job).__name__
            )

            return debug_info

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get job debug info: {str(e)}", job_id=job_id, exc_info=True
            )
            raise HTTPException(status_code=500, detail="Failed to get job debug info")


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    request: Request,
    job_id: str,
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines to return"),
    follow: bool = Query(False, description="Stream logs in real-time"),
):
    """Get logs for a specific job."""

    with RequestContext(operation="get_job_logs"):
        try:
            # Get job manager from app state
            job_manager = request.app.state.job_manager

            # Check if job exists
            job = job_manager.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            if follow:
                # Return streaming response for real-time logs
                return StreamingResponse(
                    _stream_job_logs(job, lines),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache"},
                )
            else:
                # Return static logs
                logs = _get_job_logs(job, lines)
                return {"job_id": job_id, "logs": logs, "line_count": len(logs)}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get job logs: {str(e)}", job_id=job_id, exc_info=True
            )
            raise HTTPException(status_code=500, detail="Failed to get job logs")


@router.get("/jobs/{job_id}/metrics")
async def get_job_metrics(request: Request, job_id: str):
    """Get performance metrics for a specific job."""

    with RequestContext(operation="get_job_metrics"):
        try:
            # Get job manager from app state
            job_manager = request.app.state.job_manager

            # Check if job exists
            job = job_manager.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            metrics = {"job_id": job_id, "metrics": {}, "timestamps": []}

            # Add job-specific metrics if available
            if hasattr(job, "get_metrics"):
                metrics["metrics"] = job.get_metrics()

            # Add system metrics if available
            if hasattr(job, "process") and job.process:
                try:
                    import psutil

                    process = psutil.Process(job.process.pid)
                    metrics["system"] = {
                        "cpu_percent": process.cpu_percent(),
                        "memory_info": process.memory_info()._asdict(),
                        "status": process.status(),
                        "create_time": process.create_time(),
                    }
                except (psutil.NoSuchProcess, ImportError):
                    pass

            logger.debug(f"Retrieved metrics for job {job_id}")

            return metrics

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get job metrics: {str(e)}", job_id=job_id, exc_info=True
            )
            raise HTTPException(status_code=500, detail="Failed to get job metrics")


@router.get("/system")
async def get_system_debug_info(request: Request):
    """Get system-wide debugging information."""

    with RequestContext(operation="get_system_debug"):
        try:
            # Get managers from app state
            job_manager = request.app.state.job_manager

            debug_info = {
                "timestamp": time.time(),
                "jobs": {},
                "system": {},
                "errors": error_handler.get_error_summary(),
            }

            # Add job information
            all_jobs = job_manager.get_all_jobs()
            debug_info["jobs"] = {
                "total": len(all_jobs),
                "by_type": {},
                "by_status": {},
            }

            for job in all_jobs:
                job_type = type(job).__name__
                job_status = getattr(job, "status", "unknown")

                debug_info["jobs"]["by_type"][job_type] = (
                    debug_info["jobs"]["by_type"].get(job_type, 0) + 1
                )
                debug_info["jobs"]["by_status"][job_status] = (
                    debug_info["jobs"]["by_status"].get(job_status, 0) + 1
                )

            # Add system information
            try:
                import psutil

                debug_info["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": psutil.virtual_memory()._asdict(),
                    "disk": psutil.disk_usage("/")._asdict(),
                    "process_count": len(psutil.pids()),
                }
            except ImportError:
                debug_info["system"] = {"error": "psutil not available"}

            logger.debug("Retrieved system debug info")

            return debug_info

        except Exception as e:
            logger.error(f"Failed to get system debug info: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to get system debug info"
            )


@router.get("/metrics")
async def get_metrics(request: Request):
    """Get system metrics in a structured format."""

    with RequestContext(operation="get_metrics"):
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {},
                "application": {},
                "errors": error_handler.get_error_summary(),
            }

            # System metrics
            try:
                import psutil

                metrics["system"] = {
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=1),
                        "count": psutil.cpu_count(),
                    },
                    "memory": psutil.virtual_memory()._asdict(),
                    "disk": psutil.disk_usage("/")._asdict(),
                    "processes": len(psutil.pids()),
                }
            except ImportError:
                metrics["system"]["error"] = "psutil not available"

            # Application metrics
            try:
                job_manager = request.app.state.job_manager
                all_jobs = job_manager.get_all_jobs()

                job_types = {}
                job_statuses = {}

                for job in all_jobs:
                    job_type = type(job).__name__
                    job_status = getattr(job, "status", "unknown")

                    job_types[job_type] = job_types.get(job_type, 0) + 1
                    job_statuses[job_status] = job_statuses.get(job_status, 0) + 1

                metrics["application"] = {
                    "jobs": {
                        "total": len(all_jobs),
                        "by_type": job_types,
                        "by_status": job_statuses,
                    }
                }
            except Exception as e:
                metrics["application"]["error"] = str(e)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.get("/dashboard")
async def monitoring_dashboard(request: Request):
    """Simple HTML dashboard for system monitoring."""

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arbor Monitoring Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status-healthy { color: #22c55e; }
            .status-degraded { color: #f59e0b; }
            .status-critical { color: #ef4444; }
            .status-unhealthy { color: #dc2626; }
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px 0;
                border-bottom: 1px solid #f0f0f0;
            }
            .metric:last-child {
                border-bottom: none;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #e5e7eb;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                border-radius: 10px;
                transition: width 0.3s ease;
            }
            .progress-green { background-color: #22c55e; }
            .progress-yellow { background-color: #f59e0b; }
            .progress-red { background-color: #ef4444; }
            #refresh-btn {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            }
            #refresh-btn:hover {
                background: #2563eb;
            }
            .error-list {
                max-height: 200px;
                overflow-y: auto;
                background: #fef2f2;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            .error-item {
                font-size: 12px;
                margin: 4px 0;
                padding: 4px;
                background: white;
                border-radius: 2px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌳 Arbor Monitoring Dashboard</h1>
                <p>Real-time system monitoring and health status</p>
                <button id="refresh-btn" onclick="refreshData()">Refresh Data</button>
                <span id="last-updated" style="margin-left: 20px; color: #666;"></span>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>System Status</h3>
                    <div id="system-status">Loading...</div>
                </div>

                <div class="card">
                    <h3>System Metrics</h3>
                    <div id="system-metrics">Loading...</div>
                </div>

                <div class="card">
                    <h3>Services</h3>
                    <div id="services-status">Loading...</div>
                </div>

                <div class="card">
                    <h3>Recent Errors</h3>
                    <div id="errors-summary">Loading...</div>
                </div>

                <div class="card">
                    <h3>Jobs</h3>
                    <div id="jobs-info">Loading...</div>
                </div>

                <div class="card">
                    <h3>GPU Status</h3>
                    <div id="gpu-status">Loading...</div>
                </div>
            </div>
        </div>

        <script>
            function getStatusClass(status) {
                return 'status-' + (status || 'unknown');
            }

            function getProgressColor(percent) {
                if (percent > 80) return 'progress-red';
                if (percent > 60) return 'progress-yellow';
                return 'progress-green';
            }

            function formatBytes(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }

            async function refreshData() {
                try {
                    const response = await fetch('/monitor/health/detailed');
                    const data = await response.json();

                    // Update system status
                    document.getElementById('system-status').innerHTML = `
                        <div class="metric">
                            <span>Overall Status:</span>
                            <span class="${getStatusClass(data.status)}">${data.status.toUpperCase()}</span>
                        </div>
                        <div class="metric">
                            <span>Timestamp:</span>
                            <span>${new Date(data.timestamp).toLocaleString()}</span>
                        </div>
                    `;

                    // Update system metrics
                    if (data.system) {
                        let metricsHtml = '';

                        if (data.system.cpu) {
                            metricsHtml += `
                                <div class="metric">
                                    <span>CPU Usage:</span>
                                    <span>${data.system.cpu.percent}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill ${getProgressColor(data.system.cpu.percent)}"
                                         style="width: ${data.system.cpu.percent}%"></div>
                                </div>
                            `;
                        }

                        if (data.system.memory) {
                            metricsHtml += `
                                <div class="metric">
                                    <span>Memory Usage:</span>
                                    <span>${data.system.memory.percent}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill ${getProgressColor(data.system.memory.percent)}"
                                         style="width: ${data.system.memory.percent}%"></div>
                                </div>
                                <div class="metric">
                                    <span>Memory Available:</span>
                                    <span>${formatBytes(data.system.memory.available)}</span>
                                </div>
                            `;
                        }

                        if (data.system.disk) {
                            metricsHtml += `
                                <div class="metric">
                                    <span>Disk Usage:</span>
                                    <span>${data.system.disk.percent.toFixed(1)}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill ${getProgressColor(data.system.disk.percent)}"
                                         style="width: ${data.system.disk.percent}%"></div>
                                </div>
                                <div class="metric">
                                    <span>Disk Free:</span>
                                    <span>${formatBytes(data.system.disk.free)}</span>
                                </div>
                            `;
                        }

                        document.getElementById('system-metrics').innerHTML = metricsHtml;
                    }

                    // Update services
                    if (data.services) {
                        let servicesHtml = '';
                        for (const [service, info] of Object.entries(data.services)) {
                            servicesHtml += `
                                <div class="metric">
                                    <span>${service}:</span>
                                    <span class="${getStatusClass(info.status)}">${info.status}</span>
                                </div>
                            `;
                        }
                        document.getElementById('services-status').innerHTML = servicesHtml;
                    }

                    // Update errors
                    if (data.errors) {
                        let errorsHtml = `
                            <div class="metric">
                                <span>Total Recent:</span>
                                <span>${data.errors.total}</span>
                            </div>
                        `;

                        if (data.errors.by_severity) {
                            for (const [severity, count] of Object.entries(data.errors.by_severity)) {
                                errorsHtml += `
                                    <div class="metric">
                                        <span>${severity}:</span>
                                        <span>${count}</span>
                                    </div>
                                `;
                            }
                        }

                        document.getElementById('errors-summary').innerHTML = errorsHtml;
                    }

                    // Update GPU status
                    if (data.system.gpu) {
                        let gpuHtml = '';
                        if (data.system.gpu.error) {
                            gpuHtml = `<div class="metric"><span>Status:</span><span>Not Available</span></div>`;
                        } else if (data.system.gpu.gpus) {
                            gpuHtml = `<div class="metric"><span>GPUs Found:</span><span>${data.system.gpu.gpus.length}</span></div>`;
                            data.system.gpu.gpus.forEach((gpu, i) => {
                                gpuHtml += `
                                    <div class="metric">
                                        <span>GPU ${i}:</span>
                                        <span>${gpu.name || 'Unknown'}</span>
                                    </div>
                                `;
                            });
                        }
                        document.getElementById('gpu-status').innerHTML = gpuHtml;
                    }

                    // Update jobs placeholder
                    document.getElementById('jobs-info').innerHTML = '<div class="metric"><span>Jobs info available via API</span></div>';

                    // Update timestamp
                    document.getElementById('last-updated').textContent = `Last updated: ${new Date().toLocaleTimeString()}`;

                } catch (error) {
                    console.error('Failed to refresh data:', error);
                    document.getElementById('system-status').innerHTML = '<div style="color: red;">Failed to load data</div>';
                }
            }

            // Initial load
            refreshData();

            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.websocket("/live")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring."""

    await websocket.accept()
    logger.info("Monitoring WebSocket connected")

    try:
        while True:
            # Get current health data
            try:
                # This would normally get data from request.app.state
                # For now, we'll create a simplified version
                health_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "healthy",
                    "metrics": {
                        "cpu_percent": 25.0,  # Would get from psutil
                        "memory_percent": 45.0,
                        "active_jobs": 2,
                        "recent_errors": 0,
                    },
                }

                await websocket.send_json(health_data)
                await asyncio.sleep(5)  # Send updates every 5 seconds

            except Exception as e:
                logger.error(f"Error sending health data: {str(e)}")
                await websocket.send_json(
                    {"error": str(e), "timestamp": datetime.now().isoformat()}
                )
                break

    except WebSocketDisconnect:
        logger.info("Monitoring WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")


@router.post("/test-error")
async def test_error_handling(
    request: Request, error_type: str = "generic", message: str = "Test error"
):
    """Test endpoint for error handling (development only)."""

    with RequestContext(operation="test_error"):
        logger.warning(
            "Test error endpoint called", error_type=error_type, message=message
        )

        if error_type == "validation":
            from arbor.server.utils.error_handling import ValidationError

            raise ValidationError(message, field="test_field", value="test_value")
        elif error_type == "resource":
            from arbor.server.utils.error_handling import ResourceError

            raise ResourceError(message, resource_type="memory")
        elif error_type == "model":
            from arbor.server.utils.error_handling import ModelError

            raise ModelError(message, model_name="test-model")
        elif error_type == "training":
            from arbor.server.utils.error_handling import TrainingError

            raise TrainingError(message, step=42)
        elif error_type == "config":
            from arbor.server.utils.error_handling import ConfigError

            raise ConfigError(message, config_key="test.setting")
        else:
            # Generic exception
            raise Exception(message)


# Helper functions
def _get_job_logs(job, lines: int) -> List[str]:
    """Extract logs from a job (implementation depends on job type)."""
    logs = []

    # Try to get logs from different sources
    if hasattr(job, "get_logs"):
        logs = job.get_logs(lines)
    elif hasattr(job, "process") and job.process:
        # Try to read from process stdout/stderr if available
        if hasattr(job.process, "stdout") and job.process.stdout:
            try:
                # This is a simplified implementation
                # In practice, you'd need to handle buffering properly
                logs = ["Process logs not directly available"]
            except Exception:
                logs = ["Failed to read process logs"]
    else:
        logs = ["No logs available for this job type"]

    return logs


async def _stream_job_logs(job, initial_lines: int):
    """Stream job logs in real-time."""
    # This is a simplified implementation
    # In practice, you'd need to implement proper log tailing

    # Send initial logs
    initial_logs = _get_job_logs(job, initial_lines)
    for log_line in initial_logs:
        yield f"data: {json.dumps({'type': 'log', 'content': log_line})}\n\n"

    # Stream new logs (simplified - in practice you'd tail the actual log files)
    import asyncio

    for i in range(10):  # Send 10 fake updates
        await asyncio.sleep(1)
        yield f"data: {json.dumps({'type': 'log', 'content': f'[{time.time()}] Streaming log line {i}'})}\n\n"

    # End stream
    yield f"data: {json.dumps({'type': 'end', 'content': 'Log stream ended'})}\n\n"
