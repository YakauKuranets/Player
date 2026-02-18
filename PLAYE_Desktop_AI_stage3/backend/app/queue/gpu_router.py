"""GPU routing and load balancing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class GpuSlot:
    device_id: int
    device_name: str
    queue_name: str
    active_tasks: int = 0
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    is_healthy: bool = True


class GpuRouter:
    def __init__(self):
        self._slots: list[GpuSlot] = []
        self._lock = Lock()
        self._rr_index = 0
        self._discover()

    def _discover(self):
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    slot = GpuSlot(
                        device_id=i,
                        device_name=props.name,
                        queue_name=f"gpu_{i}",
                        total_memory_mb=props.total_memory // (1024 * 1024),
                        free_memory_mb=props.total_memory // (1024 * 1024),
                    )
                    self._slots.append(slot)
                    logger.info("Discovered GPU %d: %s", i, props.name)
        except Exception:
            pass

        if not self._slots:
            self._slots.append(GpuSlot(device_id=-1, device_name="CPU", queue_name="cpu"))
            logger.info("No GPU found — using CPU queue")

    def get_best_queue(self, strategy: str = "least_loaded") -> str:
        with self._lock:
            healthy = [s for s in self._slots if s.is_healthy]
            if not healthy:
                return "celery"
            if strategy == "least_loaded":
                return min(healthy, key=lambda s: s.active_tasks).queue_name
            if strategy == "round_robin":
                slot = healthy[self._rr_index % len(healthy)]
                self._rr_index += 1
                return slot.queue_name
            return healthy[0].queue_name

    def increment(self, queue_name: str):
        with self._lock:
            for slot in self._slots:
                if slot.queue_name == queue_name:
                    slot.active_tasks += 1

    def decrement(self, queue_name: str):
        with self._lock:
            for slot in self._slots:
                if slot.queue_name == queue_name:
                    slot.active_tasks = max(0, slot.active_tasks - 1)

    def status(self) -> list[dict]:
        with self._lock:
            result = []
            for slot in self._slots:
                try:
                    import torch

                    if slot.device_id >= 0:
                        free, total = torch.cuda.mem_get_info(slot.device_id)
                        slot.free_memory_mb = free // (1024 * 1024)
                        slot.total_memory_mb = total // (1024 * 1024)
                except Exception:
                    pass
                result.append(
                    {
                        "device_id": slot.device_id,
                        "device_name": slot.device_name,
                        "queue": slot.queue_name,
                        "active_tasks": slot.active_tasks,
                        "total_memory_mb": slot.total_memory_mb,
                        "free_memory_mb": slot.free_memory_mb,
                        "healthy": slot.is_healthy,
                    }
                )
            return result


gpu_router = GpuRouter()
