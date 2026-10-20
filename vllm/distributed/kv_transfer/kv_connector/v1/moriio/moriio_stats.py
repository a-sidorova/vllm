# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the MoRIIO connector (READ mode).

One observation is one request's READ on one decode TP rank (all of its
attention and KDA layers), mirroring NIXL's per-rank, per-request transfer.

MoRI transfer statuses carry no completion timestamp, so transfer_duration is
measured from the first posted READ to the first time the worker observes every
status of the request succeeded. Observations happen where the worker already
waits on reads (step barrier, per-layer barrier, get_finished), so the value is
an upper bound on the wire time.
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

_DESCRIPTOR_BUCKETS = [
    10,
    20,
    30,
    50,
    75,
    100,
    200,
    400,
    1000,
    2000,
    4000,
    10000,
    20000,
    50000,
]


@dataclass
class MoRIIOReadXfer:
    """Bookkeeping of one request's in-flight READ on this rank."""

    start: float
    post_duration: float
    num_bytes: int
    num_descriptors: int
    pending: list = field(default_factory=list)


@dataclass
class MoRIIOKVConnectorStats(KVConnectorStats):
    """Container for MoRIIO transfer performance metrics."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
        }

    def record_transfer(
        self,
        transfer_duration: float,
        post_duration: float,
        num_bytes: int,
        num_descriptors: int,
    ):
        self.data["transfer_duration"].append(transfer_duration)
        self.data["post_duration"].append(post_duration)
        self.data["bytes_transferred"].append(num_bytes)
        self.data["num_descriptors"].append(num_descriptors)

    def record_failed_transfer(self):
        """Record a failed or timed-out MoRIIO READ."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed release notification to the producer."""
        self.data["num_failed_notifications"].append(1)

    def clone_and_reset(self) -> "MoRIIOKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        failure_counts = {
            "Num failed transfers": len(self.data["num_failed_transfers"]),
            "Num failed notifications": len(self.data["num_failed_notifications"]),
        }
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
                **failure_counts,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)

        total_mb = mb.sum()
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds if total_time_seconds else 0.0

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean().item() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean().item() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round((total_mb / n).item(), 3),
            "Throughput (MB/s)": round(float(throughput_mb_s), 3),
            "Avg number of descriptors": round(descs.mean().item(), 1),
            **failure_counts,
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MoRIIOPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        time_buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        self.histogram_xfer_time = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_xfer_time_seconds",
                documentation="Histogram of MoRIIO READ duration per request and "
                "TP rank, from first post to observed completion.",
                buckets=time_buckets[1:],
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.histogram_post_time = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_post_time_seconds",
                documentation="Histogram of host time spent posting MoRIIO READs "
                "per request and TP rank.",
                buckets=time_buckets,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        # uniform 2kb to 16gb range
        self.histogram_bytes_transferred = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_bytes_transferred",
                documentation="Histogram of bytes read per MoRIIO transfer.",
                buckets=[2 ** (10 + i) for i in range(1, 25, 2)],
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.histogram_num_descriptors = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_num_descriptors",
                documentation="Histogram of (offset, size) segments per MoRIIO "
                "transfer.",
                buckets=_DESCRIPTOR_BUCKETS,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.counter_num_failed_transfers = create_metric_per_engine(
            self._counter_cls(
                name="vllm:moriio_num_failed_transfers",
                documentation="Number of failed or timed-out MoRIIO READs.",
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.counter_num_failed_notifications = create_metric_per_engine(
            self._counter_cls(
                name="vllm:moriio_num_failed_notifications",
                documentation="Number of failed MoRIIO release notifications.",
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, key in (
            (self.histogram_xfer_time, "transfer_duration"),
            (self.histogram_post_time, "post_duration"),
            (self.histogram_bytes_transferred, "bytes_transferred"),
            (self.histogram_num_descriptors, "num_descriptors"),
        ):
            for value in transfer_stats_data[key]:
                prom_obj[engine_idx].observe(value)
        for counter_obj, key in (
            (self.counter_num_failed_transfers, "num_failed_transfers"),
            (self.counter_num_failed_notifications, "num_failed_notifications"),
        ):
            for value in transfer_stats_data[key]:
                counter_obj[engine_idx].inc(value)
