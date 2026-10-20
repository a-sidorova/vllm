# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_stats import (
    MoRIIOKVConnectorStats,
    MoRIIOPromMetrics,
    MoRIIOReadXfer,
)


class Status:
    def __init__(self, done: bool = False, failed: bool = False):
        self.done = done
        self.failed = failed

    def Succeeded(self):
        return self.done

    def Failed(self):
        return self.failed

    def Message(self):
        return "boom" if self.failed else ""

    def Code(self):
        return 1 if self.failed else 0


class FakeWrapper:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = []

    poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

    def read_remote_data(self, sizes, local_offsets, remote_offsets, session):
        return Status(done=False)

    def send_notify(self, *args, **kwargs):
        self.sent.append((args, kwargs))

    def shutdown(self):
        pass


def _make_worker() -> MoRIIOConnectorWorker:
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 4
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {}
    worker._recving_transfers_callback_addr = {}
    worker._recving_transfers_start = {}
    worker._recving_local_blocks = {}
    worker._invalid_block_ids = set()
    worker._has_mamba = True
    worker.xfer_stats = MoRIIOKVConnectorStats()
    worker._read_xfers = {}
    worker._read_bytes_posted = 0
    worker._read_descs_posted = 0
    return worker


def _track(worker, req_id: str, statuses: list) -> None:
    worker._recving_transfers[req_id] = {"layer0": statuses}
    worker._recving_transfers_callback_addr[req_id] = ("127.0.0.1", "7000", "tx")
    worker._recving_transfers_start[req_id] = float("inf")
    worker._read_xfers[req_id] = MoRIIOReadXfer(
        start=0.0,
        post_duration=0.001,
        num_bytes=2**20,
        num_descriptors=3,
        pending=list(statuses),
    )


def test_stats_reduce_and_roundtrip():
    stats = MoRIIOKVConnectorStats()
    assert stats.is_empty()
    stats.record_transfer(0.004, 0.001, 2 * 2**20, 10)
    stats.record_transfer(0.006, 0.001, 4 * 2**20, 20)
    stats.record_failed_transfer()

    # Workers ship .data to the logger, which rebuilds the stats object.
    rebuilt = MoRIIOConnector.build_kv_connector_stats(data=stats.to_dict())
    assert isinstance(rebuilt, MoRIIOKVConnectorStats)
    reduced = rebuilt.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Avg xfer time (ms)"] == 5.0
    assert reduced["Avg MB per transfer"] == 3.0
    assert reduced["Throughput (MB/s)"] == 600.0
    assert reduced["Avg number of descriptors"] == 15.0
    assert reduced["Num failed transfers"] == 1

    old = stats.clone_and_reset()
    assert stats.is_empty()
    assert old.num_successful_transfers == 2


def test_stats_aggregate_across_workers():
    a = MoRIIOKVConnectorStats()
    a.record_transfer(0.001, 0.0, 1, 1)
    b = MoRIIOKVConnectorStats()
    b.record_transfer(0.002, 0.0, 2, 2)
    b.record_failed_notification()
    merged = a.aggregate(b)
    assert merged.num_successful_transfers == 2
    assert merged.data["bytes_transferred"] == [1, 2]
    assert merged.data["num_failed_notifications"] == [1]


def test_only_failures_is_not_empty():
    stats = MoRIIOKVConnectorStats()
    stats.record_failed_transfer()
    assert not stats.is_empty()
    assert stats.reduce()["Num successful transfers"] == 0


def test_post_read_counts_bytes_and_descriptors():
    worker = _make_worker()
    worker._post_read_with_backoff(
        session=None,
        sizes=[100, 200, 300],
        local_offsets=[0, 100, 300],
        remote_offsets=[0, 100, 300],
        request_id="req",
        layer_name="layer0",
        deadline=float("inf"),
    )
    assert worker._read_bytes_posted == 600
    assert worker._read_descs_posted == 3


def test_observe_records_only_fully_landed_reads():
    worker = _make_worker()
    landed, in_flight = Status(), Status()
    _track(worker, "req", [landed, in_flight])

    landed.done = True
    worker._observe_read_completions()
    assert worker.xfer_stats.is_empty()
    assert worker._read_xfers["req"].pending == [in_flight]

    in_flight.done = True
    worker._observe_read_completions()
    assert "req" not in worker._read_xfers
    assert worker.xfer_stats.data["bytes_transferred"] == [2**20]
    assert worker.xfer_stats.data["num_descriptors"] == [3]
    assert worker.xfer_stats.data["post_duration"] == [0.001]
    assert worker.xfer_stats.data["transfer_duration"][0] > 0

    stats = worker.get_kv_connector_stats()
    assert stats is not None and stats.num_successful_transfers == 1
    assert worker.get_kv_connector_stats() is None


def test_pop_done_records_completion_and_failure():
    worker = _make_worker()
    _track(worker, "ok", [Status(done=True)])
    _track(worker, "bad", [Status(failed=True)])

    worker._pop_done_transfers()

    assert worker._read_xfers == {}
    assert worker.xfer_stats.num_successful_transfers == 1
    assert worker.xfer_stats.data["num_failed_transfers"] == [1]


class _FakeMetric:
    def __init__(self, name, documentation, labelnames, buckets=None):
        self.name = name
        self.observed: list = []
        self.incremented: list = []

    def labels(self, *labelvalues):
        return self

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value=1):
        self.incremented.append(value)


def test_prom_metrics_observe():
    metric_types = {Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric}
    prom = MoRIIOPromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        metric_types,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["m", "0"]},
    )
    stats = MoRIIOKVConnectorStats()
    stats.record_transfer(0.004, 0.001, 1024, 5)
    stats.record_failed_transfer()

    prom.observe(stats.to_dict(), engine_idx=0)

    assert prom.histogram_xfer_time[0].name == "vllm:moriio_xfer_time_seconds"
    assert prom.histogram_xfer_time[0].observed == [0.004]
    assert prom.histogram_bytes_transferred[0].observed == [1024]
    assert prom.counter_num_failed_transfers[0].incremented == [1]
