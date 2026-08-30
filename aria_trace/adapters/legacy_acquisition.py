"""Explicit bridge between legacy acquisition packets and universal envelopes."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from aria_trace.domain.packets import FramePacket

from aria_trace.domain import (
    CheckStatus,
    ComponentError,
    DataEnvelope,
    Decision,
    DiagnosticValue,
    EnvelopeIdentity,
    FailureKind,
    ImageFrame,
    ProducerRef,
    ProvenanceInfo,
    QualityCheck,
    QualityInfo,
    QualityMetric,
    SchemaRef,
    SourceRef,
    SpaceRef,
    TimePoint,
    TimingInfo,
)
from aria_trace.ports import Source


FRAME_PACKET_SCHEMA = SchemaRef("aria.legacy-acquisition-frame", "1.0")
HOST_CLOCK_METADATA_KEY = "aria_host_clock_id"
SOURCE_CLOCK_METADATA_KEY = "aria_source_clock_id"
SPACE_METADATA_KEY = "aria_coordinate_space_id"


def _compatible_metadata(metadata, key: str, value: str) -> dict:
    result = dict(metadata)
    existing = result.get(key)
    if existing is not None and str(existing) != value:
        raise ComponentError(
            "legacy-acquisition-frame-adapter",
            "metadata-conflict",
            FailureKind.INVALID_INPUT,
            "Legacy frame metadata conflicts with explicit {}".format(key),
            details={"metadata_value": existing, "explicit_value": value},
        )
    result[key] = value
    return result


def frame_packet_to_envelope(
    packet: FramePacket,
    *,
    envelope_id: str,
    trace_id: str,
    host_clock_id: str,
    source_clock_id: Optional[str],
    space: SpaceRef,
    producer: ProducerRef,
    parent_ids: Tuple[str, ...] = (),
    sources: Tuple[SourceRef, ...] = (),
) -> DataEnvelope[ImageFrame]:
    """Bridge a packet only when its formerly implicit domains are supplied."""

    if packet.source_time_ns is not None and not source_clock_id:
        raise ComponentError(
            "legacy-acquisition-frame-adapter",
            "missing-source-clock",
            FailureKind.INVALID_INPUT,
            "A frame with source_time_ns requires an explicit source clock ID",
            details={"stream_id": packet.stream_id},
        )
    if not host_clock_id.strip():
        raise ValueError("Host clock ID is required")

    metadata = _compatible_metadata(
        packet.metadata, HOST_CLOCK_METADATA_KEY, host_clock_id
    )
    metadata = _compatible_metadata(metadata, SPACE_METADATA_KEY, space.space_id)
    if source_clock_id:
        metadata = _compatible_metadata(
            metadata, SOURCE_CLOCK_METADATA_KEY, source_clock_id
        )

    source_time = (
        TimePoint(packet.source_time_ns, source_clock_id)
        if packet.source_time_ns is not None
        else None
    )
    timing_order_ok = packet.host_receive_time_ns >= packet.host_capture_time_ns
    quality = QualityInfo(
        decision=Decision.UNKNOWN,
        metrics=(
            QualityMetric("dropped_before", packet.dropped_before, "frames"),
            QualityMetric(
                "capture_to_receive_ns",
                packet.host_receive_time_ns - packet.host_capture_time_ns,
                "ns",
            ),
        ),
        checks=(
            QualityCheck(
                "host_timing_order",
                CheckStatus.PASS if timing_order_ok else CheckStatus.WARN,
                None if timing_order_ok else "receive time precedes capture time",
            ),
        ),
        warnings=(
            ("Legacy packet reported dropped frames",)
            if packet.dropped_before > 0
            else ()
        ),
    )
    return DataEnvelope(
        schema=FRAME_PACKET_SCHEMA,
        value=ImageFrame(
            stream_id=packet.stream_id,
            image=packet.image,
            metadata=metadata,
            dropped_before=packet.dropped_before,
        ),
        identity=EnvelopeIdentity(envelope_id, trace_id, parent_ids),
        timing=TimingInfo(
            source=source_time,
            capture=TimePoint(packet.host_capture_time_ns, host_clock_id),
            receive=TimePoint(packet.host_receive_time_ns, host_clock_id),
        ),
        space=space,
        provenance=ProvenanceInfo(producer=producer, sources=sources),
        quality=quality,
        diagnostics=(DiagnosticValue("legacy_packet_type", "FramePacket"),),
    )


def envelope_to_frame_packet(value: DataEnvelope[ImageFrame]) -> FramePacket:
    """Return to the legacy boundary without discarding explicit domain IDs."""

    if value.schema != FRAME_PACKET_SCHEMA or not isinstance(value.value, ImageFrame):
        raise ComponentError(
            "legacy-acquisition-frame-adapter",
            "incompatible-envelope",
            FailureKind.INCOMPATIBLE_CAPABILITY,
            "Envelope is not a legacy acquisition frame",
            details={"schema_id": value.schema.schema_id},
        )
    if value.timing.capture is None or value.timing.receive is None:
        raise ComponentError(
            "legacy-acquisition-frame-adapter",
            "missing-host-time",
            FailureKind.INVALID_INPUT,
            "Legacy FramePacket requires capture and receive timestamps",
        )
    if value.timing.capture.clock_id != value.timing.receive.clock_id:
        raise ComponentError(
            "legacy-acquisition-frame-adapter",
            "host-clock-mismatch",
            FailureKind.INVALID_INPUT,
            "Legacy FramePacket host timestamps must use one clock",
        )

    metadata = _compatible_metadata(
        value.value.metadata,
        HOST_CLOCK_METADATA_KEY,
        value.timing.capture.clock_id,
    )
    metadata = _compatible_metadata(
        metadata, SPACE_METADATA_KEY, value.space.space_id
    )
    source_time_ns = None
    if value.timing.source is not None:
        source_time_ns = value.timing.source.value_ns
        metadata = _compatible_metadata(
            metadata,
            SOURCE_CLOCK_METADATA_KEY,
            value.timing.source.clock_id,
        )
    return FramePacket(
        stream_id=value.value.stream_id,
        image=value.value.image,
        host_capture_time_ns=value.timing.capture.value_ns,
        host_receive_time_ns=value.timing.receive.value_ns,
        source_time_ns=source_time_ns,
        metadata=metadata,
        dropped_before=value.value.dropped_before,
    )


class LegacyFrameSourceAdapter(Source[ImageFrame]):
    """Expose one existing acquisition source through the universal Source port."""

    def __init__(
        self,
        source,
        *,
        trace_id: str,
        host_clock_id: str,
        source_clock_id: Optional[str],
        space: SpaceRef,
        producer: ProducerRef,
        envelope_id: Callable[[FramePacket], str],
        sources: Tuple[SourceRef, ...] = (),
    ) -> None:
        self.source = source
        self.trace_id = trace_id
        self.host_clock_id = host_clock_id
        self.source_clock_id = source_clock_id
        self.space = space
        self.producer = producer
        self.envelope_id = envelope_id
        self.sources = sources

    def start(self) -> None:
        self.source.start()

    def read(self) -> Optional[DataEnvelope[ImageFrame]]:
        packet = self.source.read()
        if packet is None:
            return None
        return frame_packet_to_envelope(
            packet,
            envelope_id=self.envelope_id(packet),
            trace_id=self.trace_id,
            host_clock_id=self.host_clock_id,
            source_clock_id=self.source_clock_id,
            space=self.space,
            producer=self.producer,
            sources=self.sources,
        )

    def stop(self) -> None:
        self.source.stop()
