"""Verify event-only saved-image messages over a local MQTT broker."""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from amqtt.broker import Broker

PORT = 18883
TOPIC = "inspection/defects"


async def run(source: Path, output: Path):
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not events:
        raise RuntimeError("Event file is empty")
    broker = Broker({
        "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{PORT}"}},
        "sys_interval": 0,
        "auth": {"allow-anonymous": True},
        "topic-check": {"enabled": False},
    })
    await broker.start()
    received = []
    connected = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            raise RuntimeError(f"Subscriber connect failed: {reason_code}")
        client.subscribe(TOPIC, qos=1)
        connected.set()

    def on_message(client, userdata, message):
        received.append(json.loads(message.payload.decode("utf-8")))

    subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="es-subscriber")
    publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="es-publisher")
    subscriber.on_connect = on_connect
    subscriber.on_message = on_message
    try:
        subscriber.connect("127.0.0.1", PORT)
        subscriber.loop_start()
        if not await asyncio.to_thread(connected.wait, 10):
            raise TimeoutError("MQTT subscriber did not connect")
        publisher.connect("127.0.0.1", PORT)
        publisher.loop_start()
        started = time.perf_counter()
        for event in events:
            result = publisher.publish(TOPIC, json.dumps(event), qos=1)
            await asyncio.to_thread(result.wait_for_publish, 10)
            if not result.is_published():
                raise TimeoutError("MQTT publish did not complete")
        deadline = time.monotonic() + 15
        while len(received) < len(events) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if received != events:
            raise AssertionError(f"MQTT event mismatch: sent={len(events)}, received={len(received)}")
        summary = {
            "source": str(source),
            "broker": f"local amqtt 127.0.0.1:{PORT}",
            "protocol": "MQTT QoS 1",
            "topic": TOPIC,
            "published_events": len(events),
            "received_events": len(received),
            "payload_bytes": sum(len(json.dumps(item).encode("utf-8")) for item in events),
            "roundtrip_batch_seconds": time.perf_counter() - started,
            "result": "pass",
            "scope": "saved-image event transport on laptop localhost; no cloud or physical device",
        }
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        publisher.loop_stop()
        subscriber.loop_stop()
        publisher.disconnect()
        subscriber.disconnect()
        await broker.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(run(Path(args.events), Path(args.out)))
