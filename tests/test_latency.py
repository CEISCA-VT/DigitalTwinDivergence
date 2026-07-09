from DigitalTwin.latency import LatencyQueue


def test_buffered_latency_changes_delivery_intervals():
    queue = LatencyQueue(200.0, jitter_ms=0.0, seed=1)
    for i in range(5):
        queue.push(i * 0.1, i)

    deliveries = []
    for i in range(10):
        deliveries.extend(queue.pop_ready(i * 0.1))

    delivery_times = [round(item.delivery_s, 3) for item in deliveries]
    intervals = [round(b - a, 3) for a, b in zip(delivery_times, delivery_times[1:])]
    assert 0.0 in intervals
    assert 0.2 in intervals
    assert deliveries[-1].queue_depth_after_pop == 0


def test_queue_depth_reports_buffered_items_after_delivery():
    queue = LatencyQueue(500.0, jitter_ms=0.0, seed=1)
    queue.push(0.0, "a")
    queue.push(0.1, "b")

    deliveries = queue.pop_ready(0.5)

    assert len(deliveries) == 1
    assert deliveries[0].queue_depth_after_pop == 1
    assert queue.depth == 1
