from bench_logger import (
    DEFAULT_MOTION_CALIBRATION,
    HOLD_SECONDS,
    INITIAL_HOLD_SECONDS,
    MotionCalibration,
    MotionSequenceController,
    SQUARE_TURN_DEGREES,
    SQUARE_TURN_SECONDS,
    SEQUENCE_REPEAT_COUNT,
    SquareSequenceController,
    STRAIGHT_DISTANCE_M,
    TURN_DEGREES,
    TURN_REPEAT_COUNT,
)


def _telemetry(enc_left: int, enc_right: int) -> dict[str, int]:
    return {"enc_left": enc_left, "enc_right": enc_right}


def test_default_motion_calibration_targets_match_current_week2_values():
    assert round(DEFAULT_MOTION_CALIBRATION.distance_target_counts(1.0), 3) == 5632.373
    assert round(DEFAULT_MOTION_CALIBRATION.turn_target_counts(360.0), 2) == 3216.88


def test_motion_sequence_starts_with_hold_step():
    motion = MotionSequenceController()
    left_cmd, right_cmd, label = motion.update(_telemetry(0, 0), now_s=0.0)
    assert (left_cmd, right_cmd) == (0.0, 0.0)
    assert label == "initial hold"


def test_motion_sequence_advances_after_hold_and_distance():
    calibration = MotionCalibration(
        encoder_counts_per_meter=100.0,
        effective_track_width_m=0.2,
    )
    motion = MotionSequenceController(calibration=calibration)

    motion.update(_telemetry(0, 0), now_s=0.0)
    left_cmd, right_cmd, label = motion.update(
        _telemetry(0, 0),
        now_s=INITIAL_HOLD_SECONDS + 0.1,
    )
    assert (left_cmd, right_cmd) != (0.0, 0.0)
    assert label == "forward 1 m (run 1/3)"

    forward_target = int(calibration.distance_target_counts(STRAIGHT_DISTANCE_M))
    left_cmd, right_cmd, label = motion.update(
        _telemetry(-forward_target, -forward_target),
        now_s=INITIAL_HOLD_SECONDS + 0.2,
    )
    assert (left_cmd, right_cmd) == (0.0, 0.0)
    assert label == "hold after forward (run 1/3)"


def test_motion_sequence_contains_expected_turn_steps():
    motion = MotionSequenceController()
    labels = [step.label for step in motion.steps]
    assert f"counterclockwise 360 #{TURN_REPEAT_COUNT} (run {SEQUENCE_REPEAT_COUNT}/{SEQUENCE_REPEAT_COUNT})" in labels
    assert "clockwise 360 (run 1/3)" in labels
    assert "backward 1 m (run 1/3)" in labels
    assert "final hold" in labels


def test_square_sequence_contains_four_sides_and_three_corners():
    motion = SquareSequenceController()
    labels = [step.label for step in motion.steps]
    assert "side 1 forward 1.00 m (square 1/1)" in labels
    assert "side 4 forward 1.00 m (square 1/1)" in labels
    assert "clockwise timed 90 deg turn (square 1/1) corner 3" in labels
    assert "final hold" in labels


def test_square_turn_uses_original_timed_corner_duration():
    motion = SquareSequenceController()
    motion.index = next(i for i, step in enumerate(motion.steps) if step.kind == "turn_timed")
    motion.step_started_s = 0.0
    motion.start_left = 1000
    motion.start_right = 2000
    motion.start_yaw_deg = 170.0

    left_cmd, right_cmd, _ = motion.update(
        {**_telemetry(1000, 2000), "y": -109.1},
        now_s=SQUARE_TURN_SECONDS - 0.01,
    )
    assert (left_cmd, right_cmd) != (0.0, 0.0)

    left_cmd, right_cmd, label = motion.update(
        {**_telemetry(1000, 2000), "y": 45.0},
        now_s=SQUARE_TURN_SECONDS + 0.01,
    )
    assert (left_cmd, right_cmd) == (0.0, 0.0)
    assert label.startswith("hold after corner")
    assert SQUARE_TURN_SECONDS == 1.65
