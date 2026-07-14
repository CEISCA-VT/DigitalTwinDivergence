from bench_logger import (
    DEFAULT_MOTION_CALIBRATION,
    HOLD_SECONDS,
    INITIAL_HOLD_SECONDS,
    MotionCalibration,
    MotionSequenceController,
    SEQUENCE_REPEAT_COUNT,
    SquareSequenceController,
    STRAIGHT_DISTANCE_M,
    TURN_DEGREES,
    TURN_REPEAT_COUNT,
)


def _telemetry(enc_left: int, enc_right: int) -> dict[str, int]:
    return {"enc_left": enc_left, "enc_right": enc_right}


def test_default_motion_calibration_targets_match_current_week2_values():
    assert round(DEFAULT_MOTION_CALIBRATION.straight_target_counts, 2) == 6646.16
    assert round(DEFAULT_MOTION_CALIBRATION.turn_target_counts, 2) == 2944.02


def test_motion_sequence_starts_with_hold_step():
    motion = MotionSequenceController()
    left_cmd, right_cmd, label = motion.update(_telemetry(0, 0), now_s=0.0)
    assert (left_cmd, right_cmd) == (0.0, 0.0)
    assert label == "initial hold"


def test_motion_sequence_advances_after_hold_and_distance():
    calibration = MotionCalibration(
        encoder_counts_per_meter=100.0,
        effective_track_width_m=0.2,
        nominal_track_width_m=0.17,
    )
    motion = MotionSequenceController(calibration=calibration)

    motion.update(_telemetry(0, 0), now_s=0.0)
    left_cmd, right_cmd, label = motion.update(
        _telemetry(0, 0),
        now_s=INITIAL_HOLD_SECONDS + 0.1,
    )
    assert (left_cmd, right_cmd) != (0.0, 0.0)
    assert label == "forward 1 m (run 1/3)"

    forward_target = int(calibration.straight_target_counts)
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
    assert "side 1 forward 0.25 m (square 1/1)" in labels
    assert "side 4 forward 0.25 m (square 1/1)" in labels
    assert "clockwise 90 deg corner 3 (square 1/1)" in labels
    assert "final hold" in labels
