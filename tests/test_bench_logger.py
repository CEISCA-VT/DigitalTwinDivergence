import bench_logger

from bench_logger import (
    DEFAULT_MOTION_CALIBRATION,
    HALF_TURN_DEGREES,
    HOLD_SECONDS,
    INITIAL_HOLD_SECONDS,
    MotionCalibration,
    MotionSequenceController,
    SQUARE_SPEED_PROFILES,
    SQUARE_TERRAIN_PROFILES,
    SQUARE_TURN_DEGREES,
    SQUARE_TURN_SECONDS,
    SQUARE_TURN_SECONDS_BY_CORNER,
    SEQUENCE_REPEAT_COUNT,
    SquareSequenceController,
    STRAIGHT_DISTANCE_M,
    TURN_DEGREES,
    apply_square_speed_profile,
    apply_square_terrain_profile,
    set_run_metadata,
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
    assert "clockwise 360 (run 1/3) quarter 4/4" in labels
    assert "backward 1 m (run 1/3)" in labels
    assert "forward 1 m after 360 (run 1/3)" in labels
    assert "clockwise 180 turnaround (run 1/3) quarter 2/2" in labels
    assert "forward 1 m back to start (run 1/3)" in labels
    assert "clockwise 180 restore heading (run 1/3) quarter 2/2" in labels
    assert not any("counterclockwise" in label for label in labels)
    assert "final hold" in labels

    bench_turns = [step for step in motion.steps if step.kind == "turn_timed"]
    assert len(bench_turns) == 8 * SEQUENCE_REPEAT_COUNT
    assert all(step.target == SQUARE_TURN_SECONDS for step in bench_turns)


def test_square_sequence_contains_four_sides_and_four_corners():
    motion = SquareSequenceController()
    labels = [step.label for step in motion.steps]
    assert "side 1 forward 1.00 m (square 1/1)" in labels
    assert "side 4 forward 1.00 m (square 1/1)" in labels
    assert "clockwise timed 90 deg turn (square 1/1) corner 3" in labels
    assert "clockwise timed 90 deg turn (square 1/1) corner 4" in labels
    assert "final hold" in labels


def test_square_turn_uses_current_smooth_timed_corner_duration():
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
    assert SQUARE_TURN_SECONDS == 1.59


def test_repeated_squares_are_continuous_and_turn_between_loops():
    original_repeats = bench_logger.SQUARE_REPEAT_COUNT
    try:
        bench_logger.SQUARE_REPEAT_COUNT = 3
        motion = SquareSequenceController()
    finally:
        bench_logger.SQUARE_REPEAT_COUNT = original_repeats

    assert sum(step.kind == "distance" for step in motion.steps) == 12
    assert sum(step.kind == "turn_timed" for step in motion.steps) == 12
    assert sum(step.label == "initial hold" for step in motion.steps) == 1
    labels = [step.label for step in motion.steps]
    assert "clockwise timed 90 deg turn (square 1/3) corner 4" in labels
    assert "side 1 forward 1.00 m (square 2/3)" in labels


def test_square_profiles_have_separate_dataset_labels():
    assert SQUARE_TERRAIN_PROFILES["smooth"].surface_label == "smooth_kitchen_floor"
    assert SQUARE_TERRAIN_PROFILES["rough"].surface_label == "rough_permeable_concrete"
    assert SQUARE_TERRAIN_PROFILES["smooth"].corner_turn_seconds is None
    assert SQUARE_TERRAIN_PROFILES["rough"].corner_turn_seconds == (1.80, 1.90, 1.75, 1.80)
    assert SQUARE_SPEED_PROFILES["low"].straight_left_cmd == -0.14
    assert SQUARE_SPEED_PROFILES["medium"].straight_left_cmd == -0.20


def test_square_profile_application_and_metadata():
    original_turn = bench_logger.SQUARE_TURN_CW_CMD
    original_turn_seconds = bench_logger.SQUARE_TURN_SECONDS
    original_turn_schedule = bench_logger.SQUARE_TURN_SECONDS_BY_CORNER
    original_straight = bench_logger.SQUARE_STRAIGHT_FORWARD_CMD
    try:
        apply_square_terrain_profile(
            "rough",
            turn_cmd=(0.078, -0.078),
        )
        apply_square_speed_profile("medium")
        set_run_metadata(
            run_id="demo",
            surface="rough_permeable_concrete",
            speed_label="medium",
        )

        assert bench_logger.SQUARE_TURN_CW_CMD == (0.078, -0.078)
        assert bench_logger.SQUARE_TURN_SECONDS == 1.65
        assert bench_logger.SQUARE_TURN_SECONDS_BY_CORNER == (1.80, 1.90, 1.75, 1.80)
        assert bench_logger.SQUARE_STRAIGHT_FORWARD_CMD == (-0.20, -0.20)
        assert bench_logger.RUN_METADATA["surface"] == "rough_permeable_concrete"
    finally:
        bench_logger.SQUARE_TURN_CW_CMD = original_turn
        bench_logger.SQUARE_TURN_SECONDS = original_turn_seconds
        bench_logger.SQUARE_TURN_SECONDS_BY_CORNER = original_turn_schedule
        bench_logger.SQUARE_STRAIGHT_FORWARD_CMD = original_straight
        set_run_metadata()


def test_rough_square_uses_later_corner_boost_without_changing_smooth():
    original_turn = bench_logger.SQUARE_TURN_CW_CMD
    original_turn_seconds = bench_logger.SQUARE_TURN_SECONDS
    original_turn_schedule = bench_logger.SQUARE_TURN_SECONDS_BY_CORNER
    try:
        apply_square_terrain_profile("rough")
        rough = SquareSequenceController()
        rough_turns = [step.target for step in rough.steps if step.kind == "turn_timed"][:4]
        assert rough_turns == [1.80, 1.90, 1.75, 1.80]

        apply_square_terrain_profile("smooth")
        smooth = SquareSequenceController()
        smooth_turns = [step.target for step in smooth.steps if step.kind == "turn_timed"][:4]
        assert smooth_turns == [SQUARE_TURN_SECONDS] * 4
        assert SQUARE_TURN_SECONDS_BY_CORNER is None
    finally:
        bench_logger.SQUARE_TURN_CW_CMD = original_turn
        bench_logger.SQUARE_TURN_SECONDS = original_turn_seconds
        bench_logger.SQUARE_TURN_SECONDS_BY_CORNER = original_turn_schedule


def test_explicit_turn_schedule_overrides_profile_schedule():
    original_turn_schedule = bench_logger.SQUARE_TURN_SECONDS_BY_CORNER
    original_turn_seconds = bench_logger.SQUARE_TURN_SECONDS
    try:
        apply_square_terrain_profile(
            "rough",
            turn_schedule=(1.81, 1.82, 1.83, 1.84),
        )
        motion = SquareSequenceController()
        turns = [step.target for step in motion.steps if step.kind == "turn_timed"][:4]
        assert turns == [1.81, 1.82, 1.83, 1.84]
    finally:
        bench_logger.SQUARE_TURN_SECONDS_BY_CORNER = original_turn_schedule
        bench_logger.SQUARE_TURN_SECONDS = original_turn_seconds
