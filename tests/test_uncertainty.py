from DigitalTwin.uncertainty import TelemetryDrivenUncertaintyEstimator, TelemetryStatisticsWindow


def test_rolling_uncertainty_features_match_proposal_contract():
    window = TelemetryStatisticsWindow(maxlen=3)
    window.observe(
        dead_reckoning_residual_m=1.0,
        accel_z=9.8,
        gyro_z=0.1,
        velocity_mps=0.2,
        packet_dt_s=0.1,
    )
    window.observe(
        dead_reckoning_residual_m=2.0,
        accel_z=10.2,
        gyro_z=0.3,
        velocity_mps=0.4,
        packet_dt_s=0.2,
    )

    features = window.features(gps_hdop=1.2, gps_satellites=10, fallback_dt_s=0.1)
    assert features.dead_reckoning_residual_m == 1.5
    assert features.model_vector().shape == (5,)


def test_process_covariance_is_positive_diagonal():
    window = TelemetryStatisticsWindow()
    features = window.features(gps_hdop=1.2, gps_satellites=10, fallback_dt_s=0.1)
    Q = TelemetryDrivenUncertaintyEstimator().process_covariance(features, 0.1)
    assert Q.shape == (3, 3)
    assert Q[0, 0] > 0
    assert Q[1, 1] > 0
    assert Q[2, 2] > 0
