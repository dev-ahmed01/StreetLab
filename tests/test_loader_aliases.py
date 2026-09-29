from streetlab_phase1.loaders import _key


def test_key_normalization():
    assert _key("Vehicle ID") == "vehicle_id"
    assert _key("Longitudinal-Speed (m/s)") == "longitudinal_speed_m_s"
