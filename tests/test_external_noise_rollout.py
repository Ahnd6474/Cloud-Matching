from external_noise_rollout import cosine_noise_schedule


def test_cosine_noise_schedule_hits_endpoints() -> None:
    values = [cosine_noise_schedule(0.1, 0.01, index, 5) for index in range(5)]
    assert values[0] == 0.1
    assert values[-1] == 0.01
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_single_external_injection_uses_start_value() -> None:
    assert cosine_noise_schedule(0.05, 0.0, 0, 1) == 0.05
