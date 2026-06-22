import pytest

from openpi.models import pi0_config


@pytest.mark.rl
@pytest.mark.smoke
def test_distributional_value_head_accepts_valid_config() -> None:
    config = pi0_config.Pi0Config(
        enable_rl_value_head=True,
        value_bins=51,
        value_range=(-1.0, 0.0),
        value_temperature=2.0,
    )

    assert config.enable_rl_value_head is True
    assert config.value_bins == 51
    assert config.value_range == (-1.0, 0.0)


@pytest.mark.rl
@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"value_bins": 0}, "value_bins must be >= 1"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
        (
            {"enable_rl_value_head": True, "value_bins": 10, "value_range": (0.0, -1.0)},
            "value_range must be increasing",
        ),
    ],
)
def test_pi0_config_rejects_invalid_rl_related_contracts(kwargs, error_match: str) -> None:
    with pytest.raises(ValueError, match=error_match):
        pi0_config.Pi0Config(**kwargs)
