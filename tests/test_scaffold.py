import pytest

from ddp_amp_bench import activation_elements


def test_activation_element_accounting_scales_with_depth() -> None:
    assert activation_elements(2, 16, 32, 4) == 2 * 16 * 32 * 4
    with pytest.raises(ValueError, match="positive"):
        activation_elements(1, 0, 32, 4)
