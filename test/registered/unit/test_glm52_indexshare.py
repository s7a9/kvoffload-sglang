from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import get_nsa_indexshare_flags


def test_glm52_indexshare_layer_mapping():
    config = SimpleNamespace(
        index_topk_freq=4,
        index_topk_pattern=None,
        index_skip_topk_offset=3,
    )
    producer_layers = {
        layer_id
        for layer_id in range(78)
        if not get_nsa_indexshare_flags(config, layer_id)[0]
    }
    assert producer_layers == {0, 1, 2, *range(6, 78, 4)}


def test_glm52_nextn_runs_only_without_reusable_topk():
    config = SimpleNamespace(
        index_topk_freq=4,
        index_topk_pattern=None,
        index_skip_topk_offset=3,
    )
    assert get_nsa_indexshare_flags(config, 78, is_nextn=True) == (True, True)


@pytest.mark.parametrize("frequency,offset", [(0, 3), (4, 0)])
def test_invalid_indexshare_config_is_rejected(frequency, offset):
    config = SimpleNamespace(
        index_topk_freq=frequency,
        index_topk_pattern=None,
        index_skip_topk_offset=offset,
    )
    with pytest.raises(ValueError):
        get_nsa_indexshare_flags(config, 0)
