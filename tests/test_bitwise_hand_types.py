import pytest

torch = pytest.importorskip("torch", reason="torch is required for bitwise hand type tests")

from modeling.bitwise_hand_types import BitwiseHandTypes


def _make_card_obs(ranks, suits=None):
    if suits is None:
        suits = [1] * len(ranks)
    return {
        "rank": torch.tensor([ranks], dtype=torch.long),
        "suit": torch.tensor([suits], dtype=torch.long),
    }


def test_contained_hand_types_detects_classic_hands():
    straight = _make_card_obs([1, 2, 3, 4, 5])
    flush = _make_card_obs([1, 4, 6, 8, 13], [2, 2, 2, 2, 2])
    full_house = _make_card_obs([1, 1, 1, 2, 2], [1, 2, 3, 4, 1])
    four_kind = _make_card_obs([4, 4, 4, 4, 1], [1, 2, 3, 4, 1])

    straight_types = BitwiseHandTypes.contained_hand_types(straight)
    flush_types = BitwiseHandTypes.contained_hand_types(flush)
    full_house_types = BitwiseHandTypes.contained_hand_types(full_house)
    four_kind_types = BitwiseHandTypes.contained_hand_types(four_kind)

    assert straight_types[0, 3]
    assert flush_types[0, 4]
    assert full_house_types[0, 6]
    assert four_kind_types[0, 2]


def test_subset_metadata_counts_pairs_and_suits():
    obs = _make_card_obs([1, 1, 2, 3, 4], [1, 2, 1, 3, 4])
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    metadata = BitwiseHandTypes.subset_metadata(obs, mask)

    assert metadata["card_count"].tolist() == [3]
    # Two copies of rank 1 and one of rank 2
    assert metadata["rank_counts"][0, 0] == 2
    assert metadata["rank_counts"].sum() == 3
    assert metadata["suit_counts"].max() == 1


def test_possible_hand_types_respects_masks():
    obs = _make_card_obs([1, 2, 3, 4, 5])
    must_include = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
    illegal = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)

    no_options = BitwiseHandTypes.possible_hand_types(obs, illegal, must_include)
    assert not no_options.any()

    all_legal = BitwiseHandTypes.possible_hand_types(
        obs,
        torch.zeros_like(illegal),
        torch.zeros_like(must_include),
    )
    expected = torch.tensor([[False, False, False, True, True, False, False, True]])
    assert torch.equal(all_legal, expected)


def test_batch_dimension_support():
    obs = {
        "rank": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 1, 2, 1]]),
        "suit": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }

    mask = torch.zeros_like(obs["rank"], dtype=torch.bool)
    hand_types = BitwiseHandTypes.possible_hand_types(obs, mask, mask)
    assert hand_types.shape == (2, 8)
    assert hand_types[0, 3] and hand_types[0, 4]
    assert hand_types[1, 0] and hand_types[1, 1]
