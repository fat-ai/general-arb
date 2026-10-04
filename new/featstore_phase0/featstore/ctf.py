"""featstore.ctf -- the ConditionalTokens id arithmetic, in Python.

An outcome token's ERC-1155 id is not arbitrary: the ConditionalTokens contract derives it
from (collateral token, condition id, outcome index set) --

    collectionId = getCollectionId(parentCollectionId = 0, conditionId, indexSet)
    positionId   = keccak256(collateralToken ++ collectionId)

-- where getCollectionId hashes (conditionId, indexSet) to a point on the alt_bn128 curve
(CTHelpers.sol). So for a known condition and collateral, the ids of outcome 0 (indexSet 1)
and outcome 1 (indexSet 2) can be COMPUTED, and "which token is outcome 0" is arithmetic,
not inference. This is the primary token -> (condition, outcome index) source; the mint
order in PositionSplit transactions verifies it.

Transcribed from gnosis/conditional-tokens-contracts CTHelpers.sol (P, B, getCollectionId
with parentCollectionId == 0, getPositionId). The contract's sqrt is x^((P+1)/4) mod P.
"""
from Crypto.Hash import keccak

P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
B = 3
_SQRT_EXP = (P + 1) // 4


def keccak256(data: bytes) -> bytes:
    h = keccak.new(digest_bits=256)
    h.update(data)
    return h.digest()


def collection_id(condition_id: bytes, index_set: int) -> bytes:
    """getCollectionId(bytes32(0), conditionId, indexSet)."""
    x1 = int.from_bytes(keccak256(condition_id + index_set.to_bytes(32, "big")), "big")
    odd = (x1 >> 255) != 0
    while True:
        x1 = (x1 + 1) % P
        yy = (x1 * x1 % P * x1 + B) % P
        y1 = pow(yy, _SQRT_EXP, P)
        if y1 * y1 % P == yy:
            break
    if (odd and y1 % 2 == 0) or (not odd and y1 % 2 == 1):
        y1 = P - y1
    if y1 % 2 == 1:
        x1 ^= 1 << 254
    return x1.to_bytes(32, "big")


def position_id(collateral: str, condition_id: str, index_set: int) -> str:
    """0x-hex ERC-1155 id of the outcome token for `index_set` (1 = outcome 0, 2 = outcome
    1, 4 = outcome 2 ...) of `condition_id` collateralised by `collateral`."""
    coll = bytes.fromhex(collateral[2:])
    cond = bytes.fromhex(condition_id[2:])
    return "0x" + keccak256(coll + collection_id(cond, index_set)).hex()


def outcome_ids(collateral: str, condition_id: str, n_outcomes: int = 2):
    """[id of outcome 0, id of outcome 1, ...]."""
    return [position_id(collateral, condition_id, 1 << i) for i in range(n_outcomes)]
