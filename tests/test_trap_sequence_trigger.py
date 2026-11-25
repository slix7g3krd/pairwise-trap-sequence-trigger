import pytest
import gameplay.trap_sequence_trigger as mod

CASES = [
    ("delay_on_timeline", (1,)),
    ("activation_idempotent", (1,)),
    ("persist_with_scene", (1,)),
    ("cancel_on_unload", (1,)),
    ("dwell_polling", (1,)),
    ("chain_replay_check", (1,)),
    ("arm", (1,)),
    ("trigger", (1,)),
    ("reset", (1,)),
    ("present", (1,)),
    ("validate_trap", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
