"""六个修复点的行为测试 + 引擎生命周期测试。"""

import pytest

import gameplay.trap_sequence_trigger as mod
from gameplay.trap_sequence_trigger import (
    ChainSpec,
    ChainStep,
    MechanismState,
    Timeline,
    TrapChainEngine,
)


# ---------------------------------------------------------------------------
# 修复点 1：延时按时间轴（秒级绝对时刻），帧率波动不错开节奏
# ---------------------------------------------------------------------------


class FakeClock:
    """可手动推进的假墙钟：引擎 advance 多少，时钟就走多少。"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_delay_on_timeline_is_frame_rate_independent():
    info_60 = mod.delay_on_timeline(1.0, 1.0 / 60.0)
    info_30 = mod.delay_on_timeline(1.0, 1.0 / 30.0)

    # 绝对触发偏移恒为 1 秒，变的只是跨越的帧数
    assert info_60["absolute_fire_at"] - info_60["scheduled_at"] == pytest.approx(1.0)
    assert info_30["absolute_fire_at"] - info_30["scheduled_at"] == pytest.approx(1.0)
    assert info_60["frames_at_this_rate"] == pytest.approx(60.0)
    assert info_30["frames_at_this_rate"] == pytest.approx(30.0)
    assert info_60["frame_rate_independent"] is True


def test_chain_keeps_timeline_under_jitter():
    spec = [
        ChainStep("spike", 0.5, "spike_pop", MechanismState.ACTIVE),
        ChainStep("door", 1.0, "open_door", MechanismState.OPEN),
    ]

    def run(frame_pattern):
        clock = FakeClock()
        engine = TrapChainEngine(scene_id="t", clock=clock)
        engine.timeline.on_advance = clock.sleep
        engine.arm("c1", spec)
        engine.trigger("c1_plate", idem_key="k1")
        for dt in frame_pattern:
            engine.timeline.advance(dt)
        fired = [
            (round(e["t"] - 1000.0, 6), e["target_id"], e["state"])
            for e in engine.events
            if e["kind"] == "mechanism_fire"
        ]
        return fired

    stable = run([1.0 / 60.0] * 90)
    jitter = run([0.05, 0.02, 0.08] * 10)

    # 两种帧率下，机关都在同样的墙钟相对时刻触发、顺序一致
    assert stable == jitter == [
        (0.5, "spike", MechanismState.ACTIVE),
        (1.0, "door", MechanismState.OPEN),
    ]


def test_timeline_schedule_and_cancel():
    tl = Timeline(FakeClock())
    hits = []
    task = tl.schedule(0.5, "s1", 0, lambda: hits.append("x"))
    tl.advance(0.4)
    assert hits == []
    task.cancelled = True
    tl.advance(0.2)
    assert hits == []  # 取消后不再触发


# ---------------------------------------------------------------------------
# 修复点 2：激活带幂等键，重复广播只生效一次
# ---------------------------------------------------------------------------


def test_activation_idempotent_dedupes():
    first = mod.activation_idempotent("run-42")
    second = mod.activation_idempotent("run-42")
    third = mod.activation_idempotent("run-42")

    assert first["accepted"] is True
    assert second["accepted"] is False and second["status"] == "duplicate"
    assert third["hits"] == 3


def test_engine_duplicate_broadcast_activates_chain_once():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("door_chain", [ChainStep("gate", 0.3, "open_door", MechanismState.OPEN)])

    r1 = engine.trigger("door_chain_plate", idem_key="broadcast-7", source="player_a")
    # 两名玩家同时踩不同压板，广播重复投递同一幂等键
    r2 = engine.trigger("door_chain_plate", idem_key="broadcast-7", source="player_b")

    assert r1["accepted"] is True
    assert r2["accepted"] is False and r2["status"] == "duplicate"
    assert len(engine.timeline.pending()) == 1  # 只调度了一条链的任务

    engine.timeline.advance(0.5)
    gate_events = [e for e in engine.events if e["kind"] == "mechanism_fire"]
    assert len(gate_events) == 1  # 门只开一次，不会开了又关
    assert engine.states["gate"] == MechanismState.OPEN


def test_state_is_set_not_toggled():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("c", [ChainStep("gate", 0.0, "open_door", MechanismState.OPEN)])
    engine.trigger("c_plate", idem_key="k")
    engine.timeline.advance(0.1)
    # 手动重放同一个动作：状态保持 OPEN，而非翻转
    engine._apply_action(ChainStep("gate", 0.0, "open_door", MechanismState.OPEN))
    assert engine.states["gate"] == MechanismState.OPEN


# ---------------------------------------------------------------------------
# 修复点 3：状态随场景持久化，卸载再加载不复位
# ---------------------------------------------------------------------------


def test_persist_with_scene_facade():
    result = mod.persist_with_scene("dungeon_a", {"gate": MechanismState.OPEN, "spike": MechanismState.IDLE})
    assert result["status"] == "persisted"
    assert result["snapshot"]["states"]["gate"] == MechanismState.OPEN


def test_scene_unload_reload_restores_state():
    engine = TrapChainEngine(scene_id="dungeon", clock=FakeClock())
    engine.arm("c", [ChainStep("gate", 0.2, "open_door", MechanismState.OPEN)])
    engine.trigger("c_plate", idem_key="k1")
    engine.timeline.advance(0.5)
    assert engine.states["gate"] == MechanismState.OPEN

    engine.unload_scene()  # 流式卸载
    assert engine.states == {} or engine.states.get("gate") is None

    engine.load_scene("dungeon")
    # 打开过的门不会重新上锁
    assert engine.states["gate"] == MechanismState.OPEN
    assert engine.chains["c"].trigger_id == "c_plate"


# ---------------------------------------------------------------------------
# 修复点 4：切场景取消任务，旧协程/任务不再执行
# ---------------------------------------------------------------------------


def test_cancel_on_unload_facade():
    sid = "dungeon_b"
    engine = mod._get_engine(sid)
    engine.reset()
    engine.arm("c", [ChainStep("spike", 0.5, "spike_pop", MechanismState.ACTIVE)])
    engine.trigger("c_plate", idem_key="k")
    assert len(engine.timeline.pending(sid)) == 1

    result = mod.cancel_on_unload(sid)
    assert result["cancelled_tasks"] == 1
    assert result["pending_after"] == 0

    engine.timeline.advance(10.0)
    assert engine.states.get("spike") != MechanismState.ACTIVE


def test_stale_task_rejected_after_reload():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("c", [ChainStep("spike", 0.5, "spike_pop", MechanismState.ACTIVE)])
    engine.trigger("c_plate", idem_key="k1")
    old_epoch = engine.epoch
    engine.unload_scene()
    assert engine.epoch == old_epoch + 1

    # 白盒：构造一条残留的旧 epoch 任务，到期必须被拒绝
    stale = engine.timeline.schedule(0.0, "s", old_epoch, lambda: None)
    engine._fire(stale, "c", ChainStep("spike", 0.0, "spike_pop", MechanismState.ACTIVE), "k1")
    assert engine.states.get("spike") != MechanismState.ACTIVE
    assert any(e["kind"] == "stale_task_rejected" for e in engine.events)


# ---------------------------------------------------------------------------
# 修复点 5：停留态轮询，站着不动压板不弹回
# ---------------------------------------------------------------------------


def test_dwell_polling_state_machine():
    r1 = mod.dwell_polling("plate_1", ["player_1"])
    r2 = mod.dwell_polling("plate_1", ["player_1"])  # 站着不动，持续轮询
    r3 = mod.dwell_polling("plate_1", ["player_1", "player_2"])
    r4 = mod.dwell_polling("plate_1", [])  # 最后一人离开

    assert r1["transition"] == "pressed" and r1["state"] == MechanismState.PRESSED
    assert r2["transition"] == "dwell" and r2["state"] == MechanismState.PRESSED
    assert r3["state"] == MechanismState.PRESSED  # 换人/多人仍保持按下
    assert r4["transition"] == "released" and r4["state"] == MechanismState.RELEASED


def test_engine_poll_keeps_plate_pressed_while_occupied():
    clock = FakeClock()
    engine = TrapChainEngine(scene_id="s", clock=clock)
    engine.timeline.on_advance = clock.sleep
    engine.arm("c", [ChainStep("spike", 0.1, "spike_pop", MechanismState.ACTIVE)])
    occupants = ["player_1"]
    engine.register_probe("c_plate", lambda: occupants)

    engine.poll(0.0)
    assert engine.states["c_plate"] == MechanismState.PRESSED
    first_trigger = len([e for e in engine.events if e["kind"] == "chain_triggered"])
    assert first_trigger == 1

    # 连续多帧站着不动：压板保持按下，不会重复触发
    for _ in range(5):
        engine.poll(0.016)
    assert engine.states["c_plate"] == MechanismState.PRESSED
    assert len([e for e in engine.events if e["kind"] == "chain_triggered"]) == 1
    assert any(e["kind"] == "plate_dwell" for e in engine.events)

    occupants = []
    engine.poll(0.016)
    assert engine.states["c_plate"] == MechanismState.RELEASED


# ---------------------------------------------------------------------------
# 修复点 6：回放校验链
# ---------------------------------------------------------------------------


def test_chain_replay_check_passes_and_detects_tamper():
    spec = [
        ChainStep("spike_a", 0.3, "spike_pop", MechanismState.ACTIVE),
        ChainStep("gate", 0.8, "open_door", MechanismState.OPEN),
    ]
    frames = [0.016, 0.05, 0.033, 0.1] * 12  # 合计 2.388s
    result = mod.chain_replay_check("c1", spec, frames)

    assert result["passed"] is True
    assert result["consistent"] is True
    assert result["all_steps_fired"] is True
    assert result["order_ok"] is True
    assert result["tamper_detected"] is True
    assert [t["target_id"] for t in result["trace"]] == ["spike_a", "gate"]
    assert [t["t"] for t in result["trace"]] == pytest.approx([0.3, 0.8])


def test_chain_replay_check_default_contract():
    result = mod.chain_replay_check(1)  # 契约形态：单参数
    assert result is not None
    assert result["consistent"] is True


def test_determinism_across_frame_rates():
    spec = [
        ChainStep("spike_a", 0.5, "spike_pop", MechanismState.ACTIVE),
        ChainStep("gate", 1.0, "open_door", MechanismState.OPEN),
    ]
    result = mod.determinism("c1", spec)
    assert result["deterministic"] is True
    assert result["patterns_tested"] == 3
    targets = [(e["t"], e["target_id"]) for e in result["reference_trace"]]
    assert targets == [(0.5, "spike_a"), (1.0, "gate")]


# ---------------------------------------------------------------------------
# 引擎其余生命周期与可观测性
# ---------------------------------------------------------------------------


def test_validate_trap_rejects_bad_spec():
    bad = mod.validate_trap({"steps": [{"target_id": "", "delay": -1, "action": "", "state": ""}]})
    assert bad["valid"] is False and bad["issues"]

    good = mod.validate_trap([ChainStep("gate", 0.2, "open_door", MechanismState.OPEN)])
    assert good["valid"] is True


def test_cycle_detection():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("a", ChainSpec("a", "plate_a", [ChainStep("plate_b", 0.1, "activate", MechanismState.TRIGGERED)]))
    engine.arm("b", ChainSpec("b", "plate_b", [ChainStep("plate_a", 0.1, "activate", MechanismState.TRIGGERED)]))
    result = engine.validate()
    assert result["valid"] is False
    assert any("引用环" in issue for issue in result["issues"])


def test_fallback_records_failure_without_raising():
    result = mod.fallback("c1", "gate", RuntimeError("jammed"))
    assert result["status"] == "fallback"
    assert mod.metrics()["fallback_count"] >= 1


def test_present_metrics_audit_shapes():
    engine = TrapChainEngine(scene_id="obs", clock=FakeClock())
    engine.arm("c", [ChainStep("gate", 0.1, "open_door", MechanismState.OPEN)])
    engine.trigger("c_plate", idem_key="k")
    engine.timeline.advance(0.2)

    view = engine.present()
    assert view["states"]["gate"] == MechanismState.OPEN
    assert engine.metrics()["event_count"] > 0
    audit = engine.audit()
    assert any(e["kind"] == "chain_triggered" for e in audit["events"])


def test_reset_cancels_pending_and_rearms():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("c", [ChainStep("spike", 0.5, "spike_pop", MechanismState.ACTIVE)])
    engine.trigger("c_plate", idem_key="k")
    result = engine.reset("c")
    assert result["status"] == "reset"
    assert engine.states["spike"] == MechanismState.ARMED
    assert engine.timeline.pending() == []


def test_one_shot_chain_blocks_second_activation():
    engine = TrapChainEngine(scene_id="s", clock=FakeClock())
    engine.arm("c", ChainSpec("c", "c_plate", [ChainStep("gate", 0.1, "open_door", MechanismState.OPEN)], one_shot=True))
    r1 = engine.trigger("c_plate", idem_key="k1")
    r2 = engine.trigger("c_plate", idem_key="k2")  # 不同幂等键，但 one_shot 拦截
    assert r1["accepted"] is True
    assert r2["accepted"] is False and r2["status"] == "oneshot_blocked"


def test_contract_facades_return_non_none():
    # 与原契约测试一致的单参数调用形态，全部要求非 None
    for fn, arg in [
        (mod.arm, 1),
        (mod.trigger, 1),
        (mod.reset, 1),
        (mod.present, 1),
        (mod.validate_trap, 1),
        (mod.metrics, 1),
        (mod.audit, 1),
        (mod.fallback, 1),
        (mod.determinism, 1),
    ]:
        assert fn(arg) is not None
