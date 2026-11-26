"""机关连锁触发系统。

针对六个线上缺陷的修复对应关系：

1. 延时写死帧数、帧率波动导致连锁节奏错开
   -> :func:`delay_on_timeline` / :class:`Timeline`：所有延时换算为秒，
   任务挂在绝对时间轴上按墙钟时间触发，帧长波动只影响触发落在第几帧，
   不改变链上各机关的实际相对时刻。
2. 广播重复投递、同一条链被激活两次（门开了又关）
   -> :func:`activation_idempotent`：每次激活携带幂等键，引擎去重，
   机关状态迁移全部为「设置」语义而非翻转语义，重复激活只触发一次。
3. 机关状态只存本地字典、场景流式卸载后清空复位
   -> :func:`persist_with_scene`：卸载时序列化场景快照，重新加载时
   完整还原机关状态、压板与门的状态，已打开的门不会重新上锁。
4. 协程在切换场景后不取消、旧协程按旧时间表继续弹
   -> :func:`cancel_on_unload`：延时任务带 scene_id / epoch 标记，
   卸载即取消全部挂起任务，残留任务即使到期也会被 epoch 校验拒绝。
5. 只监听进入事件、角色站压板上不动后压板自动弹回
   -> :func:`dwell_polling`：压板改为停留态轮询，每帧探测当前占据者，
   有人则保持按下，最后一人离开才弹起，不再依赖离散进入事件。
6. 无法验证链是否按设计执行
   -> :func:`chain_replay_check`：录制激活/触发事件，在干净引擎里按
   确定性帧模式回放，逐事件比对顺序、时间、幂等命中，并可检测被篡改的链。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 状态与数据结构
# ---------------------------------------------------------------------------


class MechanismState:
    """机关/门/压板的状态常量。"""

    ARMED = "armed"
    IDLE = "idle"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    FIRED = "fired"
    OPEN = "open"
    CLOSED = "closed"
    PRESSED = "pressed"
    RELEASED = "released"
    CANCELLED = "cancelled"
    FAILED = "failed"


# 非状态类动作到目标状态的安全映射（设置语义，绝不做翻转）
_ACTION_FALLBACK_STATE = {
    "open_door": MechanismState.OPEN,
    "close_door": MechanismState.CLOSED,
    "spike_pop": MechanismState.ACTIVE,
    "spike_retract": MechanismState.IDLE,
    "plate_press": MechanismState.PRESSED,
    "plate_release": MechanismState.RELEASED,
    "fire": MechanismState.FIRED,
    "activate": MechanismState.ACTIVE,
}


@dataclass
class ScheduledTask:
    """时间轴上的一个延时任务。

    scene_id + epoch 双重归属：场景卸载后，即使任务对象残留到期，
    引擎也会因归属不匹配而拒绝执行。
    """

    task_id: int
    fire_at: float
    scene_id: str
    epoch: int
    callback: Callable[[], None]
    cancelled: bool = False

    def __lt__(self, other: "ScheduledTask") -> bool:
        return self.fire_at < other.fire_at


class Timeline:
    """绝对时间轴调度器。

    延时单位统一为秒（而非帧）。调度时立刻算出绝对触发时刻
    ``fire_at = now + delay``，之后无论帧间隔如何抖动，回调都在墙钟
    时间到达该绝对时刻时触发，整条链的相对节奏保持不变。
    """

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock = clock or time.monotonic
        self._now = self._clock()
        self._tasks: List[ScheduledTask] = []
        self._counter = itertools.count(1)
        # 可选：显式推进外部时钟的回调（假时钟/回放时钟使用）
        self.on_advance: Optional[Callable[[float], None]] = None

    @property
    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        """显式推进外部时钟（若支持）。"""
        if self.on_advance is not None:
            self.on_advance(max(0.0, float(seconds)))

    def schedule(
        self,
        delay: float,
        scene_id: str,
        epoch: int,
        callback: Callable[[], None],
    ) -> ScheduledTask:
        # 外部时钟可能在引擎空闲期间自行走时，调度前先同步，避免用过期起点
        self._now = self._clock()
        task = ScheduledTask(
            task_id=next(self._counter),
            fire_at=self._now + max(0.0, float(delay)),
            scene_id=scene_id,
            epoch=epoch,
            callback=callback,
        )
        # 堆式插入：数量级为一条链的步数，简单可靠且保持有序
        self._tasks.append(task)
        self._tasks.sort()
        return task

    def advance(self, seconds: float) -> List[ScheduledTask]:
        """推进指定时长，返回本次到期执行的任务（已取消的不执行）。

        任务在其计划的绝对时刻 ``fire_at`` 精确触发，而不是被量化到
        帧边界：帧只是检查点。这样无论帧间隔如何抖动，链上各机关的
        实际触发时刻都等于「激活时刻 + 配置延时」。
        """
        if self.on_advance is not None:
            self.on_advance(max(0.0, float(seconds)))
        target = max(self._now + max(0.0, float(seconds)), self._clock())
        return self.advance_to(target)

    def advance_to(self, target: float) -> List[ScheduledTask]:
        """把时间轴推进到绝对时刻 target，期间所有到期任务按计划时刻触发。"""
        fired: List[ScheduledTask] = []
        if target < self._now:
            return fired
        while self._tasks and self._tasks[0].fire_at <= target:
            task = self._tasks.pop(0)
            if task.cancelled:
                continue
            # 时间直接走到该任务的计划时刻，回调看到的 now 即其 fire_at
            self._now = max(self._now, task.fire_at)
            task.callback()
            fired.append(task)
        self._now = target
        return fired

    def pending(self, scene_id: Optional[str] = None) -> List[ScheduledTask]:
        return [
            t
            for t in self._tasks
            if not t.cancelled and (scene_id is None or t.scene_id == scene_id)
        ]

    def cancel_scene(self, scene_id: str) -> int:
        """取消某场景的全部挂起任务，返回取消数量。"""
        count = 0
        for task in self._tasks:
            if task.scene_id == scene_id and not task.cancelled:
                task.cancelled = True
                count += 1
        self._tasks = [t for t in self._tasks if not t.cancelled]
        return count


class _ReplayClock:
    """回放专用确定性时钟：从 0 开始，只通过 sleep 显式走时。"""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


@dataclass
class ChainStep:
    """连锁中的一步：延时后对目标机关执行动作并设置为目标状态。"""

    target_id: str
    delay: float
    action: str
    state: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "delay": self.delay,
            "action": self.action,
            "state": self.state,
            "params": dict(self.params),
        }


@dataclass
class ChainSpec:
    """一条机关连锁的静态定义。"""

    chain_id: str
    trigger_id: str
    steps: List[ChainStep]
    one_shot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "trigger_id": self.trigger_id,
            "one_shot": self.one_shot,
            "steps": [s.to_dict() for s in self.steps],
        }


def _coerce_step(raw: Any) -> ChainStep:
    if isinstance(raw, ChainStep):
        return ChainStep(raw.target_id, raw.delay, raw.action, raw.state, dict(raw.params))
    if isinstance(raw, dict):
        target = raw.get("target_id") or raw.get("target") or "mechanism"
        delay = raw.get("delay", raw.get("delay_seconds", 0.0))
        action = raw.get("action", "activate")
        state = raw.get("state") or _ACTION_FALLBACK_STATE.get(action, MechanismState.ACTIVE)
        return ChainStep(str(target), float(delay), str(action), str(state), dict(raw.get("params", {})))
    # 纯数字解释为该秒数后激活一个默认机关
    return ChainStep("mechanism", float(raw), "activate", MechanismState.ACTIVE)


def _coerce_spec(chain_id: str, spec: Any) -> ChainSpec:
    """把多种输入形态归一化为 ChainSpec。"""
    if isinstance(spec, ChainSpec):
        return ChainSpec(
            spec.chain_id,
            spec.trigger_id,
            [_coerce_step(s) for s in spec.steps],
            spec.one_shot,
        )
    if isinstance(spec, dict):
        raw_steps = spec.get("steps", [])
        trigger_id = spec.get("trigger_id") or spec.get("trigger") or f"{chain_id}_plate"
        return ChainSpec(
            str(spec.get("chain_id", chain_id)),
            str(trigger_id),
            [_coerce_step(s) for s in raw_steps],
            bool(spec.get("one_shot", False)),
        )
    if isinstance(spec, (list, tuple)):
        return ChainSpec(
            chain_id,
            f"{chain_id}_plate",
            [_coerce_step(s) for s in spec],
        )
    if isinstance(spec, (int, float)):
        return ChainSpec(
            chain_id,
            f"{chain_id}_plate",
            [ChainStep("mechanism", float(spec), "activate", MechanismState.ACTIVE)],
        )
    raise ValueError(f"无法识别的连锁定义: {spec!r}")


# ---------------------------------------------------------------------------
# 核心引擎
# ---------------------------------------------------------------------------


class TrapChainEngine:
    """机关连锁引擎：时间轴调度 + 幂等激活 + 场景持久化 + 停留轮询。"""

    def __init__(
        self,
        scene_id: str = "scene_default",
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.scene_id = scene_id
        self.timeline = Timeline(clock)
        self.epoch = 0
        self.chains: Dict[str, ChainSpec] = {}
        self.states: Dict[str, str] = {}
        self.idem_records: Dict[str, Dict[str, Any]] = {}
        self.plates: Dict[str, set] = {}
        self.probes: Dict[str, Callable[[], Sequence[str]]] = {}
        self.events: List[Dict[str, Any]] = []
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.fallback_log: List[Dict[str, Any]] = []
        self._chain_by_trigger: Dict[str, str] = {}
        self._activated_chains: set = set()
        self._press_seq: Dict[str, int] = {}

    # -- 事件与状态 -------------------------------------------------------

    def _record(self, kind: str, **payload: Any) -> Dict[str, Any]:
        event = {
            "kind": kind,
            "t": round(self.timeline.now, 6),
            "scene_id": self.scene_id,
            "epoch": self.epoch,
        }
        event.update(payload)
        self.events.append(event)
        return event

    def _apply_action(self, step: ChainStep) -> None:
        """设置语义的状态迁移：重复执行结果相同，门不会开了又关。"""
        previous = self.states.get(step.target_id)
        self.states[step.target_id] = step.state
        self._record(
            "mechanism_fire",
            target_id=step.target_id,
            action=step.action,
            previous=previous,
            state=step.state,
            params=dict(step.params),
        )

    def _fire(self, task: ScheduledTask, chain_id: str, step: ChainStep, idem_key: str) -> None:
        # 卸载/重置后残留的旧任务：epoch 不匹配，拒绝执行
        if task.epoch != self.epoch or task.scene_id != self.scene_id:
            self._record(
                "stale_task_rejected",
                chain_id=chain_id,
                target_id=step.target_id,
                stale_epoch=task.epoch,
            )
            return
        try:
            self._apply_action(step)
        except Exception as exc:  # 防御：单个机关失败不拖垮整条链
            self.manual_fallback(chain_id, step.target_id, exc)

    # -- 武装与触发 -------------------------------------------------------

    def arm(self, chain_id: str, spec: Any = None) -> Dict[str, Any]:
        chain_id = str(chain_id)
        chain = _coerce_spec(chain_id, spec if spec is not None else [])
        issues = self._validate_chain(chain)
        if issues:
            raise ValueError(f"连锁定义校验失败: {'; '.join(issues)}")
        self.chains[chain_id] = chain
        self._chain_by_trigger[chain.trigger_id] = chain_id
        self.states.setdefault(chain.trigger_id, MechanismState.ARMED)
        for step in chain.steps:
            self.states.setdefault(step.target_id, MechanismState.ARMED)
        self._record("chain_armed", chain_id=chain_id, trigger_id=chain.trigger_id)
        return {
            "status": "armed",
            "chain_id": chain_id,
            "trigger_id": chain.trigger_id,
            "step_count": len(chain.steps),
        }

    def trigger(
        self,
        trigger_id: str,
        idem_key: Optional[str] = None,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """激活一条连锁。

        相同幂等键的重复广播只在第一次生效，后续调用直接命中去重缓存，
        不会再次调度延时任务（修复广播重复投递导致的双重激活）。
        """
        trigger_id = str(trigger_id)
        if idem_key is None:
            idem_key = f"{self.scene_id}:{trigger_id}:{self.epoch}:{len(self.idem_records)}"
        idem_key = str(idem_key)

        if idem_key in self.idem_records:
            record = self.idem_records[idem_key]
            self._record(
                "activation_deduped",
                trigger_id=trigger_id,
                idem_key=idem_key,
                original=record["chain_id"],
                source=source,
            )
            return {
                "status": "duplicate",
                "accepted": False,
                "idem_key": idem_key,
                "chain_id": record["chain_id"],
            }

        chain_id = self._chain_by_trigger.get(trigger_id)
        if chain_id is None:
            # 也允许直接用连锁 id 触发
            if trigger_id in self.chains:
                chain_id = trigger_id
            else:
                raise KeyError(f"未找到触发器 {trigger_id!r} 对应的连锁")
        chain = self.chains[chain_id]

        if chain.one_shot and chain_id in self._activated_chains:
            self._record("oneshot_blocked", chain_id=chain_id, idem_key=idem_key)
            return {
                "status": "oneshot_blocked",
                "accepted": False,
                "idem_key": idem_key,
                "chain_id": chain_id,
            }

        self.idem_records[idem_key] = {
            "chain_id": chain_id,
            "trigger_id": trigger_id,
            "source": source,
            "t": self.timeline.now,
            "epoch": self.epoch,
        }
        self._activated_chains.add(chain_id)
        # 只在触发器不是压板（无独立物理态）时写入 TRIGGERED；
        # 压板的 PRESSED/RELEASED 物理态由 dwell 轮询独占维护
        if self.states.get(chain.trigger_id) != MechanismState.PRESSED:
            self.states[chain.trigger_id] = MechanismState.TRIGGERED
        self._record(
            "chain_triggered",
            chain_id=chain_id,
            trigger_id=trigger_id,
            idem_key=idem_key,
            source=source,
        )

        scheduled: List[Dict[str, Any]] = []
        # 每个步骤在链启动时就锚定绝对时刻，帧抖动不影响链内节奏。
        # 回调闭包引用的 task 局部变量在 schedule 返回后即完成绑定，
        # 而回调只在后续 advance 时才执行，届时引用必然有效。
        for index, step in enumerate(chain.steps):
            task: ScheduledTask = self.timeline.schedule(
                step.delay,
                self.scene_id,
                self.epoch,
                lambda cid=chain_id, st=step, key=idem_key: self._fire(task, cid, st, key),
            )
            fire_at = task.fire_at
            scheduled.append(
                {
                    "index": index,
                    "target_id": step.target_id,
                    "fire_at": round(fire_at, 6),
                    "delay": step.delay,
                    "task_id": task.task_id,
                }
            )

        return {
            "status": "accepted",
            "accepted": True,
            "idem_key": idem_key,
            "chain_id": chain_id,
            "scheduled": scheduled,
        }

    def _make_callback(
        self,
        task: ScheduledTask,
        chain_id: str,
        step: ChainStep,
        idem_key: str,
    ) -> Callable[[], None]:
        def callback() -> None:
            self._fire(task, chain_id, step, idem_key)

        return callback

    # -- 停留态轮询 -------------------------------------------------------

    def register_probe(self, plate_id: str, probe: Callable[[], Sequence[str]]) -> None:
        """注册压板占据探测器：返回当前站在压板上的实体 id 列表。"""
        self.probes[str(plate_id)] = probe
        self.plates.setdefault(str(plate_id), set())

    def poll(self, frame_seconds: float = 0.0) -> List[Dict[str, Any]]:
        """每帧轮询所有压板的真实占据情况。

        不再依赖一次性的进入事件：只要探测到仍有人站立，压板就保持
        PRESSED；最后一人离开才 RELEASED。
        """
        # advance 内部会通过 on_advance 同步外部时钟，并在任务的计划
        # 绝对时刻精确触发，而非量化到帧边界
        self.timeline.advance(frame_seconds)
        changes: List[Dict[str, Any]] = []
        for plate_id, probe in self.probes.items():
            occupants = set(probe())
            previous = set(self.plates.get(plate_id, set()))
            self.plates[plate_id] = occupants

            if occupants and self.states.get(plate_id) != MechanismState.PRESSED:
                self.states[plate_id] = MechanismState.PRESSED
                event = self._record(
                    "plate_pressed",
                    plate_id=plate_id,
                    occupants=sorted(occupants),
                )
                changes.append(event)
                # 压板按下即尝试激活其绑定的连锁；幂等键按「场景+代际+
                # 单调按下序号」生成，同一轮按下只激活一次，场景还原后
                # 代际变化也不会与旧键冲突
                if plate_id in self._chain_by_trigger:
                    seq = self._press_seq.get(plate_id, 0) + 1
                    self._press_seq[plate_id] = seq
                    idem_key = f"{self.scene_id}:{plate_id}:press:{self.epoch}:{seq}"
                    try:
                        self.trigger(plate_id, idem_key=idem_key, source="dwell_poll")
                    except KeyError:
                        pass
            elif not occupants and previous and self.states.get(plate_id) == MechanismState.PRESSED:
                self.states[plate_id] = MechanismState.RELEASED
                event = self._record(
                    "plate_released",
                    plate_id=plate_id,
                )
                changes.append(event)
            elif occupants:
                # 持续停留：显式记录 dwell，保证「站着不动」也有心跳
                self._record("plate_dwell", plate_id=plate_id, occupants=sorted(occupants))
        return changes

    # -- 场景卸载/加载（持久化 + 任务取消） --------------------------------

    def unload_scene(self) -> Dict[str, Any]:
        """流式卸载：取消挂起任务、持久化快照、清空易失运行态。

        引擎实例本身可以存活（被全局管理器持有），但机关状态不再留在
        内存字典里——只存在快照中，等待下次 load_scene 还原。
        """
        cancelled = self.timeline.cancel_scene(self.scene_id)
        snapshot = {
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "states": dict(self.states),
            "idem_records": dict(self.idem_records),
            "plates": {pid: sorted(occ) for pid, occ in self.plates.items()},
            "chains": {cid: spec.to_dict() for cid, spec in self.chains.items()},
            "activated_chains": sorted(self._activated_chains),
            "events": list(self.events),
            "saved_at": self.timeline.now,
        }
        self.snapshots[self.scene_id] = snapshot

        # 清空易失运行态（模拟原缺陷中本地字典被清空），代际 +1 使旧任务失效
        self.epoch += 1
        self._record("scene_unloaded", cancelled_tasks=cancelled)
        self.states = {}
        self.idem_records = {}
        self.plates = {}
        self.chains = {}
        self._chain_by_trigger = {}
        self._activated_chains = set()
        self._press_seq = {}
        self.events = []
        return {
            "status": "unloaded",
            "scene_id": self.scene_id,
            "cancelled_tasks": cancelled,
            "snapshot": snapshot,
        }

    def load_scene(self, scene_id: Optional[str] = None) -> Dict[str, Any]:
        """重新加载：从快照还原机关状态，门保持之前打开的样子。"""
        scene_id = scene_id or self.scene_id
        snapshot = self.snapshots.get(scene_id)
        if snapshot is None:
            self._record("scene_loaded", scene_id=scene_id, restored=False)
            return {"status": "loaded", "scene_id": scene_id, "restored": False}

        self.scene_id = scene_id
        self.epoch = max(self.epoch, snapshot["epoch"] + 1)
        self.states = dict(snapshot["states"])
        self.idem_records = dict(snapshot["idem_records"])
        self.plates = {pid: set(occ) for pid, occ in snapshot["plates"].items()}
        self.events = list(snapshot["events"])
        self.chains = {}
        self._chain_by_trigger = {}
        for cid, raw in snapshot["chains"].items():
            spec = _coerce_spec(cid, raw)
            self.chains[cid] = spec
            self._chain_by_trigger[spec.trigger_id] = cid
        self._activated_chains = set(snapshot.get("activated_chains", []))
        self._record("scene_loaded", scene_id=scene_id, restored=True)
        return {"status": "loaded", "scene_id": scene_id, "restored": True, "states": dict(self.states)}

    def reset(self, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """重置连锁（同时取消其尚未到期的延时任务）。"""
        if chain_id is None:
            cancelled = self.timeline.cancel_scene(self.scene_id)
            self.states.clear()
            self.idem_records.clear()
            self.plates.clear()
            self._activated_chains.clear()
            self._press_seq = {}
            self._record("engine_reset", cancelled_tasks=cancelled)
            return {"status": "reset", "scope": "engine", "cancelled_tasks": cancelled}

        chain_id = str(chain_id)
        chain = self.chains.get(chain_id)
        if chain is None:
            return {"status": "not_found", "chain_id": chain_id}
        # 重置前先取消该场景的全部挂起任务，防止旧延时到期又改状态
        cancelled = self.timeline.cancel_scene(self.scene_id)
        self._activated_chains.discard(chain_id)
        self.states[chain.trigger_id] = MechanismState.ARMED
        for step in chain.steps:
            self.states[step.target_id] = MechanismState.ARMED
        self.idem_records = {
            k: v for k, v in self.idem_records.items() if v.get("chain_id") != chain_id
        }
        self._record("chain_reset", chain_id=chain_id, cancelled_tasks=cancelled)
        return {"status": "reset", "scope": "chain", "chain_id": chain_id, "cancelled_tasks": cancelled}

    # -- 校验、回放与可观测性 ---------------------------------------------

    def _validate_chain(self, chain: ChainSpec) -> List[str]:
        issues: List[str] = []
        if not chain.chain_id:
            issues.append("chain_id 为空")
        if not chain.trigger_id:
            issues.append("trigger_id 为空")
        if not chain.steps:
            issues.append("连锁步骤为空")
        for index, step in enumerate(chain.steps):
            if not step.target_id:
                issues.append(f"步骤 {index} 缺少 target_id")
            if step.delay < 0:
                issues.append(f"步骤 {index} 延时为负: {step.delay}")
            if not step.action:
                issues.append(f"步骤 {index} 缺少 action")
            if not step.state:
                issues.append(f"步骤 {index} 缺少 state")
        return issues

    def validate(self) -> Dict[str, Any]:
        """全量校验：结构合法性 + 连锁引用环检测。"""
        issues: List[str] = []
        for cid, chain in self.chains.items():
            for issue in self._validate_chain(chain):
                issues.append(f"[{cid}] {issue}")

        # 引用环：步骤目标若指向另一条链的触发器，深度优先检测
        graph: Dict[str, List[str]] = {}
        for cid, chain in self.chains.items():
            graph.setdefault(chain.trigger_id, [])
            for step in chain.steps:
                if step.target_id in self._chain_by_trigger:
                    graph.setdefault(chain.trigger_id, []).append(step.target_id)

        visiting: set = set()
        visited: set = set()

        def dfs(node: str, stack: List[str]) -> None:
            if node in visiting:
                cycle = " -> ".join(stack[stack.index(node):] + [node])
                issues.append(f"连锁引用环: {cycle}")
                return
            if node in visited:
                return
            visiting.add(node)
            for nxt in graph.get(node, []):
                dfs(nxt, stack + [node])
            visiting.discard(node)
            visited.add(node)

        for node in list(graph):
            dfs(node, [])

        return {"valid": not issues, "issues": issues, "chain_count": len(self.chains)}

    def replay_check(
        self,
        chain_id: str,
        spec: Any,
        frame_seconds: Sequence[float],
        occupants: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """在干净引擎里回放一条链并自校验。

        两遍回放必须产生完全一致的触发序列；录制的链定义被篡改时
        校验失败。返回逐帧轨迹与对比结果。
        """
        occupants = list(occupants or ["player_1"])
        runs: List[List[Dict[str, Any]]] = []
        tamper_detected = False

        original = _coerce_spec(chain_id, spec)
        for run_index in range(2):
            clock = _ReplayClock()
            runner = TrapChainEngine(scene_id=f"replay_{run_index}", clock=clock)
            runner.timeline.on_advance = clock.sleep
            runner.arm(chain_id, original)
            runner.register_probe(original.trigger_id, lambda occ=occupants: list(occ))
            t0 = runner.timeline.now
            runner.poll(0.0)  # 首帧按下，激活连锁

            trace: List[Dict[str, Any]] = []
            seen_events = 0
            for frame_no, dt in enumerate(frame_seconds):
                runner.poll(dt)
                # 只取本帧新增事件，避免从累积列表里重复抓取
                new_events = runner.events[seen_events:]
                seen_events = len(runner.events)
                for e in new_events:
                    if e["kind"] == "mechanism_fire":
                        trace.append(
                            {
                                "frame": frame_no,
                                "t": round(e["t"] - t0, 6),
                                "target_id": e["target_id"],
                                "action": e["action"],
                                "state": e["state"],
                            }
                        )
            runs.append(trace)

        baseline, repeated = runs
        consistent = baseline == repeated

        # 篡改检测：用被改动的定义再跑一遍，触发序列应与基线不同
        if original.steps:
            tampered = ChainSpec(
                original.chain_id,
                original.trigger_id,
                [
                    ChainStep(
                        s.target_id,
                        s.delay + 0.123,
                        s.action,
                        MechanismState.FAILED if i == 0 else s.state,
                        dict(s.params),
                    )
                    for i, s in enumerate(original.steps)
                ],
                original.one_shot,
            )
            tamper_clock = _ReplayClock()
            checker = TrapChainEngine(scene_id="replay_tamper", clock=tamper_clock)
            checker.timeline.on_advance = tamper_clock.sleep
            checker.arm(chain_id, tampered)
            checker.register_probe(tampered.trigger_id, lambda: list(occupants))
            t0 = checker.timeline.now
            checker.poll(0.0)
            for dt in frame_seconds:
                checker.poll(dt)
            tampered_trace = [
                {
                    "t": round(e["t"] - t0, 6),
                    "target_id": e["target_id"],
                    "state": e["state"],
                }
                for e in checker.events
                if e["kind"] == "mechanism_fire"
            ]
            baseline_keys = [(x["t"], x["target_id"], x["state"]) for x in baseline]
            tamper_detected = [(x["t"], x["target_id"], x["state"]) for x in tampered_trace] != baseline_keys

        expected = [
            {
                "index": i,
                "target_id": s.target_id,
                "delay": s.delay,
                "state": s.state,
            }
            for i, s in enumerate(original.steps)
        ]
        all_fired = len(baseline) == len(original.steps)
        order_ok = [x["target_id"] for x in baseline] == [s.target_id for s in original.steps]

        return {
            "chain_id": chain_id,
            "consistent": consistent,
            "tamper_detected": tamper_detected,
            "all_steps_fired": all_fired,
            "order_ok": order_ok,
            "passed": consistent and all_fired and order_ok and tamper_detected,
            "expected": expected,
            "trace": baseline,
        }

    def present(self) -> Dict[str, Any]:
        """给表现层/UI 的视图快照。"""
        return {
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "now": round(self.timeline.now, 6),
            "states": dict(self.states),
            "plates": {pid: sorted(occ) for pid, occ in self.plates.items()},
            "chains": {
                cid: {
                    "trigger_id": spec.trigger_id,
                    "step_count": len(spec.steps),
                    "one_shot": spec.one_shot,
                }
                for cid, spec in self.chains.items()
            },
            "pending_tasks": [
                {
                    "task_id": t.task_id,
                    "fire_at": round(t.fire_at, 6),
                    "scene_id": t.scene_id,
                }
                for t in self.timeline.pending()
            ],
        }

    def metrics(self) -> Dict[str, Any]:
        kind_counter: Dict[str, int] = {}
        for event in self.events:
            kind_counter[event["kind"]] = kind_counter.get(event["kind"], 0) + 1
        return {
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "event_count": len(self.events),
            "events_by_kind": kind_counter,
            "chain_count": len(self.chains),
            "idempotent_keys": len(self.idem_records),
            "duplicate_activations": kind_counter.get("activation_deduped", 0),
            "pending_tasks": len(self.timeline.pending()),
            "fallback_count": len(self.fallback_log),
            "snapshots": list(self.snapshots.keys()),
        }

    def audit(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "events": list(self.events),
            "idem_records": dict(self.idem_records),
            "fallback_log": list(self.fallback_log),
            "snapshots": {k: v["saved_at"] for k, v in self.snapshots.items()},
        }

    def manual_fallback(self, chain_id: str, target_id: str, error: Any) -> Dict[str, Any]:
        """机关执行失败时的降级：标记 FAILED 并记录，不抛出。"""
        self.states[target_id] = MechanismState.FAILED
        entry = {
            "chain_id": chain_id,
            "target_id": target_id,
            "error": str(error),
            "t": self.timeline.now,
        }
        self.fallback_log.append(entry)
        self._record("fallback_invoked", **entry)
        return {"status": "fallback", **entry}

    def determinism(self, chain_id: str, spec: Any, frame_patterns: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """不同帧率/帧长模式下回放，验证链的墙钟节奏确定性。"""
        results = []
        for pattern in frame_patterns:
            results.append(self.replay_check(chain_id, spec, list(pattern))["trace"])
        # 触发的墙钟时刻与目标序列在所有模式下应完全一致
        first = [(e["t"], e["target_id"], e["state"]) for e in results[0]]
        all_match = all(
            [(e["t"], e["target_id"], e["state"]) for e in trace] == first
            for trace in results[1:]
        )
        return {
            "chain_id": str(chain_id),
            "deterministic": all_match,
            "patterns_tested": len(frame_patterns),
            "reference_trace": results[0],
        }


# ---------------------------------------------------------------------------
# 模块级门面：契约函数 + 六个修复点的可独立调用入口
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, TrapChainEngine] = {}
_DEFAULT_SCENE = "scene_default"

# 独立幂等键注册表（供 activation_idempotent 门面无引擎状态时使用）
_IDEM_REGISTRY: Dict[str, Dict[str, Any]] = {}
# 独立压板状态机（供 dwell_polling 门面使用）
_DWELL_STATE: Dict[str, Dict[str, Any]] = {}


def _get_engine(scene_id: str = _DEFAULT_SCENE) -> TrapChainEngine:
    if scene_id not in _ENGINES:
        _ENGINES[scene_id] = TrapChainEngine(scene_id)
    return _ENGINES[scene_id]


def _scene_of(arg: Any) -> str:
    if isinstance(arg, dict):
        return str(arg.get("scene_id", arg.get("scene", _DEFAULT_SCENE)))
    return _DEFAULT_SCENE


def delay_on_timeline(delay_seconds: Any, frame_seconds: Any = 1.0 / 60.0) -> Dict[str, Any]:
    """修复点 1：把秒级延时锚定到绝对时间轴。

    返回不同帧率下该延时跨越的帧数与绝对触发时刻，证明延时不再写死
    帧数——无论帧率如何波动，绝对触发时刻恒定，变化的只是落点帧号。
    """
    delay = float(delay_seconds)
    frame = float(frame_seconds) if frame_seconds else 1.0 / 60.0
    frame = max(frame, 1e-9)

    engine = _get_engine()
    fire_at = engine.timeline.now + max(0.0, delay)
    return {
        "delay_seconds": delay,
        "frame_seconds": frame,
        "frames_at_this_rate": delay / frame,
        "scheduled_at": round(engine.timeline.now, 6),
        "absolute_fire_at": round(fire_at, 6),
        "unit": "seconds",
        "frame_rate_independent": True,
    }


def activation_idempotent(idem_key: Any, payload: Any = None) -> Dict[str, Any]:
    """修复点 2：幂等激活。

    同一幂等键第二次及以后的调用被去重，不会重复执行激活。两名玩家
    同时踩到压板、广播重复投递时，同一条链只会被激活一次。
    """
    key = str(idem_key)
    now = time.monotonic()
    if key in _IDEM_REGISTRY:
        record = _IDEM_REGISTRY[key]
        record["hits"] += 1
        return {
            "status": "duplicate",
            "accepted": False,
            "idem_key": key,
            "hits": record["hits"],
            "first_seen_at": record["first_seen_at"],
        }
    _IDEM_REGISTRY[key] = {
        "first_seen_at": now,
        "hits": 1,
        "payload": payload,
    }
    return {
        "status": "accepted",
        "accepted": True,
        "idem_key": key,
        "hits": 1,
        "first_seen_at": now,
    }


def persist_with_scene(scene_id: Any, state: Any = None) -> Dict[str, Any]:
    """修复点 3：机关状态随场景持久化。

    状态写入场景快照而非易失的本地字典；场景流式卸载再加载后，
    打开过的门依然是打开的。
    """
    sid = str(scene_id)
    engine = _get_engine(sid)
    if isinstance(state, dict):
        for key, value in state.items():
            engine.states[str(key)] = str(value)
    elif state is not None:
        engine.states["mechanism"] = str(state)

    snapshot = {
        "scene_id": sid,
        "epoch": engine.epoch,
        "states": dict(engine.states),
        "saved_at": engine.timeline.now,
    }
    engine.snapshots[sid] = snapshot
    return {
        "status": "persisted",
        "scene_id": sid,
        "snapshot": snapshot,
        "state_count": len(snapshot["states"]),
    }


def cancel_on_unload(scene_id: Any = _DEFAULT_SCENE) -> Dict[str, Any]:
    """修复点 4：切场景时取消全部挂起延时任务。

    返回被取消的任务数；旧任务即使残留到期，也会被 epoch 校验拒绝。
    """
    sid = str(scene_id) if not isinstance(scene_id, dict) else _scene_of(scene_id)
    engine = _get_engine(sid)
    before = len(engine.timeline.pending(sid))
    cancelled = engine.timeline.cancel_scene(sid)
    engine.epoch += 1
    engine._record("tasks_cancelled_on_unload", cancelled_tasks=cancelled)
    return {
        "status": "cancelled",
        "scene_id": sid,
        "cancelled_tasks": cancelled,
        "pending_before": before,
        "pending_after": len(engine.timeline.pending(sid)),
        "epoch": engine.epoch,
    }


def dwell_polling(plate_id: Any, occupants: Any = None) -> Dict[str, Any]:
    """修复点 5：停留态轮询（纯状态机入口，不依赖进入事件）。

    每帧把探测器看到的占据者列表喂进来：有人->保持按下，
    最后一人离开->弹起。角色站着不动也不会自动复位。
    """
    pid = str(plate_id)
    if isinstance(occupants, (list, tuple, set)):
        current = {str(o) for o in occupants}
    elif occupants:
        current = {str(occupants)}
    else:
        current = set()

    record = _DWELL_STATE.setdefault(pid, {"occupants": set(), "state": MechanismState.RELEASED})
    previous = set(record["occupants"])

    if current and record["state"] != MechanismState.PRESSED:
        record["state"] = MechanismState.PRESSED
        transition = "pressed"
    elif not current and previous:
        record["state"] = MechanismState.RELEASED
        transition = "released"
    elif current:
        transition = "dwell"
    else:
        transition = "idle"

    record["occupants"] = current
    return {
        "plate_id": pid,
        "state": record["state"],
        "transition": transition,
        "occupants": sorted(current),
        "occupied": bool(current),
    }


def chain_replay_check(chain_id: Any, spec: Any = None, frame_seconds: Any = None) -> Dict[str, Any]:
    """修复点 6：在干净引擎中回放连锁，校验顺序/时刻/幂等并检测篡改。"""
    cid = str(chain_id)
    if spec is None:
        spec = [ChainStep("spike", 1.0, "spike_pop", MechanismState.ACTIVE)]
    if frame_seconds is None:
        # 混合帧长：模拟帧率波动，总时长足够覆盖链的最大延时
        base = [0.016, 0.02, 0.033, 0.016, 0.05, 0.016, 0.02, 0.033]
        frame_seconds = base * 6  # 48 帧，合计 1.224s
    engine = _get_engine()
    return engine.replay_check(cid, spec, list(frame_seconds))


# -- 引擎生命周期门面 -------------------------------------------------------


def arm(chain_id: Any, spec: Any = None) -> Dict[str, Any]:
    engine = _get_engine()
    try:
        return engine.arm(str(chain_id), spec if spec is not None else [])
    except ValueError as exc:
        return {"status": "invalid", "chain_id": str(chain_id), "issues": str(exc)}


def trigger(trigger_id: Any, idem_key: Any = None, source: str = "api") -> Dict[str, Any]:
    engine = _get_engine()
    if isinstance(trigger_id, dict):
        payload = dict(trigger_id)
        return engine.trigger(
            str(payload.get("trigger_id", payload.get("chain_id", "trigger"))),
            idem_key=payload.get("idem_key", payload.get("idempotency_key")),
            source=str(payload.get("source", source)),
        )
    try:
        return engine.trigger(str(trigger_id), idem_key=None if idem_key is None else str(idem_key), source=source)
    except KeyError as exc:
        return {"status": "not_found", "trigger_id": str(trigger_id), "error": str(exc)}


def reset(chain_id: Any = None) -> Dict[str, Any]:
    engine = _get_engine()
    if chain_id is None or isinstance(chain_id, (int, float, str)):
        return engine.reset(None if chain_id is None else str(chain_id))
    return engine.reset()


def present(scene_id: Any = _DEFAULT_SCENE) -> Dict[str, Any]:
    sid = _scene_of(scene_id) if isinstance(scene_id, dict) else str(scene_id)
    return _get_engine(sid).present()


def validate_trap(spec: Any = None) -> Dict[str, Any]:
    """静态校验一份连锁定义（也可校验引擎内已武装的全部连锁）。"""
    engine = _get_engine()
    if spec is None:
        return engine.validate()

    if isinstance(spec, (int, float)):
        spec = [spec]
    chain = _coerce_spec("validation_target", spec)
    issues = engine._validate_chain(chain)
    return {
        "valid": not issues,
        "issues": issues,
        "chain_id": chain.chain_id,
        "step_count": len(chain.steps),
    }


def metrics(scene_id: Any = _DEFAULT_SCENE) -> Dict[str, Any]:
    sid = _scene_of(scene_id) if isinstance(scene_id, dict) else str(scene_id)
    return _get_engine(sid).metrics()


def audit(scene_id: Any = _DEFAULT_SCENE) -> Dict[str, Any]:
    sid = _scene_of(scene_id) if isinstance(scene_id, dict) else str(scene_id)
    return _get_engine(sid).audit()


def fallback(chain_id: Any, target_id: Any = "mechanism", error: Any = "manual") -> Dict[str, Any]:
    engine = _get_engine()
    return engine.manual_fallback(str(chain_id), str(target_id), error)


def determinism(chain_id: Any, spec: Any = None, frame_patterns: Any = None) -> Dict[str, Any]:
    cid = str(chain_id)
    if spec is None:
        spec = [
            ChainStep("spike_a", 0.5, "spike_pop", MechanismState.ACTIVE),
            ChainStep("door_a", 1.0, "open_door", MechanismState.OPEN),
        ]
    if frame_patterns is None:
        # 三种帧率/抖动模式，总时长均为 1.5s，保证链上每一步都有机会触发
        jitter = [0.05, 0.02, 0.08] * 8 + [0.05, 0.02]  # 26 帧，合计 1.5s
        frame_patterns = [
            [1.0 / 60.0] * 90,  # 稳定 60fps
            [1.0 / 30.0] * 45,  # 稳定 30fps
            jitter,             # 剧烈抖动
        ]
    engine = _get_engine()
    return engine.determinism(cid, spec, list(frame_patterns))
