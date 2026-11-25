# pairwise-trap-sequence-trigger

## 领域

游戏、图形与多媒体

## 主题

机关连锁触发

## 说明

时间轴延时、幂等激活与场景持久化。

初始状态下 `gameplay/trap_sequence_trigger.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
