# ReDAG: Learning Agentic Repair for Complex Text-to-Image Generation over Condition DAGs

面向复杂场景的 agentic 文生图：把提示词拆成带依赖关系的条件 DAG，先整图出草图并挑画布，再逐条核对条件、诊断错误、生成局部编辑候选、区域提交并验收，循环到条件全部满足；最后用修复过程中"哪块修好、哪块弄坏"的区域级信号训练执行器。

## 方法

| 步骤 | 做什么 | 模型 |
|---|---|---|
| 规划与条件表 | 提示词拆成对象、属性、动作、关系等节点，记依赖、画面位置；每条节点是一道可核对的是/否题，代码再补计数、成对、画质条件 | 规划器 qwen3.7-plus（百炼 API） |
| 草图生成与挑选 | 按原始提示词出多张整图，逐题打分，挑总分最高的做画布 | 执行器 Qwen-Image-2.1 |
| 检查与定位 | 两段确认：整图或裁剪视图先问一遍，不满足再换视图确认；实体先定位再问；计数题问数字并列框 | 评估器 Qwen3.8-27B（vLLM） |
| 诊断与指令 | 由条件表的形状分型：缺失、画了但不对、多出、并成一个；写成改谁、怎么改、改哪块 | 规则 |
| 调度 | 判据 $p/(c_e(m+m_f))>q/(1-q)$ 决定继续画还是转修；按依赖排序、互不干扰的任务并行 | 规则 |
| 编辑与区域提交 | 同一画布同一指令出多个候选，比较候选与原图提取真正改动的区域贴回画布 | 执行器 |
| 验收与复查 | 目标是否修好、原有内容是否受损、计数与人物区分是否破坏；只复查受影响的条件 | 评估器 |
| 训练执行器 | 候选按区域分别计 credit，修好的加分、弄坏的减分，与整图分合成后做 GRPO | Qwen-Image-2.1 LoRA |

## 仓库结构

```
code/draftdag/     库（包名沿用项目早期名字）
code/scripts/      命令行入口，每个文件头写了输入输出
code/tests/        不需要显卡的测试，bash code/tests/run_all.sh
code/configs/      循环与训练默认配置、机器配置样例、去重合并清单
configs/           模型来源表
data/sets/         12 个提示词集 1710 条，每条带按 id 哈希固定的 dev/test 划分
data/plans/        规划定稿 qwen3.7-plus-v85，每集一个目录，_dedupe.jsonl 记去重改动；xl_seed 另留 v4d（标注集所用）
data/checklists/   公开基准官方清单（只用于最终评分）
data/requirements/ 合成集的要求清单
data/tiers*.json   按条件数的五档分档；tier_picks*/ 每档固定抽样
data_ext/          CoReBench（Apache-2.0）与 LongBench-T2I（MIT）原始数据
labels/            标注集：seed_v4 600 道节点题，三方标注与一致率；seed-1008/img 为所标的 30 张草图
gates/             各组件的判定文件（数字、设置、是否过线）
runs/pool_s0/      草图池索引 drafts.jsonl（564 条提示词 × 16 张）、每条一张拼图、3 条样例原图
archive/           早期探索与另一条生成 agent 线，原样保留
```

代码以仓库根目录为工作区根（环境变量 `P0`，默认取 `code/` 的上一级），所有相对路径按它解析。

## 提示词集与分档

| 集 | 来源 | 条数 | 节点/条（v85） |
|---|---|---|---|
| t02 t03 t05 t10 t20 t30 t40 | 合成集，按实体数分级，带要求清单 | 100 / 150 / 150 / 120 / 120 / 120 / 120 | 10.7 / 14.1 / 23.1 / 32.4 / 45.4 / 56.7 / 69.8 |
| cb_ma cb_mi cb_mr | T2I-CoReBench C-MA / C-MI / C-MR | 90 × 3 | 19.3 / 42.2 / 34.8 |
| lb_t2i | LongBench-T2I，中位 528 词 | 500 | 55.6 |
| xl_seed | 早期的 60 条复杂场景 | 60 | 36.0 |

五档按条件数划分：`data/tiers.json` 是出草图池时用的分档（规划 v5/v6 口径，`tier_picks*/` 据此抽样），`data/tiers_v85.json`（含代码生成的计数与成对条件）和 `data/tiers_v85_planner.json`（只数规划器节点）是定稿口径。

## 各组件现状

| 组件 | 代码 | 产出 | 现状 |
|---|---|---|---|
| 规划器 | `planner.py` `schema.py` `conditions.py` `table.py` `dedupe.py`；`plan.py` `cover_plans.py` `atomize_plans.py` `dedupe_plans.py` `audit_plans.py` `checklist_coverage.py` `plan_atomicity.py` | `data/plans/*/qwen3.7-plus-v85`，`gates/planner_*_v84*.json` `coverage_official_*.json` `atomicity_*.json` `audit_v85_merged.json` | 12 集全部过三条审计线（问题超出提示词、一题多事、要求遗漏各不超过 5%）。流程：首遍规划 → 补覆盖（v5；lb_t2i 用事实清单补覆盖 v6）→ 一节点一事（v84）→ 无模型去重（v85）。CoReBench 官方清单 6008 题覆盖 95.6%，并题率 33.0% → 21.3% |
| 标注集 | `labels.py`；`make_labels.py` `label_with_model.py` `label_report.py` `make_sheet.py` `merge_ref.py` | `labels/seed_v4/` | 600 道三方标注：第二标注方 gpt-6.1-sol 与第一标注方一致率 0.84，人工抽查 100 道与第一标注方 0.88；评估器标定的参考标签定为 `labels_gpt-6.1-sol.jsonl`，计数题偏多，单独看 |
| 评估器 | `verify.py` `vlm.py` `views.py` `calibrate.py`；`serve_vlm.sh` `calibrate_eval.py` `count_probe.py` | `gates/evaluator_Qwen3.8-27B*.json`，`gates/count_probe.json` | 生产设置（计数问数字并列框、第二段救回 ≥0.7、第一段 0.5–0.8 带确认，见 `code/configs/loop.default.json` 的 verify 段）在合议参考上错判 1.6%、多画一份检出 0.81、缺失检出 0.79、部分不对检出 0.59；非计数条件错判 0.8%。缺失与部分不对两条线未过，分歧集中在计数与"大体成立"的口径，待人工裁决后重标定 |
| 草图池 | `executor_qi21.py` `executor.py`；`draw.py` `launch.py` `jobq.py` `preflight.py` `pick_tiers.py` | `runs/pool_s0/` | 564 条提示词各 16 张 1024² 草图（Qwen-Image-2.1，40 步，种子 0），每张 14.4 秒；索引与拼图在仓库里，原图另存 |
| 诊断、调度、编辑与区域提交、验收、循环 | `diagnose.py` `schedule.py` `stats.py` `commit.py` `accept.py` `loop.py`；`run_loop.py` `make_instructions.py` `edit_bench.py` `status.py` `compare.py` `judge.py` `score_checklist.py` | 无 | 写完并在 mock 环境下有端到端测试（区域提交有像素级测试）；未在显卡上跑过。`run_loop.py` 读不到评估器过线文件不启动 |
| 训练执行器 | `policy_qi21.py` `grpo.py` `credit.py` `train.py` `warmup.py`；`validate_policy.py` `warmup.py` `train_grpo.py` | 无 | 写完并有数学测试；`validate_policy.py` 三项检查过线前不训练 |

## 运行

```bash
export P0=$(pwd)
bash code/tests/run_all.sh                      # 十二组离线测试，只需 numpy、Pillow、torch（CPU 即可）
cp code/configs/machine.example.json configs/machine.json   # 写本机可用的卡
python code/scripts/preflight.py --render        # 机器自检，写 gates/preflight_<主机名>.json
python code/scripts/plan.py --set t02 --tag <标签>           # 规划，需要 DASHSCOPE_API_KEY
bash code/scripts/serve_vlm.sh                   # 起评估器
python code/scripts/calibrate_eval.py --items labels/seed_v4/items.jsonl --ref labels/seed_v4/labels_gpt-6.1-sol.jsonl --set xl_seed --tag qwen3.7-plus-v4d   # 评估器标定
python code/scripts/draw.py --set t02 --tag qwen3.7-plus-v85 --run pool --n 16        # 草图池
python code/scripts/run_loop.py --set t02 --tag qwen3.7-plus-v85 --run r0 --mode adaptive
```

API 密钥只从环境变量或根目录 `secrets.env`（`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`，已忽略）读取。模型权重放 `models/<名>`（`scripts/get_models.py`，已忽略），`scripts/setup_env.sh` 建两个环境（生成：torch + diffusers ≥ 0.41 + peft；评估：vllm）。

## 规矩

打分方式进入循环前先在标注集上过线；调参用 dev，报数用 test；写进文档的数字至少两把尺（表内分数之外另有裁判或人工），单种子、少于 100 条的差别只叫迹象；一次只改一处，配置在 `runs/<run>/args.json`；图片一律 PNG；循环内的分数不当最终结果，最终评分用 `judge.py` 与 `score_checklist.py`。
