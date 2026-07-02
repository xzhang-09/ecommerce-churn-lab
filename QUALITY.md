# 项目质量评估报告(QUALITY.md)

- 评估日期:2026-07-02
- 评估范围:`refactor/package-churn` 分支工作区(含未提交改动)
- 评估方式:通读全部源码、测试、配置与文档;实际运行测试套件(`python -m pytest -q`:19 passed);扫描硬编码密钥;核对死代码引用
- 约束:仅评估,未修改任何代码

> **注意:下方是初次评估(基线),用于日后对比。改进清单 11 项的落实情况见文末[「改进记录」](#改进记录2026-07-02)。**

---

## 总览

| 维度 | 得分(1-10) |
|---|---|
| 可维护性 | 7 |
| 可靠性 | 7 |
| 工程化程度 | 5 |
| 演进能力 | 6 |

这是一个刻意收敛到"离线分析与建模"的电商客户流失(churn)项目,整体质量明显高于同类个人/教学项目:目录结构规范、注释密集且解释了"为什么"、建模方法学(防数据泄漏、成本感知阈值、概率校准)非常扎实。主要短板集中在工程化层面:没有 CI(持续集成,即每次提交代码自动跑测试的机制)、核心编排函数是一个约 360 行的"大函数"且没有端到端测试、部分静默失败路径可能产生错误数据而不报错。

---

## 1. 可维护性:7/10

**问题:半年后的陌生开发者能否看懂并安全修改?——大体上能,但有几处会踩坑。**

### 做得好的地方

- **目录结构清晰、命名规范。** 采用 src-layout(把可安装的 Python 包放在 `src/` 目录下的标准布局,避免误导入本地文件):`src/churn/` 下按 `data / features / models / validation` 分职责,`scripts/` 只是薄封装,`tests/` 分 `unit/` 与 `integration/`。README 里的结构图与实际一致。
- **注释质量高。** 几乎每个关键决策都写了"为什么"(如 [preprocess.py:53-59](src/churn/data/preprocess.py:53) 解释了为什么必须在切分前去重、不去重会导致 recall=1.0 的假象),这是半年后读代码最需要的信息。
- **模块边界大多合理。** 特征编码拆成了 `build_feature_schema`(拟合)/ `apply_feature_schema`(应用)两步,这是能支撑"训练/推理一致性"的正确抽象。

### 具体问题(带文件路径)

1. **"大杂烩"文件:[src/churn/pipeline.py](src/churn/pipeline.py)(941 行)。** 一个文件里同时装着:候选模型定义、6 个评估指标函数、阈值选择、概率校准、交叉验证、模型对比、CLI 参数解析、以及一个约 360 行的 `main()`([pipeline.py:514](src/churn/pipeline.py:514))。`main()` 把 7 个阶段(加载→验证→预处理→特征→训练→阈值→评估)线性堆在一个函数体里,中间还夹着两处函数内 `import json/joblib`。想单独改"评估"或"产物落盘"逻辑的人必须通读整个函数才能确认不会破坏别的阶段。

2. **死代码(已无人调用的代码):**
   - [`_map_binary_series`](src/churn/features/build_features.py:21)(build_features.py:21)——全仓库无任何调用,纯死代码。
   - [`build_features()`](src/churn/features/build_features.py:111)(build_features.py:111)——生产代码零调用(pipeline 和 prepare 都直接用 schema 两步接口),唯一"使用者"是 [test_project_structure.py:46](tests/test_project_structure.py:46) 里一句 `assert callable(build_features)`,等于用测试给死代码续命。半年后的人会误以为这是特征工程主入口。

3. **轻微重复实现:** [prepare.py:25-26](src/churn/data/prepare.py:25) 又做了一次 Churn 的 Yes/No→0/1 映射,而 `preprocess_data`([preprocess.py:22-23](src/churn/data/preprocess.py:22))刚做过同样的事。防御性重复,不致错,但会让读者怀疑两处逻辑是否会分叉。

4. **部分注释是"写给 code reviewer 的历史说明"而非给未来读者的。** 如 [pipeline.py:66-69](src/churn/pipeline.py:66)("以前的 XGBoost 配置用了魔法数字…现已替换")、[pipeline.py:355](src/churn/pipeline.py:355)("相对上一版的公平性修复")。这类"相对于旧版"的叙述在 git 历史里已有记录,留在代码里会随时间变成噪音。

---

## 2. 可靠性:7/10

### 测试覆盖

- **有覆盖且质量不错的部分:** 指标计算、阈值选择、校准报告、特征 schema 防泄漏(测试了"测试集独有类别不进 schema"这个真实风险点)、缺依赖时的报错文案,共 19 个测试全部通过。合成数据的轻量冒烟测试([test_pipeline_smoke.py](tests/integration/test_pipeline_smoke.py))不依赖 Excel 原始数据,设计合理。
- **没有覆盖的部分(风险最大):**
  - `main()` 编排本身零测试——没有任何测试真正跑过 `pipeline.main()`(哪怕在小数据 + `--skip_validation` + 临时 MLflow 目录下)。MLflow 记录、产物落盘、`--tune` / `--compare_models` 分支全部裸奔。改坏 `main()` 里任何一行,测试套件不会发现。
  - [tune.py](src/churn/models/tune.py) 只有一句 `callable` 检查;`cross_val_oof`、`compare_models`、`fit_with_early_stopping` 无测试。

### 错误处理与日志

- 错误处理总体不错:文件不存在抛 `FileNotFoundError`,数据验证失败抛 `ValueError` 并把失败项记入 MLflow,缺 Great Expectations 时的报错信息直接告诉用户怎么办([pipeline.py:436-443](src/churn/pipeline.py:436))——这是好实践。
- **日志全靠 `print` + emoji**,没有 `logging` 模块、没有级别、没有落盘(除 MLflow 指标外)。离线项目可接受,但一旦要排查"上周那次跑挂在哪",只能靠终端回滚。

### 数据丢失/数据错误风险

- 数据丢失风险低:所有输出(`data/processed/`、`artifacts/`、`mlruns/`)都可从原始 Excel 重新生成。
- **静默数据错误风险是这个维度扣分的主因:**
  1. [apply_feature_schema](src/churn/features/build_features.py:83)(build_features.py:83):二值列遇到训练时没见过的取值(如上游把 "Yes/No" 改成 "Y/N")会被 `.map(mapping).fillna(0)` **静默映射成 0**,不警告不报错。配合 `--skip_validation` 使用时,整列特征悄悄变成全 0,pipeline 照常跑完并给出一份看起来正常但实际失真的模型。
  2. [impute_numeric](src/churn/data/preprocess.py:88)(preprocess.py:88-89):整列全 NaN 时中位数无定义,静默填 0,同样无警告。
- 数据验证(Great Expectations,一个用"声明式规则"校验数据表的库)本可拦截第 1 类问题,但 `--skip_validation` 是文档推荐的常用路径,绕过后没有任何兜底。

---

## 3. 工程化程度:5/10

- **一条命令能否跑起来:基本可以。** `pip install -e .` 之后一条 `churn-pipeline --input ... --target Churn` 即可跑通全流程,依赖在 [pyproject.toml](pyproject.toml) 里全部精确锁版本(`==`),可复现性好。扣分点:原始数据 Excel 不在仓库里(合理,但 README 只说"放到这个路径",没有获取方式说明),新人拿到仓库无法自证跑通。
- **自动化测试:有;CI/CD:完全没有。** 没有任何 CI 配置,测试只能靠人手动跑。更麻烦的是 [test_project_structure.py:7-19](tests/test_project_structure.py:7) **断言 `.github` 目录必须不存在**——这条"部署组件已删除"的守卫测试会直接阻止将来添加 GitHub Actions(GitHub 的 CI 服务,配置就放在 `.github/` 下)。守卫本意是防止部署代码回潮,却把 CI 也一起封死了。
- **密钥与配置安全:干净。** 全仓库扫描无硬编码密码/API key;MLflow 用本地文件后端,无外部服务凭证;`.gitignore` 正确排除了数据和产物。这一项没有失分。
- **依赖管理小问题:** `pytest`、`jupyter`、`ipykernel`、`seaborn`、`matplotlib` 混在运行时依赖里([pyproject.toml:11-27](pyproject.toml)),应放进 `[project.optional-dependencies]` 的 dev 组;`requirements.txt` 在工作区已删除但该删除尚未提交,当前分支处于未提交状态(9 个文件有改动),README/CLAUDE.md 描述的是改动后的世界,若此时有人 clone main 分支会对不上。
- 没有 lint/格式化配置(ruff/black/pre-commit 均无)。

---

## 4. 演进能力:6/10

**需求变化时能否增量修改?——加"新指标/新校准方法"容易,加"新阶段/新数据集/新交付形态"会牵动多处。**

### 边界清晰、可增量改的部分

- 换/加候选模型:只改 [get_model_specs](src/churn/pipeline.py:64) 一处。
- 改验证规则:[validate_data.py:9-29](src/churn/validation/validate_data.py:9) 把规则提成了模块级常量,一处可改。
- 调优搜索空间:[tune.py](src/churn/models/tune.py) 按模型隔离,互不影响。

### 一改就牵一发动全身的地方

1. **`main()` 是所有变更的必经之路。** 任何新阶段、新 CLI 参数、新产物都要在这个 360 行函数里找插入点,且它同时负责 MLflow 记录、打印、落盘三件事,改动的爆炸半径大。
2. **数据集特定知识散落在 4 个文件里:** 类别别名表在 [preprocess.py:29-33](src/churn/data/preprocess.py:29)、验证规则在 validate_data.py、Excel sheet 名 `"E Comm"` 硬编码在 [load_data.py:23](src/churn/data/load_data.py:23)、目标列默认值 `"Churn"` 出现在多处签名里。换一份数据集需要同时改这 4 处,且没有一处汇总说明。
3. **守卫型测试与演进方向冲突。** [test_project_structure.py](tests/test_project_structure.py) 和 [test_notebook_scope.py](tests/test_notebook_scope.py) 把"某些文件必须不存在、notebook 不许 import lightgbm"写成了测试。项目范围一旦合理扩张(比如加 CI、加批量打分脚本、在 notebook 里做模型误差分析),第一件事就是和自己的测试打架。范围约束更适合写在 CLAUDE.md/README(它们确实已经写了),不适合固化成测试。
4. **没有推理/打分模块。** `preprocessing.pkl` + MLflow 模型 + `threshold.txt` 三样产物齐全,但"如何用它们给一批新客户打分"这段逻辑不存在于任何模块——将来第一个要做打分的人必须从 `main()` 里反向拼装预处理顺序(别名→缺失指示列→schema→中位数填充→阈值),很容易漏一步造成训练/推理不一致。
5. `random_state=42` 在 [pipeline.py:586](src/churn/pipeline.py:586) 等处硬编码(未暴露为 CLI 参数),想做多种子稳定性实验要改源码。

---

## 5. 有效性一致性检查

### 项目自我定位

- **目标用户:** 做客户留存(retention)分析的数据科学家/分析师本人,以及需要"该联系哪些客户"名单的运营方。
- **核心问题:** 在约 5,000 名电商客户(约 17% 流失率)中,离线地识别高流失风险客户,并给出**可信的流失概率**和**成本最优的决策阈值**(漏掉一个真流失客户的代价 = 10 次误报),供留存干预使用。
- **关键使用场景:** ① EDA 探索;② 一条命令复现"验证→清洗→特征→训练→评估"全流程并记录到 MLflow;③ 公平对比三个模型;④ Optuna 调参;⑤ 按业务成本选阈值、校准概率,为"期望价值式定向"做准备。

### 实现与目标是否一致

- **一致性总体很高,且有难得的"反过度开发"自觉:** Docker/FastAPI/Gradio/部署曾存在,后被主动删除并写明理由。校准、OOF 阈值选择、置换重要性看似"高级",但都直接服务于"概率要可信、阈值要省钱"这个业务目标,不算镀金。
- **轻微的过度开发嫌疑:** ① 死代码 `build_features` / `_map_binary_series`(见上);② 守卫型测试(用 pytest 断言文件不存在、notebook 不许 import 某库)——这是把一次性重构验收固化成了永久负担;③ `scripts/*.py` 与 console scripts 双入口(有文档理由,可接受)。
- **关键场景遗漏:批量打分。** README 自己说目标是"who is worth contacting",但项目产出止步于测试集指标——没有任何脚本能把训练好的模型应用到一批新客户上输出名单。这是目标链条上缺的最后一环(注意:这不是"部署",离线批量打分完全在项目自我声明的范围内)。
- **10 倍规模(约 5 万行)什么先出问题:**
  1. **[select_threshold](src/churn/pipeline.py:212)(pipeline.py:212-231)最先爆。** 它对每个候选阈值(≈去重后的预测概率个数,与样本数同阶)在纯 Python 循环里各算一次完整混淆矩阵,复杂度是 O(n²)(计算量随数据量平方增长):5 千行时约 2,500 万次运算尚可,5 万行时约 25 亿次,单这一步就要跑几十分钟到数小时,而且 `main()` 里它被调用多次(每个对比模型一次 + 最终模型一次)。
  2. 其次是 `--compare_models` + `--tune` 的组合成本线性膨胀(5 折 CV × 3 模型 × Optuna trials)。
  3. 内存无忧(pandas 全量载入,5 万行 × 几十列仍是小数据),MLflow 文件后端和 Excel 读取也都还撑得住。

---

## 全项目最严重的 3 个问题

### ① 特征编码对"没见过的取值"静默归零 —— 会产出看似正常、实际失真的模型

[build_features.py:83](src/churn/features/build_features.py:83) 的 `fillna(0)` 把未知二值取值悄悄当成 0 类;[preprocess.py:89](src/churn/data/preprocess.py:89) 把全空列悄悄填 0。
**实际损失场景:** 上游导出格式变了(比如 `Complain` 从 0/1 变成 "Yes"/"No",或 Gender 改成 "M"/"F"),分析师照常用 `--skip_validation` 快速跑了一版,pipeline 全绿跑完、MLflow 记录齐全,只是这几列特征已全部变成常数 0。模型指标略降但不崩,没人起疑,基于这份模型圈出的"高风险客户名单"发给运营做留存干预——预算花在了错误的客户上,且事后极难归因。

### ② 编排层(`main()`)零测试 + 零 CI —— 回归只能靠人肉发现

360 行的 [pipeline.py:514](src/churn/pipeline.py:514) 没有任何测试跑过它,项目也没有 CI;更有 [test_project_structure.py:7](tests/test_project_structure.py:7) 断言 `.github` 不许存在,主动堵死了加 CI 的路。
**实际损失场景:** 半年后有人给 `main()` 加一个新指标,不小心改动了 `preprocessing_artifact` 的落盘时机或键名。单元测试全过(它们不碰 `main()`),问题直到某次跑了 40 分钟 Optuna 调参后在最后落盘阶段崩溃才暴露——或者更糟,不崩溃,只是 `preprocessing.pkl` 里少了 medians,下游打分时填充值全错。

### ③ 产物无版本关联 + 没有打分入口 —— 用错模型/阈值组合的风险

`artifacts/` 下的 `preprocessing.pkl`、`feature_schema.json`、`model_comparison.csv` 每次运行**原地覆盖**,与 MLflow run 没有 ID 关联;而应用模型所需的完整链条(别名→指示列→schema→中位数→模型→阈值)没有任何代码实现。
**实际损失场景:** 周一用默认参数跑了一版(阈值 0.31),周三又用 `--fn_fp_ratio 5` 试了一版(阈值 0.48)。之后要出客户名单的人从 `artifacts/` 拿 `preprocessing.pkl`(周三的)、从 MLflow 里挑了周一那个 run 的模型和 `threshold.txt`,手工拼打分脚本——预处理状态和模型来自不同的运行,阈值和成本假设对不上,名单系统性偏差,而整个过程没有任何机制能发现这种错配。

---

## 改进清单(按性价比排序:成本低、收益大者优先)

| # | 改进项 | 工作量 | 预期收益 |
|---|---|---|---|
| 1 | 删除死代码:`_map_binary_series`、`build_features()`,并同步修改 [test_project_structure.py:46](tests/test_project_structure.py:46) 的 import 检查 | ~0.5 小时 | 消除"假入口"误导,新人不再读错主路径 |
| 2 | 提交当前分支的未提交状态(9 个改动文件 + requirements.txt 删除),让仓库与文档一致 | ~0.5 小时 | 消除"文档描述与 main 分支不符"的窗口期 |
| 3 | 对未知二值取值和全 NaN 列**至少打印警告**(理想是默认报错、加 `--allow-unseen` 放行),改 [build_features.py:83](src/churn/features/build_features.py:83) 与 [preprocess.py:89](src/churn/data/preprocess.py:89) | 1–2 小时 | 直接封堵最严重问题①的静默失真路径 |
| 4 | 把 pytest/jupyter/seaborn/matplotlib/ipykernel 移入 `[project.optional-dependencies].dev` | ~0.5 小时 | 运行时安装更轻、依赖语义正确 |
| 5 | 放开 `.github` 禁令(从守卫测试中移除该项),添加最小 GitHub Actions:`pip install -e .[dev]` + `compileall` + `pytest` | 1–2 小时 | 每次提交自动验证,问题②的一半解决 |
| 6 | 为 `pipeline.main()` 写端到端测试:合成小数据 + `--skip_validation` + `--mlflow_uri` 指向 tmp 目录,断言产物文件与关键指标存在;可再覆盖 `--compare_models` 分支 | 0.5–1 天 | 问题②的另一半:编排层任何回归当场暴露 |
| 7 | 用 `sklearn.metrics.precision_recall_curve` 或累计计数向量化 [select_threshold](src/churn/pipeline.py:212),O(n²)→O(n log n) | 1–2 小时 | 消除 10 倍数据量下的第一个性能悬崖 |
| 8 | 产物与 run 关联:把 MLflow run_id 写入 `preprocessing.pkl` 和 `artifacts/` 各文件(或产物写入 `artifacts/<run_id>/`) | 2–4 小时 | 封堵问题③的"错配"风险 |
| 9 | 新增 `churn-score` 批量打分入口:加载 preprocessing.pkl + MLflow 模型 + threshold,输入新客户表,输出概率与名单 | 0.5–1 天 | 补上目标链条缺的最后一环,项目从"报告"变成"可用" |
| 10 | 拆分 [pipeline.py](src/churn/pipeline.py):指标/阈值/校准 → `models/evaluate.py`,对比 → `models/compare.py`,`main()` 拆成 6–7 个阶段函数 | 1–2 天 | 可维护性与演进能力的根本改善;建议在 #6 的端到端测试就位后再做 |
| 11 | `print` → `logging`(保留 CLI 友好格式,支持级别与落盘) | 2–4 小时 | 排障可追溯;优先级最低,离线场景收益有限 |

---

## 附:本次评估核实过的事实

- `python -m pytest -q`:19 passed(4.7s)。
- 硬编码密钥扫描(api_key/password/secret/token,含 notebook):无发现。
- `_map_binary_series` 全仓库零调用;`build_features` 仅被结构测试的 callable 断言引用。
- 数据规模:去重后 5,073 行;`mlruns/` 已 68MB(正确地被 git 忽略)。
- git 状态:`refactor/package-churn` 分支,9 个文件改动未提交(含 requirements.txt 删除)。

---

## 改进记录(2026-07-02)

改进清单 11 项已全部落实。落实后测试从 19 个增至 **25 个,全部通过**;`python -m compileall src scripts` 通过;`churn-pipeline` / `churn-prepare` / `churn-score` 三个入口均可解析。

| # | 改进项 | 状态 | 落实方式 |
|---|---|---|---|
| 1 | 删除死代码 | ✅ 已完成 | 删除 `_map_binary_series`、`build_features()`([build_features.py](src/churn/features/build_features.py));`test_project_structure.py` 的 import 检查改指向真实入口 |
| 2 | 提交未提交状态 | ⏸ 留给用户 | 分支本就有未提交改动,本轮又新增;是否提交/如何组织 commit 由用户决定 |
| 3 | 未知取值/全 NaN 列告警 | ✅ 已完成 | [apply_feature_schema](src/churn/features/build_features.py) 对未知二值取值发 `warnings.warn`;[impute_numeric](src/churn/data/preprocess.py) 对填 0 的残留 NaN 告警;新增 `test_score.py` 覆盖 |
| 4 | dev 依赖分组 | ✅ 已完成 | [pyproject.toml](pyproject.toml) 新增 `[project.optional-dependencies].dev`(pytest/jupyter/ipykernel/matplotlib/seaborn);运行时依赖精简 |
| 5 | 放开 `.github` 禁令 + CI | ✅ 已完成 | 守卫测试移除 `.github` 断言;新增 [.github/workflows/ci.yml](.github/workflows/ci.yml)(install `.[dev]` + compileall + pytest) |
| 6 | `main()` 端到端测试 | ✅ 已完成 | [tests/integration/test_pipeline_main_e2e.py](tests/integration/test_pipeline_main_e2e.py):合成数据 + tmp cwd/MLflow,断言产物与 `mlflow_run_id` 存在 |
| 7 | 向量化 `select_threshold` | ✅ 已完成 | 累计计数实现,O(n²)→O(n log n),移入 [evaluate.py](src/churn/models/evaluate.py);与暴力版对拍 300×4×2 组合**零差异** |
| 8 | 产物与 run 关联 | ✅ 已完成 | `preprocessing.pkl` 写入 `mlflow_run_id`;`churn-score` 从**同一 run** 加载模型/预处理/阈值,杜绝错配 |
| 9 | `churn-score` 批量打分 | ✅ 已完成 | 新增 [score.py](src/churn/models/score.py) + [scripts/score_customers.py](scripts/score_customers.py) + console script;端到端验证:训练→按 run 加载→打分→按风险降序名单 |
| 10 | 拆分 `pipeline.py` | ✅ 已完成 | 拆出 `models/{estimators,evaluate,training,compare}.py`;`main()` 由 ~360 行降为 ~70 行阶段编排;`pipeline` 重导出保持公共 API 稳定;新增 `test_module_layout.py` 锁定布局 |
| 11 | `print` → `logging` | ✅ 已完成 | 全包改用 `logging.getLogger(__name__)`;新增 [logging_utils.py](src/churn/logging_utils.py) 的 `configure_logging()`(仅配置 `churn` logger,不影响第三方日志噪声),CLI 入口调用 |

**尚未处理(明确不在本轮范围):**

- #2 提交:分支处于未提交状态(现约 20 个文件改动/新增),由用户决定提交时机与 commit 划分。
- 附带发现的 `.Rhistory` 杂散文件仍未加入 `.gitignore`(与改进清单无关,提示性事项)。

**落实后复评(简评):** 可维护性与演进能力因 god-file 拆分、死代码清除、日志化而明显提升;可靠性因编排层端到端测试 + 静默失真告警而提升;工程化因 CI + dev 依赖分组 + 批量打分入口补齐而提升。三大严重问题(静默归零、编排零测试/零 CI、产物错配)均已针对性封堵。建议下次以已提交的 `main` 分支为基准做一次完整复评并更新上方总览评分。
