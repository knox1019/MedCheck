# Agent 评测报告

> 由 `uv run python -m scripts.run_agent_eval` 自动生成，请勿手工修改。

- 模型：`cn:hy3`
- 评测集：18 条
- 平均工具调用 1.1 次 / 平均轮数 1.9

## 指标

| 指标 | 含义 | 结果 |
| --- | --- | --- |
| 工具选择正确 | 该调的调了，且没调不该调的 | 18/18 = 100.0% |
| 行为正确 | 该答的答了、该拒的拒了 | 18/18 = 100.0% |
| 引用标注率 | 作答时答案里有可映射的 [n] | 10/11 = 90.9% |
| 多余调用 | 调了既不在期望集也不在白名单的工具 | 0 条 |
| 收敛率 | 没有撞上轮数上限 | 18/18 = 100.0% |
| 延迟中位 | 单条耗时 | 7.3s |

## 逐条结果

| id | 问题 | 期望工具 | 实际工具 | 轮数 | 行为 | 引用 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ag01 | 华法林的禁忌是什么 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag02 | 阿司匹林会不会把胃吃坏 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag03 | 华法林和什么药不能一起吃 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag04 | 布洛芬和阿司匹林能一起吃吗 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag05 | 二甲双胍怎么吃要注意什么 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag06 | 氯吡格雷的禁忌 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag07 | 格列本脲的禁忌是什么 | search_drug_label | search_drug_label | 2 | 拒答 | 无 | ✅ |
| ag08 | 甲钴胺能长期吃吗 | search_drug_label | search_drug_label | 2 | 拒答 | 无 | ✅ |
| ag09 | 阿昔洛韦的注意事项 | search_drug_label | search_drug_label | 2 | 拒答 | 无 | ✅ |
| ag10 | 你好，你能做什么 | 无 | 无 | 1 | 作答 | 无 | ✅ |
| ag11 | 今天杭州天气怎么样 | 无 | 无 | 1 | 拒答 | 无 | ✅ |
| ag12 | 我报告里转氨酶偏高，和我在吃的阿托伐他汀有关系吗 | check_indicators | check_indicators | 2 | 作答 | 有 | ✅ |
| ag13 | 我的尿酸高，是不是吃药吃的 | check_indicators | check_indicators | 2 | 拒答 | 无 | ✅ |
| ag14 | 血红蛋白偏低要注意什么 | check_indicators | check_indicators | 2 | 拒答 | 无 | ✅ |
| ag15 | 我在吃华法林和阿司匹林，最近牙龈出血，要注意什么 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag16 | 华法林和阿司匹林一起吃有什么风险 | search_drug_label | search_drug_label | 2 | 作答 | 有 | ✅ |
| ag17 | 报告里胆固醇也高了，和我吃的药有关吗 | check_indicators | check_indicators | 2 | 作答 | 有 | ✅ |
| ag18 | 华法林是饭前吃还是饭后吃 | search_drug_label | search_drug_label | 2 | 拒答 | 无 | ✅ |
