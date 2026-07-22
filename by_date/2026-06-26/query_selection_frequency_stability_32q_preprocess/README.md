# Query Selection Frequency and Stable-Set Convergence

分析对象：固定同一组文档时，不同 query 触发的 selected token 集合。

- 当前缓存：每个 example 32 个 query。
- frequency histogram：统计每个 document token 在 32 个 query 中被选中 0..32 次。
- convergence：随机采样 2/4/8/16/32 个 query 做交集，统计 stable intersection ratio。
- 本补充实验只覆盖 Target-QK(preprocess KV), passages=10。

## Query 构造

- 每个 example 固定同一组 passages。
- 32 个 query 由三部分组成：
  - 1 个原始 native question；
  - 10 个 native_template，对原始问题添加检索/证据相关提示；
  - 21 个 control_other_example，直接复用其他 example 的原始问题，作为无关 query control。
- 因此这个设置比“全是相关改写 query”更严格：如果仍然有大量 token 被稳定选中，说明这部分 token 更像 document-intrinsic stable update set。

Example 0 的原始问题：

```text
Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
```

Example 0 的 32 个 query：

```text
[00] native: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[01] native_template: What information in the passages helps answer: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[02] native_template: Which entity or event is most relevant to this question: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[03] native_template: Answer this using the same documents: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[04] native_template: What evidence in the documents supports the answer to: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[05] native_template: Identify the passages that are useful for answering: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[06] native_template: Which facts should be retrieved from this document set for: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[07] native_template: Find the supporting context needed to solve: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[08] native_template: Using only the provided passages, reason about: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[09] native_template: What parts of the document are likely necessary for this question: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[10] native_template: Select the relevant evidence before answering: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
[11] control_other_example: What was the person who provided evidence to suggest the existence of the neutron a participant of?
[12] control_other_example: In which borough was Callum McManaman born?
[13] control_other_example: Who is the child of the Italian navigator who sailed for England and explored the eastern coast of the continent Francisco Bojado's birthplace is located?
[14] control_other_example: Who is the spouse of the actor who played hannibal smith in the a team?
[15] control_other_example: When was the last time Peter Till's sports team beat the winner of the 1894-95 FA Cup?
[16] control_other_example: Who is the child of the cast member of Green Ice?
[17] control_other_example: How many times did plague occur in the birthplace of La Silvia's composer?
[18] control_other_example: When did military instruction start at the place where Larry Alcala was educated?
[19] control_other_example: What is the experimental satellite being forerunner to communication satellite of INSAT-4CR's manufacturer called?
[20] control_other_example: Who was the first African American student at the university Robert Khayat was educated at?
[21] control_other_example: Who gives out the prize named after the author of Lectures on Jurisprudence?
[22] control_other_example: Who is Frances Freeling Broderip's sibling?
[23] control_other_example: Who is the spouse of the person who voices Jarvis in Iron Man?
[24] control_other_example: What is the source of the river that is the mouth of the Caledon River?
[25] control_other_example: What is the name of the famous bridge located in the birthplace of the composer of Nulla in mundo pax sincera?
[26] control_other_example: When did the party that gained control of Congress in the midterm elections in 1946 take control of the government branch that determines the rules of the US House?
[27] control_other_example: Where did the producer of Julius Caesar study or work?
[28] control_other_example: Mehmet Hayri Tarhan's birthplace is the capital of what municipality?
[29] control_other_example: What is the performer of Heartbeat named after?
[30] control_other_example: The author of Elizabeth and After attended which university?
[31] control_other_example: What year did the publisher of Labyrinth end?
```

## Passages=10 Summary

| selector | rate | queries | never selected | selected all-1 | selected all | polarized 0 or all |
|---|---:|---:|---:|---:|---:|---:|
| Target-QK(preprocess KV) | 0.1 | 32 | 0.7639 | 0.0035 | 0.0581 | 0.8220 |
| Target-QK(preprocess KV) | 0.2 | 32 | 0.5871 | 0.0075 | 0.1217 | 0.7088 |
| Target-QK(preprocess KV) | 0.3 | 32 | 0.4539 | 0.0107 | 0.1954 | 0.6493 |
| Target-QK(preprocess KV) | 0.5 | 32 | 0.2579 | 0.0153 | 0.3699 | 0.6278 |

## Figures

- `token_frequency_hist_passages10_target-qkpreprocess_kv.png`
- `stable_set_convergence_passages10.png`
