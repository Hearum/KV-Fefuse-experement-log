# Query-dependent recomputation overlap

- 固定样本: musique example 1
- 固定文档: system + 前 3 个 passage
- context tokens: 3604, system tokens: 792
- 重算比例: 0.2, block size: 16

## related_same_docs

- token_jaccard: mean=1.0000, min=1.0000, max=1.0000
- token_overlap: mean=1.0000, min=1.0000, max=1.0000
- block_jaccard: mean=1.0000, min=1.0000, max=1.0000
- block_overlap: mean=1.0000, min=1.0000, max=1.0000
- chunk_cosine: mean=1.0000, min=1.0000, max=1.0000

0. What was the person who provided evidence to suggest the existence of the neutron a participant of?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
1. Which wartime project did James Chadwick work on?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
2. Who discovered the neutron in 1932?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
3. What report did James Chadwick write the final draft of in 1941?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
4. Where did James Chadwick move in early 1944?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
5. What Nobel Prize did James Chadwick receive?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
6. Who suggested using deuterons to determine the neutron mass?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
7. What laboratory did Chadwick work at when searching for the neutron?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]

## unrelated_control

- token_jaccard: mean=1.0000, min=1.0000, max=1.0000
- token_overlap: mean=1.0000, min=1.0000, max=1.0000
- block_jaccard: mean=1.0000, min=1.0000, max=1.0000
- block_overlap: mean=1.0000, min=1.0000, max=1.0000
- chunk_cosine: mean=1.0000, min=1.0000, max=1.0000

0. Who plays the wife of the producer of Here Comes the Boom in Grown Ups?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
1. In which borough was Callum McManaman born?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
2. Who is the child of the Italian navigator who sailed for England and explored the eastern coast of the continent Francisco Bojado's birthplace is located?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
3. Who is the spouse of the actor who played hannibal smith in the a team?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
4. When was the last time Peter Till's sports team beat the winner of the 1894-95 FA Cup?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
5. Who is the child of the cast member of Green Ice?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
6. How many times did plague occur in the birthplace of La Silvia's composer?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]
7. When did military instruction start at the place where Larry Alcala was educated?
   - chunk distribution: [0.0, 0.0, 0.6068, 0.3932]
   - first selected positions: [2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218]

## 初步解读

- 如果 token/block overlap 高，说明同一文档下不同 query 触发的重算区域较稳定，可能由文档结构/sink/局部重要 token 主导。
- 如果 overlap 低但 chunk cosine 高，说明 query 会改变具体 token，但倾向于落在相似 chunk 区域。
- related 与 unrelated control 的差异可以用来区分 query-specific 选择和 query-independent 文档偏置。
