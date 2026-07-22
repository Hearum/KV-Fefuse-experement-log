FusionRAG 的重算为 token selection 本质上prefill的是稀疏 Query 重算，但是在这里的对第i个token的重算还是走的标准注意力：
标准注意力为：
A = softmax( QK^⊤ / √d ), O = AV

序列长度为 N 时，QK^⊤ 和 AV 的复杂度都是：
O(N^2 d)

稀疏注意力对每个 token i 只保留集合 S_i：
O_i = Σ_{j∈S_i} ( exp(q_i^⊤ k_j / √d) / Σ_{t∈S_i} exp(q_i^⊤ k_t / √d) ) v_j

若每个 Query 只关注 k << N 个 Key，复杂度可降为：
O(N k d)

motivation：对于token i，若被选中重算，我们看到重算前后的kv变化分别是20%和40%。意味着本身token i的kv（preprocess/raw）就有一定的信息量，我们只需要对它这部分的kv信息进行补充即可。一个节省计算的开销想法说我想使用稀疏注意力来完成重算，k_i,v_i。

我打算初步参考moba的稀疏注意力机制
MoBA 把上下文分成 KV blocks：
B_b = { k_s : s ∈ b }

每个 block 的代表 Key 就是平均值：
k̄_b = 1/|B_b| Σ_{s∈B_b} k_s

Query 与每个 block representative 做点积：
S_{t,b} = q_t^⊤ k̄_b

再选择 Top-k blocks：
I_t = TopK_b(S_{t,b}, k) ∪ {current block}
通过这样，我们不再重算<i 部分的所有 token，而是重算部分token，减少计算量，先验证一下这个思路的在musqiue数据集端到端效果

我们在offline阶段已经预先处理好的每一个doc的attention score/top 15%的token，这里你可以利用这部分token
