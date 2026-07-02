# 21 · Graph & Network Analysis

When the *relationships between* entities carry the signal — not just the entities'
own features. Social networks, fraud rings, supply chains, recommendation graphs,
knowledge graphs, molecules, infrastructure.

## The mental shift
Tabular ML assumes rows are independent. Graphs explicitly model that they're not:
nodes (entities) connected by edges (relationships), possibly directed, weighted,
and typed. The moment "who is connected to whom" matters, a graph representation
usually beats flattening it into features.

## Classic network analysis (no ML required)
- **Centrality** — who is important, and in which sense:
  - *degree* (most connections), *betweenness* (bridges between clusters — remove
    them and the network fragments), *closeness* (short paths to everyone),
    *eigenvector/PageRank* (connected to important nodes). Choosing the right
    centrality *is* the analysis.
- **Community detection** — find clusters/modules (Louvain, Leiden, label
  propagation). Uncovers segments, fraud rings, functional groups.
- **Path & connectivity** — shortest paths, connected components, network diameter,
  bottlenecks.
- **Link structure** — reciprocity, triadic closure, assortativity (do similar
  nodes connect?).

These alone answer a lot: influencer identification, fraud-ring discovery,
resilience/bottleneck analysis, and organisational-network insight.

## Predictive tasks on graphs
- **Link prediction** — will an edge form? (friend/product recommendation, drug
  interactions.) From simple heuristics (common neighbours, Adamic-Adar) to
  embeddings.
- **Node classification** — label nodes using their features *and* their neighbours'
  (fraud/bot detection, where fraudsters [cluster](06-unsupervised-learning.md)).
- **Graph classification** — a label for a whole graph (molecule toxicity).

## Embeddings & Graph Neural Networks
- **Shallow embeddings** (node2vec, DeepWalk) — random walks → node vectors you feed
  to any downstream model. Cheap, strong baseline.
- **Graph Neural Networks (GNNs)** — GCN, GraphSAGE, GAT: learn by aggregating
  ("message passing") information from each node's neighbourhood. State of the art
  when nodes have rich features and the graph is large; GraphSAGE scales inductively
  to unseen nodes. Watch for over-smoothing (too many layers make all nodes look
  alike) and the engineering cost of graph batching.

## Where it earns its keep
- **Fraud & AML** — rings, mule accounts, and shared-attribute linkage that
  per-transaction models miss; graph features routinely lift fraud detection.
- **Recommendation** — the user-item graph; GNN-based recommenders are now common.
- **Knowledge graphs** — entity/relationship stores powering search and increasingly
  **GraphRAG** for grounding LLMs ([10](10-nlp-and-llms.md)).
- **Ops & supply chain** — dependency and flow analysis, single points of failure.

## Tools
`NetworkX` (analysis, prototyping, teaching — not for millions of edges), `igraph`
(faster, community detection), `PyG`/`DGL` (GNNs), `node2vec`, and graph databases
`Neo4j`/`Memgraph` (Cypher queries) when the graph is the primary data store.

## Practical cautions
- Graph algorithms scale badly; know the complexity before running betweenness on a
  10M-node graph (approximate it).
- Define the graph deliberately — what is a node, what is an edge, is it directed/
  weighted? The modelling choice dominates the result.
- Beware [temporal leakage](03-data-and-feature-engineering.md): for link prediction, split by *time*, not random edges.

---

## Python example — fraud ring detection with NetworkX

```python
"""
Graph analysis for fraud ring detection.
Builds a transaction graph and uses community detection + centrality
to surface suspicious clusters and bridge accounts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    print("Install networkx: pip install networkx")
    raise

rng = np.random.default_rng(42)

# ── Build synthetic transaction graph ─────────────────────────────────────────
# Accounts 0-199 are normal; accounts 200-219 form a fraud ring (dense connections)
G = nx.DiGraph()
n_normal, n_fraud = 200, 20
all_accounts = list(range(n_normal + n_fraud))
G.add_nodes_from(all_accounts)
G.nodes[all_accounts[0]]["is_fraud"] = False

# Normal transactions: sparse random connections
for _ in range(300):
    src = rng.integers(0, n_normal)
    dst = rng.integers(0, n_normal)
    if src != dst:
        G.add_edge(src, dst, weight=rng.uniform(10, 500))

# Fraud ring: dense connections among accounts 200-219
fraud_accounts = list(range(n_normal, n_normal + n_fraud))
for i in fraud_accounts:
    for j in rng.choice(fraud_accounts, size=5, replace=False):
        if i != j:
            G.add_edge(i, j, weight=rng.uniform(1000, 5000))

# Bridge accounts: connect fraud ring to normal network
for bridge in rng.choice(n_normal, size=3, replace=False):
    G.add_edge(bridge, rng.choice(fraud_accounts))

# ── Centrality features ───────────────────────────────────────────────────────
pagerank     = nx.pagerank(G, alpha=0.85)
betweenness  = nx.betweenness_centrality(G)
in_degree    = dict(G.in_degree())

# Flag top-10 by betweenness — often mule/bridge accounts
top_between  = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
fraud_in_top = [n for n in top_between if n >= n_normal]
print(f"Top-10 betweenness nodes: {top_between}")
print(f"Fraud ring accounts in top-10 betweenness: {fraud_in_top}")

# ── Community detection ────────────────────────────────────────────────────────
G_undirected = G.to_undirected()
communities  = louvain_communities(G_undirected, seed=42)
sizes = sorted([len(c) for c in communities], reverse=True)
print(f"\nCommunity sizes (largest first): {sizes[:8]}")

# Flag communities where most members are in known fraud range
for comm in communities:
    fraud_frac = sum(1 for n in comm if n >= n_normal) / len(comm)
    if fraud_frac > 0.5 and len(comm) >= 5:
        print(f"  Suspicious community: {len(comm)} nodes, {fraud_frac:.0%} fraud accounts")

# ── Plot (sample subgraph) ────────────────────────────────────────────────────
subgraph_nodes = list(range(0, 20)) + fraud_accounts
H = G.subgraph(subgraph_nodes).copy()
colors = ["red" if n >= n_normal else "steelblue" for n in H.nodes()]
pos = nx.spring_layout(H, seed=42)
fig, ax = plt.subplots(figsize=(8, 5))
nx.draw_networkx(H, pos, node_color=colors, node_size=200,
                 font_size=6, arrows=True, ax=ax,
                 edge_color="gray", alpha=0.7)
ax.set_title("Transaction graph sample\nRed = fraud ring, Blue = normal")
ax.axis("off")
plt.tight_layout()
plt.savefig("fraud_ring_graph.png", dpi=120)
plt.close()
print("\nPlot saved: fraud ring is visibly denser than normal subgraph.")
```

---

## Cross-references

- [37](37-cybersecurity.md) — lateral movement detection with graphs
- [31](31-fintech.md) — AML fraud ring detection
- [08](08-recommendation-systems.md) — user-item graph for recommenders
