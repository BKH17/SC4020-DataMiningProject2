# src/task1_apriori/apriori.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as apriori_fp, association_rules
import matplotlib.pyplot as plt
import networkx as nx


def load_transactions(path: Path) -> list[list[str]]:
    """Read transactions from CSV written by prep.py (symptoms column is ';'-joined)."""
    df = pd.read_csv(path)
    tx: list[list[str]] = []
    for s in df["symptoms"].fillna(""):
        items = [t for t in str(s).split(";") if t]
        if items:
            tx.append(items)
    return tx


def to_onehot(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode transactions for mlxtend.apriori()."""
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def plot_support_conf(rules: pd.DataFrame, out_png: Path) -> None:
    if rules.empty:
        return
    plt.figure()
    plt.scatter(rules["support"], rules["confidence"], alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def plot_lift_hist(rules: pd.DataFrame, out_png: Path) -> None:
    if rules.empty:
        return
    plt.figure()
    rules["lift"].plot(kind="hist", bins=30)
    plt.xlabel("Lift")
    plt.title("Distribution of Rule Lift")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def build_rules_graph(rules: pd.DataFrame, out_png: Path, top_k: int = 30) -> None:
    """Lightweight rule graph using top-k by lift."""
    if rules.empty:
        return
    r = rules.sort_values("lift", ascending=False).head(top_k)
    G = nx.DiGraph()
    for _, row in r.iterrows():
        a = ", ".join(sorted(list(row["antecedents"])))
        b = ", ".join(sorted(list(row["consequents"])))
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b, weight=row["lift"])
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=600)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def prune_redundant_rules(rules: pd.DataFrame, margin: float = 0.02) -> pd.DataFrame:
    """
    Drop multi-antecedent rules whose lift is not better than the best single-antecedent
    rule for the same RHS by at least `margin` (relative).
    """
    if rules.empty:
        return rules

    rules = rules.copy()
    rules["a_size"] = rules["antecedents"].apply(len)
    rhs_key = rules["consequents"].apply(lambda s: tuple(sorted(s)))
    single = rules[rules["a_size"] == 1].copy()
    best_single = single.groupby(rhs_key)["lift"].max()

    keep_mask = ~rhs_key.isin(best_single.index)

    compare_lift = rhs_key.map(best_single)
    improved = rules["lift"] > (1.0 + margin) * compare_lift.fillna(-1)
    keep_mask = keep_mask | improved

    return rules[keep_mask].drop(columns=["a_size"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine Apriori itemsets and association rules")
    ap.add_argument("--transactions", type=Path, required=True, help="data/interim/transactions.csv")
    ap.add_argument("--min_support", type=float, default=0.10)
    ap.add_argument("--min_conf", type=float, default=0.60)
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/task1"))
    ap.add_argument("--top_k", type=int, default=50, help="for top-lift/graph previews")
    ap.add_argument("--min_transactions", type=int, default=200, help="fail if dataset is too small")
    ap.add_argument("--rhs_len1_only", action="store_true", help="keep only rules with a single consequent")
    ap.add_argument("--prune_redundant", action="store_true", help="drop multi-antecedent rules that add < margin lift")
    ap.add_argument("--redundancy_margin", type=float, default=0.02, help="relative lift margin (default 0.02 = 2%)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load & basic guardrails
    transactions = load_transactions(args.transactions)
    n_tx = len(transactions)
    if n_tx == 0:
        raise SystemExit("No transactions found. Did you run prep.py?")
    if n_tx < args.min_transactions:
        raise SystemExit(
            f"Only {n_tx} transactions (< {args.min_transactions}). "
            "Increase data or lower minsup before mining."
        )
    print(f"Transactions: {n_tx}")

    # Apriori
    basket = to_onehot(transactions)
    itemsets = apriori_fp(
        basket, min_support=args.min_support, use_colnames=True, max_len=args.max_len
    ).sort_values("support", ascending=False)
    itemsets.to_csv(args.outdir / "frequent_itemsets.csv", index=False)

    # Association rules
    rules = association_rules(itemsets, metric="confidence", min_threshold=args.min_conf)

    # Optional: filter RHS length == 1
    if args.rhs_len1_only:
        rules = rules[rules["consequents"].apply(lambda s: len(s) == 1)].copy()

    # Optional: redundancy pruning
    if args.prune_redundant:
        rules = prune_redundant_rules(rules, margin=args.redundancy_margin)

    # Final formatting/exports
    rules = rules.sort_values(["lift", "confidence"], ascending=[False, False]).copy()
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules.to_csv(args.outdir / "rules.csv", index=False)

    # Convenience views
    rules.head(args.top_k).to_csv(args.outdir / "rules_top_lift.csv", index=False)
    rules.sort_values("confidence", ascending=False).head(args.top_k).to_csv(
        args.outdir / "rules_top_confidence.csv", index=False
    )
    if args.rhs_len1_only:
        rules.to_csv(args.outdir / "rules_rhs1.csv", index=False)
    if args.prune_redundant:
        rules.to_csv(args.outdir / "rules_pruned.csv", index=False)

    # Plots
    plot_support_conf(rules, args.outdir / "support_vs_conf.png")
    plot_lift_hist(rules, args.outdir / "lift_hist.png")
    build_rules_graph(rules, args.outdir / "rules_graph.png", top_k=min(args.top_k, 30))

    print(f"Itemsets: {len(itemsets)}, Rules: {len(rules)} → {args.outdir}")


if __name__ == "__main__":
    main()
