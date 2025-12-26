import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def build_na_report(
    df: pd.DataFrame,
    cols: list[str],
    sample_n: int = 5,
    top_k: int = 15,
    all_unique_threshold: int = 40,
) -> str:
    sep = "─" * 100
    lines: list[str] = []

    def H(title: str):
        lines.append(title)
        lines.append(sep)

    def blank():
        lines.append("")

    def bullet(prefix: str, key: str, val: str):
        lines.append(f"{prefix} {key:<14}: {val}")

    # =========================
    # Header
    # =========================
    H("DF N/A + TYPE SUMMARY REPORT")
    bullet("•", "Rows", f"{len(df):,}")
    bullet("•", "Columns", f"{len(cols)}")
    blank()

    # =========================
    # Global N/A summary
    # =========================
    H("1) GLOBAL N/A COUNTS (sorted)")
    na_counts = df[cols].isna().sum().sort_values(ascending=False)
    na_pct = (na_counts / len(df) * 100).round(2) if len(df) else na_counts * 0
    na_summary = pd.DataFrame({"na_count": na_counts, "na_pct": na_pct})
    lines.append(na_summary.to_string())
    blank()

    # =========================
    # Per-column details
    # =========================
    H("2) PER-COLUMN DETAILS")

    for idx, k in enumerate(cols, start=1):
        s = df[k]
        na_count = int(s.isna().sum())
        na_percent = (na_count / len(df) * 100) if len(df) else 0.0
        dtype = str(s.dtype)

        # Column header
        lines.append(f"[{idx:02d}] {k}")
        lines.append("")

        # Core info
        bullet("├─", "dtype", dtype)
        bullet("├─", "NA", f"{na_count:,} ({na_percent:.2f}%)")

        # Numeric vs non-numeric section
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")

            bullet("├─", "type", "numeric")
            lines.append("│  ├─ stats")
            lines.append(f"│  │  ├─ min/max   : {s_num.min(skipna=True)} / {s_num.max(skipna=True)}")
            lines.append(f"│  │  ├─ mean      : {s_num.mean(skipna=True)}")
            lines.append(f"│  │  ├─ median    : {s_num.median(skipna=True)}")
            lines.append(f"│  │  └─ q25/q75   : {s_num.quantile(0.25)} / {s_num.quantile(0.75)}")

        else:
            # booleans + categoricals
            n_unique = int(s.nunique(dropna=True))
            bullet("├─", "type", "categorical/bool")
            bullet("├─", "unique", f"{n_unique:,} (non-NA)")

            # Show all unique values only if small; otherwise show top counts
            if n_unique <= all_unique_threshold:
                uniq = list(map(str, s.dropna().astype("string").unique()))
                lines.append("│  ├─ categories (all)")
                if len(uniq) == 0:
                    lines.append("│  │  └─ (none)")
                else:
                    for j, v in enumerate(uniq, start=1):
                        prefix = "│  │  └─" if j == len(uniq) else "│  │  ├─"
                        lines.append(f"{prefix} {v}")
            else:
                vc = s.astype("string").value_counts(dropna=False).head(top_k)
                lines.append(f"│  ├─ value_counts (top {top_k})")
                # indent the printed table
                for line in vc.to_string().splitlines():
                    lines.append("│  │  " + line)

        # Sample rows with NA
        if sample_n > 0 and na_count > 0:
            sample = df.loc[s.isna(), cols].head(sample_n)
            lines.append("│")
            lines.append(f"├─ sample rows where [{k}] is NA (first {sample_n})")
            for line in sample.to_string(index=False).splitlines():
                lines.append("│  " + line)
            lines.append("└─ end sample")
        else:
            lines.append("└─ sample rows: none")

        lines.append("")          # blank line between columns
        lines.append(sep)
        lines.append("")

    return "\n".join(lines)


def save_report(text: str, fname: str = "report.txt") -> None:
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
