"""Dashboard hasil clustering time series (DTW, CDTW, DDTW, WDTW) – data sales transaction."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Clustering Time Series DTW", page_icon="📈", layout="wide")

PAL = px.colors.qualitative.Bold
METRIC_LABEL = {"silhouette": "Silhouette (↑)", "chi": "Calinski-Harabasz (↑)", "dbi": "Davies-Bouldin (↓)", "balance": "Balance score (↑)"}
MAXIMIZE = {"silhouette": True, "chi": True, "dbi": False, "balance": True}

st.markdown(
    """
<style>
.hero{padding:26px 32px;border-radius:20px;margin-bottom:18px;background:linear-gradient(120deg,#6366f1,#ec4899 60%,#f59e0b)}
.hero h1{color:#fff;margin:0;font-size:2rem}.hero p{color:#fff;opacity:.92;margin:6px 0 0}
.card{padding:16px 18px;border-radius:16px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);height:100%}
.card .k{font-size:.75rem;opacity:.75;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:1.45rem;font-weight:700;margin-top:4px}.card .s{font-size:.82rem;opacity:.8}
</style>
""",
    unsafe_allow_html=True,
)


def card(col, k, v, s=""):
    col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>', unsafe_allow_html=True)


def style(fig, h=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h, margin=dict(t=50, b=10))
    return fig


@st.cache_data
def load_results():
    df = pd.read_csv(BASE / "hasil_clustering.csv")
    if "distance_full" not in df:
        df["distance_full"] = np.where(df["param"].fillna("-") == "-", df["distance"], df["distance"] + " " + df["param"].astype(str))
    if "config_id" not in df:
        df["config_id"] = df["matriks"] + " | " + df["method"] + " | " + df["setting"]
    df["linkage"] = df["linkage"].fillna("")
    df = df.drop_duplicates("config_id").reset_index(drop=True)
    return df


@st.cache_resource
def load_npz():
    p = BASE / "data_dashboard.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


res = load_results()
npz = load_npz()

# ---------------- Sidebar ----------------
st.sidebar.title("🎛️ Filter")
page = st.sidebar.radio("Halaman", ["🏠 Ringkasan", "📊 Eksplorasi Data", "🔥 Eksplorasi Metrik", "🏆 Peringkat Konfigurasi", "🧬 Profil Cluster", "📚 Metodologi"])
opt = lambda c: sorted(res[c].dropna().unique())
f_data = st.sidebar.multiselect("Jenis data", opt("data"), default=opt("data"))
f_dist = st.sidebar.multiselect("Jarak", opt("distance_full"), default=opt("distance_full"))
f_meth = st.sidebar.multiselect("Metode klaster", opt("method"), default=opt("method"))
kmin = st.sidebar.slider("Minimal jumlah cluster", 2, 10, 2, help="K=2 sering menang di Silhouette karena trivial; naikkan untuk melihat solusi yang lebih informatif.")
noise_max = st.sidebar.slider("Maks. noise DBSCAN (%)", 0, 100, 100)

F = res[res["data"].isin(f_data) & res["distance_full"].isin(f_dist) & res["method"].isin(f_meth) & (res["n_cluster"] >= kmin) & (res["noise_pct"] <= noise_max)].copy()

st.markdown('<div class="hero"><h1>Clustering Time Series Penjualan Produk</h1><p>DTW · CDTW · DDTW · WDTW × K-Means · K-Medoids · Hierarchical · DBSCAN × 3 jenis data</p></div>', unsafe_allow_html=True)
if F.empty:
    st.warning("Tidak ada konfigurasi yang lolos filter.")
    st.stop()


def rank_table(df, metrics):
    d = df.dropna(subset=metrics).copy()
    for m in metrics:
        d["r_" + m] = d[m].rank(ascending=not MAXIMIZE[m], method="min")
    d["rata_rank"] = d[["r_" + m for m in metrics]].mean(axis=1)
    return d.sort_values(["rata_rank"] + [m for m in metrics[:1]], ascending=[True, not MAXIMIZE[metrics[0]]]).reset_index(drop=True)


def best_pivot(df, metric, idx="distance_full", col="method"):
    agg = "max" if MAXIMIZE[metric] else "min"
    return df.pivot_table(index=idx, columns=col, values=metric, aggfunc=agg)


# ---------------- Halaman ----------------
if page == "🏠 Ringkasan":
    R = rank_table(F, ["silhouette", "chi", "dbi"])
    c = st.columns(4)
    card(c[0], "Konfigurasi dievaluasi", f"{len(F):,}", f"{F['distance_full'].nunique()} jarak · {F['method'].nunique()} metode · {F['data'].nunique()} data")
    if not R.empty:
        b = R.iloc[0]
        card(c[1], "Terbaik (rata-rata rank)", b["distance_full"], f"{b['method']} · {b['data']} · {b['setting']}")
    s = F.loc[F["silhouette"].idxmax()]
    card(c[2], "Silhouette tertinggi", f"{s['silhouette']:.3f}", f"{s['distance_full']} · {s['method']} · {s['setting']}")
    dmin = F.dropna(subset=["dbi"])
    if not dmin.empty:
        d = dmin.loc[dmin["dbi"].idxmin()]
        card(c[3], "DBI terendah", f"{d['dbi']:.3f}", f"{d['distance_full']} · {d['method']} · {d['setting']}")
    st.write("")
    a, b2 = st.columns(2)
    a.plotly_chart(style(px.box(F, x="method", y="silhouette", color="method", color_discrete_sequence=PAL, title="Sebaran Silhouette per metode klaster").update_layout(showlegend=False)), width='stretch')
    b2.plotly_chart(style(px.box(F, x="distance_full", y="silhouette", color="distance", color_discrete_sequence=PAL, title="Sebaran Silhouette per jarak").update_layout(showlegend=False)), width='stretch')
    pv = best_pivot(F, "silhouette")
    st.plotly_chart(style(px.imshow(pv, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis", title="Silhouette terbaik: jarak × metode klaster")), width='stretch')
    st.plotly_chart(style(px.scatter(F.dropna(subset=["dbi"]), x="dbi", y="silhouette", color="method", symbol="data", hover_name="config_id", color_discrete_sequence=PAL, title="Silhouette vs DBI", log_x=True)), width='stretch')

elif page == "📊 Eksplorasi Data":
    if npz is None:
        st.info("Letakkan `data_dashboard.npz` (hasil ekspor Colab) di folder yang sama dengan `app.py`.")
    else:
        X = npz["X_original"]
        codes = npz["product_code"]
        c = st.columns(3)
        card(c[0], "Jumlah produk", f"{X.shape[0]:,}")
        card(c[1], "Panjang deret", f"{X.shape[1]} minggu")
        card(c[2], "Total transaksi", f"{X.sum():,.0f}")
        wk = pd.DataFrame({"Minggu": np.arange(X.shape[1]), "Total": X.sum(0), "Rata-rata": X.mean(0)})
        st.plotly_chart(style(px.area(wk, x="Minggu", y="Total", title="Total transaksi per minggu", color_discrete_sequence=["#ec4899"])), width='stretch')
        a, b = st.columns(2)
        tot = pd.Series(X.sum(1), name="Total")
        a.plotly_chart(style(px.histogram(tot, x="Total", nbins=40, title="Distribusi total transaksi per produk", color_discrete_sequence=["#6366f1"])), width='stretch')
        top = pd.DataFrame({"Produk": codes, "Rata-rata": X.mean(1)}).nlargest(10, "Rata-rata")
        b.plotly_chart(style(px.bar(top, x="Rata-rata", y="Produk", orientation="h", title="10 produk dengan rata-rata tertinggi", color="Rata-rata", color_continuous_scale="Sunsetdark").update_yaxes(autorange="reversed")), width='stretch')
        pick = st.multiselect("Bandingkan produk", list(codes), default=list(codes[np.argsort(-X.mean(1))[:3]]))
        if pick:
            long = pd.DataFrame({p: X[list(codes).index(p)] for p in pick}).reset_index().melt("index", var_name="Produk", value_name="Transaksi")
            st.plotly_chart(style(px.line(long, x="index", y="Transaksi", color="Produk", color_discrete_sequence=PAL, title="Pola mingguan produk terpilih")), width='stretch')

elif page == "🔥 Eksplorasi Metrik":
    m = st.selectbox("Metrik", ["silhouette", "chi", "dbi", "balance"], format_func=METRIC_LABEL.get)
    if F[m].notna().sum() == 0:
        st.info("Kolom ini belum terisi di file hasil.")
    else:
        for d in sorted(F["data"].unique()):
            pv = best_pivot(F[F["data"] == d], m)
            fig = px.imshow(pv, text_auto=".3f" if m != "chi" else ".0f", aspect="auto", color_continuous_scale="RdYlGn_r" if m == "dbi" else "Viridis", title=f"Data {d} — nilai terbaik per sel")
            st.plotly_chart(style(fig), width='stretch')
        st.subheader("Profil terhadap jumlah cluster (K-Means, K-Medoids, Hierarchical)")
        KF = F[F["method"] != "DBSCAN"]
        if not KF.empty:
            c1, c2 = st.columns(2)
            dsel = c1.selectbox("Jarak", sorted(KF["distance_full"].unique()))
            dd = c2.selectbox("Data", sorted(KF["data"].unique()))
            sub = KF[(KF["distance_full"] == dsel) & (KF["data"] == dd)].copy()
            sub["seri"] = np.where(sub["linkage"] != "", sub["method"] + " – " + sub["linkage"], sub["method"])
            st.plotly_chart(style(px.line(sub.sort_values("k"), x="k", y=m, color="seri", markers=True, color_discrete_sequence=PAL, title=f"{METRIC_LABEL[m]} vs K")), width='stretch')
        DF = F[F["method"] == "DBSCAN"]
        if not DF.empty:
            st.subheader("DBSCAN: epsilon, noise, dan kualitas")
            fig = px.scatter(DF.dropna(subset=["silhouette"]), x="eps", y="silhouette", size="n_cluster", color="noise_pct", color_continuous_scale="Turbo", hover_name="config_id", title="Silhouette vs epsilon (warna = % noise, ukuran = jumlah cluster)")
            st.plotly_chart(style(fig), width='stretch')

elif page == "🏆 Peringkat Konfigurasi":
    ms = st.multiselect("Metrik yang dipakai dalam rata-rata rank", ["silhouette", "chi", "dbi", "balance"], default=["silhouette", "chi", "dbi"], format_func=METRIC_LABEL.get)
    if not ms:
        st.stop()
    R = rank_table(F, ms)
    if R.empty:
        st.warning("Tidak ada baris dengan semua metrik terpilih terisi.")
        st.stop()
    top = st.slider("Top-N", 5, min(50, len(R)), min(15, len(R)))
    cols = ["config_id", "n_cluster", "noise_pct"] + ms + ["rata_rank"]
    st.dataframe(R.head(top)[cols].rename(columns={"rata_rank": "rata-rata rank"}).style.background_gradient(subset=["rata-rata rank"], cmap="RdYlGn_r").format(precision=3), width='stretch', hide_index=True)
    fig = go.Figure()
    for _, r in R.head(5).iterrows():
        vals = []
        for m in ms:
            lo, hi = R[m].min(), R[m].max()
            z = (r[m] - lo) / (hi - lo if hi > lo else 1)
            vals.append(z if MAXIMIZE[m] else 1 - z)
        names = [METRIC_LABEL[m].split(" (")[0] + ("" if MAXIMIZE[m] else " (dibalik)") for m in ms]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself", name=r["config_id"][:45]))
    st.plotly_chart(style(fig.update_layout(title="Radar 5 teratas (0–1, makin luas makin baik)"), 520), width='stretch')
    st.plotly_chart(style(px.bar(R.groupby("method")["rata_rank"].min().reset_index(), x="method", y="rata_rank", color="method", color_discrete_sequence=PAL, title="Rata-rata rank terbaik per metode klaster (makin kecil makin baik)").update_layout(showlegend=False)), width='stretch')

elif page == "🧬 Profil Cluster":
    if npz is None or not any(k.startswith("lab_") for k in npz.files):
        st.info("Halaman ini butuh `data_dashboard.npz` dari sel ekspor Colab (berisi data deret waktu dan label cluster konfigurasi teratas).")
    else:
        avail = F[F["lab_key"].isin(npz.files)] if "lab_key" in F else F.iloc[0:0]
        if avail.empty:
            st.warning("Label cluster hanya disimpan untuk konfigurasi teratas hasil ekspor; longgarkan filter.")
            st.stop()
        R = rank_table(avail, ["silhouette", "chi", "dbi"]) if avail[["chi", "dbi"]].notna().all().all() else avail.sort_values("silhouette", ascending=False)
        cfg = st.selectbox("Konfigurasi", R["config_id"].tolist())
        row = R[R["config_id"] == cfg].iloc[0]
        lab = npz[row["lab_key"]]
        X, codes = npz["X_original"], npz["product_code"]
        basis = st.radio("Skala tampilan", ["Original", "Min-Max per produk", "Z-score per produk"], horizontal=True)
        Y = X.copy().astype(float)
        if basis.startswith("Min"):
            Y = (Y - Y.min(1, keepdims=True)) / np.where(np.ptp(Y, 1, keepdims=True) == 0, 1, np.ptp(Y, 1, keepdims=True))
        elif basis.startswith("Z"):
            Y = (Y - Y.mean(1, keepdims=True)) / np.where(Y.std(1, keepdims=True) == 0, 1, Y.std(1, keepdims=True))
        c = st.columns(4)
        card(c[0], "Silhouette", f"{row['silhouette']:.3f}")
        card(c[1], "Jumlah cluster", f"{len(set(lab) - {-1})}")
        card(c[2], "Noise", f"{(lab == -1).mean() * 100:.1f}%")
        card(c[3], "DBI / CHI", f"{row['dbi']:.2f} / {row['chi']:.0f}" if pd.notna(row["dbi"]) else "–")
        fig = go.Figure()
        for i, cl in enumerate(sorted(set(lab))):
            idx = np.where(lab == cl)[0]
            col = "#94a3b8" if cl == -1 else PAL[i % len(PAL)]
            nm = "Noise" if cl == -1 else f"Cluster {cl}"
            mu, sd = Y[idx].mean(0), Y[idx].std(0)
            x = np.arange(Y.shape[1])
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name=f"{nm} (n={len(idx)})", line=dict(color=col, width=3)))
        st.plotly_chart(style(fig.update_layout(title="Rata-rata ± 1 SD per cluster", xaxis_title="Minggu", yaxis_title="Transaksi"), 480), width='stretch')
        a, b = st.columns(2)
        cnt = pd.Series(lab).value_counts().sort_index().rename_axis("cluster").reset_index(name="jumlah")
        cnt["cluster"] = cnt["cluster"].astype(str)
        a.plotly_chart(style(px.bar(cnt, x="cluster", y="jumlah", color="cluster", color_discrete_sequence=PAL, title="Anggota per cluster").update_layout(showlegend=False)), width='stretch')
        cl_sel = b.selectbox("Lihat anggota cluster", sorted(set(lab)))
        mem = np.where(lab == cl_sel)[0]
        b.dataframe(pd.DataFrame({"Product_Code": codes[mem], "Rata-rata": X[mem].mean(1).round(2), "Total": X[mem].sum(1).round(0)}), width='stretch', hide_index=True, height=260)

else:
    st.subheader("Jarak antar deret waktu")
    st.markdown("Untuk dua deret $x=(x_1,\\dots,x_n)$ dan $y=(y_1,\\dots,y_n)$, matriks biaya kumulatif dibangun rekursif:")
    st.latex(r"D(i,j)=c(i,j)+\min\{D(i-1,j),\,D(i,j-1),\,D(i-1,j-1)\},\quad D(0,0)=0,\ D(i,0)=D(0,j)=\infty")
    st.latex(r"\text{DTW: } c(i,j)=(x_i-y_j)^2,\qquad DTW(x,y)=\sqrt{D(n,n)}")
    st.latex(r"\text{CDTW (Sakoe–Chiba): } D(i,j)=\infty \ \text{ jika } |i-j|>r,\quad r\in\{2,3,5\}")
    st.latex(r"\text{DDTW: } x'_i=\tfrac12(x_i-x_{i-1})+\tfrac14(x_{i+1}-x_{i-1}),\ \text{lalu DTW pada } x',y'")
    st.latex(r"\text{WDTW: } c(i,j)=w_{|i-j|}(x_i-y_j)^2,\quad w_a=\frac{1}{1+e^{-g(a-n/2)}},\ g\in\{0.05,0.1,0.5,1.0\}")
    st.subheader("Representasi untuk K-Means, CHI, dan DBI")
    st.markdown("Matriks jarak $D$ ($811\\times811$) diubah menjadi koordinat 5 dimensi dengan *metric MDS* agar K-Means, CHI, dan DBI dapat dihitung.")
    st.latex(r"\min_{Z}\ \sum_{i<j}\left(d_{ij}-\lVert z_i-z_j\rVert\right)^2")
    st.subheader("Metrik validasi")
    st.latex(r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},\quad Sil=\frac1n\sum_{i=1}^n s(i)\quad(\text{dari matriks jarak asli})")
    st.latex(r"CHI=\frac{\operatorname{tr}(B_K)/(K-1)}{\operatorname{tr}(W_K)/(n-K)}\qquad DBI=\frac1K\sum_{k=1}^K\max_{l\ne k}\frac{S_k+S_l}{d(c_k,c_l)}")
    st.latex(r"\text{Balance}=\frac{-\sum_k p_k\ln p_k}{\ln K},\quad p_k=\frac{n_k}{n}")
    st.latex(r"\text{rata-rata rank}_c=\frac1{|M|}\sum_{m\in M}R_m(c)")
    st.caption("Pada DBSCAN, titik noise (label −1) dikeluarkan sebelum menghitung ketiga metrik. Konfigurasi dengan < 2 cluster tidak dievaluasi.")
