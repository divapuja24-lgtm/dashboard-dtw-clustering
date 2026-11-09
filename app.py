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
/* ---------- Global theme ---------- */
:root { --ink:#e8f0ff; --muted:#9fb0ca; --navy:#0b1220; --panel:#111d31; --line:rgba(148,163,184,.18); --cyan:#38bdf8; }
.stApp { background:linear-gradient(145deg,#0b1220 0%,#101a2d 55%,#0c1728 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(11,18,32,.82); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111d31 0%,#172642 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#dce8fb; }
[data-testid="stSidebar"] h1 { color:#f8fbff; font-size:1.45rem; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#dce8fb; font-weight:600; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
.hero { padding:30px 34px; border-radius:22px; margin:4px 0 24px; background:linear-gradient(115deg,#17345c 0%,#155e75 55%,#2563a6 100%); border:1px solid rgba(125,211,252,.22); box-shadow:0 16px 38px rgba(0,0,0,.18); }
.hero h1 { color:#fff; margin:0; font-size:2.15rem; font-weight:750; letter-spacing:-.035em; }
.hero p { color:#d8f2ff; opacity:.95; margin:9px 0 0; font-size:1rem; }
.section-kicker { color:#67e8f9; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:750; margin-bottom:5px; }
.card { padding:19px 20px; border-radius:18px; background:linear-gradient(145deg,rgba(25,43,69,.96),rgba(17,29,49,.96)); border:1px solid rgba(148,163,184,.2); min-height:112px; box-shadow:0 8px 22px rgba(0,0,0,.12); }
.card .k { color:#a9bdd8; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.card .v { color:#f8fbff; font-size:1.42rem; font-weight:750; margin-top:8px; line-height:1.25; overflow-wrap:anywhere; }
.card .s { color:#a9bdd8; font-size:.78rem; margin-top:7px; line-height:1.4; }
[data-testid="stMetric"] { background:rgba(17,29,49,.72); border:1px solid var(--line); padding:14px 16px; border-radius:15px; }
[data-testid="stTabs"] button { font-weight:650; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.stPlotlyChart { border:1px solid rgba(148,163,184,.12); border-radius:16px; padding:5px; background:rgba(17,29,49,.35); }
hr { border-color:var(--line); }
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
st.sidebar.markdown("## 🎛️ Panel kontrol")
st.sidebar.caption("Atur data dan konfigurasi yang ingin dibandingkan.")
page = st.sidebar.radio("NAVIGASI", ["🏠 Ringkasan", "📊 Eksplorasi Data", "🔥 Eksplorasi Metrik", "🏆 Peringkat Konfigurasi", "🧬 Profil Cluster", "📚 Metodologi"])
opt = lambda c: sorted(res[c].dropna().unique())
f_data = st.sidebar.multiselect("Jenis data", opt("data"), default=opt("data"))

# Pilih keluarga metode jarak terlebih dahulu, lalu parameter untuk CDTW/WDTW.
distance_col = "distance" if "distance" in res.columns else "distance_full"
all_distances = list(res[distance_col].dropna().unique())
canonical = [d for d in ["DTW", "CDTW", "DDTW", "WDTW"] if d in all_distances]
other_distances = sorted([d for d in all_distances if d not in canonical])
distance_options = canonical + other_distances
f_dist_family = st.sidebar.multiselect("Metode jarak", distance_options, default=distance_options,
    help="Pilih keluarga metode. Parameter CDTW dan WDTW diatur terpisah di bawah.")

parameterized = res[res[distance_col].isin(f_dist_family)].copy()
parameterized = parameterized[parameterized[distance_col].isin(["CDTW", "WDTW"])]
if not parameterized.empty:
    # distance_full sudah memuat nama keluarga + parameter (mis. CDTW r=2).
    param_options = sorted(parameterized["distance_full"].dropna().unique())
    f_params = st.sidebar.multiselect("Parameter CDTW / WDTW", param_options, default=param_options,
        help="Pilih konfigurasi radius CDTW atau bobot WDTW yang ingin dibandingkan.")
else:
    f_params = []

f_meth = st.sidebar.multiselect("Metode klaster", opt("method"), default=opt("method"))
kmin = st.sidebar.slider("Minimal jumlah cluster", 2, 10, 2, help="Atur batas minimum jumlah cluster yang ditampilkan.")
noise_max = st.sidebar.slider("Maksimum noise DBSCAN (%)", 0, 100, 100)

mask = res["data"].isin(f_data) & res[distance_col].isin(f_dist_family) & res["method"].isin(f_meth) & (res["n_cluster"] >= kmin) & (res["noise_pct"] <= noise_max)
parameterized_rows = res[distance_col].isin(["CDTW", "WDTW"])
param_mask = ~parameterized_rows | res["distance_full"].isin(f_params)
mask &= param_mask
F = res[mask].copy()

st.markdown('<div class="hero"><div class="section-kicker">ANALYTICS DASHBOARD</div><h1>Clustering Time Series Penjualan Produk</h1><p>Eksplorasi, evaluasi, dan perbandingan konfigurasi clustering berbasis DTW</p></div>', unsafe_allow_html=True)
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
    st.markdown("<div class='section-kicker'>OVERVIEW</div>", unsafe_allow_html=True)
    st.subheader("Ringkasan hasil analisis")
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
    st.markdown("<div class='section-kicker'>DATA INSIGHT</div>", unsafe_allow_html=True)
    st.subheader("Eksplorasi data penjualan")
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
    st.markdown("<div class='section-kicker'>MODEL EVALUATION</div>", unsafe_allow_html=True)
    st.subheader("Eksplorasi metrik validasi")
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
    st.markdown("<div class='section-kicker'>MODEL RANKING</div>", unsafe_allow_html=True)
    st.subheader("Peringkat konfigurasi terbaik")
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
    st.markdown("<div class='section-kicker'>CLUSTER INSIGHT</div>", unsafe_allow_html=True)
    st.subheader("Profil dan karakteristik cluster")
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
    st.markdown("<div class='section-kicker'>METHOD NOTES</div>", unsafe_allow_html=True)
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
