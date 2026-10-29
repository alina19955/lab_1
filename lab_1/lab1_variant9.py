"""
Лабораторная работа 1, вариант 9: все 50 пунктов.
Файл должен лежать рядом с lab1_part12.py (там реализованы пункты 1-13).

Запуск:  python lab1_variant9.py Lab1_Dataset.xlsx IMG_4811.jpeg
Результаты: output/ (таблицы и картинки с датой в имени), figs/ (рисунки для отчёта), results.json.
Блоки TensorFlow / Keras / PyTorch выполняются, только если библиотеки установлены;
иначе те же вычисления выполняются средствами NumPy (результат тот же).
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats, signal, interpolate, ndimage, sparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from PIL import Image, ImageOps
import lab1_part12 as L

VARIANT, SEED = 9, 42
DATASET = sys.argv[1] if len(sys.argv) > 1 else "Lab1_Dataset.xlsx"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "IMG_4811.jpeg"
os.makedirs("figs", exist_ok=True)
R = {}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})


def fig_save(fig, name):
    fig.savefig(f"figs/{name}.png", dpi=140, bbox_inches="tight")
    L.save_figure(fig, name)                      # п.3: сохранение с датой и временем в имени
    plt.close(fig)


# ---------------------------------------------------------------- п.13
L.set_seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- п.1-7
def extract_variant(path, variant):
    raw = pd.read_excel(path, header=None)
    labels = {i: str(v).strip() for i, v in enumerate(raw.iloc[0]) if isinstance(v, str)}
    start = [i for i, v in labels.items() if v == f"Вариант {variant}"][0]
    later = [i for i in labels if i > start]
    end = min(later) if later else raw.shape[1]
    block = raw.iloc[2:, start:end].copy()
    block.columns = [str(c).strip() for c in raw.iloc[1, start:end]]
    return block.reset_index(drop=True), (start, end)

df_raw, span = extract_variant(DATASET, VARIANT)
df = df_raw.astype("float64")                       # п.7: явный тип float64
R["span"] = span; R["n_raw"] = len(df); R["cols"] = list(df.columns)
R["issues"] = {k: v for k, v in L.check_data(df).items()}          # п.4
R["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
R["head_raw"] = df.head(5).round(6).values.tolist()

# ---------------------------------------------------------------- п.8
# Меню (L.preprocessing_menu) интерактивное; для воспроизводимости применяем действия из меню автоматически.
empty_rows = df.isna().all(axis=1)
R["dropped_empty_rows"] = int(empty_rows.sum())
df = df[~empty_rows].reset_index(drop=True)                          # "выбросить строки"
x_nan = df["x"].isna(); R["x_filled"] = int(x_nan.sum())
df.loc[x_nan, "x"] = df["x"].iloc[0] + df.index[x_nan]               # "указанное значение": шаг сетки x равен 1
z_nan = df["z"].isna(); R["z_filled"] = int(z_nan.sum())
df = L.fill_prev(df)                                                  # "предыдущее значение"
assert not df.isna().any().any()
R["n"] = len(df); R["head_clean"] = df.head(5).round(6).values.tolist()

# п.1-3: запись и чтение во всех форматах, автоимена столбцов
saved = {f: L.save_table(df, f"v{VARIANT}", f) for f in ("xlsx", "csv", "txt", "mat")}
R["saved_files"] = list(saved.values())
chk = {f: list(L.read_table(p).shape) for f, p in saved.items()}
nohdr = os.path.join(L.OUT_DIR, "no_header_test.csv"); pd.DataFrame(np.random.rand(3, 2)).to_csv(nohdr, header=False, index=False)
R["read_back_shapes"] = chk; R["auto_names"] = list(L.read_table(nohdr).columns)


# ---------------------------------------------------------------- п.9-12
X = L.to_numpy(df); n = len(X); R["X_shape"] = list(X.shape)
Xmm, p_mm = L.minmax_scale(X); Xab, p_ab = L.minmax_scale_ab(X, -5, 5); Xz, p_z = L.zscore_scale(X)
np.savez("scaler_params.npz", **{f"{p['type']}_{k}": v for p in (p_mm, p_ab, p_z) for k, v in p.items() if k != "type"})
R["scale_first_rows"] = {"minmax": Xmm[:2, 0].round(6).tolist(), "minmax_ab": Xab[:2, 0].round(6).tolist(), "zscore": Xz[:2, 0].round(6).tolist()}
R["inv_err"] = {k: float(np.abs(L.inverse_scale(xs, p) - X).max()) for k, xs, p in
                (("minmax", Xmm, p_mm), ("minmax_ab", Xab, p_ab), ("zscore", Xz, p_z))}
parts = L.split_data(X, (75, 10, 15), name="X"); R["split"] = {k: len(v) for k, v in parts.items()}

x, z = X[:, 0], X[:, 1]; t = np.arange(n)

# ---------------------------------------------------------------- п.14
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(t, x, color="C0"); ax[0].set_title("Столбец x"); ax[1].plot(t, z, color="C1"); ax[1].set_title("Столбец z")
for a in ax: a.set_xlabel("индекс наблюдения")
fig_save(fig, "01_raw")

# ---------------------------------------------------------------- п.16
Xs = np.sort(X, axis=0); R["sorted10"] = Xs[:10].round(6).tolist()

# ---------------------------------------------------------------- п.18 (NumPy и Pandas)
def mode_info(s):
    m = s.mode(); first = float(m.iloc[0]); return first, int((s == first).sum()), len(m)
stat = {}
for c, col in zip("xz", (x, z)):
    s = pd.Series(col)
    np_v = (np.mean(col), np.var(col, ddof=1), np.median(col)); pd_v = (s.mean(), s.var(), s.median())
    assert np.allclose(np_v, pd_v)
    mo = mode_info(s)
    stat[c] = dict(mean=float(np_v[0]), var=float(np_v[1]), median=float(np_v[2]), mode=mo[0], mode_freq=mo[1], n_modes=mo[2])
R["stats"] = stat

# ---------------------------------------------------------------- п.15, 18: гистограммы + статистики
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for j, (c, col) in enumerate(zip("xz", (x, z))):
    ax[0, j].hist(col, bins=15, color="steelblue", edgecolor="w"); ax[0, j].set_title(f"Гистограмма {c}")
    ax[1, j].hist(col, bins=15, density=True, color="seagreen", edgecolor="w"); ax[1, j].set_title(f"Нормированная гистограмма {c} (плотность)")
    st = stat[c]
    for a in (ax[0, j], ax[1, j]):
        a.axvline(st["mean"], color="red", label=f"среднее {st['mean']:.2f}")
        a.axvline(st["median"], color="orange", ls="--", label=f"медиана {st['median']:.2f}")
        a.axvline(st["mode"], color="k", ls=":", label=f"мода {st['mode']:.2f}")
    ax[1, j].legend(fontsize=8)
fig_save(fig, "02_hist")

# ---------------------------------------------------------------- п.17 ECDF
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, c, col in zip(ax, "xz", (x, z)):
    s = np.sort(col); a.step(s, np.arange(1, n + 1) / n, where="post"); a.set_title(f"ЭФР {c}"); a.set_ylabel("F(x)")
fig_save(fig, "03_ecdf")

# ---------------------------------------------------------------- п.19 доверительные интервалы
alpha = .05; ci = {}
for c, col in zip("xz", (x, z)):
    m, s2 = col.mean(), col.var(ddof=1)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1); se = np.sqrt(s2 / n)
    var_ci = ((n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, n - 1), (n - 1) * s2 / stats.chi2.ppf(alpha / 2, n - 1))
    ci[c] = dict(mean=[float(m - tcrit * se), float(m + tcrit * se)], var=[float(v) for v in var_ci])
R["ci"] = ci

# ---------------------------------------------------------------- п.20-22
R["cov"] = np.cov(X.T).tolist(); R["corr"] = np.corrcoef(X.T).tolist()
r, p = stats.pearsonr(x, z); R["pearson"] = [float(r), float(p)]
shifts = np.arange(-20, 21)
def xc(k):
    return np.corrcoef(x[k:], z[:n - k])[0, 1] if k >= 0 else np.corrcoef(x[:n + k], z[-k:])[0, 1]
cc = np.array([xc(k) for k in shifts]); R["xcorr_max"] = [int(shifts[cc.argmax()]), float(cc.max())]
R["xcorr_min"] = [int(shifts[cc.argmin()]), float(cc.min())]; R["xcorr0"] = float(cc[20])
fig, ax = plt.subplots(figsize=(7, 3.8)); ax.stem(shifts, cc); ax.set_xlabel("сдвиг, элементов"); ax.set_ylabel("коэффициент корреляции")
ax.set_title("Взаимная корреляция x и z"); fig_save(fig, "04_xcorr")

# ---------------------------------------------------------------- п.23-26
grad = np.gradient(X, axis=0); R["grad_shape"] = list(grad.shape)
conv = np.convolve(x, z, mode="full"); R["conv_len"] = len(conv)
R["dot"] = float(np.dot(x, z)); R["cross2d"] = float(x[0] * z[1] - x[1] * z[0]); R["cross3d"] = np.cross(x[:3], z[:3]).tolist()
R["norms"] = {c: dict(l1=float(np.linalg.norm(col, 1)), l2=float(np.linalg.norm(col, 2))) for c, col in zip("xz", (x, z))}

# ---------------------------------------------------------------- п.27
ks_u = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min()))
ks_n = stats.kstest(z, "norm", args=(z.mean(), z.std(ddof=1))); sh = stats.shapiro(z)
R["tests"] = dict(ks_uniform=[float(ks_u.statistic), float(ks_u.pvalue)], ks_norm=[float(ks_n.statistic), float(ks_n.pvalue)],
                  shapiro=[float(sh.statistic), float(sh.pvalue)], skew_z=float(stats.skew(z)), kurt_z=float(stats.kurtosis(z)))

# ---------------------------------------------------------------- п.28-30
fig, ax = plt.subplots(3, 2, figsize=(11, 10))
for j, (c, col) in enumerate(zip("xz", (x, z))):
    f, tt, Sxx = signal.spectrogram(col, fs=1, nperseg=32, noverlap=16)
    im = ax[0, j].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="auto"); ax[0, j].set_title(f"Спектрограмма {c}")
    ax[0, j].set_ylabel("частота"); ax[0, j].set_xlabel("время (индекс)"); ax[0, j].grid(False); plt.colorbar(im, ax=ax[0, j], label="дБ")
    fp, Pxx = signal.periodogram(col, fs=1); ax[1, j].semilogy(fp, Pxx + 1e-12); ax[1, j].set_title(f"Периодограмма {c}"); ax[1, j].set_xlabel("частота")
    F = np.fft.rfft(col - col.mean()); ff = np.fft.rfftfreq(n, 1); ax[2, j].plot(ff, np.abs(F) / n * 2); ax[2, j].set_title(f"АЧХ {c} (|FFT|)"); ax[2, j].set_xlabel("частота")
    R.setdefault("fft_peak", {})[c] = float(ff[1:][np.abs(F)[1:].argmax()])
fig.tight_layout(); fig_save(fig, "05_spectral")

# ---------------------------------------------------------------- п.31-32
tt = np.linspace(0, n - 1, 5 * n); fig, ax = plt.subplots(1, 2, figsize=(11, 3.9)); R["interp_err"] = {}
for a, c, col in zip(ax, "xz", (x, z)):
    cub = interpolate.interp1d(t, col, kind="cubic"); spl = interpolate.UnivariateSpline(t, col, k=3)
    a.plot(t, col, "o", ms=3, label="исходные точки"); a.plot(tt, cub(tt), label="кубическая"); a.plot(tt, spl(tt), "--", label="сплайн"); a.set_title(f"Столбец {c}"); a.legend(fontsize=8)
    R["interp_err"][c] = dict(cubic=float(np.abs(cub(t) - col).max()), spline=float(np.abs(spl(t) - col).max()))
fig_save(fig, "06_interp")

# ---------------------------------------------------------------- п.33
masks = {"pos": X > 0, "neg": X < 0, "zero": X == 0, "m11": (X >= -1) & (X <= 1)}
R["masks"] = {c: {k: int(m[:, j].sum()) for k, m in masks.items()} for j, c in enumerate("xz")}

# ---------------------------------------------------------------- п.34-35
sk_mm = MinMaxScaler().fit(X); sk_z = StandardScaler().fit(X)
Xmm_sk, Xz_sk = sk_mm.transform(X), sk_z.transform(X)
R["scale_cmp"] = dict(minmax_diff=float(np.abs(Xmm - Xmm_sk).max()), zscore_diff=float(np.abs(Xz - Xz_sk).max()),
    zscore_ratio=float(np.mean((Xz_sk / Xz)[np.abs(Xz) > 1e-9])), sqrt_ratio=float(np.sqrt((n - 1) / n)),
    inv_own_mm=float(np.abs(L.inverse_scale(Xmm, p_mm) - X).max()), inv_sk_mm=float(np.abs(sk_mm.inverse_transform(Xmm_sk) - X).max()),
    inv_own_z=float(np.abs(L.inverse_scale(Xz, p_z) - X).max()), inv_sk_z=float(np.abs(sk_z.inverse_transform(Xz_sk) - X).max()))
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for j, c in enumerate("xz"):
    ax[0, j].plot(Xmm[:, j], label="своё MinMax"); ax[0, j].plot(Xmm_sk[:, j], "--", label="MinMaxScaler"); ax[0, j].set_title(f"MinMax, столбец {c}"); ax[0, j].legend(fontsize=8)
    ax[1, j].plot(Xz[:, j], label="своё z-score (ddof=1)"); ax[1, j].plot(Xz_sk[:, j], "--", label="StandardScaler (ddof=0)"); ax[1, j].set_title(f"Стандартизация, столбец {c}"); ax[1, j].legend(fontsize=8)
fig_save(fig, "07_scaling")
cmp_df = pd.DataFrame({"x_minmax_own": Xmm[:, 0], "x_minmax_sk": Xmm_sk[:, 0], "z_minmax_own": Xmm[:, 1], "z_minmax_sk": Xmm_sk[:, 1],
                       "x_zscore_own": Xz[:, 0], "x_zscore_sk": Xz_sk[:, 0], "z_zscore_own": Xz[:, 1], "z_zscore_sk": Xz_sk[:, 1]})
R["cmp_file"] = L.save_table(cmp_df, "scaling_comparison", "csv")

# ---------------------------------------------------------------- п.36-37
def rolling_windows(a, w): return sliding_window_view(a, w, axis=0)       # (n-w+1, ..., w)
def moving_average(a, w): return rolling_windows(a, w).mean(axis=-1)
W_ = 10; R["win_shape"] = list(rolling_windows(x, W_).shape); R["ma_shape"] = list(moving_average(X, W_).shape)

# ---------------------------------------------------------------- п.38-42 (тензоры; TF/PyTorch при наличии)
A = rng.integers(0, 10, size=(3, n)); Wt = rng.normal(size=(2, 4)); Bt = rng.uniform(size=(3, 4))
res_np = A @ X @ Wt + Bt; R["tensor"] = dict(X=list(X.shape), A=list(A.shape), W=list(Wt.shape), B=list(Bt.shape), AXWB=list(res_np.shape))
libs = {}
try:
    import tensorflow as tf
    tx, ta, tw, tb = (tf.constant(v, dtype=tf.float64) for v in (X, A, Wt, Bt))
    libs["tensorflow"] = float(np.abs(tf.matmul(tf.matmul(ta, tx), tw).numpy() + Bt - res_np).max())
except Exception as e: libs["tensorflow"] = None
try:
    import torch
    tx, ta, tw, tb = (torch.tensor(v, dtype=torch.float64) for v in (X, A, Wt, Bt))
    libs["torch"] = float(np.abs((ta @ tx @ tw + tb).numpy() - res_np).max())
except Exception as e: libs["torch"] = None
R["tensor_libs_diff_vs_numpy"] = libs

# ---------------------------------------------------------------- п.43-45
T = rng.normal(size=(3, 10)); P = rng.uniform(size=(3, 10)); Q = rng.uniform(.5, 2, size=(3, 10))
V_np = np.abs(np.sin(T) - np.exp(P) * np.sqrt(Q)); R["V_shape"] = list(V_np.shape); vl = {}
try:
    import keras; from keras import ops
    kT, kP, kQ = (ops.convert_to_tensor(v) for v in (T, P, Q))
    vl["keras"] = float(np.abs(ops.convert_to_numpy(ops.abs(ops.sin(kT) - ops.exp(kP) * ops.sqrt(kQ))) - V_np).max())
except Exception: vl["keras"] = None
try:
    import torch
    tT, tP, tQ = (torch.tensor(v) for v in (T, P, Q)); vl["torch"] = float(np.abs(torch.abs(torch.sin(tT) - torch.exp(tP) * torch.sqrt(tQ)).numpy() - V_np).max())
except Exception: vl["torch"] = None
R["V_libs_diff_vs_numpy"] = vl

# ---------------------------------------------------------------- п.46-47
S = sparse.random(100_000, 100_000, density=1e-5, format="csr", random_state=np.random.default_rng(SEED))
R["sparse"] = dict(shape=list(S.shape), nnz=int(S.nnz), density=float(S.nnz / 1e10), mb=float((S.data.nbytes + S.indices.nbytes + S.indptr.nbytes) / 2**20))
M = rng.normal(size=(100, 100)); det = np.linalg.det(M); Minv = np.linalg.inv(M)
R["inverse"] = dict(det=float(det), err=float(np.abs(M @ Minv - np.eye(100)).max()), cond=float(np.linalg.cond(M)))

# ---------------------------------------------------------------- п.48: тензор F через представления (views)
chan = ["X", "MinMax", "ZScore", ">0", "<0", "==0", "[-1,1]"]
Ft = np.zeros((n, len(chan), 2))                                   # измерения: наблюдения x каналы x столбцы
views = [Ft[:, i, :] for i in range(len(chan))]                     # представления, а не копии
assert all(v.base is Ft for v in views)
for v, src in zip(views, (X, Xmm, Xz, *masks.values())): v[...] = src
R["F_shape"] = list(Ft.shape); R["F_views_share_memory"] = all(np.shares_memory(v, Ft) for v in views)

# ---------------------------------------------------------------- п.49: пайплайн scaler -> PCA
sets = {"X + маски (10)": [0, 3, 4, 5, 6], "только маски (8)": [3, 4, 5, 6], "все каналы F (14)": list(range(7))}
scalers = {"MinMax": MinMaxScaler, "Standard": StandardScaler}; rows = []; ll_curves = {}
for sn, sc in scalers.items():
    for fn, idx in sets.items():
        Z = Ft[:, idx, :].reshape(n, -1)
        pipe = Pipeline([("scale", sc()), ("pca", PCA(n_components=2, random_state=SEED))]).fit(Z)
        ev2 = float(pipe["pca"].explained_variance_ratio_.sum()); Zs = pipe["scale"].transform(Z)
        ks = list(range(1, min(Z.shape[1] - 1, 8) + 1)); ll = []
        for k in ks:
            try:
                pk = PCA(n_components=k, random_state=SEED).fit(Zs)
                # при почти нулевой остаточной дисперсии правдоподобие вырождается (деление на ~0) - такие k пропускаем
                ll.append(float(pk.score(Zs)) if pk.noise_variance_ > 1e-2 * np.mean(Zs.var(axis=0)) else np.nan)
            except Exception: ll.append(np.nan)
        ll_curves[(sn, fn)] = (ks, ll); rows.append(dict(scaler=sn, feats=fn, p=Z.shape[1], ev2=ev2, ll2=ll[1] if len(ll) > 1 else float('nan')))
R["pipelines"] = rows; best = max(rows, key=lambda r: r["ev2"]); R["pipe_best"] = best
fig, ax = plt.subplots(figsize=(8, 4.6))
for (sn, fn), (ks, ll) in ll_curves.items():
    ax.plot(ks, ll, "o-", label=f"{sn} + PCA, {fn}", lw=2.5 if (sn, fn) == (best["scaler"], best["feats"]) else 1)
ax.set_xlabel("число главных компонент"); ax.set_ylabel("средняя лог-правдоподобность (PCA.score)"); ax.set_title("Логарифмическое правдоподобие PCA"); ax.legend(fontsize=7)
fig_save(fig, "08_pca_ll")

# ---------------------------------------------------------------- п.50: изображение
im = ImageOps.exif_transpose(Image.open(IMAGE)).convert("RGB"); R["img_orig"] = list(im.size)
im.thumbnail((1280, 1280), Image.LANCZOS); img = np.asarray(im, dtype=float); h, w, _ = img.shape; R["img_size"] = [w, h]
u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
chs = [img[:, :, i] for i in range(3)]                              # каналы в отдельных массивах
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(u8(img)); ax[0].set_title("Исходное")
for a, ch, nm, cm in zip(ax[1:], chs, "RGB", ("Reds", "Greens", "Blues")): a.imshow(ch, cmap=cm, vmin=0, vmax=255); a.set_title(f"Канал {nm}")
for a in ax: a.axis("off")
fig_save(fig, "09_channels")
rot = ndimage.rotate(img, 40, reshape=True, order=1, cval=0); gray = img @ np.array([.299, .587, .114])
fig, ax = plt.subplots(1, 2, figsize=(11, 6)); ax[0].imshow(u8(rot)); ax[0].set_title(f"Поворот на 40° ({rot.shape[1]}×{rot.shape[0]})")
ax[1].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[1].set_title("Чёрно-белое")
for a in ax: a.axis("off")
fig_save(fig, "10_rotate_gray")
hist, edges = np.histogram(gray, bins=256, range=(0, 255)); cum = np.cumsum(hist) / hist.sum()
t1, t2 = int(np.searchsorted(cum, 1 / 3)), int(np.searchsorted(cum, 2 / 3)); R["thresholds"] = [t1, t2]
fig, ax = plt.subplots(figsize=(7.5, 3.8)); ax.bar(edges[:-1], hist, width=1, color="gray")
for tv in (t1, t2): ax.axvline(tv, color="red", ls="--")
ax.set_xlabel("яркость"); ax.set_ylabel("число пикселей"); ax.set_title(f"Гистограмма ЧБ-изображения (пороги {t1} и {t2})"); fig_save(fig, "11_hist_gray")
bands = [gray < t1, (gray >= t1) & (gray < t2), gray >= t2]; R["band_share"] = [float(b.mean()) for b in bands]
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
for a, b, nm in zip(ax, bands, ("тёмные", "средние", "светлые")): a.imshow(np.where(b, gray, 0), cmap="gray", vmin=0, vmax=255); a.set_title(f"Пиксели: {nm}"); a.axis("off")
fig_save(fig, "12_split")
yy, xx = np.ogrid[:h, :w]; circ = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (min(h, w) / 2) ** 2
fig, ax = plt.subplots(figsize=(6, 7.5)); ax.imshow(u8(img * circ[..., None])); ax.axis("off"); ax.set_title("Чёрная круговая рамка"); fig_save(fig, "13_circle")
noisy = np.clip(img + rng.normal(0, 25, img.shape), 0, 255); blur = ndimage.gaussian_filter(img, sigma=(3, 3, 0))
sharp = np.clip(img + 1.5 * (img - ndimage.gaussian_filter(img, sigma=(2, 2, 0))), 0, 255)
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
for a, im_, nm in zip(ax, (noisy, blur, sharp), ("Случайный шум (σ=25)", "Гауссово размытие (σ=3)", "Sharpening (unsharp mask)")): a.imshow(u8(im_)); a.set_title(nm); a.axis("off")
fig_save(fig, "14_noise_blur_sharp")
ks_img = (5, 20, 50); recs = {k: np.zeros_like(img) for k in ks_img}; var_cum = []
for i, ch in enumerate(chs):
    pc = PCA(n_components=50, random_state=SEED, svd_solver="randomized").fit(ch); Z = pc.transform(ch); var_cum.append(np.cumsum(pc.explained_variance_ratio_))
    for k in ks_img: recs[k][:, :, i] = Z[:, :k] @ pc.components_[:k] + pc.mean_
vc = np.mean(var_cum, axis=0); R["img_pca"] = {str(k): float(vc[k - 1]) for k in ks_img}
fig, ax = plt.subplots(1, 4, figsize=(17, 5.5)); ax[0].imshow(u8(img)); ax[0].set_title("Исходное")
for a, k in zip(ax[1:], ks_img): a.imshow(u8(recs[k])); a.set_title(f"PCA, {k} комп. ({vc[k-1]*100:.1f}% дисперсии)")
for a in ax: a.axis("off")
fig_save(fig, "15_img_pca")

json.dump(R, open("results1.json", "w"), ensure_ascii=False, indent=1, default=float)
print("готово; результаты в results1.json")
