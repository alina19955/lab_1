"""
Лабораторная работа 1. Разделы 1-2 (пункты 1-13).
Вариант: 9.

Запуск:  python lab1_part12.py путь_к_датасету.xlsx
"""
import os
import random
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # чтобы сохранение рисунков работало без окна; уберите, если нужны окна
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# ----------------------------------------------------------------------------
# НАСТРОЙКИ
# ----------------------------------------------------------------------------
VARIANT = 9
SEED = 42

# Способ выбора столбцов варианта (пункт 6). Выберите ОДИН:
# а) задать имена столбцов явно (надёжнее всего), например ["x9", "y9"]:
VARIANT_COLUMNS = None
# б) если в датасете у каждого варианта подряд идут COLS_PER_VARIANT столбцов
#    (вариант 1 -> столбцы 0,1; вариант 9 -> столбцы 16,17). Это ПРЕДПОЛОЖЕНИЕ,
#    сверьте со структурой своего файла.
COLS_PER_VARIANT = 2

OUT_DIR = "output"


# ----------------------------------------------------------------------------
# Пункт 13. Фиксация seed
# ----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except (ImportError, AttributeError):
        pass


# ----------------------------------------------------------------------------
# Пункты 1-3. Чтение / запись
# ----------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _first_row_is_header(raw: pd.DataFrame) -> bool:
    """Первая строка - заголовок, если ни одна её ячейка не является числом."""
    first = raw.iloc[0]
    return pd.to_numeric(first, errors="coerce").isna().all()


def _auto_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
    return df


def _read_mat(path: str) -> pd.DataFrame:
    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    # наш собственный формат записи: переменная 'columns' + по переменной на столбец
    if "columns" in mat:
        names = []
        for c in np.ravel(mat["columns"]):
            while isinstance(c, np.ndarray):   # loadmat вкладывает строки в массивы
                c = c.ravel()[0]
            names.append(str(c).strip())
        data = {n: np.ravel(mat[_safe_name(n)]) for n in names}
        return pd.DataFrame(data)
    for k in keys:
        arr = np.asarray(mat[k])
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return _auto_names(pd.DataFrame(arr))
    raise ValueError("В .mat-файле не найдено числовой 2D-матрицы")


def read_table(path: str) -> pd.DataFrame:
    """Чтение .xlsx, .csv, .txt, .mat. Без заголовков - имена col_1, col_2, ..."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return _read_mat(path)
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
    elif ext in (".csv", ".txt"):
        raw = pd.read_csv(path, header=None, sep=None, engine="python")
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    if _first_row_is_header(raw):
        df = raw.iloc[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in raw.iloc[0]]
        df = _soft_numeric(df)
    else:
        df = _auto_names(raw)
    return df


def _soft_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит столбец к числам, только если в нём нет потерь (иначе оставляет как есть,
    чтобы проверка из пункта 4 нашла некорректные данные)."""
    out = df.copy()
    for c in out.columns:
        conv = pd.to_numeric(out[c], errors="coerce")
        if conv.notna().sum() == out[c].notna().sum():
            out[c] = conv
    return out


def _safe_name(name: str) -> str:
    """Имя столбца -> допустимое имя переменной MATLAB."""
    s = re.sub(r"\W", "_", str(name))
    return s if re.match(r"[A-Za-z]", s) else "v_" + s


def save_table(df: pd.DataFrame, base_name: str, fmt: str, folder: str = OUT_DIR) -> str:
    """Запись с сохранением заголовков; в имени файла дата и время."""
    os.makedirs(folder, exist_ok=True)
    fmt = fmt.lower().lstrip(".")
    path = os.path.join(folder, f"{base_name}_{_timestamp()}.{fmt}")
    if fmt == "xlsx":
        df.to_excel(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "txt":
        df.to_csv(path, index=False, sep="\t")
    elif fmt == "mat":
        payload = {_safe_name(c): df[c].to_numpy() for c in df.columns}
        payload["columns"] = np.array([str(c) for c in df.columns], dtype=object)
        savemat(path, payload)
    else:
        raise ValueError(f"Неподдерживаемый формат: {fmt}")
    return path


def save_figure(fig, base_name: str, folder: str = OUT_DIR) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{base_name}_{_timestamp()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ----------------------------------------------------------------------------
# Пункты 4-7. Проверка, вывод, выбор столбцов, типы
# ----------------------------------------------------------------------------
def check_data(df: pd.DataFrame) -> dict:
    """Проверка на пустые ячейки, некорректные данные и несовпадение типов."""
    issues = {"empty": {}, "invalid": {}, "type_mismatch": {}}
    for c in df.columns:
        n_empty = int(df[c].isna().sum())
        if n_empty:
            issues["empty"][c] = n_empty

        col = df[c]
        if col.dtype == object:
            conv = pd.to_numeric(col, errors="coerce")
            bad = col.notna() & conv.isna()          # не удалось привести к числу
            if bad.any():
                issues["type_mismatch"][c] = int(bad.sum())
        elif np.issubdtype(col.dtype, np.number):
            n_inf = int(np.isinf(col.astype(float)).sum())
            if n_inf:
                issues["invalid"][c] = n_inf          # бесконечности

    for kind, text in (("empty", "пустые ячейки"),
                       ("invalid", "некорректные значения (inf)"),
                       ("type_mismatch", "несовпадение типов (нечисловые данные)")):
        for c, n in issues[kind].items():
            warnings.warn(f"Столбец '{c}': {text}, штук: {n}")
    return issues


def show_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", 50, "display.max_columns", None,
                           "display.width", 200):
        print(df)


def select_variant(df: pd.DataFrame, variant: int = VARIANT,
                   columns=VARIANT_COLUMNS, cols_per_variant: int = COLS_PER_VARIANT) -> pd.DataFrame:
    """Копия только со столбцами варианта; исходный df и файл не меняются."""
    if columns is not None:
        return df[list(columns)].copy()
    start = (variant - 1) * cols_per_variant
    return df.iloc[:, start:start + cols_per_variant].copy()


def show_dtypes(df: pd.DataFrame) -> None:
    print("Типы данных Pandas:")
    print(df.dtypes)


# ----------------------------------------------------------------------------
# Пункт 8. Меню обработки проблем
# ----------------------------------------------------------------------------
def drop_bad_rows(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def cast_types(df):
    return df.apply(pd.to_numeric, errors="coerce")  # непреобразуемое станет NaN


def fill_mean(df):
    return df.fillna(df.mean(numeric_only=True))


def fill_prev(df):
    return df.ffill()


def fill_next(df):
    return df.bfill()


def fill_zero(df):
    return df.fillna(0)


def fill_value(df, value):
    return df.fillna(value)


MENU = {
    "1": "Выбросить строки с некорректными данными",
    "2": "Выполнить приведение типов",
    "3": "Заполнить пустые ячейки средним по столбцу",
    "4": "Заполнить пустые ячейки предыдущим значением",
    "5": "Заполнить пустые ячейки последующим значением",
    "6": "Заполнить пустые ячейки нулями",
    "7": "Заполнить пустые ячейки указанным значением",
    "0": "Ничего не делать / выход",
}


def preprocessing_menu(df: pd.DataFrame) -> pd.DataFrame:
    """Интерактивное меню; повторяется, пока проблемы не устранены или выбран выход."""
    while True:
        issues = check_data(df)
        if not any(issues.values()):
            print("Проблем в данных не обнаружено.")
            return df
        print("\nОбнаружены проблемы:", {k: v for k, v in issues.items() if v})
        for k, v in MENU.items():
            print(f"  {k}. {v}")
        choice = input("Выберите действие: ").strip()
        if choice == "0":
            return df
        elif choice == "1":
            df = drop_bad_rows(df)
        elif choice == "2":
            df = cast_types(df)
        elif choice == "3":
            df = fill_mean(df)
        elif choice == "4":
            df = fill_prev(df)
        elif choice == "5":
            df = fill_next(df)
        elif choice == "6":
            df = fill_zero(df)
        elif choice == "7":
            df = fill_value(df, float(input("Значение: ").replace(",", ".")))
        else:
            print("Неверный пункт меню.")


# ----------------------------------------------------------------------------
# Пункт 9. DataFrame -> NumPy
# ----------------------------------------------------------------------------
def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)


# ----------------------------------------------------------------------------
# Пункты 10-11. Масштабирование и обратное преобразование (по столбцам)
# Параметры масштаба возвращаются и сохраняются в словаре params.
# ----------------------------------------------------------------------------
def minmax_scale(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(axis=0), x.max(axis=0)
    return (x - mn) / (mx - mn), {"type": "minmax", "min": mn, "max": mx}


def minmax_scale_ab(x: np.ndarray, a: float, b: float):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(axis=0), x.max(axis=0)
    return a + (x - mn) * (b - a) / (mx - mn), {"type": "minmax_ab", "min": mn, "max": mx, "a": a, "b": b}


def zscore_scale(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(axis=0), x.std(axis=0, ddof=1)  # выборочное СКО (ddof=1)
    return (x - mu) / sigma, {"type": "zscore", "mean": mu, "std": sigma}


def inverse_scale(xs: np.ndarray, params: dict) -> np.ndarray:
    t = params["type"]
    if t == "minmax":
        return xs * (params["max"] - params["min"]) + params["min"]
    if t == "minmax_ab":
        a, b = params["a"], params["b"]
        return (xs - a) * (params["max"] - params["min"]) / (b - a) + params["min"]
    if t == "zscore":
        return xs * params["std"] + params["mean"]
    raise ValueError(f"Неизвестный тип масштабирования: {t}")


# ----------------------------------------------------------------------------
# Пункт 12. Разбиение на train / val / test
# ----------------------------------------------------------------------------
def split_data(arr: np.ndarray, ratios, name: str = "data", shuffle: bool = False) -> dict:
    """ratios: доли, проценты или пропорции, например (1, 1, 1), (75, 10, 15), (0.75, 0.1, 0.15).
    Возвращает словарь с ключами {name}_train, {name}_val, {name}_test."""
    arr = np.asarray(arr)
    r = np.asarray(ratios, dtype=float)
    if r.size != 3 or (r < 0).any() or r.sum() == 0:
        raise ValueError("Нужно ровно три неотрицательных числа")
    if shuffle:
        arr = arr[np.random.permutation(len(arr))]  # seed зафиксирован в set_seed
    cuts = np.round(np.cumsum(r) / r.sum() * len(arr)).astype(int)[:2]
    train, val, test = np.split(arr, cuts)
    return {f"{name}_train": train, f"{name}_val": val, f"{name}_test": test}


# ----------------------------------------------------------------------------
# Демонстрация работы
# ----------------------------------------------------------------------------
def main(path: str) -> None:
    set_seed(SEED)                                   # п.13

    df_full = read_table(path)                       # п.1, 4
    check_data(df_full)                              # п.4
    show_table(df_full)                              # п.5

    df = select_variant(df_full)                     # п.6 (исходный файл не меняется)
    show_dtypes(df)                                  # п.7

    df = preprocessing_menu(df)                      # п.8
    X = to_numpy(df)                                 # п.9

    # п.10-11: масштабирование и обратное преобразование
    X_mm, p_mm = minmax_scale(X)
    X_ab, p_ab = minmax_scale_ab(X, -1, 1)
    X_z, p_z = zscore_scale(X)
    for xs, p in ((X_mm, p_mm), (X_ab, p_ab), (X_z, p_z)):
        print(p["type"], "обратное преобразование верно:", np.allclose(inverse_scale(xs, p), X))

    parts = split_data(X, (75, 10, 15), name="X")    # п.12
    print({k: v.shape for k, v in parts.items()})

    # п.2-3: запись таблицы во всех форматах и сохранение рисунка
    for fmt in ("xlsx", "csv", "txt", "mat"):
        print("Сохранено:", save_table(df, f"variant{VARIANT}", fmt))
    fig, ax = plt.subplots()
    ax.plot(X)
    ax.set_title(f"Данные варианта {VARIANT}")
    ax.legend(df.columns)
    print("Сохранено:", save_figure(fig, f"variant{VARIANT}_plot"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Использование: python lab1_part12.py путь_к_датасету")
    main(sys.argv[1])
