import io
import json
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rdacca_hp import __version__, permu_hp, rdacca_hp
from rdacca_hp.utils import VEGAN_DISTANCE_METHODS, coerce_distance_input


st.set_page_config(page_title="rdacca_hp Online Analysis System", layout="wide")


# =========================
# Language settings
# =========================
_, lang_col = st.columns([7, 1.4])
with lang_col:
    language_label = st.selectbox(
        "Language",
        ["English", "中文"],
        index=0,
        key="language_selector",
        label_visibility="collapsed",
    )

LANG = "zh" if language_label == "中文" else "en"


def tr(en: str, zh: str) -> str:
    return zh if LANG == "zh" else en


TYPE_LABELS = {
    "continuous": {
        "en": "Continuous variable",
        "zh": "连续变量",
    },
    "categorical": {
        "en": "Unordered categorical variable",
        "zh": "无序分类变量",
    },
    "ordered": {
        "en": "Ordered factor",
        "zh": "有序因子",
    },
    "exclude": {
        "en": "Exclude from analysis",
        "zh": "不参与分析",
    },
}


def type_label(code: str) -> str:
    return TYPE_LABELS[code][LANG]


def label_to_type_code(label: str) -> str:
    for code, names in TYPE_LABELS.items():
        if label in names.values():
            return code
    raise ValueError(f"Unknown variable type: {label}")


DV_MODE_LABELS = {
    "distance_matrix": {
        "en": "Precomputed distance matrix",
        "zh": "预先计算的距离矩阵",
    },
    "distance_vector": {
        "en": "Precomputed distance vector",
        "zh": "预先计算的距离向量",
    },
    "response_matrix": {
        "en": "Raw species/community matrix",
        "zh": "原始物种/群落矩阵",
    },
}


def dv_mode_label(code: str) -> str:
    return DV_MODE_LABELS[code][LANG]


# =========================
# Utility functions
# =========================
def read_table(uploaded_file, use_first_col_as_index: bool) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    index_col = 0 if use_first_col_as_index else None

    if name.endswith(".csv"):
        return pd.read_csv(
            uploaded_file,
            keep_default_na=False,
            index_col=index_col,
        )
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(
            uploaded_file,
            index_col=index_col,
        )
    raise ValueError(tr("Only CSV / Excel files are supported.", "仅支持 CSV / Excel 文件。"))


def _valid_condensed_length(n_values: int) -> bool:
    """Check whether length satisfies condensed distance vector: n * (n - 1) / 2."""
    if n_values <= 0:
        return False
    n = (1 + np.sqrt(1 + 8 * n_values)) / 2
    n_int = int(round(n))
    return n_int >= 2 and n_int * (n_int - 1) // 2 == n_values


def _drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp = tmp.replace(r"^\s*$", np.nan, regex=True)
    tmp = tmp.dropna(axis=0, how="all")
    tmp = tmp.dropna(axis=1, how="all")
    return tmp


def _numeric_vector_from_series(s: pd.Series) -> Optional[np.ndarray]:
    """
    Extract a numeric vector from one row or one column.
    Leading non-numeric labels such as distance/dist/value are allowed;
    non-numeric values in the middle are not allowed.
    """
    s = pd.Series(s).dropna()
    s = s.astype(str).str.strip()
    s = s[s != ""]

    if len(s) == 0:
        return None

    numeric = pd.to_numeric(s, errors="coerce")

    while len(numeric) > 0 and pd.isna(numeric.iloc[0]):
        s = s.iloc[1:]
        numeric = pd.to_numeric(s, errors="coerce")

    if len(numeric) == 0 or numeric.isna().any():
        return None

    return numeric.to_numpy(dtype=float)


def _is_index_like_vector(arr: np.ndarray) -> bool:
    if arr.ndim != 1 or len(arr) == 0:
        return False
    n = len(arr)
    return np.array_equal(arr, np.arange(n)) or np.array_equal(arr, np.arange(1, n + 1))


def read_distance_vector(uploaded_file) -> np.ndarray:
    """
    Read a condensed distance vector and support common exported formats:
    - single column without header
    - single column with any header
    - index + distance two-column table
    - one-row vector
    - two-dimensional exported table where one valid condensed vector can be identified

    The final extracted numeric length must satisfy n*(n-1)/2.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file, header=None, keep_default_na=False, dtype=object)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(uploaded_file, header=None, dtype=object)
    else:
        raise ValueError(tr("Only CSV / Excel files are supported.", "仅支持 CSV / Excel 文件。"))

    raw = _drop_empty_rows_cols(raw)
    if raw.empty:
        raise ValueError(tr("The distance vector file is empty.", "距离向量文件为空。"))

    candidates: List[Tuple[int, str, np.ndarray]] = []

    # 1) Prefer column-wise parsing: single column or index + distance two-column table.
    for col in raw.columns:
        arr = _numeric_vector_from_series(raw[col])
        if arr is None or not _valid_condensed_length(len(arr)):
            continue

        score = 10
        if _is_index_like_vector(arr):
            score -= 8
        if raw.shape[1] == 1:
            score += 5
        candidates.append((score, f"column_{col}", arr))

    # 2) Support one-row vector.
    for idx in raw.index:
        arr = _numeric_vector_from_series(raw.loc[idx, :])
        if arr is None or not _valid_condensed_length(len(arr)):
            continue

        score = 12
        if _is_index_like_vector(arr):
            score -= 8
        candidates.append((score, f"row_{idx}", arr))

    # 3) Fallback: flatten whole table, with low score to avoid preferring index+distance flattening.
    flat = _numeric_vector_from_series(pd.Series(np.asarray(raw).reshape(-1)))
    if flat is not None and _valid_condensed_length(len(flat)):
        candidates.append((3, "flattened_table", flat))

    if not candidates:
        numeric_all = pd.to_numeric(pd.Series(np.asarray(raw).reshape(-1)), errors="coerce")
        n_numeric = int(numeric_all.notna().sum())
        raise ValueError(
            tr(
                "Could not identify a valid condensed distance vector from the file. "
                f"The number of numeric cells is {n_numeric}, but the vector length must satisfy n*(n-1)/2; "
                "for example, 29 samples require 406 distance values.",
                "未能从文件中识别合法的 condensed distance vector。"
                f"当前可转换为数值的单元格数量为 {n_numeric}，"
                "但距离向量长度必须满足 n*(n-1)/2，例如 29 个样点需要 406 个距离值。",
            )
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2].astype(float)


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    tokens = (
        text.replace("，", ",")
        .replace("、", ",")
        .replace(" ", ",")
        .split(",")
    )
    out = []
    for x in tokens:
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_col_list(text: str) -> List[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.replace("，", ",").split(",") if x.strip()]


def drop_rows_by_r_position(df: pd.DataFrame, rows_1based: List[int]) -> pd.DataFrame:
    if not rows_1based:
        return df

    positions = []
    n = len(df)
    for r in rows_1based:
        if r < 1 or r > n:
            raise ValueError(
                tr(
                    f"Row number {r} is out of range; the current table has {n} rows.",
                    f"行号 {r} 超出范围，当前表共有 {n} 行。",
                )
            )
        positions.append(r - 1)

    return df.drop(df.index[positions])


def drop_rows_and_cols_by_r_position(mat: pd.DataFrame, rows_1based: List[int]) -> pd.DataFrame:
    """For a distance matrix, delete rows and columns at the same time."""
    if not rows_1based:
        return mat

    positions = []
    n = len(mat)
    for r in rows_1based:
        if r < 1 or r > n:
            raise ValueError(
                tr(
                    f"Row number {r} is out of range; the current distance matrix has {n} samples.",
                    f"行号 {r} 超出范围，当前距离矩阵共有 {n} 个样点。",
                )
            )
        positions.append(r - 1)

    mat2 = mat.drop(mat.index[positions], axis=0)
    mat2 = mat2.drop(mat2.columns[positions], axis=1)
    return mat2


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.apply(pd.to_numeric, errors="raise")
    row_sums = df_num.sum(axis=1)
    rel = df_num.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


def preprocess_normal_tables(
    dv: pd.DataFrame,
    iv: pd.DataFrame,
    dv_drop_rows_1based: List[int],
    iv_drop_rows_1based: List[int],
    iv_drop_cols: List[str],
    apply_hellinger: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normal response matrix mode: RDA / CCA / dbRDA-response matrix."""
    dv2 = dv.copy()
    iv2 = iv.copy()

    if dv_drop_rows_1based:
        dv2 = drop_rows_by_r_position(dv2, dv_drop_rows_1based)

    if iv_drop_rows_1based:
        iv2 = drop_rows_by_r_position(iv2, iv_drop_rows_1based)

    if iv_drop_cols:
        missing = [c for c in iv_drop_cols if c not in iv2.columns]
        if missing:
            raise ValueError(
                tr(
                    f"The explanatory variable table does not contain these columns: {missing}",
                    f"解释变量表中不存在这些列：{missing}",
                )
            )
        iv2 = iv2.drop(columns=iv_drop_cols)

    if len(dv2) != len(iv2):
        raise ValueError(
            tr(
                "After preprocessing, the response table and explanatory table have different numbers of rows.",
                "预处理后响应变量表与解释变量表的行数不一致。",
            )
        )

    if not dv2.index.equals(iv2.index):
        dv2 = dv2.copy()
        iv2 = iv2.copy()
        dv2.index = range(len(dv2))
        iv2.index = range(len(iv2))

    if apply_hellinger:
        dv2 = hellinger_transform(dv2)

    return dv2, iv2


def preprocess_distance_matrix(
    dv: pd.DataFrame,
    iv: pd.DataFrame,
    dv_drop_rows_1based: List[int],
    iv_drop_rows_1based: List[int],
    iv_drop_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """dbRDA distance matrix mode."""
    dv2 = dv.copy()
    iv2 = iv.copy()

    if iv_drop_rows_1based:
        iv2 = drop_rows_by_r_position(iv2, iv_drop_rows_1based)

    if iv_drop_cols:
        missing = [c for c in iv_drop_cols if c not in iv2.columns]
        if missing:
            raise ValueError(
                tr(
                    f"The explanatory variable table does not contain these columns: {missing}",
                    f"解释变量表中不存在这些列：{missing}",
                )
            )
        iv2 = iv2.drop(columns=iv_drop_cols)

    if dv_drop_rows_1based:
        dv2 = drop_rows_and_cols_by_r_position(dv2, dv_drop_rows_1based)

    dv_arr = np.asarray(dv2, dtype=float)

    if dv_arr.shape[0] != dv_arr.shape[1]:
        raise ValueError(tr("The distance matrix must be square.", "距离矩阵必须是方阵。"))

    if len(iv2) != dv_arr.shape[0]:
        raise ValueError(
            tr(
                "The number of samples in the distance matrix does not match the explanatory variable table.",
                "距离矩阵样点数与解释变量表行数不一致。",
            )
        )

    return dv_arr, iv2


def preprocess_distance_vector(
    dv_vec: np.ndarray,
    iv: pd.DataFrame,
    dv_drop_rows_1based: List[int],
    iv_drop_rows_1based: List[int],
    iv_drop_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """dbRDA condensed vector mode. Convert to square matrix first, then delete rows/columns."""
    iv2 = iv.copy()

    if iv_drop_rows_1based:
        iv2 = drop_rows_by_r_position(iv2, iv_drop_rows_1based)

    if iv_drop_cols:
        missing = [c for c in iv_drop_cols if c not in iv2.columns]
        if missing:
            raise ValueError(
                tr(
                    f"The explanatory variable table does not contain these columns: {missing}",
                    f"解释变量表中不存在这些列：{missing}",
                )
            )
        iv2 = iv2.drop(columns=iv_drop_cols)

    if dv_drop_rows_1based:
        dist_mat = coerce_distance_input(dv_vec)
        dist_df = pd.DataFrame(dist_mat)
        dist_df = drop_rows_and_cols_by_r_position(dist_df, dv_drop_rows_1based)
        dist_mat = np.asarray(dist_df, dtype=float)
        n_distance_samples = dist_mat.shape[0]
        dv_output: np.ndarray = dist_mat
    else:
        n_values = len(dv_vec)
        n_distance_samples = int(round((1 + np.sqrt(1 + 8 * n_values)) / 2))
        dv_output = np.asarray(dv_vec, dtype=float)

    if len(iv2) != n_distance_samples:
        raise ValueError(
            tr(
                "The number of samples represented by the distance vector does not match the explanatory variable table.",
                "距离向量对应的样点数与解释变量表行数不一致。",
            )
        )

    return dv_output, iv2


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


def format_permutation_result_for_web(result: pd.DataFrame) -> pd.DataFrame:
    """Separate the R-style p-value and significance marks for table output."""
    formatted = result.copy()
    p_column = "Pr(>I)"

    if p_column not in formatted.columns:
        return formatted

    raw_values = formatted[p_column]
    if pd.api.types.is_numeric_dtype(raw_values):
        p_values = pd.to_numeric(raw_values, errors="raise")
        significance = p_values.map(
            lambda p: "***"
            if p <= 0.001
            else "**"
            if p <= 0.01
            else "*"
            if p <= 0.05
            else ""
        )
    else:
        parsed = raw_values.astype(str).str.extract(
            r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\*{1,3})?\s*$"
        )
        invalid = parsed[0].isna()
        if invalid.any():
            invalid_values = raw_values.loc[invalid].astype(str).tolist()
            raise ValueError(
                f"Cannot parse permutation p-value output: {invalid_values}"
            )

        p_values = pd.to_numeric(parsed[0], errors="raise")
        significance = parsed[1].fillna("")

    formatted[p_column] = p_values.to_numpy()
    formatted["Significance"] = significance.to_numpy()
    return formatted


def dataframe_to_excel_bytes(
    total_explained_variation,
    hier_part: pd.DataFrame,
    var_part: Optional[pd.DataFrame],
    perm_result: Optional[pd.DataFrame],
    params: Dict,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(
            {"total_explained_variation": [total_explained_variation]}
        )
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        hier_part.to_excel(writer, sheet_name="hier_part")

        if var_part is not None:
            var_part.to_excel(writer, sheet_name="var_part")

        if perm_result is not None:
            perm_result.to_excel(writer, sheet_name="permutation_result")

        params_df = pd.DataFrame(
            [
                {
                    "parameter": k,
                    "value": json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v,
                }
                for k, v in params.items()
            ]
        )
        params_df.to_excel(writer, sheet_name="parameters", index=False)

    return output.getvalue()


def auto_detect_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            detected[col] = "continuous"
        else:
            detected[col] = "categorical"
    return detected


def build_factor_settings(
    iv: pd.DataFrame,
    enable_groups: bool,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    st.subheader(tr("Explanatory variable type settings", "解释变量类型设置"))
    st.caption(
        tr(
            "All explanatory variables are listed below. You can modify variable types directly in the table. For ordered factors, specify the level order.",
            "系统已自动列出所有解释变量。你可以直接在表格中修改变量类型；若为有序因子，请填写顺序。",
        )
    )

    detected = auto_detect_variable_types(iv)

    col_var = tr("Variable", "变量名")
    col_detected = tr("Detected type", "系统识别")
    col_user = tr("User setting", "用户设置")
    col_group = tr("Predictor group", "解释变量组")
    col_order = tr("Ordered levels", "有序水平顺序")

    config_data = {
        col_var: list(iv.columns),
        col_detected: [type_label(detected[col]) for col in iv.columns],
        col_user: [type_label(detected[col]) for col in iv.columns],
    }
    if enable_groups:
        # Separate names preserve the ordinary per-variable analysis until the
        # user assigns the same group name to multiple variables.
        config_data[col_group] = [str(col) for col in iv.columns]
    config_data[col_order] = ["" for _ in iv.columns]
    config_df = pd.DataFrame(config_data)

    for i, col in enumerate(iv.columns):
        if detected[col] == "categorical":
            unique_levels = pd.Series(iv[col].astype(str)).dropna().unique().tolist()
            if len(unique_levels) >= 2:
                config_df.loc[i, col_order] = " > ".join(unique_levels)

    column_config = {
        col_var: st.column_config.TextColumn(col_var, disabled=True),
        col_detected: st.column_config.TextColumn(col_detected, disabled=True),
        col_user: st.column_config.SelectboxColumn(
            col_user,
            options=[
                type_label("continuous"),
                type_label("categorical"),
                type_label("ordered"),
                type_label("exclude"),
            ],
            required=True,
        ),
        col_order: st.column_config.TextColumn(
            col_order,
            help=tr(
                "Required only when the variable is set as an ordered factor, for example: None > Few > Many.",
                "仅当“用户设置”为“有序因子”时需要填写，例如：None > Few > Many。",
            ),
        ),
    }
    if enable_groups:
        column_config[col_group] = st.column_config.TextColumn(
            col_group,
            required=True,
            help=tr(
                "Variables with the same group name are analyzed as one predictor group.",
                "填写相同组名的变量将作为同一个解释变量组参与分析。",
            ),
        )

    edited_df = st.data_editor(
        config_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config=column_config,
        key=f"factor_editor_{'grouped' if enable_groups else 'ungrouped'}",
    )

    categorical_factors = []
    ordered_factors = {}
    user_type_map = {}
    predictor_group_map = {}

    for _, row in edited_df.iterrows():
        col = row[col_var]
        user_type_code = label_to_type_code(str(row[col_user]))
        user_type_map[col] = user_type_code

        if enable_groups and user_type_code != "exclude":
            group_name = str(row[col_group]).strip()
            if not group_name:
                raise ValueError(
                    tr(
                        f"Variable {col} has no predictor group name.",
                        f"变量 {col} 未填写解释变量组名称。",
                    )
                )
            predictor_group_map[col] = group_name

        if user_type_code == "categorical":
            categorical_factors.append(col)

        elif user_type_code == "ordered":
            ordered_text = str(row[col_order]).strip()
            if not ordered_text:
                raise ValueError(
                    tr(
                        f"Variable {col} is set as an ordered factor, but no level order was provided.",
                        f"变量 {col} 被设为有序因子，但未填写水平顺序。",
                    )
                )
            levels = [x.strip() for x in ordered_text.split(">") if x.strip()]
            if len(levels) < 2:
                raise ValueError(
                    tr(
                        f"Ordered factor {col} requires at least two levels.",
                        f"变量 {col} 的有序水平顺序至少需要两个水平。",
                    )
                )
            ordered_factors[col] = levels

    return categorical_factors, ordered_factors, user_type_map, predictor_group_map


def filter_iv_by_user_types(iv: pd.DataFrame, user_type_map: Dict[str, str]) -> pd.DataFrame:
    keep_cols = [col for col in iv.columns if user_type_map.get(col) != "exclude"]
    if not keep_cols:
        raise ValueError(
            tr(
                "No explanatory variables are available for analysis. Please keep at least one variable.",
                "解释变量表中没有可用于分析的变量，请至少保留一个变量。",
            )
        )
    return iv[keep_cols].copy()


def build_grouped_predictors(
    iv: pd.DataFrame,
    predictor_group_map: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    grouped_columns: Dict[str, List[str]] = {}
    for col in iv.columns:
        group_name = predictor_group_map.get(col, str(col)).strip()
        grouped_columns.setdefault(group_name, []).append(col)

    if len(grouped_columns) < 2:
        raise ValueError(
            tr(
                "Grouped analysis requires at least two predictor groups.",
                "分组分析至少需要两个解释变量组。",
            )
        )

    return {
        group_name: iv[columns].copy()
        for group_name, columns in grouped_columns.items()
    }


def figure_to_bytes(fig, file_format: str) -> bytes:
    buffer = io.BytesIO()
    save_kwargs = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = 300
    fig.savefig(buffer, **save_kwargs)
    buffer.seek(0)
    return buffer.getvalue()


def make_bar_chart(df: pd.DataFrame, value_col: str, title: str, file_prefix: str):
    plot_df = df.copy().sort_values(value_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df.index.astype(str), plot_df[value_col].values)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(tr("Variables", "变量"), fontsize=10)
    ax.set_ylabel(value_col, fontsize=10)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    _, format_col, download_col = st.columns([6, 1.2, 1.4])
    with format_col:
        selected_format = st.selectbox(
            tr("Download format", "下载格式"),
            options=["PNG", "SVG", "PDF"],
            index=0,
            key=f"format_{file_prefix}",
            label_visibility="collapsed",
        )

    fmt = selected_format.lower()
    mime_map = {
        "png": "image/png",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
    }

    with download_col:
        st.download_button(
            tr("Download", "下载图"),
            data=figure_to_bytes(fig, fmt),
            file_name=f"{file_prefix}.{fmt}",
            mime=mime_map[fmt],
            key=f"download_{file_prefix}_{fmt}",
            use_container_width=True,
        )

    st.pyplot(fig)
    plt.close(fig)


def preview_object(x: Union[pd.DataFrame, np.ndarray], title: str):
    st.subheader(title)
    if isinstance(x, pd.DataFrame):
        st.dataframe(x, use_container_width=True)
    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            preview_df = pd.DataFrame({"distance_vector": arr})
        else:
            preview_df = pd.DataFrame(arr)
        st.dataframe(preview_df, use_container_width=True)


def friendly_error_message(e: Exception) -> str:
    msg = str(e)

    if "行数不一致" in msg or "索引不一致" in msg or "different numbers of rows" in msg:
        return tr(
            "The response data and explanatory variable table are inconsistent. Please check row counts, sample order, or row deletion settings.",
            "响应数据与解释变量表不一致。请检查两张表的行数、样点顺序或删除行设置。",
        )

    if "距离矩阵必须是方阵" in msg or "must be square" in msg:
        return tr(
            "The uploaded distance matrix is not square. Please check the data format.",
            "你上传的距离矩阵不是方阵，请检查数据格式。",
        )

    if "距离矩阵样点数与解释变量表行数不一致" in msg or "distance matrix does not match" in msg:
        return tr(
            "The number of samples in the distance matrix does not match the explanatory variable table.",
            "距离矩阵的样点数与解释变量表行数不一致，请检查输入。",
        )

    if "距离向量对应的样点数与解释变量表行数不一致" in msg or "distance vector does not match" in msg:
        return tr(
            "The number of samples represented by the distance vector does not match the explanatory variable table.",
            "距离向量对应的样点数与解释变量表行数不一致，请检查输入。",
        )

    if "square symmetric distance matrix" in msg or "valid condensed distance vector" in msg:
        return tr(
            "For dbRDA, the response input must be a valid square distance matrix or a condensed distance vector.",
            "dbRDA 模式下，响应输入必须是合法的距离矩阵（方阵）或 condensed distance vector。",
        )

    if "could not convert" in msg or "Unable to parse string" in msg:
        return tr(
            "Some values cannot be converted to numeric values. Please check the response table, distance data, or columns that should be numeric.",
            "数据中存在无法转换为数值的内容。请检查响应变量表、距离数据或需要数值化的列。",
        )

    if "requires non-negative" in msg or "should be non-negative" in msg:
        return tr(
            "The selected distance or CCA requires non-negative response values.",
            "所选距离方式或 CCA 要求响应数据为非负值。",
        )

    if "produced NaN or Inf" in msg:
        return tr(
            "Distance calculation produced NaN or Inf. Check for invalid or all-zero samples.",
            "距离计算产生 NaN 或 Inf，请检查无效值或全零样点。",
        )

    if "解释变量表中不存在这些列" in msg or "explanatory variable table does not contain" in msg:
        return msg

    if "有序因子" in msg or "ordered factor" in msg:
        return msg

    if "仅支持 CSV / Excel 文件" in msg or "Only CSV / Excel" in msg:
        return msg

    return tr(f"Analysis failed: {msg}", f"分析失败：{msg}")


# =========================
# Page
# =========================
st.title(tr("rdacca_hp Online Analysis System", "rdacca_hp 在线分析系统"))
st.caption(
    tr(
        f"rdacca_hp {__version__} · Hierarchical partitioning, variation partitioning, and permutation tests for RDA / CCA / dbRDA.",
        f"rdacca_hp {__version__} · 用于 RDA / CCA / dbRDA 的层次分解、变异分解与置换检验",
    )
)

with st.sidebar:
    st.header(tr("Analysis parameters", "分析参数"))

    method = st.selectbox(
        tr("Method", "方法"),
        ["RDA", "CCA", "dbRDA"],
        index=0,
        help=tr(
            "RDA is commonly used for linear constrained ordination; CCA is based on correspondence analysis; dbRDA is distance-based RDA.",
            "RDA：常用于线性约束排序；CCA：常用于对应分析框架；dbRDA：基于距离的 RDA。",
        ),
    )

    r2_type = st.selectbox(
        tr("Statistic type", "统计量类型"),
        ["adjR2", "R2"],
        index=0,
        help=tr(
            "adjR2 is adjusted explained variation; R2 is raw explained variation.",
            "adjR2 为调整后的解释率；R2 为原始解释率。",
        ),
    )

    scale = st.checkbox(
        "scale=True",
        value=False,
        help=tr(
            "Pass the scale parameter to rdacca_hp. Usually keep the default unless you need to match a specific workflow.",
            "传递给 rdacca_hp 的 scale 参数。通常保持默认即可；如需与特定分析流程保持一致时再启用。",
        ),
    )

    var_part = st.checkbox(
        tr("Calculate variation partitioning", "计算 variation partitioning"),
        value=True,
        help=tr(
            "Whether to calculate the variation partitioning table.",
            "是否计算变异分解结果表。",
        ),
    )

    use_predictor_groups = st.checkbox(
        tr("Analyze predictor groups", "按解释变量组分析"),
        value=False,
        help=tr(
            "Enable this to assign variables to named groups in the variable settings table.",
            "启用后，可在变量设置表中填写组名；相同组名的变量作为一个解释变量组。",
        ),
    )
    if use_predictor_groups:
        st.warning(
            tr(
                "Grouped analysis passes explanatory variables as predictor groups. "
                "Permutation tests then follow the group/list input logic, so Pr(>I) "
                "may differ from ordinary data-frame input even when each variable is "
                "kept as a separate group.",
                "按解释变量组分析会将解释变量作为 group/list 输入。此时 permutation test "
                "遵循分组输入的置换逻辑；即使每个变量单独成组，Pr(>I) 也可能与普通解释变量表输入不同。",
            )
        )

    run_permutation = st.checkbox(
        tr("Run permutation test", "运行 permutation test"),
        value=False,
        help=tr(
            "Whether to run the permutation test. More permutations usually require more time.",
            "是否进行置换检验。置换次数越大，运行通常越慢。",
        ),
    )

    permutations = 1000
    if run_permutation:
        permutations = st.number_input(
            tr("Permutations", "置换次数 permutations"),
            min_value=9,
            max_value=100000,
            value=1000,
            step=1,
            help=tr(
                "Default is 1000, following rdacca.hp semantics: 999 randomized runs plus 1 observed value in the empirical distribution.",
                "默认 1000，与 rdacca.hp 习惯一致：999 次随机置换 + 1 次观测值参与经验分布。",
            ),
        )

    cca_n_perm = 1000
    if method == "CCA" and r2_type == "adjR2":
        cca_n_perm = st.number_input(
            tr("CCA adjusted R2 permutations", "CCA 调整 R2 的内部置换次数"),
            min_value=9,
            max_value=100000,
            value=1000,
            step=1,
            help=tr(
                "This is n_perm for RsquareAdj.cca and is separate from the outer permu_hp test.",
                "这是计算 CCA 调整 R2 的 n_perm，与外层 permu_hp 显著性置换次数不同。",
            ),
        )

    st.markdown("---")
    st.subheader(tr("Response data input", "响应数据输入"))

    distance_method: Optional[str] = None
    dbrdatype = "dbrda"
    sqrt_dist = False
    add: Union[bool, str] = False

    if method == "dbRDA":
        dv_input_mode = st.selectbox(
            tr("Response data format", "响应数据输入形式"),
            ["response_matrix", "distance_matrix", "distance_vector"],
            format_func=dv_mode_label,
            index=0,
            help=tr(
                "dbRDA can use precomputed distances or a raw response matrix whose distances are calculated by rdacca_hp.",
                "dbRDA 可使用预先计算的距离，也可上传原始响应矩阵并由 rdacca_hp 计算距离。",
            ),
        )

        if dv_input_mode == "response_matrix":
            distance_method = st.selectbox(
                tr("Distance method", "距离方式"),
                list(VEGAN_DISTANCE_METHODS),
                index=list(VEGAN_DISTANCE_METHODS).index("bray"),
                help=tr(
                    "Distance is calculated inside rdacca_hp using vegan-compatible definitions.",
                    "距离由 rdacca_hp 在包内按 vegan 兼容定义计算。",
                ),
            )

        dbrdatype = st.selectbox(
            "dbrdatype",
            ["dbrda", "capscale"],
            index=0,
            help=tr(
                "dbrda is the rdacca.hp default; capscale uses the PCoA-based route.",
                "dbrda 是 rdacca.hp 的默认方式；capscale 使用基于 PCoA 的计算路径。",
            ),
        )
        sqrt_dist = st.checkbox(
            "sqrt_dist=True",
            value=False,
            help=tr(
                "Take the square root of dissimilarities before dbRDA.",
                "在 dbRDA 前对距离值开平方。",
            ),
        )
        add_option = st.selectbox(
            tr("Distance correction", "距离校正"),
            ["none", "lingoes", "cailliez"],
            index=0,
            help=tr(
                "Optional euclidification correction matching vegan.",
                "可选的距离欧氏化校正，与 vegan 参数一致。",
            ),
        )
        add = False if add_option == "none" else add_option
    else:
        dv_input_mode = "response_matrix"

    st.markdown("---")
    st.subheader(tr("Preprocessing options", "预处理选项"))

    if method == "dbRDA" and dv_input_mode == "distance_vector":
        dv_use_first_col_as_index = False
        st.caption(
            tr(
                "Distance vector mode: the response data will automatically identify headers and index-like columns; no response index setting is needed.",
                "当前为距离向量输入模式：响应数据会自动识别表头和索引列，不需要设置响应数据索引。",
            )
        )
    else:
        dv_use_first_col_as_index = st.checkbox(
            tr("Response data: use first column as sample index", "响应变量表：将第一列作为样点名索引"),
            value=True,
            help=tr(
                "Enable this if the first column in the response table or distance matrix contains sample names. Distance vector mode does not need this setting.",
                "如果响应变量表或距离矩阵的第一列是样点名，建议勾选。距离向量模式不需要设置。",
            ),
        )

    iv_use_first_col_as_index = st.checkbox(
        tr("Explanatory table: use first column as sample index", "解释变量表：将第一列作为样点名索引"),
        value=True,
        help=tr(
            "Enable this if the first column in the explanatory variable table contains sample names.",
            "如果解释变量表第一列是样点名，建议勾选。",
        ),
    )

    if method == "dbRDA" and dv_input_mode in ["distance_matrix", "distance_vector"]:
        apply_hellinger = False
        st.caption(
            tr(
                "Current mode uses distance input for dbRDA; Hellinger transformation is not applicable.",
                "当前为 dbRDA 距离输入模式，Hellinger 转换不适用。",
            )
        )
    elif method == "dbRDA" and dv_input_mode == "response_matrix":
        apply_hellinger = False
        st.caption(
            tr(
                "Current mode uses a raw response matrix; rdacca_hp will calculate the selected distance. Hellinger preprocessing is not applied here.",
                "当前为 dbRDA 原始响应矩阵模式；rdacca_hp 将计算所选距离，网页不额外进行 Hellinger 转换。",
            )
        )
    else:
        apply_hellinger = st.checkbox(
            tr("Apply Hellinger transformation to response data", "对响应变量表做 Hellinger 转换"),
            value=False,
            help=tr(
                "Use only when Hellinger transformation is needed for a community/species abundance matrix. Do not use it for already-transformed data, ordinary continuous responses, or typical CCA workflows.",
                "仅在需要对群落/物种丰度矩阵进行 Hellinger 转换时勾选；已转换数据、普通连续响应变量或 CCA 通常不需要。",
            ),
        )

    dv_drop_rows_text = st.text_input(
        tr("Delete rows from response data (R-style row numbers)", "删除响应数据中的行（按 R 行号）"),
        value="",
        help=tr("For example: 8 or 8,12", "例如：8 或 8,12"),
    )

    iv_drop_rows_text = st.text_input(
        tr("Delete rows from explanatory table (R-style row numbers)", "删除解释变量表中的行（按 R 行号）"),
        value="",
        help=tr("For example: 8 or 8,12", "例如：8 或 8,12"),
    )

    iv_drop_cols_text = st.text_input(
        tr("Delete columns from explanatory table (column names)", "删除解释变量表中的列（按列名）"),
        value="",
        help=tr("For example: dfs or dfs,temp", "例如：dfs 或 dfs,temp"),
    )

    random_state_text = ""
    if run_permutation or (method == "CCA" and r2_type == "adjR2"):
        st.markdown("---")
        random_state_text = st.text_input(
            "random_state",
            value="",
            help=tr(
                "Leave blank by default to avoid fixing the random seed. Enter an integer such as 42 only when reproducible results are needed.",
                "默认留空，表示不固定随机种子；需要复现实验结果时再填写整数，例如 42。",
            ),
        )

col1, col2 = st.columns(2)

with col1:
    if method == "dbRDA" and dv_input_mode == "distance_vector":
        dv_label = tr("Upload distance vector (CSV / Excel)", "上传距离向量（CSV / Excel）")
    elif method == "dbRDA" and dv_input_mode == "distance_matrix":
        dv_label = tr("Upload distance matrix (CSV / Excel)", "上传距离矩阵（CSV / Excel）")
    elif method == "dbRDA":
        dv_label = tr(
            "Upload raw species/community matrix (CSV / Excel; not a distance matrix)",
            "上传原始物种/群落矩阵（CSV / Excel；不是距离矩阵）",
        )
    else:
        dv_label = tr("Upload response table (CSV / Excel)", "上传响应变量表（CSV / Excel）")

    dv_file = st.file_uploader(
        dv_label,
        type=["csv", "xlsx", "xls"],
        key="dv",
    )

with col2:
    iv_file = st.file_uploader(
        tr("Upload explanatory variable table (CSV / Excel)", "上传解释变量表（CSV / Excel）"),
        type=["csv", "xlsx", "xls"],
        key="iv",
    )

if method == "dbRDA":
    st.info(
        tr(
            "For the recommended input mode, upload the original species/community table directly. Do not calculate or enter a distance matrix in Excel. rdacca_hp calculates the selected distance inside the package.",
            "推荐直接上传原始物种/群落数据表，无需在 Excel 中计算或填写距离矩阵；rdacca_hp 将在包内按所选方法自动计算距离。",
        )
    )
else:
    st.info(
        tr(
            "The response table is usually a species/community matrix; the explanatory table is usually an environmental factor matrix.",
            "响应变量表（dv）通常是物种/群落矩阵；解释变量表（iv）通常是环境因子矩阵。",
        )
    )

if dv_file is not None and iv_file is not None:
    try:
        iv_raw = read_table(iv_file, use_first_col_as_index=iv_use_first_col_as_index)

        if method == "dbRDA" and dv_input_mode == "distance_vector":
            dv_raw = read_distance_vector(dv_file)
        else:
            dv_raw = read_table(dv_file, use_first_col_as_index=dv_use_first_col_as_index)

        with st.expander(tr("Raw data preview", "原始数据预览"), expanded=False):
            tab1, tab2 = st.tabs(
                [
                    tr("Raw response / distance data", "原始响应/距离数据"),
                    tr("Raw explanatory table", "原始解释变量表"),
                ]
            )
            with tab1:
                preview_object(dv_raw, tr("Raw response / distance data", "原始响应/距离数据"))
            with tab2:
                st.dataframe(iv_raw, use_container_width=True)

        dv_drop_rows = parse_int_list(dv_drop_rows_text)
        iv_drop_rows = parse_int_list(iv_drop_rows_text)
        iv_drop_cols = parse_col_list(iv_drop_cols_text)

        if method == "dbRDA" and dv_input_mode == "distance_matrix":
            dv, iv = preprocess_distance_matrix(
                dv=dv_raw,
                iv=iv_raw,
                dv_drop_rows_1based=dv_drop_rows,
                iv_drop_rows_1based=iv_drop_rows,
                iv_drop_cols=iv_drop_cols,
            )
        elif method == "dbRDA" and dv_input_mode == "distance_vector":
            dv, iv = preprocess_distance_vector(
                dv_vec=dv_raw,
                iv=iv_raw,
                dv_drop_rows_1based=dv_drop_rows,
                iv_drop_rows_1based=iv_drop_rows,
                iv_drop_cols=iv_drop_cols,
            )
        else:
            dv, iv = preprocess_normal_tables(
                dv=dv_raw,
                iv=iv_raw,
                dv_drop_rows_1based=dv_drop_rows,
                iv_drop_rows_1based=iv_drop_rows,
                iv_drop_cols=iv_drop_cols,
                apply_hellinger=apply_hellinger,
            )

        with st.expander(tr("Preprocessed data preview", "预处理后的数据预览"), expanded=False):
            tab3, tab4 = st.tabs(
                [
                    tr("Processed response / distance data", "处理后响应/距离数据"),
                    tr("Processed explanatory table", "处理后解释变量表"),
                ]
            )
            with tab3:
                preview_object(dv, tr("Processed response / distance data", "处理后响应/距离数据"))
            with tab4:
                st.dataframe(iv, use_container_width=True)

        (
            categorical_factors,
            ordered_factors,
            user_type_map,
            predictor_group_map,
        ) = build_factor_settings(iv, enable_groups=use_predictor_groups)
        iv_filtered = filter_iv_by_user_types(iv, user_type_map)
        iv_for_analysis: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        if use_predictor_groups:
            iv_for_analysis = build_grouped_predictors(
                iv_filtered,
                predictor_group_map,
            )
            group_sizes: Dict[str, int] = {}
            for col in iv_filtered.columns:
                group_name = predictor_group_map.get(col, str(col)).strip()
                group_sizes[group_name] = group_sizes.get(group_name, 0) + 1
            if group_sizes and all(size == 1 for size in group_sizes.values()):
                st.warning(
                    tr(
                        "Each predictor is currently assigned to its own group. "
                        "This is still treated as grouped/list input and may give "
                        "different permutation p-values from ungrouped data-frame input. "
                        "To match permu.hp(dv, env), turn off grouped analysis.",
                        "当前每个解释变量都被分配到单独的组。系统仍会按分组/list 输入处理，"
                        "置换检验的 p 值可能不同于未分组的普通解释变量表输入。"
                        "如果需要对应 permu.hp(dv, env)，请关闭“按解释变量组分析”。",
                    )
                )
        else:
            iv_for_analysis = iv_filtered

        active_categorical_factors = [
            name for name in categorical_factors if name in iv_filtered.columns
        ]
        active_ordered_factors = {
            name: levels
            for name, levels in ordered_factors.items()
            if name in iv_filtered.columns
        }

        random_state: Optional[int] = None
        if random_state_text.strip():
            random_state = int(random_state_text.strip())

        if st.button(tr("Run analysis", "开始分析"), type="primary"):
            status_placeholder = st.empty()
            status_placeholder.info(
                tr(
                    "Running analysis. Please wait; permutation tests with many permutations may take a long time.",
                    "正在运行分析，请稍候。置换次数较大时可能需要较长时间。",
                )
            )

            with st.spinner(tr("Running rdacca_hp analysis...", "正在运行 rdacca_hp 分析...")):
                result = rdacca_hp(
                    dv=dv,
                    iv=iv_for_analysis,
                    method=method,
                    type=r2_type,
                    scale=scale,
                    var_part=var_part,
                    n_perm=int(cca_n_perm),
                    add=add,
                    sqrt_dist=sqrt_dist,
                    dbrdatype=dbrdatype,
                    distance=distance_method,
                    random_state=random_state,
                    categorical_factors=active_categorical_factors,
                    ordered_factors=active_ordered_factors,
                )

            status_placeholder.empty()
            st.success(tr("rdacca_hp analysis completed.", "rdacca_hp 分析完成。"))

            st.subheader(tr("Total explained variation", "总解释率"))
            total_value = float(result.total_explained_variation)
            st.markdown(
                f"""
                <div style="
                    font-size: 2.2rem;
                    font-weight: 650;
                    line-height: 1.2;
                    margin: 0.2rem 0 1.0rem 0;
                ">
                    {total_value:.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Hierarchical Partitioning")
            st.dataframe(result.hier_part, use_container_width=True)

            st.subheader(tr("Bar charts", "柱状图展示"))
            chart_tab1, chart_tab2 = st.tabs(["Individual", "I.perc(%)"])
            with chart_tab1:
                make_bar_chart(result.hier_part, "Individual", "Individual Bar Chart", "individual_bar_chart")
            with chart_tab2:
                make_bar_chart(result.hier_part, "I.perc(%)", "I.perc(%) Bar Chart", "i_perc_bar_chart")

            if var_part and result.var_part is not None:
                st.subheader("Variation Partitioning")
                st.dataframe(result.var_part, use_container_width=True)

            perm_result = None
            if run_permutation:
                status_placeholder.info(tr("Running permutation test. Please wait.", "正在运行 permutation test，请稍候。"))
                with st.spinner(tr("Running permutation test...", "正在运行 permutation test...")):
                    raw_perm_result = permu_hp(
                        dv=dv,
                        iv=iv_for_analysis,
                        method=method,
                        type=r2_type,
                        permutations=int(permutations),
                        scale=scale,
                        n_perm=int(cca_n_perm),
                        add=add,
                        sqrt_dist=sqrt_dist,
                        dbrdatype=dbrdatype,
                        distance=distance_method,
                        categorical_factors=active_categorical_factors,
                        ordered_factors=active_ordered_factors,
                        verbose=False,
                        random_state=random_state,
                    )
                    perm_result = format_permutation_result_for_web(raw_perm_result)

                status_placeholder.empty()

                st.subheader("Permutation Test Result")
                st.dataframe(perm_result, use_container_width=True)

            params = {
                "rdacca_hp_version": __version__,
                "language": LANG,
                "method": method,
                "dv_input_mode": dv_input_mode,
                "type": r2_type,
                "distance": distance_method,
                "dbrdatype": dbrdatype if method == "dbRDA" else None,
                "sqrt_dist": sqrt_dist if method == "dbRDA" else None,
                "add": add if method == "dbRDA" else None,
                "scale": scale,
                "var_part": var_part,
                "cca_n_perm": int(cca_n_perm)
                if method == "CCA" and r2_type == "adjR2"
                else None,
                "run_permutation": run_permutation,
                "permutations": int(permutations),
                "random_state": random_state,
                "dv_use_first_col_as_index": dv_use_first_col_as_index,
                "iv_use_first_col_as_index": iv_use_first_col_as_index,
                "apply_hellinger": apply_hellinger,
                "dv_drop_rows_r_style": dv_drop_rows,
                "iv_drop_rows_r_style": iv_drop_rows,
                "iv_drop_cols": iv_drop_cols,
                "use_predictor_groups": use_predictor_groups,
                "predictor_group_map": predictor_group_map,
                "user_type_map": user_type_map,
                "categorical_factors": active_categorical_factors,
                "ordered_factors": active_ordered_factors,
            }

            with st.expander(tr("Current parameter record", "当前参数记录"), expanded=False):
                st.code(json.dumps(params, ensure_ascii=False, indent=2), language="json")

            excel_bytes = dataframe_to_excel_bytes(
                total_explained_variation=result.total_explained_variation,
                hier_part=result.hier_part,
                var_part=result.var_part if var_part else None,
                perm_result=perm_result,
                params=params,
            )

            st.download_button(
                label=tr("Download Excel results", "下载 Excel 结果"),
                data=excel_bytes,
                file_name="rdacca_hp_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(friendly_error_message(e))

else:
    st.warning(tr("Please upload response / distance data and the explanatory variable table first.", "请先上传响应/距离数据和解释变量表。"))
