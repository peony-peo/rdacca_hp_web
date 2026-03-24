import io
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from rdacca_hp import rdacca_hp, permu_hp


st.set_page_config(page_title="rdacca_hp 在线分析系统", layout="wide")


# =========================
# 工具函数
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
    raise ValueError("仅支持 CSV / Excel 文件。")


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
            raise ValueError(f"行号 {r} 超出范围，当前表共有 {n} 行。")
        positions.append(r - 1)

    return df.drop(df.index[positions])


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.apply(pd.to_numeric, errors="raise")
    row_sums = df_num.sum(axis=1)
    rel = df_num.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


def preprocess_tables(
    dv: pd.DataFrame,
    iv: pd.DataFrame,
    dv_drop_rows_1based: List[int],
    iv_drop_rows_1based: List[int],
    iv_drop_cols: List[str],
    apply_hellinger: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dv2 = dv.copy()
    iv2 = iv.copy()

    if dv_drop_rows_1based:
        dv2 = drop_rows_by_r_position(dv2, dv_drop_rows_1based)

    if iv_drop_rows_1based:
        iv2 = drop_rows_by_r_position(iv2, iv_drop_rows_1based)

    if iv_drop_cols:
        missing = [c for c in iv_drop_cols if c not in iv2.columns]
        if missing:
            raise ValueError(f"解释变量表中不存在这些列：{missing}")
        iv2 = iv2.drop(columns=iv_drop_cols)

    if len(dv2) != len(iv2):
        raise ValueError("预处理后响应变量表与解释变量表的行数不一致。")

    if not dv2.index.equals(iv2.index):
        dv2 = dv2.copy()
        iv2 = iv2.copy()
        dv2.index = range(len(dv2))
        iv2.index = range(len(iv2))

    if apply_hellinger:
        dv2 = hellinger_transform(dv2)

    return dv2, iv2


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


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
            [{"parameter": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
             for k, v in params.items()]
        )
        params_df.to_excel(writer, sheet_name="parameters", index=False)

    return output.getvalue()


def auto_detect_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            detected[col] = "连续变量"
        else:
            detected[col] = "无序分类变量"
    return detected


def build_factor_settings(iv: pd.DataFrame) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    st.subheader("解释变量类型设置")
    st.caption("系统已自动列出所有解释变量。你可以直接在表格中修改变量类型；若为有序因子，请填写顺序。")

    detected = auto_detect_variable_types(iv)

    config_df = pd.DataFrame({
        "变量名": list(iv.columns),
        "系统识别": [detected[col] for col in iv.columns],
        "用户设置": [detected[col] for col in iv.columns],
        "有序水平顺序": ["" for _ in iv.columns],
    })

    for i, col in enumerate(iv.columns):
        if detected[col] == "无序分类变量":
            unique_levels = pd.Series(iv[col].astype(str)).dropna().unique().tolist()
            if len(unique_levels) >= 2:
                config_df.loc[i, "有序水平顺序"] = " > ".join(unique_levels)

    edited_df = st.data_editor(
        config_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "变量名": st.column_config.TextColumn("变量名", disabled=True),
            "系统识别": st.column_config.TextColumn("系统识别", disabled=True),
            "用户设置": st.column_config.SelectboxColumn(
                "用户设置",
                options=["连续变量", "无序分类变量", "有序因子", "不参与分析"],
                required=True,
            ),
            "有序水平顺序": st.column_config.TextColumn(
                "有序水平顺序",
                help="仅当“用户设置”为“有序因子”时需要填写，例如：None > Few > Many",
            ),
        },
        key="factor_editor",
    )

    user_type_map = dict(zip(edited_df["变量名"], edited_df["用户设置"]))

    categorical_factors = []
    ordered_factors = {}

    for _, row in edited_df.iterrows():
        col = row["变量名"]
        user_type = row["用户设置"]

        if user_type == "无序分类变量":
            categorical_factors.append(col)

        elif user_type == "有序因子":
            ordered_text = str(row["有序水平顺序"]).strip()
            if not ordered_text:
                raise ValueError(f"变量 {col} 被设为有序因子，但未填写水平顺序。")
            levels = [x.strip() for x in ordered_text.split(">") if x.strip()]
            if len(levels) < 2:
                raise ValueError(f"变量 {col} 的有序水平顺序至少需要两个水平。")
            ordered_factors[col] = levels

    return categorical_factors, ordered_factors, user_type_map


def filter_iv_by_user_types(iv: pd.DataFrame, user_type_map: Dict[str, str]) -> pd.DataFrame:
    keep_cols = [col for col in iv.columns if user_type_map.get(col) != "不参与分析"]
    if not keep_cols:
        raise ValueError("解释变量表中没有可用于分析的变量，请至少保留一个变量。")
    return iv[keep_cols].copy()


def make_bar_chart(df: pd.DataFrame, value_col: str, title: str):
    plot_df = df.copy().sort_values(value_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df.index.astype(str), plot_df[value_col].values)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Variables", fontsize=10)
    ax.set_ylabel(value_col, fontsize=10)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def friendly_error_message(e: Exception) -> str:
    msg = str(e)

    if "Row indices" in msg or "行数不一致" in msg or "索引不一致" in msg:
        return "响应变量表与解释变量表不一致。请检查两张表的行数、样点顺序或删除行设置。"

    if "could not convert" in msg or "Unable to parse string" in msg:
        return "数据中存在无法转换为数值的内容。请检查响应变量表或需要数值化的列。"

    if "解释变量表中不存在这些列" in msg:
        return msg

    if "有序因子" in msg:
        return msg

    if "仅支持 CSV / Excel 文件" in msg:
        return msg

    return f"分析失败：{msg}"


# =========================
# 页面
# =========================
st.title("rdacca_hp 在线分析系统")
st.caption("用于 RDA / CCA / dbRDA 的层次分解、变异分解与置换检验")

with st.sidebar:
    st.header("分析参数")

    method = st.selectbox(
        "方法",
        ["RDA", "CCA", "dbRDA"],
        index=0,
        help="RDA：常用于线性约束排序；CCA：常用于对应分析框架；dbRDA：基于距离的 RDA。",
    )

    r2_type = st.selectbox(
        "统计量类型",
        ["adjR2", "R2"],
        index=0,
        help="adjR2 为调整后的解释率；R2 为原始解释率。",
    )

    scale = st.checkbox(
        "scale=True",
        value=False,
        help="是否对解释变量进行标准化。通常保持默认即可。",
    )

    var_part = st.checkbox(
        "计算 variation partitioning",
        value=True,
        help="是否计算变异分解结果表。",
    )

    run_permutation = st.checkbox(
        "运行 permutation test",
        value=True,
        help="是否进行置换检验。置换次数越大，运行通常越慢。",
    )

    permutations = st.number_input(
        "置换次数 permutations",
        min_value=9,
        max_value=100000,
        value=1000,
        step=1,
        help="置换次数越大，p 值通常越稳定，但运行时间也会更长。",
    )

    st.markdown("---")
    st.subheader("预处理选项")

    use_first_col_as_index = st.checkbox(
        "将第一列作为样点名索引",
        value=True,
        help="如果数据文件第一列是样点名，建议勾选。",
    )

    apply_hellinger = st.checkbox(
        "对响应变量表做 Hellinger 转换",
        value=False,
        help="群落/物种矩阵做 RDA 时常用。",
    )

    dv_drop_rows_text = st.text_input(
        "删除响应变量表中的行（按 R 行号）",
        value="",
        help="例如：8 或 8,12",
    )

    iv_drop_rows_text = st.text_input(
        "删除解释变量表中的行（按 R 行号）",
        value="",
        help="例如：8 或 8,12",
    )

    iv_drop_cols_text = st.text_input(
        "删除解释变量表中的列（按列名）",
        value="",
        help="例如：dfs 或 dfs,temp",
    )

    st.markdown("---")
    random_state_text = st.text_input(
        "random_state（可空）",
        value="42",
        help="用于控制置换检验的随机种子，便于复现。",
    )

col1, col2 = st.columns(2)

with col1:
    dv_file = st.file_uploader(
        "上传响应变量表（CSV / Excel）",
        type=["csv", "xlsx", "xls"],
        key="dv",
    )

with col2:
    iv_file = st.file_uploader(
        "上传解释变量表（CSV / Excel）",
        type=["csv", "xlsx", "xls"],
        key="iv",
    )

st.info(
    "响应变量表（dv）通常是物种/群落矩阵；解释变量表（iv）通常是环境因子矩阵。"
    "系统会自动列出解释变量，并由你逐列确认变量类型。"
)

if dv_file is not None and iv_file is not None:
    try:
        dv_raw = read_table(dv_file, use_first_col_as_index=use_first_col_as_index)
        iv_raw = read_table(iv_file, use_first_col_as_index=use_first_col_as_index)

        st.subheader("原始数据预览")
        tab1, tab2 = st.tabs(["原始响应变量表", "原始解释变量表"])
        with tab1:
            st.dataframe(dv_raw, use_container_width=True)
        with tab2:
            st.dataframe(iv_raw, use_container_width=True)

        dv_drop_rows = parse_int_list(dv_drop_rows_text)
        iv_drop_rows = parse_int_list(iv_drop_rows_text)
        iv_drop_cols = parse_col_list(iv_drop_cols_text)

        dv, iv = preprocess_tables(
            dv=dv_raw,
            iv=iv_raw,
            dv_drop_rows_1based=dv_drop_rows,
            iv_drop_rows_1based=iv_drop_rows,
            iv_drop_cols=iv_drop_cols,
            apply_hellinger=apply_hellinger,
        )

        st.subheader("预处理后的数据预览")
        tab3, tab4 = st.tabs(["处理后响应变量表", "处理后解释变量表"])
        with tab3:
            st.dataframe(dv, use_container_width=True)
        with tab4:
            st.dataframe(iv, use_container_width=True)

        categorical_factors, ordered_factors, user_type_map = build_factor_settings(iv)
        iv_for_analysis = filter_iv_by_user_types(iv, user_type_map)

        random_state: Optional[int] = None
        if random_state_text.strip():
            random_state = int(random_state_text.strip())

        if st.button("开始分析", type="primary"):
            status_placeholder = st.empty()
            status_placeholder.info("正在运行分析，请稍候。置换次数较大时可能需要较长时间。")

            with st.spinner("正在运行 rdacca_hp 分析..."):
                result = rdacca_hp(
                    dv=dv,
                    iv=iv_for_analysis,
                    method=method,
                    type=r2_type,
                    scale=scale,
                    var_part=var_part,
                    categorical_factors=[x for x in categorical_factors if x in iv_for_analysis.columns],
                    ordered_factors={k: v for k, v in ordered_factors.items() if k in iv_for_analysis.columns},
                )

            status_placeholder.empty()
            st.success("rdacca_hp 分析完成。")

            st.subheader("总解释率")
            st.write(result.total_explained_variation)

            st.subheader("Hierarchical Partitioning")
            st.dataframe(result.hier_part, use_container_width=True)

            st.subheader("柱状图展示")
            chart_tab1, chart_tab2 = st.tabs(["Individual", "I.perc(%)"])
            with chart_tab1:
                make_bar_chart(result.hier_part, "Individual", "Individual Bar Chart")
            with chart_tab2:
                make_bar_chart(result.hier_part, "I.perc(%)", "I.perc(%) Bar Chart")

            if var_part and result.var_part is not None:
                st.subheader("Variation Partitioning")
                st.dataframe(result.var_part, use_container_width=True)

            perm_result = None
            if run_permutation:
                status_placeholder.info("正在运行 permutation test，请稍候。")
                with st.spinner("正在运行 permutation test..."):
                    perm_result = permu_hp(
                        dv=dv,
                        iv=iv_for_analysis,
                        method=method,
                        type=r2_type,
                        permutations=int(permutations),
                        scale=scale,
                        categorical_factors=[x for x in categorical_factors if x in iv_for_analysis.columns],
                        ordered_factors={k: v for k, v in ordered_factors.items() if k in iv_for_analysis.columns},
                        verbose=False,
                        random_state=random_state,
                    )

                status_placeholder.empty()

                st.subheader("Permutation Test Result")
                st.dataframe(perm_result, use_container_width=True)

            params = {
                "method": method,
                "type": r2_type,
                "scale": scale,
                "var_part": var_part,
                "run_permutation": run_permutation,
                "permutations": int(permutations),
                "random_state": random_state,
                "use_first_col_as_index": use_first_col_as_index,
                "apply_hellinger": apply_hellinger,
                "dv_drop_rows_r_style": dv_drop_rows,
                "iv_drop_rows_r_style": iv_drop_rows,
                "iv_drop_cols": iv_drop_cols,
                "user_type_map": user_type_map,
                "categorical_factors": [x for x in categorical_factors if x in iv_for_analysis.columns],
                "ordered_factors": {k: v for k, v in ordered_factors.items() if k in iv_for_analysis.columns},
            }

            st.subheader("当前参数记录")
            st.code(json.dumps(params, ensure_ascii=False, indent=2), language="json")

            excel_bytes = dataframe_to_excel_bytes(
                total_explained_variation=result.total_explained_variation,
                hier_part=result.hier_part,
                var_part=result.var_part if var_part else None,
                perm_result=perm_result,
                params=params,
            )

            st.download_button(
                label="下载 Excel 结果",
                data=excel_bytes,
                file_name="rdacca_hp_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(friendly_error_message(e))

else:
    st.warning("请先上传响应变量表和解释变量表。")