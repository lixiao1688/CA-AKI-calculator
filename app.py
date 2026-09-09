# -*- coding: utf-8 -*-
"""
CA-AKI 早期风险预测计算器（随机森林模型）
Streamlit 应用 —— 多中心外部验证的造影剂相关急性肾损伤(CA-AKI)预测工具。
仅供研究参考，不构成临床决策依据。
"""
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="CA-AKI 风险预测计算器", page_icon="🩺", layout="wide")

# ---------- 加载模型包 ----------
@st.cache_resource
def load_bundle():
    return joblib.load("rf_deploy_bundle.pkl")


bundle = load_bundle()
model = bundle["model"]
feature_order = bundle["feature_order"]
bounds = bundle["winsorize_bounds"]
medians = bundle["medians"]
threshold = float(bundle["threshold"])

# ---------- 标签与映射 ----------
CONTRAST = {
    1: "碘佛醇注射液 (Ioversol)",
    2: "碘海醇注射液 (Iohexol)",
    3: "碘克沙醇注射液 (Iodixanol, 等渗)",
    4: "碘帕醇注射液 (Iopamidol)",
    5: "碘普罗胺注射液 (Iopromide)",
    6: "碘普罗胺300 (Iopromide 300)",
    7: "碘普罗胺370 (Iopromide 370)",
}
MEDS = {
    "diuretic": "利尿剂",
    "beta_lactam": "β-内酰胺类抗生素",
    "beta_blocker": "β-受体阻滞剂",
    "ccb": "钙通道拮抗剂 (CCB)",
    "acei": "ACEI（血管紧张素转换酶抑制剂）",
    "arb": "ARB（血管紧张素Ⅱ受体阻滞剂）",
    "statin": "他汀类降脂药",
    "antiplatelet": "抗血小板药",
    "nsaid": "非甾体抗炎药 (NSAID)",
    "vitamin_c": "维生素 C",
}


def predict(feat: dict) -> float:
    x = np.zeros((1, len(feature_order)), dtype=float)
    for i, f in enumerate(feature_order):
        x[0, i] = feat[f]
    # winsorize（与训练一致）
    for f, (lo, hi) in bounds.items():
        if f in feature_order:
            i = feature_order.index(f)
            x[0, i] = np.clip(x[0, i], lo, hi)
    # 缺失插补（保险）
    for f, med in medians.items():
        if f in feature_order:
            i = feature_order.index(f)
            if np.isnan(x[0, i]):
                x[0, i] = med
    return float(model.predict_proba(x)[0, 1])


# ---------- 界面 ----------
st.title("造影剂相关急性肾损伤（CA-AKI）早期风险预测计算器")
st.caption("基于 24,843 例多中心队列、经外部验证的随机森林模型。内部 CV AUC 0.930；外部验证 AUC 0.807。")
st.markdown("---")

col_l, col_r = st.columns([1.15, 0.85])

with col_l:
    st.subheader("① 基本信息与造影剂")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("年龄（岁）", 18, 100, int(medians["age"]))
    sex = c2.radio("性别", ["男", "女"], horizontal=True)
    surgery = c3.radio("是否手术", ["否", "是"], horizontal=True)

    c4, c5 = st.columns(2)
    contrast_code = c4.selectbox(
        "造影剂种类", list(CONTRAST.keys()),
        format_func=lambda k: CONTRAST[k], index=2)
    contrast_dose = c5.number_input("造影剂剂量（mL）", 0, 5000,
                                    int(medians["contrast_dose"]))

    st.subheader("② 检验指标")
    labs = {
        "cr": ("血肌酐 (μmol/L)", 0.0, 2000.0, medians["cr"]),
        "egfr": ("估算肾小球滤过率 eGFR (mL/min/1.73 m²)", 0.0, 500.0, medians["egfr"]),
        "bun": ("尿素氮 BUN (mmol/L)", 0.0, 60.0, medians["bun"]),
        "ua": ("尿酸 (μmol/L)", 0.0, 1000.0, medians["ua"]),
        "ast": ("天冬氨酸转氨酶 AST (U/L)", 0.0, 500.0, medians["ast"]),
        "alt": ("丙氨酸转氨酶 ALT (U/L)", 0.0, 500.0, medians["alt"]),
        "tbil": ("总胆红素 (μmol/L)", 0.0, 200.0, medians["tbil"]),
        "alp": ("碱性磷酸酶 ALP (U/L)", 0.0, 500.0, medians["alp"]),
        "ibil": ("间接胆红素 (μmol/L)", 0.0, 200.0, medians["ibil"]),
        "dbil": ("直接胆红素 (μmol/L)", 0.0, 100.0, medians["dbil"]),
        "ggt": ("γ-谷氨酰转肽酶 GGT (U/L)", 0.0, 500.0, medians["ggt"]),
    }
    lab_vals = {}
    for i, (k, (label, lo, hi, dflt)) in enumerate(labs.items()):
        if i % 3 == 0:
            lc = st.columns(3)
        lab_vals[k] = lc[i % 3].number_input(label, lo, hi, float(dflt))

    st.subheader("③ 造影前合并用药（勾选 = 使用）")
    med_vals = {}
    for i, (k, label) in enumerate(MEDS.items()):
        if i % 3 == 0:
            mc = st.columns(3)
        med_vals[k] = int(mc[i % 3].checkbox(label))

with col_r:
    st.subheader("④ 预测结果")
    st.markdown("")
    # 组装特征向量（顺序与训练一致）
    feat = {
        "age": age,
        "contrast_dose": contrast_dose,
        "cr": lab_vals["cr"], "bun": lab_vals["bun"], "ua": lab_vals["ua"],
        "ast": lab_vals["ast"], "alt": lab_vals["alt"], "tbil": lab_vals["tbil"],
        "alp": lab_vals["alp"], "ibil": lab_vals["ibil"], "dbil": lab_vals["dbil"],
        "ggt": lab_vals["ggt"], "egfr": lab_vals["egfr"],
        "sex": 1 if sex == "男" else 0,
        "surgery": 1 if surgery == "是" else 0,
        **med_vals,
        "contrast_type": contrast_code - 1,  # 1-7 -> 0-6
    }
    if st.button("计算风险", type="primary", use_container_width=True):
        p = predict(feat)
        pct = p * 100
        high = p >= threshold
        st.metric("CA-AKI 预测风险概率", f"{pct:.1f}%")
        st.progress(min(p, 1.0))
        if high:
            st.error(f"⚠️ 高风险（≥ {threshold*100:.0f}%）：建议加强肌酐监测、"
                     f"评估并调整肾毒性合并用药、充分水化。")
        else:
            st.success(f"✅ 低风险（< {threshold*100:.0f}%）：常规监测即可。")
        st.caption(f"筛查阈值 {threshold*100:.0f}%（外验灵敏度 0.764 / 特异度 0.666）。")

    st.markdown("---")
    st.caption("⚠️ 本工具仅供科研参考，不构成临床诊断或治疗决策依据；"
               "模型基于中国多中心回顾性数据，外推到其他人群需谨慎。")

st.markdown("---")
st.markdown(
    "**模型说明**：随机森林（300 棵树，类别加权），26 个造影前特征"
    "（11 项检验指标 + 年龄 + 造影剂种类与剂量 + 性别 + 手术 + 10 类用药）。"
    "CA-AKI 定义为造影后 48 小时内血肌酐升高 ≥0.3 mg/dL（26.5 μmol/L）或 7 天内升至基线 ≥1.5 倍（KDIGO）。"
)
