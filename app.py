# -*- coding: utf-8 -*-
"""
CA-AKI Risk Prediction Calculator (random forest)
Streamlit app — interpretable, drug-aware model for early prediction of
contrast-associated acute kidney injury (CA-AKI). Research use only.
"""
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="CA-AKI Risk Prediction Calculator", page_icon="🩺", layout="wide")

# ---------- load model bundle ----------
@st.cache_resource
def load_bundle():
    return joblib.load("rf_deploy_bundle.pkl")


bundle = load_bundle()
model = bundle["model"]
feature_order = bundle["feature_order"]
bounds = bundle["winsorize_bounds"]
medians = bundle["medians"]
threshold = float(bundle["threshold"])

CONTRAST = {
    1: ("Ioversol", "碘佛醇注射液"),
    2: ("Iohexol", "碘海醇注射液"),
    3: ("Iodixanol (iso-osmolar)", "碘克沙醇注射液（等渗）"),
    4: ("Iopamidol", "碘帕醇注射液"),
    5: ("Iopromide", "碘普罗胺注射液"),
    6: ("Iopromide 300", "碘普罗胺300"),
    7: ("Iopromide 370", "碘普罗胺370"),
}

LABS = {
    "cr": ("Serum creatinine (μmol/L)", "血肌酐 (μmol/L)"),
    "egfr": ("eGFR (mL/min/1.73 m²)", "估算肾小球滤过率 eGFR (mL/min/1.73 m²)"),
    "bun": ("Blood urea nitrogen (mmol/L)", "尿素氮 BUN (mmol/L)"),
    "ua": ("Uric acid (μmol/L)", "尿酸 (μmol/L)"),
    "ast": ("AST (U/L)", "天冬氨酸转氨酶 AST (U/L)"),
    "alt": ("ALT (U/L)", "丙氨酸转氨酶 ALT (U/L)"),
    "tbil": ("Total bilirubin (μmol/L)", "总胆红素 (μmol/L)"),
    "alp": ("ALP (U/L)", "碱性磷酸酶 ALP (U/L)"),
    "ibil": ("Indirect bilirubin (μmol/L)", "间接胆红素 (μmol/L)"),
    "dbil": ("Direct bilirubin (μmol/L)", "直接胆红素 (μmol/L)"),
    "ggt": ("GGT (U/L)", "γ-谷氨酰转肽酶 GGT (U/L)"),
}
LAB_RANGES = {
    "cr": (0.0, 2000.0), "egfr": (0.0, 500.0), "bun": (0.0, 60.0),
    "ua": (0.0, 1000.0), "ast": (0.0, 500.0), "alt": (0.0, 500.0),
    "tbil": (0.0, 200.0), "alp": (0.0, 500.0), "ibil": (0.0, 200.0),
    "dbil": (0.0, 100.0), "ggt": (0.0, 500.0),
}

MEDS = {
    "diuretic": ("Diuretic", "利尿剂"),
    "beta_lactam": ("β-lactam antibiotics", "β-内酰胺类抗生素"),
    "beta_blocker": ("β-blockers", "β-受体阻滞剂"),
    "ccb": ("Calcium-channel blocker (CCB)", "钙通道拮抗剂 (CCB)"),
    "acei": ("ACE inhibitor", "ACEI（血管紧张素转换酶抑制剂）"),
    "arb": ("Angiotensin-receptor blocker (ARB)", "ARB（血管紧张素Ⅱ受体阻滞剂）"),
    "statin": ("Statin", "他汀类降脂药"),
    "antiplatelet": ("Antiplatelet agent", "抗血小板药"),
    "nsaid": ("NSAID", "非甾体抗炎药 (NSAID)"),
    "vitamin_c": ("Vitamin C", "维生素 C"),
}


def predict(feat: dict) -> float:
    x = np.zeros((1, len(feature_order)), dtype=float)
    for i, f in enumerate(feature_order):
        x[0, i] = feat[f]
    for f, (lo, hi) in bounds.items():
        if f in feature_order:
            i = feature_order.index(f)
            x[0, i] = np.clip(x[0, i], lo, hi)
    for f, med in medians.items():
        if f in feature_order:
            i = feature_order.index(f)
            if np.isnan(x[0, i]):
                x[0, i] = med
    return float(model.predict_proba(x)[0, 1])


# ---------- language (English default) ----------
lang = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0)
en = (lang == "English")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Research use only — not for clinical decision-making."
    if en else "仅供研究参考，不构成临床决策依据。"
)

st.title("CA-AKI Risk Prediction Calculator" if en else "造影剂相关急性肾损伤（CA-AKI）风险预测计算器")
st.caption(
    "Interpretable, drug-aware random-forest model (internal CV AUC 0.930; external validation AUC 0.807)."
    if en else
    "基于多中心队列、经外部验证的可解释随机森林模型（内部 CV AUC 0.930；外部验证 AUC 0.807）。"
)
st.markdown("---")

col_l, col_r = st.columns([1.15, 0.85])

with col_l:
    st.subheader("1. Demographics & contrast agent" if en else "① 基本信息与造影剂")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)" if en else "年龄（岁）", 18, 100, int(medians["age"]))
    sex = c2.radio("Sex" if en else "性别",
                   ["Male", "Female"] if en else ["男", "女"], horizontal=True)
    surgery = c3.radio("Surgery" if en else "是否手术",
                       ["No", "Yes"] if en else ["否", "是"], horizontal=True)

    c4, c5 = st.columns(2)
    contrast_code = c4.selectbox(
        "Contrast agent type" if en else "造影剂种类",
        list(CONTRAST.keys()),
        format_func=lambda k: CONTRAST[k][0] if en else CONTRAST[k][1],
        index=2)
    contrast_dose = c5.number_input("Contrast dose (mL)" if en else "造影剂剂量（mL）",
                                    0, 5000, int(medians["contrast_dose"]))

    st.subheader("2. Laboratory values" if en else "② 检验指标")
    lab_vals = {}
    for i, (k, (enl, zhl)) in enumerate(LABS.items()):
        if i % 3 == 0:
            lc = st.columns(3)
        lo, hi = LAB_RANGES[k]
        lab_vals[k] = lc[i % 3].number_input(enl if en else zhl, lo, hi, float(medians[k]))

    st.subheader("3. Pre-contrast medications" if en else "③ 造影前合并用药")
    st.caption("Check = used prior to contrast" if en else "勾选 = 造影前使用")
    med_vals = {}
    for i, (k, (enl, zhl)) in enumerate(MEDS.items()):
        if i % 3 == 0:
            mc = st.columns(3)
        med_vals[k] = int(mc[i % 3].checkbox(enl if en else zhl))

with col_r:
    st.subheader("4. Prediction" if en else "④ 预测结果")
    st.markdown("")
    feat = {
        "age": age, "contrast_dose": contrast_dose,
        "cr": lab_vals["cr"], "bun": lab_vals["bun"], "ua": lab_vals["ua"],
        "ast": lab_vals["ast"], "alt": lab_vals["alt"], "tbil": lab_vals["tbil"],
        "alp": lab_vals["alp"], "ibil": lab_vals["ibil"], "dbil": lab_vals["dbil"],
        "ggt": lab_vals["ggt"], "egfr": lab_vals["egfr"],
        "sex": 1 if (sex in ("Male", "男")) else 0,
        "surgery": 1 if (surgery in ("Yes", "是")) else 0,
        **med_vals,
        "contrast_type": contrast_code - 1,
    }
    if st.button("Calculate risk" if en else "计算风险", type="primary", use_container_width=True):
        p = predict(feat)
        pct = p * 100
        high = p >= threshold
        st.metric("Predicted CA-AKI risk" if en else "CA-AKI 预测风险概率", f"{pct:.1f}%")
        st.progress(min(p, 1.0))
        if high:
            st.error(
                f"⚠️ High risk (≥ {threshold*100:.0f}%): consider intensified creatinine "
                f"monitoring, review of nephrotoxic co-medications, and adequate hydration."
                if en else
                f"⚠️ 高风险（≥ {threshold*100:.0f}%）：建议加强肌酐监测、评估并调整肾毒性合并用药、充分水化。"
            )
        else:
            st.success(
                f"✅ Low risk (< {threshold*100:.0f}%): routine monitoring."
                if en else
                f"✅ 低风险（< {threshold*100:.0f}%）：常规监测即可。"
            )
        st.caption(
            f"Screening threshold {threshold*100:.0f}% (external sensitivity 0.764 / specificity 0.666)."
            if en else
            f"筛查阈值 {threshold*100:.0f}%（外验灵敏度 0.764 / 特异度 0.666）。"
        )

st.markdown("---")
st.caption(
    "⚠️ Research use only — this tool does not constitute clinical diagnosis or treatment advice. "
    "The model was derived from a Chinese multicenter retrospective cohort; extrapolation to other "
    "populations requires caution."
    if en else
    "⚠️ 本工具仅供科研参考，不构成临床诊断或治疗决策依据；模型基于中国多中心回顾性数据，外推到其他人群需谨慎。"
)
st.markdown(
    "**Model**: random forest (300 trees, class-weighted), 26 pre-contrast features "
    "(11 laboratory values + age + contrast type & dose + sex + surgery + 10 medication classes). "
    "CA-AKI defined by KDIGO criteria (serum creatinine rise ≥0.3 mg/dL within 48 h or ≥1.5× baseline within 7 days)."
    if en else
    "**模型说明**：随机森林（300 棵树，类别加权），26 个造影前特征"
    "（11 项检验指标 + 年龄 + 造影剂种类与剂量 + 性别 + 手术 + 10 类用药）。"
    "CA-AKI 定义为造影后 48 小时内血肌酐升高 ≥0.3 mg/dL（26.5 μmol/L）或 7 天内升至基线 ≥1.5 倍（KDIGO）。"
)
