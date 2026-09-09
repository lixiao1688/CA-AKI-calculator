# CA-AKI 早期风险预测计算器

基于多中心外部验证的随机森林模型，用于造影剂相关急性肾损伤（CA-AKI）的早期风险预测。
内部 5 折交叉验证 AUC 0.930；独立外部验证（千佛山医院）AUC 0.807。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | Streamlit 网页计算器 |
| `rf_deploy_bundle.pkl` | 模型部署包（模型 + 预处理 + 阈值） |
| `requirements.txt` | Python 依赖 |

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将本目录（`app.py`、`rf_deploy_bundle.pkl`、`requirements.txt`）推送到一个 GitHub 仓库；
2. 登录 [share.streamlit.io](https://share.streamlit.io) → New app → 选择该仓库；
3. Main file path 填 `app.py` → Deploy。

## 模型说明

- **模型**：随机森林（`n_estimators=300`，`class_weight="balanced"`，`random_state=42`），
  在毓璜顶医院 24,843 例推导队列（CA-AKI 2,181 例，8.78%）上全量训练。
- **特征**：26 个造影前特征（11 项检验指标 + 年龄 + 造影剂种类与剂量 + 性别 + 手术 + 10 类用药）。
- **结局**：CA-AKI，KDIGO 标准（造影后 48h 内肌酐升高 ≥26.5 μmol/L 或 7 天内 ≥1.5 倍基线）。
- **阈值**：筛查阈值 0.25（外验灵敏度 0.764 / 特异度 0.666）。
- **诚实性能**：内部 CV AUC 0.930；外部验证 AUC 0.807（95% CI 0.778–0.833）。

⚠️ 仅供研究参考，不构成临床决策依据。
