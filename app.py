import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# ==============================
# 字型設定 (確保中文正常顯示)
# ==============================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 載入模型
# ==============================
m1 = joblib.load("xgb_mortality.joblib")
m2 = joblib.load("xgM_ALL.joblib")

# ==============================
# 頁面設定
# ==============================
st.set_page_config(page_title="CABG Mortality Prediction", layout="wide")
st.title("CABG 病人死亡機率預測系統")
st.write("請輸入病患相關參數，系統將使用兩個模型進行預測，並顯示 SHAP 特徵重要性分析。")

# ==============================
# 欄位依 HTML 順序排列
# ==============================
ordered_features = [
    # 日期類不輸入，直接計算
    "angina class, CCS", "IABP\n insertion", "Complete revascularization", "SvG for non-LAD",
    "EuroSCORE Additive", "Number of disease vessels", "Number of grafts", "術前EF%",
    "術後EF%", "手術時間", "ICU \n日數", "total CPB time  (mins)", "性別", "年齡",
    "身高", "體重", "BMI", "Smoking", "DM", "Pump support ", "Quit \nSmoking",
    "Stroke", "HTN", "OPCAB", "Post-bypass MI", "ESRD", "Cerebral deficits", "Hyperlipidemia",
    "Hyperkalemia", "COPD", "呼吸器\n>24 hr", "Respiratory failure", "MIDCAB", "年齡 ≥ 60",
    "Repeat sternotomy for revision of bleeding", "Pump Arrest", "Sepsis", "LM>50%",
    "Gastrointestinal Bleeding", "住院\n天數", "術前住院天數", "術後住院天數"
]

model1_features = [
    "EuroSCORE Additive", "Complete revascularization", "DM",
    "IABP\n insertion", "Quit \nSmoking", "Respiratory failure",
    "MIDCAB", "OPCAB", "Number of grafts", "Number of disease vessels",
    "Cerebral deficits", "Post-bypass MI", "HTN", "住院\n天數",
    "Hyperlipidemia", "ESRD", "呼吸器\n>24 hr", "術後住院天數",
    "年齡 ≥ 60", "SvG for non-LAD", "術前住院天數",
    "Repeat sternotomy for revision of bleeding", "Smoking", "術後EF%",
    "手術時間", "身高", "ICU \n日數"
]

model2_features = model1_features + [
    "性別", "total CPB time  (mins)", "體重", "LM>50%", "BMI", "年齡",
    "angina class, CCS", "術前EF%", "Pump support ", "Stroke",
    "Gastrointestinal Bleeding", "Hyperkalemia", "Sepsis", "Pump Arrest", "COPD"
]

# ==============================
# 欄位輸入方式設定
# ==============================
binary_features = [
    "DM", "Smoking", "Quit \nSmoking", "HTN", "ESRD", "COPD",
    "Respiratory failure", "MIDCAB", "OPCAB", "Cerebral deficits",
    "Post-bypass MI", "Hyperlipidemia", "呼吸器\n>24 hr",
    "Repeat sternotomy for revision of bleeding", "Pump support ",
    "Stroke", "Gastrointestinal Bleeding", "Hyperkalemia",
    "Sepsis", "Pump Arrest", "LM>50%", "年齡 ≥ 60"
]

category_features = {
    "性別": ["男", "女"],
    "angina class, CCS": [0, 3, 4],
    "IABP\n insertion": [1, 2, 3],
    "Complete revascularization": [1, 2],
    "SvG for non-LAD": [1, 2, 3],
}

# ==============================
# Streamlit 表單輸入
# ==============================
st.subheader("輸入病人資料")
input_data = {}
cols = st.columns(3)

for idx, feature in enumerate(ordered_features):
    with cols[idx % 3]:
        if feature in binary_features:
            choice = st.radio(feature, ["無", "有"], horizontal=True)
            input_data[feature] = 1 if choice == "有" else 0

        elif feature in category_features:
            choice = st.selectbox(feature, category_features[feature])
            if feature == "性別":
                input_data[feature] = 1 if choice == "男" else 0
            else:
                input_data[feature] = choice

        else:
            input_data[feature] = st.number_input(feature, min_value=0, value=0)

# ==============================
# 預測
# ==============================
if st.button("開始預測"):
    df1 = pd.DataFrame([[input_data[f] for f in model1_features]], columns=model1_features)
    df2 = pd.DataFrame([[input_data[f] for f in model2_features]], columns=model2_features)

    # 模型1
    pred1 = m1.predict_proba(df1)[:, 1][0]
    explainer1 = shap.TreeExplainer(m1)
    shap_values1 = explainer1(df1)

    # 模型2
    pred2 = m2.predict_proba(df2)[:, 1][0]
    explainer2 = shap.TreeExplainer(m2)
    shap_values2 = explainer2(df2)

    st.success(f"模型1 (xgb_mortality) 預測死亡機率: {pred1:.3f}")
    st.success(f"模型2 (xgM_ALL) 預測死亡機率: {pred2:.3f}")

    # SHAP 表格 - 加上排名
    st.subheader("模型1 SHAP 特徵重要性")
    shap_df1 = pd.DataFrame(
        list(zip(df1.columns, shap_values1.values[0])),
        columns=["Feature", "SHAP Value"]
    ).sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
    shap_df1.index += 1
    shap_df1.index.name = "Rank"
    st.dataframe(shap_df1)

    st.subheader("模型2 SHAP 特徵重要性")
    shap_df2 = pd.DataFrame(
        list(zip(df2.columns, shap_values2.values[0])),
        columns=["Feature", "SHAP Value"]
    ).sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
    shap_df2.index += 1
    shap_df2.index.name = "Rank"
    st.dataframe(shap_df2)

    # SHAP 瀑布圖
    st.subheader("SHAP 瀑布圖")

    def plot_shap_waterfall(shap_values):
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    plot_shap_waterfall(shap_values1)
    plot_shap_waterfall(shap_values2)
