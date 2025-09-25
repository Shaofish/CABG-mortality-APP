import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# ==============================
# 字型設定
# ==============================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 避免特殊符號錯亂
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
st.title("CABG Mortality Prediction System")
st.write("Please enter patient parameters. The system will predict mortality risk using two models and display SHAP analysis.")

# ==============================
# 中英文對照
# ==============================
cn2en = {
    "angina class, CCS": "Angina_class_CCS",
    "IABP\n insertion": "IABP_insertion",
    "Complete revascularization": "Complete_revascularization",
    "SvG for non-LAD": "SVG_non_LAD",
    "EuroSCORE Additive": "EuroSCORE_Additive",
    "Number of disease vessels": "Num_disease_vessels",
    "Number of grafts": "Num_grafts",
    "術前EF%": "PreOp_EF_percent",
    "術後EF%": "PostOp_EF_percent",
    "手術時間": "Surgery_time",
    "ICU \n日數": "ICU_days",
    "total CPB time  (mins)": "Total_CPB_time",
    "性別": "Sex",
    "年齡": "Age",
    "身高": "Height",
    "體重": "Weight",
    "BMI": "BMI",
    "Smoking": "Smoking",
    "DM": "Diabetes",
    "Pump support ": "Pump_support",
    "Quit \nSmoking": "Quit_smoking",
    "Stroke": "Stroke",
    "HTN": "Hypertension",
    "OPCAB": "OPCAB",
    "Post-bypass MI": "PostBypass_MI",
    "ESRD": "ESRD",
    "Cerebral deficits": "Cerebral_deficits",
    "Hyperlipidemia": "Hyperlipidemia",
    "Hyperkalemia": "Hyperkalemia",
    "COPD": "COPD",
    "呼吸器\n>24 hr": "Ventilator_gt_24hr",
    "Respiratory failure": "Respiratory_failure",
    "MIDCAB": "MIDCAB",
    "年齡 ≥ 60": "Age_ge_60",
    "Repeat sternotomy for revision of bleeding": "Repeat_sternotomy_bleeding",
    "Pump Arrest": "Pump_arrest",
    "Sepsis": "Sepsis",
    "LM>50%": "LM_gt_50",
    "Gastrointestinal Bleeding": "GI_bleeding",
    "住院\n天數": "InHospital_days",
    "術前住院天數": "PreOp_days",
    "術後住院天數": "PostOp_days"
}
# 反轉
en2cn = {v: k for k, v in cn2en.items()}

# ==============================
# 欄位依 HTML 順序排列 (英文顯示)
# ==============================
ordered_features_en = [
    "Angina_class_CCS", "IABP_insertion", "Complete_revascularization", "SVG_non_LAD",
    "EuroSCORE_Additive", "Num_disease_vessels", "Num_grafts", "PreOp_EF_percent",
    "PostOp_EF_percent", "Surgery_time", "ICU_days", "Total_CPB_time", "Sex", "Age",
    "Height", "Weight", "BMI", "Smoking", "Diabetes", "Pump_support", "Quit_smoking",
    "Stroke", "Hypertension", "OPCAB", "PostBypass_MI", "ESRD", "Cerebral_deficits", "Hyperlipidemia",
    "Hyperkalemia", "COPD", "Ventilator_gt_24hr", "Respiratory_failure", "MIDCAB", "Age_ge_60",
    "Repeat_sternotomy_bleeding", "Pump_arrest", "Sepsis", "LM_gt_50",
    "GI_bleeding", "InHospital_days", "PreOp_days", "PostOp_days"
]

model1_features_en = [
    "EuroSCORE_Additive", "Complete_revascularization", "Diabetes",
    "IABP_insertion", "Quit_smoking", "Respiratory_failure",
    "MIDCAB", "OPCAB", "Num_grafts", "Num_disease_vessels",
    "Cerebral_deficits", "PostBypass_MI", "Hypertension", "InHospital_days",
    "Hyperlipidemia", "ESRD", "Ventilator_gt_24hr", "PostOp_days",
    "Age_ge_60", "SVG_non_LAD", "PreOp_days",
    "Repeat_sternotomy_bleeding", "Smoking", "PostOp_EF_percent",
    "Surgery_time", "Height", "ICU_days"
]

model2_features_en = model1_features_en + [
    "Sex", "Total_CPB_time", "Weight", "LM_gt_50", "BMI", "Age",
    "Angina_class_CCS", "PreOp_EF_percent", "Pump_support", "Stroke",
    "GI_bleeding", "Hyperkalemia", "Sepsis", "Pump_arrest", "COPD"
]

# ==============================
# 欄位輸入方式設定 (英文)
# ==============================
binary_features = [
    "Diabetes", "Smoking", "Quit_smoking", "Hypertension", "ESRD", "COPD",
    "Respiratory_failure", "MIDCAB", "OPCAB", "Cerebral_deficits",
    "PostBypass_MI", "Hyperlipidemia", "Ventilator_gt_24hr",
    "Repeat_sternotomy_bleeding", "Pump_support",
    "Stroke", "GI_bleeding", "Hyperkalemia",
    "Sepsis", "Pump_arrest", "LM_gt_50", "Age_ge_60"
]

category_features = {
    "Sex": ["Male", "Female"],
    "Angina_class_CCS": [0, 3, 4],
    "IABP_insertion": [1, 2, 3],
    "Complete_revascularization": [1, 2],
    "SVG_non_LAD": [1, 2, 3],
}

# ==============================
# Streamlit 表單輸入
# ==============================
st.subheader("Enter Patient Data")
input_data_en = {}
cols = st.columns(3)

for idx, feature in enumerate(ordered_features_en):
    with cols[idx % 3]:
        if feature in binary_features:
            choice = st.radio(feature, ["No", "Yes"], horizontal=True)
            input_data_en[feature] = 1 if choice == "Yes" else 0

        elif feature in category_features:
            choice = st.selectbox(feature, category_features[feature])
            if feature == "Sex":
                input_data_en[feature] = 1 if choice == "Male" else 0
            else:
                input_data_en[feature] = choice

        else:
            input_data_en[feature] = st.number_input(feature, min_value=0, value=0)

# ==============================
# 預測
# ==============================
if st.button("Predict"):
    # 轉回中文欄位名稱給模型
    df1 = pd.DataFrame([[input_data_en[f] for f in model1_features_en]],
                       columns=[en2cn[f] for f in model1_features_en])
    df2 = pd.DataFrame([[input_data_en[f] for f in model2_features_en]],
                       columns=[en2cn[f] for f in model2_features_en])

    # 模型1
    pred1 = m1.predict_proba(df1)[:, 1][0]
    explainer1 = shap.TreeExplainer(m1)
    shap_values1 = explainer1(df1)
    shap_values1.feature_names = model1_features_en  # 換成英文

    # 模型2
    pred2 = m2.predict_proba(df2)[:, 1][0]
    explainer2 = shap.TreeExplainer(m2)
    shap_values2 = explainer2(df2)
    shap_values2.feature_names = model2_features_en  # 換成英文

    st.success(f"Model 1 (xgb_mortality) Predicted Mortality Risk: {pred1:.3f}")
    st.success(f"Model 2 (xgM_ALL) Predicted Mortality Risk: {pred2:.3f}")

    # SHAP 表格 - 顯示英文
    st.subheader("Model 1 SHAP Feature Importance")
    shap_df1 = pd.DataFrame(
        list(zip(model1_features_en, shap_values1.values[0])),
        columns=["Feature", "SHAP Value"]
    ).sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
    shap_df1.index += 1
    shap_df1.index.name = "Rank"
    st.dataframe(shap_df1)

    st.subheader("Model 2 SHAP Feature Importance")
    shap_df2 = pd.DataFrame(
        list(zip(model2_features_en, shap_values2.values[0])),
        columns=["Feature", "SHAP Value"]
    ).sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
    shap_df2.index += 1
    shap_df2.index.name = "Rank"
    st.dataframe(shap_df2)

    # SHAP 瀑布圖 (英文)
    st.subheader("SHAP Waterfall Plots")

    def plot_shap_waterfall(shap_values):
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    plot_shap_waterfall(shap_values1)
    plot_shap_waterfall(shap_values2)
