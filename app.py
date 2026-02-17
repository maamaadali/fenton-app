import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go

# تنظیمات صفحه
st.set_page_config(page_title="Smart Fenton Optimizer", layout="wide")

# عنوان و توضیحات
st.title("💧 سیستم هوشمند کنترل تصفیه فاضلاب (فنتون)")
st.markdown("""
این سیستم هوشمند با بررسی **۹۰۰۰ سناریوی مختلف**، بهترین شرایط عملیاتی را برای حذف آنتی‌بیوتیک آموکسی‌سیلین پیدا می‌کند.
""")
st.markdown("---")

# منوی سمت راست (ورودی‌ها)
st.sidebar.header("⚙️ تنظیمات ورودی")
C0_input = st.sidebar.number_input("غلظت ورودی (mg/L)", min_value=10.0, max_value=1000.0, value=189.0, step=1.0)
Vol_Tank = st.sidebar.number_input("حجم مخزن (لیتر)", value=1000)

# ثوابت
Standard_Limit = 5.0
MW_FeSO4 = 278.0  # g/mol
MW_H2O2 = 34.0    # g/mol
Purity_H2O2 = 0.30 # 30%

# تابع محاسبه نرخ واکنش
def calculate_k(pH, H2O2, Fe):
    f_pH = np.exp(-((pH - 3.0)**2) / 0.5)
    f_H2O2 = (H2O2 / (5 + H2O2)) * np.exp(-H2O2/30)
    f_Fe = (Fe / (0.5 + Fe))
    k_base = 0.25
    return k_base * f_pH * f_H2O2 * f_Fe

# دکمه اجرا
if st.sidebar.button("🚀 شروع بهینه‌سازی"):
    
    # 1. ساخت شبکه جستجو (9000 حالت)
    pH_range = np.linspace(2, 5, 30)
    H2O2_range = np.linspace(1, 20, 30)
    Fe_range = np.linspace(0.1, 2, 10)
    
    G_pH, G_H2O2, G_Fe = np.meshgrid(pH_range, H2O2_range, Fe_range, indexing='ij')
    
    All_k = calculate_k(G_pH, G_H2O2, G_Fe)
    
    # پیدا کردن بهترین نقطه
    max_k_idx = np.unravel_index(np.argmax(All_k, axis=None), All_k.shape)
    Max_k = All_k[max_k_idx]
    
    Opt_pH = pH_range[max_k_idx[0]]
    Opt_H2O2 = H2O2_range[max_k_idx[1]]
    Opt_Fe = Fe_range[max_k_idx[2]]

    # 2. شبیه‌سازی زمانی
    def model(C, t):
        dCdt = -Max_k * C
        return dCdt

    t = np.linspace(0, 60, 100)
    C = odeint(model, C0_input, t)
    C = C.flatten()

    clean_indices = np.where(C <= Standard_Limit)[0]
    if len(clean_indices) > 0:
        Req_Time = t[clean_indices[0]]
        status_msg = f"تصفیه کامل در {Req_Time:.1f} دقیقه"
    else:
        Req_Time = 60.0
        status_msg = "تصفیه کامل نشد"

    Final_Eff = (C0_input - C[-1]) / C0_input * 100

    # 3. محاسبه جرم مواد
    Mass_Fe = Opt_Fe * MW_FeSO4 * (Vol_Tank / 1000)
    Mass_H2O2 = (Opt_H2O2 * MW_H2O2 * (Vol_Tank / 1000)) / Purity_H2O2

    # --- نمایش نتایج ---
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("pH بهینه", f"{Opt_pH:.2f}")
    col2.metric("دوز آب‌اکسیژنه", f"{Opt_H2O2:.2f} mM")
    col3.metric("دوز آهن", f"{Opt_Fe:.2f} mM")
    col4.metric("زمان فرآیند", f"{Req_Time:.1f} min", delta=status_msg)

    st.subheader("📋 دستورالعمل اجرایی (SOP)")
    st.info(f"مبنای محاسبات: برای {Vol_Tank} لیتر فاضلاب")
    
    recipe_data = {
        "پارامتر": ["نقطه تنظیم pH", "کاتالیست (پودر سولفات آهن)", "اکسیدکننده (آب‌اکسیژنه ۳۰٪)"],
        "مقدار علمی": [f"pH = {Opt_pH:.2f}", f"{Opt_Fe:.2f} mM", f"{Opt_H2O2:.2f} mM"],
        "مقدار اجرایی (توزین)": ["تزریق اسید تا رسیدن به عدد", f"افزودن {Mass_Fe:.1f} گرم پودر", f"افزودن {Mass_H2O2:.1f} گرم مایع"]
    }
    st.table(pd.DataFrame(recipe_data))

    # نمودارها
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.subheader("📈 نمودار غلظت-زمان")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=C, mode='lines', name='غلظت', line=dict(color='royalblue', width=3)))
        fig1.add_hline(y=Standard_Limit, line_dash="dash", line_color="red", annotation_text="حد مجاز (5 mg/L)")
        fig1.update_layout(xaxis_title="زمان (دقیقه)", yaxis_title="غلظت (mg/L)", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col_graph2:
        st.subheader("🔍 فضای جستجوی هوشمند")
        Slice_k = All_k[:, :, max_k_idx[2]]
        fig2 = go.Figure(data=go.Contour(
            z=Slice_k.T,
            x=pH_range,
            y=H2O2_range,
            colorscale='Viridis',
            colorbar=dict(title='سرعت واکنش')
        ))
        fig2.add_trace(go.Scatter(x=[Opt_pH], y=[Opt_H2O2], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='نقطه بهینه'))
        fig2.update_layout(xaxis_title="pH", yaxis_title="غلظت آب‌اکسیژنه", height=400)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 لطفاً از منوی سمت راست، غلظت را وارد کرده و دکمه 'شروع بهینه‌سازی' را بزنید.")
