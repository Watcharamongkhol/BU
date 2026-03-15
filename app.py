import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Kidney Disease AI Screening",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# MEDICAL UI STYLE
# -------------------------

st.markdown("""
<style>

/* Background */

.stApp{
background:#f6fbf9;
font-family:'Segoe UI',sans-serif;
color:#000;
}

/* Title */

.main-title{
font-size:42px;
font-weight:700;
color:#0a8f6a;
}

.subtitle{
font-size:18px;
color:#333;
margin-bottom:25px;
}

/* Section Title */

.section-title{
color:#0a8f6a;
font-size:26px;
font-weight:700;
margin-bottom:15px;
}

/* Card */

.card{
background:white;
padding:28px;
border-radius:14px;
box-shadow:0 4px 14px rgba(0,0,0,0.08);
margin-bottom:25px;
}

/* Accuracy Box */

.metric{
background:#eaf7f3;
padding:16px;
border-radius:10px;
text-align:center;
font-size:20px;
color:#0a8f6a;
font-weight:700;
margin-bottom:20px;
}

/* Labels */

label{
color:#0a8f6a !important;
font-weight:600 !important;
font-size:16px !important;
}

/* Text Input */

.stTextInput input{
background:#ffffff !important;
border:2px solid #0a8f6a !important;
border-radius:8px !important;
color:#000000 !important;
padding:8px !important;
}

/* Number Input */

.stNumberInput input{
background:#ffffff !important;
border:2px solid #0a8f6a !important;
border-radius:8px !important;
color:#000000 !important;
}

/* Select Box */

.stSelectbox div[data-baseweb="select"]{
background:#ffffff !important;
border:2px solid #0a8f6a !important;
border-radius:8px !important;
color:#000 !important;
}

/* Focus */

input:focus{
border:2px solid #14b88f !important;
box-shadow:0 0 6px rgba(10,143,106,0.4) !important;
}

/* Button */

.stButton>button{
background:#0a8f6a;
color:white;
height:50px;
border-radius:8px;
font-size:18px;
font-weight:600;
border:none;
width:100%;
}

.stButton>button:hover{
background:#067a59;
}

/* Result text */

.result-text{
color:#000;
font-size:18px;
}

/* Alert */

div[data-testid="stAlert"]{
color:black !important;
font-size:18px !important;
font-weight:600 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.markdown('<div class="main-title">🏥 Kidney Disease AI Screening</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ระบบคัดกรองโรคไตด้วย AI สำหรับสถานพยาบาล</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("kidney_dataset.csv")
df = df.dropna()

target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(X)

if y.dtype == "object":
    y = y.astype("category").cat.codes

# -------------------------
# TRAIN MODEL
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.markdown(f"""
<div class="metric">
🧠 AI Model Accuracy : {accuracy*100:.2f} %
</div>
""", unsafe_allow_html=True)

# -------------------------
# PATIENT INFO
# -------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:
    patient_name = st.text_input("ชื่อ - สกุล")

with col2:
    age = st.number_input("อายุ",1,120)

with col3:
    gender = st.selectbox("เพศ",["ชาย","หญิง"])

col4,col5 = st.columns(2)

with col4:
    weight = st.number_input("น้ำหนัก (kg)",0.0,200.0)

with col5:
    height = st.number_input("ส่วนสูง (cm)",0.0,250.0)

if height > 0:
    bmi = weight / ((height/100)**2)
    st.info(f"BMI : {bmi:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# LAB RESULT
# -------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Laboratory Results</div>', unsafe_allow_html=True)

input_dict = {}

cols = st.columns(3)

for i,col in enumerate(X.columns):

    with cols[i%3]:

        input_dict[col] = st.number_input(
            col.replace("_"," "),
            min_value=0.0,
            max_value=500.0,
            value=0.0
        )

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PREDICTION
# -------------------------

if st.button("🔬 วิเคราะห์โรคไตด้วย AI"):

    input_df = pd.DataFrame([input_dict])

    for column in X.columns:
        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df[X.columns]

    prediction = model.predict(input_df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 ผลการวิเคราะห์</div>', unsafe_allow_html=True)

    st.markdown(f'<p class="result-text">👤 ผู้ป่วย : {patient_name}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-text">🎂 อายุ : {age} ปี</p>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ พบความเสี่ยงโรคไต")
    else:
        st.success("✅ ไม่พบความเสี่ยงโรคไต")

    st.markdown('<p class="result-text">ผลลัพธ์เป็นเพียงการคัดกรองเบื้องต้น</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# GRAPH
# -------------------------

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Kidney Function Indicators</div>', unsafe_allow_html=True)

    creatinine = input_dict.get("Creatinine",0)
    bun = input_dict.get("BUN",0)
    gfr = input_dict.get("GFR",0)

    df_graph = pd.DataFrame({
        "Indicator":["Creatinine","BUN","GFR"],
        "Value":[creatinine,bun,gfr]
    })

    fig, ax = plt.subplots()

    bars = ax.bar(df_graph["Indicator"], df_graph["Value"])

    ax.set_ylabel("Value")
    ax.set_title("Kidney Function")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x()+0.25, yval+0.5, round(yval,2))

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)