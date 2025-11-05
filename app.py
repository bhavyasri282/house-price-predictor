import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ========== TRAIN MODEL ==========
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ========== APP CONFIG ==========
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

def set_bg(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True
    )

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

def format_inr(value):
    return f"₹{value:,.0f}"

if "page" not in st.session_state:
    st.session_state.page = 1
if "data" not in st.session_state:
    st.session_state.data = {}

# ========== AREA COORDINATES ==========
areas_info = {
    "Gandimaisamma": {"coords": {"lat": 17.5537, "lon": 78.4581}},
    "Medchal": {"coords": {"lat": 17.6293, "lon": 78.4804}},
    "Maisammaguda": {"coords": {"lat": 17.5789, "lon": 78.4875}},
    "Kompally": {"coords": {"lat": 17.5362, "lon": 78.4755}}
}

# ========== PAGE 1 ==========
if st.session_state.page == 1:
    set_bg("https://images.unsplash.com/photo-1600585154340-be6161a56a0c")
    st.title("🏠 House Price Predictor")
    st.subheader("Step 1️⃣: Select Area")

    area = st.selectbox("📍 Choose Area:", list(areas_info.keys()))
    if st.button("Next ➡️"):
        st.session_state.data["Area"] = area
        next_page()

# ========== PAGE 2 ==========
elif st.session_state.page == 2:
    set_bg("https://images.unsplash.com/photo-1599420186946-7b5f7d5e69e8")
    st.title("🏘 Step 2️⃣: House & Population Details")

    house_age = st.slider("🏠 House Age (years):", 1, 100, 10)
    ave_rooms = st.number_input("🛏 Average Rooms:", 1.0, 15.0, 5.0)
    ave_bedrooms = st.number_input("🛏 Average Bedrooms:", 1.0, 10.0, 2.0)
    population = st.number_input("👨‍👩‍👧‍👦 Population:", 100.0, 10000.0, 500.0)
    ave_occup = st.number_input("👪 Average Occupancy:", 1.0, 10.0, 3.0)

    if st.button("Next ➡️"):
        st.session_state.data.update({
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrooms,
            "Population": population,
            "AveOccup": ave_occup
        })
        next_page()

    st.button("⬅️ Back", on_click=prev_page)

# ========== PAGE 3 ==========
elif st.session_state.page == 3:
    set_bg("https://images.unsplash.com/photo-1570129477492-45c003edd2be")
    st.title("🏡 Step 3️⃣: Income & Street Details")

    avg_income = st.number_input("💰 Average Income (in ₹):", 10000.0, 10000000.0, 500000.0)
    street = st.text_input("🏘 Street Name:", "Sri Nagar Colony")
    distance = st.number_input("📏 Distance from main road (in meters):", 0.0, 10000.0, 500.0)
    near_road = st.radio("🚗 Is the house near a main road?", ["Yes", "No"])

    if st.button("Next ➡️"):
        st.session_state.data.update({
            "AvgIncome": avg_income,
            "Street": street,
            "DistanceFromRoad": distance,
            "NearRoad": near_road
        })
        next_page()

    st.button("⬅️ Back", on_click=prev_page)

# ========== PAGE 4 ==========
elif st.session_state.page == 4:
    set_bg("https://images.unsplash.com/photo-1600585153944-8d0011163aa7")
    st.title("🔮 Step 4️⃣ : Prediction Result")

    data = st.session_state.data
    st.subheader("📋 Review your details before prediction:")
    st.json(data)

    if st.button("🔮 Predict Price"):
        avg_income_inr = float(data.get("AvgIncome", 0.0))
        medinc_usd10k = (avg_income_inr / 83.0) / 10000.0

        coords = areas_info[data["Area"]]["coords"]
        latitude = coords["lat"]
        longitude = coords["lon"]

        X = np.array([[medinc_usd10k,
                       data["HouseAge"],
                       data["AveRooms"],
                       data["AveBedrms"],
                       data["Population"],
                       data["AveOccup"],
                       latitude,
                       longitude]])

        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        price_usd = pred * 100000.0
        price_inr = price_usd * 83.0

        # ✅ Strong Indian market scaling (₹50L–₹5Cr realistic)
        base_scale = {"Gandimaisamma": 1.8, "Medchal": 2.0, "Maisammaguda": 2.3, "Kompally": 3.0}
        price_inr *= base_scale.get(data["Area"], 2.0)

        if data.get("NearRoad") == "Yes":
            price_inr *= 0.95

        dist = float(data.get("DistanceFromRoad", 0.0))
        price_inr *= (1.0 + min(dist / 3000.0 * 0.10, 0.10))

        street_boosts = {
            "RTC Colony": 1.03, "Sri Nagar Colony": 1.04, "Anjaneya Nagar": 1.02,
            "Rajiv Gandhi Nagar": 1.05, "Venkateshwara Hills": 1.06,
            "Teachers Colony": 1.02, "Green Meadows": 1.06,
            "Sai Ram Nagar": 1.03, "Subash Nagar": 1.02,
            "Dhulapally Road": 1.01, "Petbasheerabad": 1.05, "Godavari Homes": 1.06
        }
        price_inr *= street_boosts.get(data["Street"], 1.02)

        low = max(price_inr * 0.90, 5000000)
        high = min(price_inr * 1.10, 50000000)

        st.markdown("### 🏠 Estimated House Price (approx)")
        st.success(f"📍 {data['Street']}, {data['Area']}")
        st.info(f"Range: {format_inr(low)} — {format_inr(high)}")
        st.write(f"**Final Estimated Price:** {format_inr(price_inr)} ✅")

        # ✅ New classy Indian home image
        st.image(
            "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
            caption=f"Sample house – {data['Area']}",
            use_container_width=True
        )

    st.button("⬅️ Back", on_click=prev_page)
