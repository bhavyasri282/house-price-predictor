import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------- TRAIN MODEL ----------
cal = fetch_california_housing()
X = pd.DataFrame(cal.data, columns=cal.feature_names)
y = cal.target

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------- APP CONFIG ----------
st.set_page_config(page_title="House Price Predictor", layout="centered")

def set_bg(url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.45);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"

# initialise session state
if "page" not in st.session_state:
    st.session_state.page = 1
if "data" not in st.session_state:
    st.session_state.data = {}

# ---------- AREA INFO ----------
areas_info = {
    "Gandimaisamma": {"coords": {"lat": 17.5537, "lon": 78.4581}, "base": 0.55},
    "Medchal":       {"coords": {"lat": 17.6293, "lon": 78.4804}, "base": 0.62},
    "Maisammaguda":  {"coords": {"lat": 17.5789, "lon": 78.4875}, "base": 0.70},
    "Kompally":      {"coords": {"lat": 17.5362, "lon": 78.4755}, "base": 0.85},
}

# ---------- PAGE 1 ----------
if st.session_state.page == 1:
    set_bg(
        "https://images.unsplash.com/photo-1599669454699-248895e0d826?"
        "auto=format&fit=crop&w=1950&q=80"
    )
    st.title("House Price Predictor")
    st.markdown(
        "<h3 style='color:white; text-align:center;'>Step 1: Select Area</h3>",
        unsafe_allow_html=True,
    )
    area = st.selectbox("Choose Area:", list(areas_info.keys()))
    if st.button("Next"):
        st.session_state.data["Area"] = area
        next_page()

# ---------- PAGE 2 ----------
elif st.session_state.page == 2:
    set_bg(
        "https://images.unsplash.com/photo-1618778599502-9e9a51ce2c85?"
        "auto=format&fit=crop&w=1950&q=80"
    )
    st.title("Step 2: House & Population Details")
    house_age = st.slider("House Age (years):", 1, 100, 15)
    ave_rooms = st.slider("Total Rooms:", 3, 20, 6)
    ave_bedrms = st.slider("Bedrooms:", 1, 10, 3)
    population = st.slider("Block Population:", 100, 10000, 1400)
    ave_occup = st.slider("Avg People per House:", 1.0, 10.0, 3.0, step=0.1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Next"):
            st.session_state.data.update(
                {
                    "HouseAge": house_age,
                    "AveRooms": ave_rooms,
                    "AveBedrms": ave_bedrms,
                    "Population": population,
                    "AveOccup": ave_occup,
                }
            )
            next_page()
    with col2:
        st.button("Back", on_click=prev_page)

# ---------- PAGE 3 ----------
elif st.session_state.page == 3:
    set_bg(
        "https://images.unsplash.com/photo-1600585154526-990dced4db0d?"
        "auto=format&fit=crop&w=1950&q=80"
    )
    st.title("Step 3: Income & Location Details")
    income_lakhs = st.number_input(
        "Median Annual Household Income (₹ Lakhs):",
        5.0,
        60.0,
        12.0,
        help="Typical range in these areas: ₹8–25 L",
    )
    street = st.text_input("Street / Community Name:", "Sri Nagar Colony")
    distance = st.slider("Distance from main road (m):", 0, 5000, 200)
    near_road = st.radio("Facing / Near main road?", ["Yes", "No"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Next"):
            st.session_state.data.update(
                {
                    "IncomeLakhs": income_lakhs,
                    "Street": street,
                    "DistanceFromRoad": distance,
                    "NearRoad": near_road,
                }
            )
            next_page()
    with col2:
        st.button("Back", on_click=prev_page)

# ---------- PAGE 4 (PREDICTION) ----------
elif st.session_state.page == 4:
    set_bg(
        "https://images.unsplash.com/photo-1600566753376-2e5c0a1a7f2a?"
        "auto=format&fit=crop&w=1950&q=80"
    )
    st.title("Prediction Result")
    data = st.session_state.data
    st.subheader("Your entered details:")
    st.json(data)

    if st.button("Predict Price"):
        # ---------- 1. Convert income (lakhs → model units) ----------
        income_usd = data["IncomeLakhs"] * 100_000 / 84.0      # ₹L → USD
        medinc_10k = income_usd / 10_000                     # model expects tens of thousands

        # ---------- 2. Build feature vector (exact order) ----------
        coords = areas_info[data["Area"]]["coords"]
        feature_vec = np.array([[
            medinc_10k,                 # MedInc
            data["HouseAge"],           # HouseAge
            data["AveRooms"],           # AveRooms
            data["AveBedrms"],          # AveBedrms
            data["Population"],         # Population
            data["AveOccup"],           # AveOccup
            coords["lat"],              # Latitude
            coords["lon"],              # Longitude
        ]])

        # ---------- 3. Scale & predict ----------
        feature_scaled = scaler.transform(feature_vec)
        pred = model.predict(feature_scaled)[0]          # in 100k USD
        price_usd = pred * 100_000
        price_inr = price_usd * 84.0                     # back to INR

        # ---------- 4. Hyderabad-north scaling ----------
        price_inr *= areas_info[data["Area"]]["base"]

        # ---- Room / bedroom boost ----
        price_inr *= (1 + (data["AveRooms"] - 6) * 0.025)   # +2.5 % per extra room
        price_inr *= (1 + (data["AveBedrms"] - 3) * 0.03)   # +3 % per extra bedroom

        # ---- Population penalty (dense = cheaper) ----
        price_inr *= (1 - min((data["Population"] - 1400) / 8000, 0.15))

        # ---- Near-road premium ----
        if data.get("NearRoad") == "Yes":
            price_inr *= 1.10

        # ---- Distance penalty ----
        dist = float(data.get("DistanceFromRoad", 0))
        price_inr *= (1 - min(dist / 4000 * 0.12, 0.12))

        # ---- Street premium ----
        street_boosts = {
            "RTC Colony": 1.06, "Sri Nagar Colony": 1.08, "Anjaneya Nagar": 1.05,
            "Rajiv Gandhi Nagar": 1.09, "Venkateshwara Hills": 1.12,
            "Teachers Colony": 1.05, "Green Meadows": 1.13,
            "Godavari Homes": 1.14, "Petbasheerabad": 1.07,
        }
        price_inr *= street_boosts.get(data["Street"].strip(), 1.04)

        # ---------- 5. Final clamp ----------
        price_inr = np.clip(price_inr, 5_000_000, 15_000_000)

        low  = max(price_inr * 0.92, 5_000_000)
        high = min(price_inr * 1.08, 15_000_000)

        # ---------- 6. Show result ----------
        st.markdown("### Estimated Price Range")
        st.success(f"{data['Street']}, {data['Area']}")
        st.info(f"**Likely Range:** {format_inr(low)} – {format_inr(high)}")
        st.markdown(
            f"<h2 style='color:#00c853;'>Best Estimate: {format_inr(price_inr)}</h2>",
            unsafe_allow_html=True,
        )

        # ---------- 7. Dynamic house image ----------
        if price_inr <= 7_500_000:
            img = "https://images.unsplash.com/photo-1568605117036-5fe57e8e0a96?"
            caption = f"Typical 2-3 BHK in {data['Area']}"
        elif price_inr <= 11_000_000:
            img = "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?"
            caption = f"Modern 3-4 BHK in {data['Area']}"
        else:
            img = "https://images.unsplash.com/photo-1613977257592-92f3009f2add?"
            caption = f"Premium villa-style home in {data['Area']}"
        img += "auto=format&fit=crop&w=1470&q=80"
        st.image(img, caption=caption, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_page)
    with col2:
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
