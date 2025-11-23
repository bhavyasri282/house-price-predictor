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
    st.session_state.page = 0   # LOGIN PAGE FIRST

if "data" not in st.session_state:
    st.session_state.data = {}

if "user" not in st.session_state:
    st.session_state.user = None

# ---------- LOGIN PAGE ----------
def login_page():
    st.title("🔐 Login to Continue")

    st.markdown("### Enter Your Details to Proceed")
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone (optional)")

    colA, colB = st.columns(2)

    with colA:
        if st.button("Continue"):
            if name.strip() == "" or email.strip() == "":
                st.error("Please enter both Name & Email!")
            else:
                st.session_state.user = {
                    "name": name,
                    "email": email,
                    "phone": phone
                }
                st.session_state.page = 1
                st.rerun()

    with colB:
        if st.button("Continue with Google"):
            st.session_state.user = {
                "name": "Google User",
                "email": "google@login.com",
                "phone": None
            }
            st.session_state.page = 1
            st.rerun()

# ---------- PAGE 0 ----------
if st.session_state.page == 0:
    login_page()
    st.stop()

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

    st.info(f"Logged in as: **{st.session_state.user['name']}** ({st.session_state.user['email']})")

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

    st.info(f"Logged in as: **{st.session_state.user['name']}** ({st.session_state.user['email']})")

    house_age = st.slider("House Age (years):", 1, 100, 15)
    ave_rooms = st.slider("Total Rooms:", 3, 20, 6)
    ave_bedrms = st.slider("Bedrooms:", 1, 10, 3)
    population = st.slider("Block Population:", 100, 10000, 1400)
    ave_occup = st.slider("Avg People per House:", 1.0, 10.0, 3.0, step=0.1)

    plot_size = st.slider("Plot Size (sq yards):", 50, 1500, 200)

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
                    "PlotSize": plot_size,
                }
            )
            next_page()
    with col2:
        st.button("Back", on_click=prev_page)

# ---------- PAGE 3 ----------
elif st.session_state.page == 3:
    set_bg(
        "https://images.unsplash.com/photo-1600566753376-2e5c0a1a7f2a?"
        "auto=format&fit=crop&w=1950&q=80"
    )
    st.title("Step 3: Income & Location Details")

    st.info(f"Logged in as: **{st.session_state.user['name']}** ({st.session_state.user['email']})")

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

    st.info(f"Logged in as: **{st.session_state.user['name']}** ({st.session_state.user['email']})")

    data = st.session_state.data
    st.subheader("Your entered details:")
    st.json(data)

    if st.button("Predict Price"):
        income_usd = data["IncomeLakhs"] * 100_000 / 84.0
        medinc_10k = income_usd / 10_000

        coords = areas_info[data["Area"]]["coords"]
        feature_vec = np.array([[medinc_10k, data["HouseAge"], data["AveRooms"],
                                 data["AveBedrms"], data["Population"],
                                 data["AveOccup"], coords["lat"], coords["lon"]]])

        feature_scaled = scaler.transform(feature_vec)
        pred = model.predict(feature_scaled)[0]

        price_usd = pred * 100_000
        price_inr = price_usd * 84.0

        price_inr *= areas_info[data["Area"]]["base"]

        price_inr *= (1 + (data["AveRooms"] - 6) * 0.025)
        price_inr *= (1 + (data["AveBedrms"] - 3) * 0.03)

        price_inr *= (1 - min((data["Population"] - 1400) / 8000, 0.15))

        if data.get("NearRoad") == "Yes":
            price_inr *= 1.10

        dist = float(data.get("DistanceFromRoad", 0))
        price_inr *= (1 - min(dist / 4000 * 0.12, 0.12))

        street_boosts = {
            "RTC Colony": 1.06, "Sri Nagar Colony": 1.08, "Anjaneya Nagar": 1.05,
            "Rajiv Gandhi Nagar": 1.09, "Venkateshwara Hills": 1.12,
            "Teachers Colony": 1.05, "Green Meadows": 1.13,
            "Godavari Homes": 1.14, "Petbasheerabad": 1.07,
        }
        price_inr *= street_boosts.get(data["Street"].strip(), 1.04)

        price_inr = np.clip(price_inr, 5_000_000, 15_000_000)

        size = data.get("PlotSize", 200)
        size_factor = 1 + ((size - 200) / 800)
        price_inr *= size_factor

        low  = max(price_inr * 0.92, 5_000_000)
        high = min(price_inr * 1.08, 15_000_000)

        st.markdown("### Estimated Price Range")
        st.success(f"{data['Street']}, {data['Area']}")
        st.info(f"**Likely Range:** {format_inr(low)} – {format_inr(high)}")
        st.markdown(
            f"<h2 style='color:#00c853;'>Best Estimate: {format_inr(price_inr)}</h2>",
            unsafe_allow_html=True,
        )

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

# ---------- EXTRA FEATURES (Construction, Amenities, EMI, Breakdown, PDF) ----------
if st.session_state.page == 4:
    st.markdown("---")
    st.markdown("## Additional Options (Construction type & Amenities)")

    construction_type = st.selectbox(
        "Construction Type:",
        ["Independent House", "Apartment", "Duplex", "Villa"]
    )

    amenities_list = [
        "Car Parking", "Modular Kitchen", "Interior Work Done",
        "Borewell / Manjeera Water", "Solar Panels", "Garden", "Security / Gated Community"
    ]
    amenities = st.multiselect("Select available amenities:", amenities_list)

    st.markdown("### EMI Calculator (optional)")
    with st.expander("Open EMI Calculator"):
        loan_percent = st.slider("Loan as % of predicted price:", 10, 100, 70)
        annual_rate = st.number_input("Annual interest rate (%, p.a.):", 6.0, 15.0, 8.5)
        loan_years = st.slider("Loan tenure (years):", 1, 30, 15)

    st.markdown("### Other utilities")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Recompute & Show All (with new options)"):

            income_usd = st.session_state.data["IncomeLakhs"] * 100_000 / 84.0
            medinc_10k = income_usd / 10_000

            coords = areas_info[st.session_state.data["Area"]]["coords"]
            feature_vec = np.array([[medinc_10k,
                                     st.session_state.data["HouseAge"],
                                     st.session_state.data["AveRooms"],
                                     st.session_state.data["AveBedrms"],
                                     st.session_state.data["Population"],
                                     st.session_state.data["AveOccup"],
                                     coords["lat"], coords["lon"]]])

            feature_scaled = scaler.transform(feature_vec)
            pred = model.predict(feature_scaled)[0]

            price_usd = pred * 100_000
            base_price_inr = price_usd * 84.0

            price_inr = base_price_inr * areas_info[st.session_state.data["Area"]]["base"]

            rooms_boost = (1 + (st.session_state.data["AveRooms"] - 6) * 0.025)
            beds_boost = (1 + (st.session_state.data["AveBedrms"] - 3) * 0.03)
            price_inr *= rooms_boost
            price_inr *= beds_boost

            pop_penalty = (1 - min((st.session_state.data["Population"] - 1400) / 8000, 0.15))
            price_inr *= pop_penalty

            near_road_mult = 1.10 if st.session_state.data.get("NearRoad") == "Yes" else 1.0
            price_inr *= near_road_mult

            dist = float(st.session_state.data.get("DistanceFromRoad", 0))
            dist_mult = (1 - min(dist / 4000 * 0.12, 0.12))
            price_inr *= dist_mult

            street_boosts = {
                "RTC Colony": 1.06, "Sri Nagar Colony": 1.08, "Anjaneya Nagar": 1.05,
                "Rajiv Gandhi Nagar": 1.09, "Venkateshwara Hills": 1.12,
                "Teachers Colony": 1.05, "Green Meadows": 1.13,
                "Godavari Homes": 1.14, "Petbasheerabad": 1.07,
            }
            street_mult = street_boosts.get(st.session_state.data["Street"].strip(), 1.04)
            price_inr *= street_mult

            price_inr = np.clip(price_inr, 5_000_000, 15_000_000)

            size = st.session_state.data.get("PlotSize", 200)
            size_factor = 1 + ((size - 200) / 800)
            price_inr *= size_factor

            construction_map = {
                "Apartment": 0.95,
                "Independent House": 1.10,
                "Duplex": 1.15,
                "Villa": 1.25,
            }
            construction_mult = construction_map.get(construction_type, 1.0)
            price_inr *= construction_mult

            amenities_map = {
                "Car Parking": 1.03,
                "Modular Kitchen": 1.04,
                "Interior Work Done": 1.05,
                "Borewell / Manjeera Water": 1.02,
                "Solar Panels": 1.05,
                "Garden": 1.03,
                "Security / Gated Community": 1.08,
            }
            amenities_mult = 1.0
            for a in amenities:
                amenities_mult *= amenities_map.get(a, 1.0)
            price_inr *= amenities_mult

            price_inr = np.clip(price_inr, 5_000_000, 15_000_000)

            breakdown = []
            breakdown.append(("Base (model) INR", base_price_inr))
            breakdown.append(("Area base multiplier", areas_info[st.session_state.data["Area"]]["base"]))
            breakdown.append(("Rooms boost multiplier", rooms_boost))
            breakdown.append(("Bedrooms boost multiplier", beds_boost))
            breakdown.append(("Population penalty multiplier", pop_penalty))
            breakdown.append(("Near road multiplier", near_road_mult))
            breakdown.append(("Distance multiplier", dist_mult))
            breakdown.append(("Street multiplier", street_mult))
            breakdown.append(("Plot size multiplier", size_factor))
            breakdown.append(("Construction multiplier", construction_mult))
            breakdown.append(("Amenities combined multiplier", amenities_mult))

            final_low = max(price_inr * 0.92, 5_000_000)
            final_high = min(price_inr * 1.08, 15_000_000)

            st.markdown("## Recomputed Estimate (including Construction & Amenities)")
            st.info(f"**Estimate:** {format_inr(price_inr)}  — Range: {format_inr(final_low)} – {format_inr(final_high)}")

            df_break = pd.DataFrame(breakdown, columns=["Factor", "Value"])
            def pretty(v):
                if isinstance(v, (int, float)):
                    if 0.01 < v < 3.0:
                        return f"x{v:.3f}"
                    else:
                        return format_inr(v)
                return str(v)
            df_break["Display"] = df_break["Value"].apply(pretty)
            st.markdown("### Price Breakdown")
            st.table(df_break[["Factor", "Display"]])

            st.markdown("### EMI Calculation")
            loan_amount = price_inr * (loan_percent / 100.0)
            monthly_rate = annual_rate / 100.0 / 12.0
            n_months = loan_years * 12
            if monthly_rate > 0:
                emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n_months / ((1 + monthly_rate) ** n_months - 1)
            else:
                emi = loan_amount / n_months

            st.write(f"Loan amount (approx): {format_inr(loan_amount)}")
            st.write(f"Monthly EMI (approx): {format_inr(emi)}")

            st.markdown("### Download Simple Report (PDF)")
            try:
                from fpdf import FPDF
                use_fpdf = True
            except Exception:
                use_fpdf = False

            report_text_lines = [
                "House Price Predictor - Simple Report",
                "------------------------------------",
                f"Area: {st.session_state.data['Area']}",
                f"Street: {st.session_state.data.get('Street','')}",
                f"Construction Type: {construction_type}",
                f"Amenities: {', '.join(amenities) if amenities else 'None'}",
                f"Plot Size (sq yards): {st.session_state.data.get('PlotSize', 'NA')}",
                f"Predicted Price (best): {format_inr(price_inr)}",
                f"Likely Range: {format_inr(final_low)} - {format_inr(final_high)}",
                "",
                "Generated by House Price Predictor",
            ]

            if use_fpdf:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in report_text_lines:
                    pdf.cell(0, 8, txt=line, ln=1)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("Download PDF report", data=pdf_bytes, file_name="house_report.pdf", mime="application/pdf")
            else:
                report_text = "\n".join(report_text_lines)
                st.download_button("Download text report", data=report_text, file_name="house_report.txt", mime="text/plain")

    with colB:
        if st.button("Reset Additional Options"):
            st.experimental_rerun()

# (CODE ENDS)
