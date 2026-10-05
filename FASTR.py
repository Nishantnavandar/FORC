import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FASTR | Market Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# ASTRONOMY THEME CSS INJECTION
# ==========================================
st.markdown("""
    <style>
    /* Deep space background and global typography */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a2344 0%, #050711 60%, #000000 100%);
        color: #E0E6ED;
        font-family: 'Inter', sans-serif;
    }
    
    /* Futuristic Floating Cards */
    div.stCard {
        background: rgba(20, 25, 45, 0.4) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(10px);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Typography Overrides */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .accent-text {
        background: -webkit-linear-gradient(45deg, #4F46E5, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Custom divider for sections */
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0));
        margin: 60px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 10 SECTIONS IMPLEMENTATION
# ==========================================

# --- 10. Navigation Bar (Simulated via columns) ---
nav_col1, nav_col2, nav_col3 = st.columns([2, 6, 2])
with nav_col1:
    st.markdown("### 🚀 FASTR")
with nav_col2:
    st.markdown("<div style='text-align: center;'>Products | Pricing | About Us | More ▾</div>", unsafe_allow_html=True)
with nav_col3:
    st.button("Login", use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- 1. Hero Page ---
hero_col1, hero_col2, hero_col3 = st.columns([1, 8, 1])
with hero_col2:
    st.markdown("<h1 style='text-align: center; font-size: 4rem;'><span class='accent-text'>Discover. Research. Execute.</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #94A3B8;'>One connected ecosystem for professional market intelligence.</p>", unsafe_allow_html=True)
    
    # Placeholder for the Pinterest-style market video
    st.markdown("""
        <div style="width: 100%; height: 400px; background: rgba(15, 20, 35, 0.6); border: 1px solid #2d3748; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin-top: 20px;">
            <p style="color: #64748b;">[ Dynamic Market/Space Video Visual Placeholder ]</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 2. Product View / "Our Ecosystem" ---
st.markdown("<h2 style='text-align: center;'>What You Get</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 30px;'>Our ecosystem.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["X-Ray", "Scanner", "Derivatives", "Macro Pulse", "Advanced Charts"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### X-Ray")
        st.markdown("""
        * **360° Diagnostics:** Complete technical & fundamental data in one place.
        * **Momentum & Range:** RS, RM, and 52-week price position.
        * **Candle Intelligence:** Auto-identified single/double formations.
        * **Institutional Activity:** Bulk deals and mutual fund holdings.
        * **Quant Matrix:** Alpha, Beta, CAGR, and volatility metrics.
        """)
    with col2:
        st.image("https://via.placeholder.com/800x450/0f1423/ffffff?text=X-Ray+Dashboard+Screenshot", use_column_width=True)

with tab2:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### Scanner")
        st.markdown("""
        * **Custom Discovery:** Find stocks matching your exact setup.
        * **Technical Filters:** Screen by trend, momentum, and indicators.
        * **Fundamental Overlays:** Combine financial health with price action.
        * **Volume & OI:** Scan based on institutional footprints.
        * **Real-time execution:** Live condition tracking.
        """)
    with col2:
        st.image("https://via.placeholder.com/800x450/0f1423/ffffff?text=Scanner+Dashboard+Screenshot", use_column_width=True)

with tab3:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### Derivatives")
        st.markdown("""
        * **Demat Connectivity:** Link your broker directly.
        * **Advanced Option Chain:** Deep data on calls, puts, and strikes.
        * **Strategy Builder:** Create custom F&O setups.
        * **Algo Creation:** Auto-execute trades when conditions trigger.
        * **Live Portfolio:** Monitor baskets, orders, and current positions.
        """)
    with col2:
        st.image("https://via.placeholder.com/800x450/0f1423/ffffff?text=Derivatives+UI+Screenshot+(Placeholder)", use_column_width=True)

with tab4:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### Macro Pulse")
        st.markdown("""
        * **Global Context:** Track international and domestic indices.
        * **Economic Indicators:** Live PMI, Forex reserves, and more.
        * **Fixed Income:** Monitor bond rates and yields.
        * **Currency & Commodities:** Track global forex and commodity trends.
        * **Volatility Indices:** Understand broader market fear/greed.
        """)
    with col2:
        st.image("https://via.placeholder.com/800x450/0f1423/ffffff?text=Macro+Pulse+Dashboard+Screenshot", use_column_width=True)

with tab5:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### Advanced Charts")
        st.markdown("""
        * **Smart Visuals:** Pro-grade charting integrated with platform data.
        * **Auto-Pattern Marking:** System highlights specific candle formations.
        * **Deep Overlays:** Overlay volume, OI, and technicals.
        * **Seamless Drawing:** Advanced annotation tools.
        * **Actionable Insights:** Chart + intelligence, not just visuals.
        """)
    with col2:
        st.image("https://via.placeholder.com/800x450/0f1423/ffffff?text=Advanced+Charts+Screenshot", use_column_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 3. Why Us ---
st.markdown("<h2 style='text-align: center;'>Why Us</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94A3B8; font-weight: 400;'>Data is everywhere. Clarity isn't.</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>We bring market data, analysis and execution together — without the noise.</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 4. Pricing ---
st.markdown("<h2 style='text-align: center;'>Pricing</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>Choose the duration that fits your schedule. Bundle discounts automatically applied at checkout.</p>", unsafe_allow_html=True)

# Interactive duration selector
duration = st.radio("", ["Monthly", "Quarterly", "Half-yearly", "Annual"], horizontal=True)

p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
with p_col2:
    if duration == "Monthly":
        st.info("Monthly Plan: Full access to all 5 modules. Billed month-to-month.")
    elif duration == "Quarterly":
        st.success("Quarterly Plan: Save 10%. Full access to all modules.")
    elif duration == "Half-yearly":
        st.success("Half-yearly Plan: Save 15%. Full access to all modules.")
    elif duration == "Annual":
        st.success("Annual Plan: Save 25%. Full access. Best value for professionals.")
    st.button("Start Your Journey", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 5. Customer Tie-Up ---
st.markdown("<h2 style='text-align: center;'>Trusted By</h2>", unsafe_allow_html=True)
st.markdown("""
    <div style="display: flex; justify-content: space-around; opacity: 0.6; padding: 20px;">
        <h3>[Logo 1]</h3>
        <h3>[Logo 2]</h3>
        <h3>[Logo 3]</h3>
        <h3>[Logo 4]</h3>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 6. Numbers at a Glance ---
st.markdown("<h2 style='text-align: center;'>By The Numbers</h2>", unsafe_allow_html=True)
n_col1, n_col2, n_col3, n_col4 = st.columns(4)
n_col1.metric("Markets Covered", "Global + Domestic")
n_col2.metric("Stocks Tracked", "5,000+")
n_col3.metric("Data Points", "Millions/sec")
n_col4.metric("Platform Uptime", "99.99%")

st.markdown("<hr>", unsafe_allow_html=True)

# --- 7. Endorsees ---
st.markdown("<h2 style='text-align: center;'>What Professionals Say</h2>", unsafe_allow_html=True)
e_col1, e_col2, e_col3 = st.columns(3)
with e_col1:
    st.markdown("""
        <div class="stCard" style="text-align: center;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: #38BDF8; margin: 0 auto;"></div>
            <br>
            <p>"The X-Ray tool completely changed how I filter noise before placing a trade."</p>
            <strong>- Portfolio Manager</strong>
        </div>
    """, unsafe_allow_html=True)
with e_col2:
    st.markdown("""
        <div class="stCard" style="text-align: center;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: #818CF8; margin: 0 auto;"></div>
            <br>
            <p>"Finally, a platform that connects strategy building directly to algo execution."</p>
            <strong>- Derivatives Trader</strong>
        </div>
    """, unsafe_allow_html=True)
with e_col3:
    st.markdown("""
        <div class="stCard" style="text-align: center;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: #4F46E5; margin: 0 auto;"></div>
            <br>
            <p>"Macro Pulse gives me the global context I need before looking at domestic setups."</p>
            <strong>- Equity Researcher</strong>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 8. Tagline ---
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>See the market. <span class='accent-text'>Think faster.</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #94A3B8; max-width: 800px; margin: 0 auto;'>One connected platform for sharper market intelligence, deeper analysis and more confident execution.</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 9. FAQs ---
st.markdown("<h2 style='text-align: center;'>Frequently Asked Questions</h2>", unsafe_allow_html=True)
faq_col1, faq_col2, faq_col3 = st.columns([1, 6, 1])
with faq_col2:
    with st.expander("Can I link my existing Demat account?"):
        st.write("Yes, the Derivatives module allows you to connect supported broker accounts for seamless live execution and position monitoring.")
    with st.expander("Does X-Ray cover fundamental data or just technicals?"):
        st.write("Both. X-Ray is a 360° diagnostic system covering technical momentum, cash-flow analysis, balance sheets, and institutional ownership.")
    with st.expander("Can the Advanced Charts mark candle patterns automatically?"):
        st.write("Yes, if you select a pattern, the system will scan historical data and highlight exactly where those formations occurred on the chart.")

st.markdown("<hr>", unsafe_allow_html=True)

# --- 10. Last Info / Footer ---
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    st.markdown("### 🚀 FASTR")
    st.markdown("""
    <p style='color: #94A3B8; font-size: 0.9rem;'>
    304, 3rd Floor, Palai Commercial Complex,<br>
    SB Road, Dadar West, Mumbai - 400 028.
    </p>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("#### Products")
    st.markdown("""
    <ul style='list-style: none; padding-left: 0; color: #94A3B8; font-size: 0.9rem;'>
        <li><a style='color: #94A3B8; text-decoration: none;' href='#'>X-Ray</a></li>
        <li><a style='color: #94A3B8; text-decoration: none;' href='#'>Scanner</a></li>
        <li><a style='color: #94A3B8; text-decoration: none;' href='#'>Derivatives</a></li>
        <li><a style='color: #94A3B8; text-decoration: none;' href='#'>Macro Pulse</a></li>
        <li><a style='color: #94A3B8; text-decoration: none;' href='#'>Advanced Charts</a></li>
    </ul>
    """, unsafe_allow_html=True)

with f_col3:
    st.markdown("#### Company")
    st.markdown("""
    <ul style='list-style: none; padding-left: 0; color: #94A3B8; font-size: 0.9rem;'>
        <li>About Us</li>
        <li>Customers</li>
        <li>Why Us</li>
        <li>Login</li>
    </ul>
    """, unsafe_allow_html=True)

with f_col4:
    st.markdown("#### Connect")
    st.markdown("""
    <div style='color: #94A3B8; font-size: 0.9rem;'>
        WhatsApp | Instagram | X<br>
        LinkedIn | Facebook | YouTube<br><br>
        <em>Media Coverage</em>
    </div>
    """, unsafe_allow_html=True)
