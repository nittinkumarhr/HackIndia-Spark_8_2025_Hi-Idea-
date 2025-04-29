import streamlit as st
from landing import landing_page
import main2

# Set Page Configurations (MUST BE FIRST)
st.set_page_config(
    page_title="Air Drawing App",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom right, #1e1b4b, #4c1d95, #1e1b4b);
        color: white;
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 9999px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #34d399;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2);
    }
    .feature-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s;
    }
    .feature-card:hover {
        background-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-5px);
    }
    .step-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .step-number {
        width: 4rem;
        height: 4rem;
        border-radius: 9999px;
        background: linear-gradient(to right, #8b5cf6, #6366f1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .tool-button {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 0.375rem;
        padding: 0.5rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
    }
    .tool-button:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .tool-button.active {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .canvas-container {
        background-color: #1f2937;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        position: relative;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
        margin-top: 2rem;
    }
    
    /* Animation keyframes */
    @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    
    @keyframes draw {
        0% { stroke-dashoffset: 200; }
        100% { stroke-dashoffset: 0; }
    }
    
    @keyframes ping {
        0% { transform: scale(1); opacity: 1; }
        75%, 100% { transform: scale(2); opacity: 0; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🔹 Initialize session state variables to prevent AttributeError
def initialize_session_state():
    """Initialize or reset session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    if 'calibrated' not in st.session_state:
        st.session_state.calibrated = False  # Default value
    
    # Add a new state to track whether to return to landing page
    if 'return_to_landing' not in st.session_state:
        st.session_state.return_to_landing = False

# Page Routing Function
def route_pages():
    """Handle page routing based on session state."""
    # Check if we need to return to landing page
    if st.session_state.return_to_landing:
        st.session_state.page = 'landing'
        st.session_state.return_to_landing = False
    
    # Route to appropriate page
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'dashboard':
        # Modify main2.run_streamlit_app to support back button
        main2.run_streamlit_app(return_callback=return_to_landing)
    elif st.session_state.page =='signDrawing':
        # Modify main2.run_streamlit_app to support back button
        main2.run_streamlit_app(return_callback=return_to_landing)

# Callback function to return to landing page
def return_to_landing():
    """Set the flag to return to landing page."""
    st.session_state.return_to_landing = True
    st.rerun()

# Main execution
def main():
    # Initialize session state
    initialize_session_state()
    
    # Route pages
    route_pages()

# Run the main function
if __name__ == "__main__":
    main()