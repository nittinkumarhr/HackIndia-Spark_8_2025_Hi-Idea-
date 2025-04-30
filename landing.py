import streamlit as st

def change_page_to_dashboard():
    """Function to change the page state to dashboard"""
    st.session_state.page = 'dashboard'

def change_page_to_sign():
    """Function to change the page state to sign"""
    st.session_state.page = 'signDrawing'

def landing_page():
    """Landing page for the Air Drawing App"""
    
    
    st.markdown("""
    <style>
        /* Global styles */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        section[data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }
        
        .stApp {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Hero section */
        .hero-title {
            font-size: 3.8rem;
            font-weight: 700;
            background: linear-gradient(to right, #4FACFE, #00F2FE, #4FACFE);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            animation: shine 3s linear infinite, float 6s ease-in-out infinite;
            text-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
            position: relative;
        }
        
        @keyframes shine {
            to {
                background-position: 200% center;
            }
        }
        
        @keyframes float {
            0% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
            100% {
                transform: translateY(0px);
            }
        }
        
        .sparkle {
            position: absolute;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        
        .sparkle span {
            position: absolute;
            width: 7px;
            height: 7px;
            background-color: transparent;
            border-radius: 50%;
        }
        
        .sparkle span::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            transform: scale(0);
            background: radial-gradient(#4FACFE, transparent);
            border-radius: 50%;
            animation: sparkleAnimation 3s infinite;
        }
        
        @keyframes sparkleAnimation {
            0% {
                transform: scale(0);
                opacity: 0;
            }
            20% {
                transform: scale(1);
                opacity: 1;
            }
            100% {
                transform: scale(1.5);
                opacity: 0;
            }
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
            color: rgba(255, 255, 255, 0.8);
            max-width: 800px;
            margin: 0 auto 2rem auto;
        }
        
        /* Buttons */
        .action-button {
            background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
            color: white;
            font-weight: 600;
            padding: 0.8rem 1.2rem;
            border-radius: 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1.2rem;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
            width: 100%;
            display: block;
            text-align: center;
            margin: 0 auto;
        }

        .sign-button {
            background: linear-gradient(90deg, #FF5E62 0%, #FF9966 100%);
            color: white;
            font-weight: 600;
            padding: 0.8rem 1.2rem;
            border-radius: 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1.2rem;
            box-shadow: 0 4px 15px rgba(255, 94, 98, 0.4);
            width: 100%;
            display: block;
            text-align: center;
            margin: 0 auto;
        }
        
        .action-button:hover, .sign-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 20px rgba(79, 172, 254, 0.6);
        }

        .sign-button:hover {
            box-shadow: 0 7px 20px rgba(255, 94, 98, 0.6);
        }
        
        /* Feature cards */
        .feature-section {
            margin: 4rem 0;
        }
        
        .feature-card {
            background: rgba(30, 30, 30, 0.6);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(79, 172, 254, 0.3);
        }
        
        .feature-card h3 {
            color: #4FACFE;
            margin-bottom: 1rem;
            font-size: 1.4rem;
        }
        
        /* How it works */
        .step-card {
            background: rgba(30, 30, 30, 0.6);
            border-radius: 10px;
            padding: 2rem 1.5rem;
            margin-bottom: 1rem;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.1);
            height: 100%;
        }
        
        .step-number {
            position: absolute;
            top: -20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
            color: white;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-weight: bold;
            font-size: 1.2rem;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        }
        
        .step-card h3 {
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
            color: #4FACFE;
        }
        
        /* Section headers */
        .section-header {
            font-size: 2.2rem;
            font-weight: 600;
            margin: 3rem 0 2.5rem;
            text-align: center;
            color: white;
            position: relative;
        }
        
        .section-header::after {
            content: "";
            position: absolute;
            bottom: -15px;
            left: 50%;
            transform: translateX(-50%);
            width: 50px;
            height: 4px;
            background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
            border-radius: 2px;
        }
        
        /* About section */
        .about-content {
            background: rgba(30, 30, 30, 0.6);
            border-radius: 10px;
            padding: 2rem;
            margin: 0 auto;
            max-width: 800px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            line-height: 1.6;
        }
        
        /* Footer */
        .footer {
            margin-top: 5rem;
            text-align: center;
            padding: 2rem 0;
            color: rgba(255, 255, 255, 0.6);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            .hero-subtitle {
                font-size: 1.2rem;
            }
        }

        /* Hide original buttons */
        button[data-testid='baseButton-secondary'] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section with dynamic sparkle effect
    st.markdown("""
    <div style="text-align: center; margin-bottom: 4rem; position: relative; overflow: hidden;">
        <div class="sparkle" id="sparkleContainer"></div>
        <h1 class="hero-title">
            ✨ Air Drawing App
        </h1>
    </div>

    <script>
        // Create sparkles dynamically
        const container = document.getElementById('sparkleContainer');
        for (let i = 0; i < 20; i++) {
            const sparkle = document.createElement('span');
            // Random position
            sparkle.style.left = Math.random() * 100 + '%';
            sparkle.style.top = Math.random() * 100 + '%';
            // Random delay
            sparkle.style.animationDelay = Math.random() * 5 + 's';
            container.appendChild(sparkle);
        }
    </script>
    """, unsafe_allow_html=True)
    
    # Call to action buttons
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
      
        st.button("Start Drawing", on_click=change_page_to_dashboard, key="start_drawing_btn", use_container_width=True, type="secondary")
    
    with col3:
        st.button("Sign Drawing", on_click=change_page_to_sign, key="sign_drawing_btn", use_container_width=True, type="secondary")
    
    # Features section
    st.markdown("<h2 class='section-header'>✨ Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Virtual Drawing</h3>
            <p>Draw in the air using any colored object. Perfect for presentations, teaching, and creative expression.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 AI Analysis</h3>
            <p>Get your drawings analyzed by AI for mathematical solutions, DSA explanations, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🖌️ Multiple Colors</h3>
            <p>Choose from blue, green, red, and yellow to create vibrant and expressive drawings.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("<h2 class='section-header'>🚀 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <h3>Allow Camera Access</h3>
            <p>Grant permission to your webcam when prompted.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <h3>Use a Colored Object</h3>
            <p>Hold a bright colored object (like a marker cap or sticky note) in front of the camera.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <h3>Start Drawing!</h3>
            <p>Move your colored object to draw on the virtual canvas.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About section
    st.markdown("<h2 class='section-header'>💡 About</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-content">
        <p>
            The Air Drawing App uses computer vision to track colored objects through your webcam. 
            This project combines OpenCV for computer vision, Streamlit for the user interface, 
            and Gemini AI for drawing analysis. Perfect for teachers, students, and anyone who wants 
            to explain concepts visually.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Made with ❤️ by Air Drawing Team | © 2025</p>
    </div>
    """, unsafe_allow_html=True)