import streamlit as st
import tempfile
from vegetabledetection import load_yolonas_process_each_frame  # Assuming vegetabledetection.py is in the same directory

def main():
    # Page Configuration
    st.set_page_config(
        page_title="VegGPT: AI Recipe Generator",
        page_icon="🍴",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Styling
    st.markdown("""
    <style>
        /* Add some custom style */
        h1 {
            color: green;
            text-align: center;
        }
        .reportview-container {
            background: url("https://www.toptal.com/designers/subtlepatterns/patterns/food.png");
        }
        .big-font {
            font-size:50px !important;
        }
        .stButton>button {
            width: 100%;
            height: 50px;
            background-color: #5a9;
            color: white;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #f0e68c;
        }
    </style>
    """, unsafe_allow_html=True)

    # Layout Containers
    header_container = st.container()
    menu_container = st.sidebar.container()

    # Header
    with header_container:
        st.image('https://your-logo-url.com/logo.png', width=200)
        st.title('VegGPT: Your AI Recipe Buddy 🥦🍲🤖')

    # Menu
    with menu_container:
        st.write('## 🌈 Navigation')
        app_mode = st.selectbox("", ["About", "Generate Recipe"])
    
    # About
    if app_mode == "About":
        st.markdown("""
        ## About VegGPT
        **VegGPT** is your go-to AI companion for generating delicious vegetarian recipes! Simply upload an image or a video of your available vegetables, and let VegGPT suggest a recipe for you.
        
        ### Technologies Used
        - **Streamlit**: For the web app
        - **PyTorch**: For the ML models
        - **OpenAI's GPT**: For text generation
        - **OpenCV**: For image processing
        """)
    
    # Generate Recipe
    elif app_mode == "Generate Recipe":
        st.markdown("""
        ## Generate a New Recipe 🍲
        Ready to cook something awesome? Upload a video or an image of the available vegetables, and let VegGPT take care of the rest!
        """)

        # Upload Widget
        st.markdown("### 🎥 Upload Video or Image")
        video_file_buffer = st.file_uploader("", type=["mp4", "jpg", "png", "jpeg"])

        # Button to Generate Recipe
        st.markdown("### 🍲 Let's Cook!")
        generate_button = st.button("Generate Recipe")

        # Frame Placeholder
        stframe = st.empty()

        if generate_button:
            if video_file_buffer is not None:
                tffile = tempfile.NamedTemporaryFile(delete=False) 
                tffile.write(video_file_buffer.read())
                load_yolonas_process_each_frame(tffile.name, stframe)
            else:
                st.warning("Please upload a video or image to proceed.")
    
if __name__ == '__main__':
    main()