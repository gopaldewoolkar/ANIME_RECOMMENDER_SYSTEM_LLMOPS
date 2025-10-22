import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

st.set_page_config(page_title="Anime Recommender", layout="wide")
load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()

pipeline = init_pipeline()

st.title("🎌 Anime Recommender System")

query = st.text_input("Enter your anime preferences (e.g., 'light-hearted anime with school settings')")

if query:
    with st.spinner("Fetching recommendations for you..."):
        try:
            response = pipeline.recommend(query)
            st.markdown("### Recommendations")
            st.write(response)
        except Exception as e:
            st.error(f" Failed to fetch recommendations: {str(e)}")
