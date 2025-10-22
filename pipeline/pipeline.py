from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self, persist_dir="chroma_db"):
        try:
            logger.info("Initializing Recommendation Pipeline")

            vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir)
            retriever = vector_builder.load_vector_store().as_retriever()
            logger.info("-- Retriever initialized successfully --")

            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)
            logger.info("Pipeline initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize pipeline")
            raise CustomException("Error during pipeline initialization", e)

    def recommend(self, query: str) -> str:
        try:
            logger.info(f"Received query: {query}")
            recommendation = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated successfully")
            return recommendation

        except Exception as e:
            logger.exception("Failed to get recommendation")
            raise CustomException("Error during recommendation generation", e)
