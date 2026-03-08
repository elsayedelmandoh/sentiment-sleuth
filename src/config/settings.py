from dotenv import load_dotenv
import os

# Load environment variables from project .env (if present)
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "Sentiment Sleuth")
PROJECT_DESCRIPTION = os.getenv(
	"PROJECT_DESCRIPTION", "ML-Powered Amazon Review Sentiment Analysis"
)
