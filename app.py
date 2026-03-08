import streamlit as st
import joblib
import re
import html
import unicodedata
import emoji
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

st.set_page_config(page_title="Sentiment Sleuth", page_icon="📦", layout="centered")
