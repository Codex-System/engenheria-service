import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

class Settings:
    # Centraliza todas as chaves e configurações em uma classe
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()