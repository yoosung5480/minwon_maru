import os
from dotenv import load_dotenv

# 루트 디렉토리 기준으로 env 경로 설정
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
