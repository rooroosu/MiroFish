"""
Configuration management
Loads all configuration from the project root .env file
"""

import os
from dotenv import load_dotenv
from .model_routing import (
    OPENROUTER_BASE_URL,
    PRIMARY_QWEN_MODEL,
    BALANCED_QWEN_MODEL,
    REVIEW_QWEN_MODEL,
    is_placeholder_secret,
)

# Load project root .env file
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=False)
else:
    # If no .env at root, try loading from environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON config - disable ASCII escaping so non-ASCII chars display directly (not as \uXXXX)
    JSON_AS_ASCII = False

    # LLM configuration (unified OpenAI format / OpenRouter Qwen-only by default)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', OPENROUTER_BASE_URL)
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', PRIMARY_QWEN_MODEL)
    LLM_BOOST_API_KEY = os.environ.get('LLM_BOOST_API_KEY')
    LLM_BOOST_BASE_URL = os.environ.get('LLM_BOOST_BASE_URL', OPENROUTER_BASE_URL)
    LLM_BOOST_MODEL_NAME = os.environ.get('LLM_BOOST_MODEL_NAME', BALANCED_QWEN_MODEL)
    MIROFISH_REVIEW_MODEL = os.environ.get('MIROFISH_REVIEW_MODEL', REVIEW_QWEN_MODEL)
    MIROFISH_FALLBACK_MODELS = os.environ.get(
        'MIROFISH_FALLBACK_MODELS',
        f'{BALANCED_QWEN_MODEL},qwen/qwen-plus,{PRIMARY_QWEN_MODEL}'
    )
    MIROFISH_MEMORY_BACKEND = os.environ.get('MIROFISH_MEMORY_BACKEND', 'local_qmd')

    # Document translation (Korean -> English on ingest)
    # Sub-agents read source docs many times; KO Hangul fragments hard under BPE
    # (~3x tokens vs English equivalent). Translate once at ingest, cache as
    # `<name>.en.md`, downstream agents read English.
    MIROFISH_AUTO_TRANSLATE = os.environ.get('MIROFISH_AUTO_TRANSLATE', 'true').lower() == 'true'
    MIROFISH_TRANSLATION_MODEL = os.environ.get('MIROFISH_TRANSLATION_MODEL', 'qwen/qwen-turbo')
    MIROFISH_TRANSLATION_KO_THRESHOLD = float(
        os.environ.get('MIROFISH_TRANSLATION_KO_THRESHOLD', '0.05')
    )

    # Zep configuration
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    @classmethod
    def is_local_memory_backend(cls):
        return cls.MIROFISH_MEMORY_BACKEND.lower() in {'local_qmd', 'local', 'local_qmd_backend'}

    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # default overlap size

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if is_placeholder_secret(cls.LLM_API_KEY):
            errors.append("LLM_API_KEY is not configured")
        if not cls.is_local_memory_backend() and is_placeholder_secret(cls.ZEP_API_KEY):
            errors.append("ZEP_API_KEY is not configured")
        return errors
