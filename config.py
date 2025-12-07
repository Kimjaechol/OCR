"""
Legal Document OCR - Configuration Module
==========================================
Centralized configuration management using Pydantic Settings
"""

import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # ===========================================
    # Redis Configuration
    # ===========================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery"
    )

    # ===========================================
    # API Keys
    # ===========================================
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for LLM correction"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (production)"
    )

    # ===========================================
    # Model Paths
    # ===========================================
    got_model_path: str = Field(
        default="./weights/GOT-OCR2_0",
        description="Path to GOT-OCR model"
    )
    yolo_model_path: str = Field(
        default="./weights/yolo_table_best.pt",
        description="Path to YOLO table detection model"
    )

    # ===========================================
    # Server Configuration
    # ===========================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # ===========================================
    # Processing Configuration
    # ===========================================
    max_pages_per_batch: int = Field(
        default=50,
        description="Maximum pages to process in a single batch"
    )
    max_file_size_mb: int = Field(
        default=500,
        description="Maximum upload file size in MB"
    )
    concurrent_workers: int = Field(
        default=1,
        description="Number of concurrent Celery workers"
    )
    gpu_memory_fraction: float = Field(
        default=0.8,
        description="Fraction of GPU memory to use"
    )

    # ===========================================
    # OCR Configuration
    # ===========================================
    ocr_dpi: int = Field(
        default=300,
        description="DPI for PDF to image conversion"
    )
    ocr_confidence_threshold: float = Field(
        default=0.4,
        description="Minimum confidence for table detection"
    )
    enable_gemini_correction: bool = Field(
        default=True,
        description="Enable Gemini LLM correction"
    )

    # ===========================================
    # Storage Configuration
    # ===========================================
    upload_dir: str = Field(default="./temp_uploads")
    output_dir: str = Field(default="./outputs")
    log_dir: str = Field(default="./logs")

    # ===========================================
    # Security
    # ===========================================
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",  # Vite dev server
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
        ]
    )

    # ===========================================
    # Pricing
    # ===========================================
    price_per_page: int = Field(
        default=50,
        description="Price per page in KRW"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create directories on module load
def init_directories():
    """Initialize required directories"""
    settings = get_settings()
    for dir_path in [settings.upload_dir, settings.output_dir, settings.log_dir]:
        os.makedirs(dir_path, exist_ok=True)


# Initialize directories
init_directories()
