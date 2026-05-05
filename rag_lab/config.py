"""Typed settings, model factories, logging, and LangSmith env wiring."""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault(
    "USER_AGENT",
    "rag-agent-lab/1.0 (Agentic RAG strategy comparison)",
)

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: SecretStr | None = None

    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "rag-agent-tutorial"

    ollama_base_url: str = "http://localhost:11434"

    embedding_provider: str = Field(
        default="ollama",
        description="ollama | google",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("OLLAMA_EMBED_MODEL", "EMBEDDING_MODEL"),
    )
    google_embed_model: str = Field(
        default="models/gemini-embedding-001",
        validation_alias=AliasChoices("GOOGLE_EMBEDDING_MODEL"),
    )

    devto_username: str = "fortune-ndlovu"

    chroma_dir: str = "./.chroma"
    chroma_collection: str = "fortune_devto"
    sqlite_threads_path: str = "./.threads.sqlite"

    chat_provider: str = Field(default="google", description="google | ollama")
    chat_model: str = "google_genai:gemini-2.5-flash-lite"

    judge_provider: str = Field(default="google", description="google | ollama")
    judge_model: str = "google_genai:gemini-2.5-flash"

    eval_dataset_name: str = "fortune-devto-qa-v1"
    devto_listing_per_page: int = 1000

    @field_validator("langsmith_tracing", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return True
        s = str(v).lower().strip()
        return s in ("1", "true", "yes", "on")

    @model_validator(mode="after")
    def _validate_google_and_export_key(self) -> Settings:
        cp = self.chat_provider.lower().strip()
        jp = self.judge_provider.lower().strip()
        ep = self.embedding_provider.lower().strip()

        needs_google = cp == "google" or jp == "google" or ep == "google"
        if needs_google and self.google_api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY is required when CHAT_PROVIDER, JUDGE_PROVIDER, or "
                "EMBEDDING_PROVIDER is google (or chat_model is a google_genai:* id)."
            )
        if self.google_api_key is not None:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key.get_secret_value()
        return self


def setup_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def setup_tracing(settings: Settings) -> None:
    if not settings.langsmith_api_key:
        _log.info("LangSmith disabled (no LANGSMITH_API_KEY).")
        return
    key = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    _log.info(
        "LangSmith tracing enabled → https://smith.langchain.com (project=%s)",
        settings.langsmith_project,
    )


def get_embeddings(settings: Settings):
    ep = settings.embedding_provider.lower().strip()
    if ep == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    if ep == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.google_embed_model)
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        "Use ollama or google."
    )


def get_vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(settings),
        persist_directory=settings.chroma_dir,
    )


def _init_chat_for_provider(
    *,
    model_id: str,
    provider: str,
    ollama_base_url: str,
) -> Any:
    p = provider.lower().strip()
    if p == "ollama":
        return init_chat_model(
            model_id,
            model_provider="ollama",
            base_url=ollama_base_url,
        )
    if p == "google":
        mid = model_id.strip()
        if mid.startswith("google_genai:") or mid.startswith("google_vertex_ai:"):
            return init_chat_model(mid)
        return init_chat_model(f"google_genai:{mid}")
    raise ValueError(
        f"Unsupported chat provider {provider!r}. Use google or ollama."
    )


def get_chat_model(settings: Settings):
    return _init_chat_for_provider(
        model_id=settings.chat_model,
        provider=settings.chat_provider,
        ollama_base_url=settings.ollama_base_url,
    )


def get_judge_model(settings: Settings):
    return _init_chat_for_provider(
        model_id=settings.judge_model,
        provider=settings.judge_provider,
        ollama_base_url=settings.ollama_base_url,
    )


def load_settings() -> Settings:
    return Settings()
