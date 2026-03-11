from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr

from app.config import Settings
from app.rag.prompt import HR_PROMPT
from app.rag.router import (
    QueryRoute,
    combined_relevance_score,
    detect_policy_hints,
    is_low_signal_heading,
    lexical_overlap_ratio,
    route_question,
    tokenize,
)


class VectorStoreProtocol(Protocol):
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float]]:
        ...

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float]]:
        ...


class LLMProtocol(Protocol):
    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        ...


DocScorePair: TypeAlias = tuple[Any, float]


@dataclass
class RetrievalConfig:
    top_k: int = 6
    lambda_mult: float = 0.35
    score_threshold: float = 1.6
    min_contexts: int = 3
    fetch_multiplier: int = 8


class RAGService:
    def __init__(self, settings: Settings, config: RetrievalConfig | None = None):
        self.settings = settings
        self.config = config or RetrievalConfig()
        self._embedding: Any | None = None
        self._vectorstore: VectorStoreProtocol | None = None
        self._llm: LLMProtocol | None = None

    @property
    def is_ready(self) -> bool:
        return self._embedding is not None and self._vectorstore is not None and self._llm is not None

    def _ensure_ready(self) -> None:
        if self.is_ready:
            return

        embedding = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=SecretStr(self.settings.google_api_key),
        )
        vectorstore = PGVector(
            embeddings=embedding,
            collection_name=self.settings.collection_name,
            connection=self.settings.connection_string,
            use_jsonb=True,
        )
        llm = ChatGoogleGenerativeAI(
            model=self.settings.generation_model,
            api_key=SecretStr(self.settings.google_api_key),
            temperature=0.15,
            max_tokens=2048,
        )
        self._embedding = embedding
        self._vectorstore = vectorstore
        self._llm = llm

    def _get_vectorstore(self) -> VectorStoreProtocol:
        self._ensure_ready()
        if self._vectorstore is None:
            raise RuntimeError("Vector store is not initialized.")
        return self._vectorstore

    def _get_llm(self) -> LLMProtocol:
        self._ensure_ready()
        if self._llm is None:
            raise RuntimeError("LLM is not initialized.")
        return self._llm

    @staticmethod
    def _dedupe_pairs(pairs: list[DocScorePair]) -> list[DocScorePair]:
        deduped: list[DocScorePair] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()

        for doc, score in pairs:
            metadata = getattr(doc, "metadata", {}) or {}
            chunk_id = str(metadata.get("chunk_id", "")).strip()
            text_key = doc.page_content.strip()[:200]

            if chunk_id and chunk_id in seen_ids:
                continue
            if text_key in seen_texts:
                continue

            if chunk_id:
                seen_ids.add(chunk_id)
            seen_texts.add(text_key)
            deduped.append((doc, score))

        return deduped

    @staticmethod
    def _normalize_markdown_text(text: str) -> str:
        return " ".join(text.replace("*", " ").replace("#", " ").split()).strip().lower()

    def _is_low_signal_pair(self, doc: Any, question_tokens: set[str]) -> bool:
        text = doc.page_content.strip()
        if not text:
            return True

        heading = str(doc.metadata.get("heading", "")).strip()
        normalized_text = self._normalize_markdown_text(text)
        normalized_heading = self._normalize_markdown_text(heading)
        word_count = len(text.replace("\n", " ").split())

        content_overlap = lexical_overlap_ratio(question_tokens, text)
        heading_overlap = lexical_overlap_ratio(question_tokens, heading)
        overlap_signal = content_overlap + heading_overlap

        # Drop metadata-like chunks where body is effectively just the title.
        if normalized_heading and word_count <= 14 and normalized_text == normalized_heading:
            return True

        # Drop very short chunks with weak lexical evidence for the user question.
        if word_count < 18 and overlap_signal < 0.18:
            return True

        # Drop generic sections unless they have strong overlap with the query.
        if is_low_signal_heading(heading) and word_count < 80 and content_overlap < 0.12:
            return True

        return False

    def _apply_precision_filter(
        self,
        pairs: list[DocScorePair],
        question_tokens: set[str],
        min_keep: int,
    ) -> list[DocScorePair]:
        filtered = [pair for pair in pairs if not self._is_low_signal_pair(pair[0], question_tokens)]
        if len(filtered) >= min_keep:
            return filtered
        return pairs

    def _retrieve_policy_hint_pairs(self, question: str, policies: list[str]) -> list[DocScorePair]:
        vectorstore = self._get_vectorstore()
        pairs: list[DocScorePair] = []
        for policy_name in policies:
            pairs.extend(
                vectorstore.max_marginal_relevance_search_with_score(
                    query=question,
                    k=2,
                    fetch_k=8,
                    lambda_mult=0.2,
                    filter={"filename": policy_name},
                )
            )
        return pairs

    def _retrieve_route_pairs(self, question: str, route: QueryRoute) -> list[DocScorePair]:
        vectorstore = self._get_vectorstore()
        pairs: list[DocScorePair] = []
        for policy_name in route.policies:
            pairs.extend(
                vectorstore.similarity_search_with_score(
                    question,
                    k=6,
                    filter={"filename": policy_name},
                )
            )
            pairs.extend(
                vectorstore.max_marginal_relevance_search_with_score(
                    query=question,
                    k=3,
                    fetch_k=10,
                    lambda_mult=0.25,
                    filter={"filename": policy_name},
                )
            )
        return pairs

    def _retrieve_pairs(self, question: str, top_k_override: int | None = None) -> tuple[list[DocScorePair], str | None]:
        vectorstore = self._get_vectorstore()

        effective_top_k = top_k_override or self.config.top_k
        route = route_question(question)

        if route is not None:
            question_tokens = tokenize(question)
            routed_pairs = self._dedupe_pairs(self._retrieve_route_pairs(question, route))
            routed_pairs = [pair for pair in routed_pairs if pair[1] <= (self.config.score_threshold + 0.15)] or routed_pairs

            ranked_routed: list[tuple[Any, float, float]] = []
            for doc, score in routed_pairs:
                text_lower = doc.page_content.lower()
                heading = str(doc.metadata.get("heading", "")).strip().lower()
                full_text = f"{heading}\n{text_lower}"

                content_overlap = lexical_overlap_ratio(question_tokens, doc.page_content)
                heading_overlap = lexical_overlap_ratio(question_tokens, heading)
                required_hits = sum(1 for term in route.required_terms if term in full_text)
                negative_hits = sum(1 for term in route.negative_terms if term in full_text)

                if required_hits == 0 and (content_overlap + heading_overlap) < 0.08:
                    continue

                blended = combined_relevance_score(question_tokens, doc.page_content, heading, score)
                rerank_score = blended - 0.10 * required_hits + 0.14 * negative_hits
                ranked_routed.append((doc, score, rerank_score))

            if not ranked_routed:
                ranked_routed = [(doc, score, score) for doc, score in routed_pairs]

            ranked_routed.sort(key=lambda item: item[2])
            route_top_k = top_k_override or route.top_k
            selected_pairs = [(doc, score) for doc, score, _ in ranked_routed[:route_top_k]]
            selected_pairs = self._apply_precision_filter(
                selected_pairs,
                question_tokens=question_tokens,
                min_keep=min(route.min_contexts, route_top_k),
            )

            if len(selected_pairs) < route.min_contexts:
                global_pairs = vectorstore.similarity_search_with_score(
                    question,
                    k=max(effective_top_k, route.min_contexts * 2),
                )
                merged_pairs = self._dedupe_pairs(selected_pairs + global_pairs)
                merged_pairs = self._apply_precision_filter(
                    merged_pairs,
                    question_tokens=question_tokens,
                    min_keep=route.min_contexts,
                )
                selected_pairs = merged_pairs[: max(route.min_contexts, route_top_k)]

            return selected_pairs, route.name

        raw_mmr = vectorstore.max_marginal_relevance_search_with_score(
            query=question,
            k=effective_top_k,
            lambda_mult=self.config.lambda_mult,
            fetch_k=max(effective_top_k * self.config.fetch_multiplier, effective_top_k + 8),
        )
        filtered_mmr = [(doc, score) for doc, score in raw_mmr if score <= self.config.score_threshold]

        hint_policies = detect_policy_hints(question)
        hint_policy_set = set(hint_policies)
        hint_pairs = self._retrieve_policy_hint_pairs(question, hint_policies) if hint_policies else []

        if hint_policy_set:
            filtered_mmr = [pair for pair in filtered_mmr if pair[0].metadata.get("filename") in hint_policy_set]

        selected_pairs = self._dedupe_pairs(hint_pairs + filtered_mmr)

        if len(selected_pairs) < self.config.min_contexts and hint_policy_set:
            hint_backfill: list[DocScorePair] = []
            for policy_name in hint_policies:
                hint_backfill.extend(
                    vectorstore.similarity_search_with_score(
                        question,
                        k=2,
                        filter={"filename": policy_name},
                    )
                )
            selected_pairs = self._dedupe_pairs(selected_pairs + hint_backfill)

        if len(selected_pairs) < self.config.min_contexts:
            raw_sim = vectorstore.similarity_search_with_score(
                question,
                k=max(effective_top_k, self.config.min_contexts * 2),
            )
            selected_pairs = self._dedupe_pairs(selected_pairs + raw_sim)

        if len(selected_pairs) < self.config.min_contexts:
            selected_pairs = self._dedupe_pairs(raw_mmr)[: self.config.min_contexts]

        question_tokens = tokenize(question)
        ranked_pairs: list[tuple[Any, float, float]] = []
        for doc, score in selected_pairs:
            heading = str(doc.metadata.get("heading", "")).strip()
            blended = combined_relevance_score(question_tokens, doc.page_content, heading, score)
            ranked_pairs.append((doc, score, blended))

        if not ranked_pairs:
            ranked_pairs = [(doc, score, score) for doc, score in selected_pairs]

        ranked_pairs.sort(key=lambda item: item[2])

        if hint_policy_set:
            hinted_pairs = [pair for pair in ranked_pairs if pair[0].metadata.get("filename") in hint_policy_set]
            other_pairs = [pair for pair in ranked_pairs if pair[0].metadata.get("filename") not in hint_policy_set]
            hint_quota = min(4, len(hinted_pairs))
            selected_ranked = hinted_pairs[:hint_quota] + other_pairs[: max(0, effective_top_k - hint_quota)]
            selected_pairs = [(doc, score) for doc, score, _ in selected_ranked]
            if len(selected_pairs) < effective_top_k:
                selected_pairs.extend(
                    [
                        (doc, score)
                        for doc, score, _ in hinted_pairs[
                            hint_quota : hint_quota + (effective_top_k - len(selected_pairs))
                        ]
                    ]
                )
        else:
            selected_pairs = [(doc, score) for doc, score, _ in ranked_pairs[:effective_top_k]]

        selected_pairs = self._apply_precision_filter(
            selected_pairs,
            question_tokens=question_tokens,
            min_keep=max(1, min(self.config.min_contexts, effective_top_k)),
        )[:effective_top_k]

        return selected_pairs, None

    @staticmethod
    def _to_source_item(index: int, doc: Any, score: float) -> dict[str, Any]:
        metadata = getattr(doc, "metadata", {}) or {}
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 280:
            preview = preview[:280] + "..."
        return {
            "index": index,
            "filename": metadata.get("filename", "Unknown"),
            "heading": metadata.get("heading", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "score": round(float(score), 4),
            "preview": preview,
        }

    def retrieve_debug(self, question: str, top_k_override: int | None = None) -> dict[str, Any]:
        t0 = time.time()
        pairs, route_name = self._retrieve_pairs(question, top_k_override=top_k_override)
        retrieval_time = time.time() - t0

        return {
            "route": route_name,
            "sources": [self._to_source_item(i, doc, score) for i, (doc, score) in enumerate(pairs, 1)],
            "timings": {"retrieval": round(retrieval_time, 3)},
        }

    def ask(self, question: str, include_sources: bool = True, top_k_override: int | None = None) -> dict[str, Any]:
        llm = self._get_llm()
        t_start = time.time()

        t_retrieval_start = time.time()
        pairs, route_name = self._retrieve_pairs(question, top_k_override=top_k_override)
        t_retrieval = time.time() - t_retrieval_start

        contexts = [doc.page_content.strip() for doc, _ in pairs if doc.page_content.strip()]
        context_text = "\n\n".join(f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))
        prompt = HR_PROMPT.format(context=context_text, question=question)

        t_generation_start = time.time()
        response = llm.invoke(prompt)
        t_generation = time.time() - t_generation_start
        answer = response.content if hasattr(response, "content") else str(response)

        sources = []
        if include_sources:
            sources = [self._to_source_item(i, doc, score) for i, (doc, score) in enumerate(pairs, 1)]

        return {
            "answer": answer,
            "route": route_name,
            "timings": {
                "retrieval": round(t_retrieval, 3),
                "generation": round(t_generation, 3),
                "total": round(time.time() - t_start, 3),
            },
            "sources": sources,
        }
