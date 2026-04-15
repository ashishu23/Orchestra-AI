import asyncio
from typing import Literal

import httpx


class Embedder:
    """
    Async text embedder supporting Google gemini-embedding-001 and
    sentence-transformers as a local fallback.
    """

    def __init__(
        self,
        provider: Literal["google", "sentence-transformers"],
        model: str,
        api_key: str | None = None,
        output_dimensionality: int | None = None,
    ):
        self.provider = provider
        self._model_name = model
        self._api_key = api_key or ""
        self._output_dimensionality = output_dimensionality

        if provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        if self.provider == "google":
            return await self._embed_google(texts)
        elif self.provider == "sentence-transformers":
            return await self._embed_st(texts)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def _embed_google(self, texts: list[str]) -> list[list[float]]:
        # v1 only supports embedContent (singular); batchEmbedContents is v1beta only.
        # Run requests concurrently to avoid serial slowness on large batches.
        model_id = self._model_name.removeprefix("models/")
        url = (
            f"https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model_id}:embedContent"
        )

        body: dict = {
            "content": {"parts": [{"text": ""}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        if self._output_dimensionality is not None:
            body["outputDimensionality"] = self._output_dimensionality

        async def _one(client: httpx.AsyncClient, text: str) -> list[float]:
            try:
                resp = await client.post(
                    url,
                    json={**body, "content": {"parts": [{"text": text}]}},
                    params={"key": self._api_key},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
            except httpx.HTTPStatusError as e:
                # Redact API key from URL before surfacing the error
                safe = str(e).replace(self._api_key, "***")
                raise RuntimeError(f"Google embedding error: {safe}") from None

        async with httpx.AsyncClient(timeout=60.0) as client:
            return list(await asyncio.gather(*[_one(client, t) for t in texts]))

    async def _embed_st(self, texts: list[str]) -> list[list[float]]:
        def _sync():
            vecs = self._st.encode(texts, convert_to_numpy=True)
            return vecs.tolist()

        return await asyncio.to_thread(_sync)
