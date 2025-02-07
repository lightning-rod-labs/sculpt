from .base import _BaseSculptor
from .utils import DEFAULT_SYSTEM_PROMPT
from typing import Dict, Any, Optional, List
import openai
import asyncio
from tqdm import tqdm


class AsyncSculptor(_BaseSculptor):
    """
    Asynchronous Sculptor that uses an asynchronous OpenAI client.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
        model: str = "gpt-4o-mini",
        async_openai_client: Optional[openai.AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        instructions: Optional[str] = "",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        template: Optional[str] = "",
        input_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            schema=schema,
            model=model,
            instructions=instructions,
            system_prompt=system_prompt,
            template=template,
            input_keys=input_keys,
        )
        if async_openai_client:
            self.async_openai_client = async_openai_client
        else:
            self.async_openai_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def get_client_base_url(self) -> str:
        return self.async_openai_client.base_url

    async def sculpt(self, data: Dict[str, Any], merge_input: bool = True, retries: int = 3) -> Dict[str, Any]:
        """Processes a single data item asynchronously using the LLM."""
        last_error = None
        for attempt in range(retries):
            params = self._build_request_params(data, attempt)
            try:
                resp = await self.async_openai_client.chat.completions.create(**params)
                return self._parse_response(resp, data, merge_input)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                continue
        raise RuntimeError(f"LLM API call failed after {retries} attempts. Last error: {last_error}")

    async def sculpt_batch(
        self,
        data_list: List[Dict[str, Any]],
        show_progress: bool = True,
        merge_input: bool = True,
        retries: int = 3,
        return_exceptions: bool = True,
        include_exceptions_in_results: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Processes multiple data items asynchronously,
        leveraging asyncio.gather (and optionally tqdm for progress).
        """
        if hasattr(data_list, "to_dict"):
            data_list = data_list.to_dict("records")
        tasks = [
            self.sculpt(item, merge_input=merge_input, retries=retries)
            for item in data_list
        ]
        if show_progress:
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing items"):
                results.append(await coro)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        if include_exceptions_in_results:
            return results
        else:
            return [r for r in results if not isinstance(r, Exception)]
