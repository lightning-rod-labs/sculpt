import time
from functools import partial
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import openai
from .base import _BaseSculptor
from .utils import DEFAULT_SYSTEM_PROMPT


class Sculptor(_BaseSculptor):
    """
    Synchronous Sculptor that uses a synchronous OpenAI client.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
        model: str = "gpt-4o-mini",
        openai_client: Optional[openai.OpenAI] = None,
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
        if openai_client:
            self.openai_client = openai_client
        else:
            self.openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_client_base_url(self) -> str:
        return self.openai_client.base_url

    def sculpt(self, data: Dict[str, Any], merge_input: bool = True, retries: int = 3) -> Dict[str, Any]:
        """Processes a single data item using the LLM synchronously."""
        last_error = None
        for attempt in range(retries):
            params = self._build_request_params(data, attempt)
            try:
                resp = self.openai_client.chat.completions.create(**params)
                return self._parse_response(resp, data, merge_input)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        raise RuntimeError(f"LLM API call failed after {retries} attempts. Last error: {last_error}")

    def sculpt_batch(
        self,
        data_list: List[Dict[str, Any]],
        n_workers: int = 1,
        show_progress: bool = True,
        merge_input: bool = True,
        retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Processes multiple data items using the LLM, with optional parallelization.

        Args:
            data_list: List of data dictionaries to process
            n_workers: Number of parallel workers (default: 1). If > 1, enables parallel processing
            show_progress: Whether to show a progress bar (default: True)
            merge_input: If True, merges input data with extracted fields (default: True)
            retries: Number of times to retry failed attempts (default: 3)
        """

        if hasattr(data_list, "to_dict"):
            data_list = data_list.to_dict("records")
        # Create a partial function with fixed merge_input parameter
        sculpt_with_merge = partial(self.sculpt, merge_input=merge_input, retries=retries)

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                if show_progress:
                    results = list(
                        tqdm(
                            executor.map(sculpt_with_merge, data_list),
                            total=len(data_list),
                            desc="Processing items"
                        )
                    )
                else:
                    results = list(executor.map(sculpt_with_merge, data_list))
        else:
            results = []
            iterator = tqdm(data_list, desc="Processing items") if show_progress else data_list
            for item in iterator:
                results.append(sculpt_with_merge(item))

        return results

