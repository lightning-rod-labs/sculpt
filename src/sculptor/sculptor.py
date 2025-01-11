import json
from typing import Dict, Any, Optional, List, Type, Union
from .utils import load_config
import openai
from string import Template

ALLOWED_TYPES = {
    "string": str,
    "number": float,
    "boolean": bool,
    "integer": int,
    "object": dict,
    "array": list,
    "enum": str,
    "anyOf": str
}

DEFAULT_INSTRUCTIONS = "Extract the following fields from the following data according to the provided schema. Follow the description and examples in the schema carefully."

DEFAULT_SYSTEM_PROMPT = "You are an AI extracting information into JSON format."

class Sculptor:
    """
    Extracts structured data from text using large language models (LLMs).
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
        """
        Initializes the Sculptor for LLM interaction and data extraction.

        Args:
            schema (Optional[Dict[str, Dict[str, Any]]]): Fields to extract, types, and descriptions.
            model (str): LLM model to use (default: "gpt-4o-mini").
            openai_client (Optional[openai.OpenAI]): OpenAI client (default: creates one with OPENAI_API_KEY or api_key).
            api_key (Optional[str]): OpenAI API key (default: uses OPENAI_API_KEY environment variable).
            base_url (Optional[str]): Base URL for the OpenAI API (default: OpenAI default).
            instructions (Optional[str]): Instructions prepended to the prompt (default: "").
            system_prompt (Optional[str]): System prompt for the LLM (default: DEFAULT_SYSTEM_PROMPT).
            template (Optional[str]): Template for formatting input data in the prompt (default: "").
            input_keys (Optional[List[str]]): Keys to include if no template is provided (default: None).
        """
        self.model = model
        
        if openai_client:
            self.openai_client = openai_client
        else:
            self.openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.instructions = instructions.strip()
        self.system_prompt = system_prompt
        self.template = template.strip()
        self.input_keys = input_keys
        self.schema: Dict[str, Dict[str, Any]] = {}

        # Load schema if provided
        if schema:
            self._load_schema(schema)

    def _load_schema(self, schema: Dict[str, Dict[str, Any]]):
        """Loads the schema, validating the types and structure."""
        for field_name, field_data in schema.items():
            field_type = field_data.get("type")
            description = field_data.get("description", "")
            items = field_data.get("items")
            enum = field_data.get("enum")

            self.add(
                name=field_name,
                field_type=field_type,
                description=description,
                items=items,
                enum=enum
            )

    def add(
        self,
        name: str,
        field_type: Union[str, Type] = str,
        description: str = "",
        items: Optional[Union[str, Type]] = None,
        enum: Optional[List[Any]] = None
    ):
        """
        Adds a field to the extraction schema.

        Args:
            name (str): The name of the field.
            field_type (Union[str, Type]): The type of the field (default: str).
            description (str): A description of the field (default: "").
            items (Optional[Union[str, Type]]): The type of items in the array, if field_type is list (default: None).
            enum (Optional[List[Any]]): Allowed values if field_type is "enum" (default: None).
        """

        if isinstance(field_type, str):
            field_type = field_type.lower()

        if field_type not in ALLOWED_TYPES and field_type not in ALLOWED_TYPES.values():
            allowed_types_str = ", ".join(
                [f"`{t}`" for t in ALLOWED_TYPES]
            )
            raise ValueError(
                f"Invalid type '{field_type}'. Allowed types are: {allowed_types_str}"
            )

        if field_type == list or field_type == "array":
            if items is None:
                raise ValueError(
                    "For 'array' type, 'items' must specify the type of items in the array."
                )
            if isinstance(items, str) and items not in ALLOWED_TYPES:
                raise ValueError(
                    f"Invalid items type '{items}'. Allowed item types are: {', '.join(ALLOWED_TYPES.keys())}"
                )
            if not isinstance(items, str) and items not in ALLOWED_TYPES.values():
                raise ValueError(
                    f"Invalid items type. Allowed item types are: {', '.join([t.__name__ for t in ALLOWED_TYPES.values()])}"
                )

        if field_type == "enum" and enum is None:
            raise ValueError("For 'enum' type, 'enum' must specify a list of allowed values.")

        self.schema[name] = {
            "type": field_type,
            "description": description,
            "items": items,
            "enum": enum,
        }

    @classmethod
    def from_config(cls, filepath: str) -> "Sculptor":
        """Creates a Sculptor instance from a config file (JSON or YAML)."""
        config = load_config(filepath)
        return cls(**config)

    def _build_schema_for_llm(self) -> Dict[str, Any]:
        """Builds a JSON schema for the LLM during inference."""
        properties = {}
        for field_name, meta in self.schema.items():
            field_type = meta["type"]

            if isinstance(field_type, type):
                field_type = [k for k, v in ALLOWED_TYPES.items() if v == field_type][0]

            # Add null type for numeric fields to avoid defaulting to 0
            needs_null = field_type in ["number", "integer"]

            if field_type == "array":
                item_type = meta["items"]
                if isinstance(item_type, type):
                    item_type = [k for k, v in ALLOWED_TYPES.items() if v == item_type][0]

                if item_type == "enum":
                    properties[field_name] = {
                        "type": "array",
                        "items": {"type": "string", "enum": meta["enum"]},
                    }
                else:
                    properties[field_name] = {
                        "type": "array",
                        "items": {"type": item_type},
                    }
            elif field_type == "enum":
                properties[field_name] = {"type": "string", "enum": meta["enum"]}
            elif field_type == "anyOf":
                properties[field_name] = {"anyOf": meta["anyOf"]}
            else:
                properties[field_name] = {
                    "type": [field_type, "null"] if needs_null else field_type
                }

        return {
            "name": "extract_fields",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(self.schema.keys()),
                "additionalProperties": False,
            },
        }

    def _format_input_data(self, data: Dict[str, Any]) -> str:
        """Formats the input data according to template or keys."""
        if self.template:
            # Convert values to strings and handle None values
            safe_data = {k: str(v) if v is not None else '' for k, v in data.items()}
            try:
                return self.template.format(**safe_data)
            except KeyError as e:
                raise KeyError(f"Template key {e} not found in provided data")
        
        # Use input_keys if provided, otherwise use all data keys
        keys_to_use = self.input_keys if self.input_keys else data.keys()
        return "\n".join(f"{k}: {data.get(k, '')}" for k in keys_to_use)

    def _build_user_message(self, data: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """Constructs the user message for the LLM prompt."""
        message_parts = [
            f"INSTRUCTIONS \n```{self.instructions}```",
            f"INPUT \n```{self._format_input_data(data)}```",
            f"SCHEMA \n```{json.dumps(schema["schema"], indent=2)}```",
        ]
        
        return "\n\n".join(message_parts)

    def sculpt(self, data: Dict[str, Any], merge_input: bool = True) -> Dict[str, Any]:
        """Processes a single data item using the LLM."""
        schema_for_llm = self._build_schema_for_llm()
        
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_message(data, schema_for_llm)},
                ],
                response_format={"type": "json_schema", "json_schema": schema_for_llm},
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()
            extracted = json.loads(content)
            if isinstance(extracted, list) and len(extracted) == 1:
                extracted = extracted[0]  # Some models wrap the output in a list
            
            # Clean up any whitespace in keys
            extracted = {k.strip(): v for k, v in extracted.items()}
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        if not merge_input:
            return extracted
        
        # Check for field conflicts
        conflicts = set(data.keys()) & set(extracted.keys())
        if conflicts:
            import warnings
            warnings.warn(f"The following fields will be overwritten: {conflicts}")
        
        # Merge while giving priority to extracted fields
        return {**data, **extracted}

    def sculpt_batch(
        self,
        data_list: List[Dict[str, Any]],
        n_workers: int = 1,
        show_progress: bool = True,
        merge_input: bool = True,
    ) -> List[Dict[str, Any]]:
        """Processes multiple data items using the LLM, with optional parallelization.

        Args:
            data_list: List of data dictionaries to process
            n_workers: Number of parallel workers (default: 1). If > 1, enables parallel processing
            show_progress: Whether to show progress bar (default: True)
            merge_input: If True, merges input data with extracted fields (default: True)
        """
        from tqdm import tqdm
        from functools import partial

        if hasattr(data_list, "to_dict"):
            data_list = data_list.to_dict("records")
        # Create a partial function with fixed merge_input parameter
        sculpt_with_merge = partial(self.sculpt, merge_input=merge_input)

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