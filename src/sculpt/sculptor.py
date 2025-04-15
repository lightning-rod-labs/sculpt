import json
from typing import Dict, Any, Optional, List, Type, Union
from .utils import load_config
import openai
from string import Template
import inspect
import copy
import time
from asgiref.sync import sync_to_async

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
            # Check for both 'field_type' and 'type' keys
            field_type = field_data.get("field_type") or field_data.get("type")
            description = field_data.get("description", "")
            items = field_data.get("items")
            enum = field_data.get("enum")
            self.add(name=field_name, field_type=field_type, description=description, items=items, enum=enum)

    def add(
        self,
        name: str,
        field_type: Union[str, type] = str,
        description: str = "",
        items: Optional[Union[str, type, Dict[str, Any]]] = None,
        enum: Optional[List[Any]] = None,
    ):
        """
        Adds a field to the extraction schema.

        Args:
            name (str): The field name.
            field_type (Union[str, type]): e.g. "string", "boolean", "array", "enum",
                                           or a Python type like str, bool, list, etc.
            description (str): Write a short description for the field.
            items: Used if field_type="array". Either another type/dict or "enum".
            enum: A list of valid enumerated values, used if field_type="enum"
                  or if (field_type="array" and items="enum"), or if you want the
                  entire array enumerated as a top-level.
        """

        def normalize_type(t: Union[str, type]) -> str:
            if isinstance(t, type):
                t_lower = t.__name__.lower()
            else:
                t_lower = t.lower()

            if t_lower in ("str", "string"):
                return "string"
            elif t_lower in ("bool", "boolean"):
                return "boolean"
            elif t_lower in ("int", "integer"):
                return "integer"
            elif t_lower in ("float", "number"):
                return "number"
            elif t_lower in ("dict", "object"):
                return "object"
            elif t_lower in ("list", "array"):
                return "array"
            elif t_lower in ("enum", "anyof"):
                return t_lower
            else:
                raise ValueError(
                    f"Unsupported or invalid type '{t_lower}'. "
                    "Allowed types: string, boolean, integer, number, object, array, enum, anyOf"
                )

        field_type_str = normalize_type(field_type)
        processed_items = None

        # Handle arrays specially
        if field_type_str == "array":
            # (A) If user wants a top-level enumerated array
            #     e.g.: add(name="skills", field_type="array", enum=[["a"],["b"]]) => entire array must be one of these
            if enum and items is None:
                # We'll just store it; the final schema = {type: "array", enum: [...of arrays...] }
                pass  # processed_items remains None, so "items" won't appear
            else:
                # (B) Normal array approach (or array of enumerated items)
                if items is None:
                    raise ValueError("For 'array' type, you must provide 'items' or supply 'enum' for an enumerated array.")

                # If items="enum", that means each array item is enumerated
                if isinstance(items, str) and items.lower() == "enum":
                    # e.g. items="enum", enum=["infiltration","time_travel"]
                    # => each item is type="string" with that enum
                    if not enum:
                        raise ValueError(
                            "For 'array' + items='enum', please provide an `enum` list for the allowed item values."
                        )
                    processed_items = {
                        "type": "string",
                        "enum": enum,
                    }
                    enum = None  # so the parent doesn't claim 'enum' as well
                elif isinstance(items, dict):
                    # Possibly a nested object or more complex definition
                    processed_items = items
                else:
                    # items is a str/type => convert to "string", "integer", etc.
                    processed_items = normalize_type(items)

        # If field_type="enum" but no enum array is provided
        if field_type_str == "enum" and not enum:
            raise ValueError("For 'enum' type, you must provide a list of allowed values via `enum`.")

        # Lastly, store the field definition
        self.schema[name] = {
            "type": field_type_str,
            "description": description,
            "items": processed_items,
            "enum": enum,
        }

    @classmethod
    def from_config(cls, filepath: str, **kwargs: Any) -> "Sculptor":
        """Creates a Sculptor instance from a config file (JSON or YAML)."""
        config = load_config(filepath)

        # Get the parameters of the Sculptor.__init__ method
        init_params = inspect.signature(cls.__init__).parameters
        # Filter the config to only include valid parameters
        filtered_config = {k: v for k, v in config.items() if k in init_params}

        # Merge the filtered config with any keyword arguments passed in,
        # with kwargs taking precedence
        combined_config = {**filtered_config, **kwargs}

        return cls(**combined_config)

    def _build_schema_for_llm(self) -> Dict[str, Any]:
        """
        Builds the final JSON Schema for the LLM, using self.schema (which is
        already normalized to valid schema strings: 'string', 'boolean', etc.)
        """

        def build_subschema(meta: Dict[str, Any]) -> Dict[str, Any]:
            # Copy so we never modify the original in self.schema
            node = copy.deepcopy(meta)
            node_type = node["type"]  # Guaranteed to exist from add()

            schema_def: Dict[str, Any] = {}
            if "description" in node:
                schema_def["description"] = node["description"]

            if node_type == "object":
                schema_def["type"] = "object"
                # If you want an object with sub-properties, add them under node["properties"]
                # by calling add(...) for each sub-field; otherwise you can define them manually
                props = node.get("properties", {})
                nested_props = {}
                for prop_name, prop_meta in props.items():
                    nested_props[prop_name] = build_subschema(prop_meta)

                schema_def["properties"] = nested_props
                schema_def["additionalProperties"] = False
                schema_def["required"] = list(nested_props.keys())

            elif node_type == "array":
                schema_def["type"] = "array"
                if "items" not in node or node["items"] is None:
                    raise ValueError("Array schema is missing 'items' definition.")
                items_meta = node["items"]

                # If items is a dict, we might have nested objects/arrays
                if isinstance(items_meta, dict):
                    schema_def["items"] = build_subschema(items_meta)
                else:
                    # Otherwise, items_meta is a string like "string", "integer", "boolean"...
                    schema_def["items"] = {"type": items_meta}

            elif node_type in ("string", "number", "boolean", "integer"):
                schema_def["type"] = node_type

            elif node_type == "enum":
                schema_def["type"] = "string"
                if "enum" not in node or node["enum"] is None:
                    raise ValueError("Missing 'enum' array for enum type.")
                schema_def["enum"] = node["enum"]

            elif node_type == "anyof":
                schema_def["anyOf"] = node.get("anyOf", [])

            else:
                raise ValueError(f"Unsupported type '{node_type}' in build_subschema.")

            return schema_def

        # Build top-level object with each field in self.schema
        top_properties = {}
        for field_name, field_meta in self.schema.items():
            top_properties[field_name] = build_subschema(field_meta)

        # Return the final top-level schema
        return {
            "name": "extract_fields",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": top_properties,
                "required": list(top_properties.keys()),
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
            f"SCHEMA \n```{json.dumps(schema['schema'], indent=2)}```",
        ]
        
        return "\n\n".join(message_parts)

    def sculpt(self, data: Dict[str, Any], merge_input: bool = True, retries: int = 3, suppress_errors: bool = False) -> Dict[str, Any]:
        """Processes a single data item using the LLM."""
        schema_for_llm = self._build_schema_for_llm()
        
        last_error = None
        for attempt in range(retries):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self._build_user_message(data, schema_for_llm)},
                    ],
                    response_format = (
                        {"type": "json_object", "json_schema": schema_for_llm}
                        if ("deepseek" in str(self.openai_client.base_url).lower() or 
                            "deepseek" in str(self.model).lower())
                        else {"type": "json_schema", "json_schema": schema_for_llm}
                    ),
                    temperature=attempt * 0.1,  # Increase temperature by 0.1 for each retry
                )
                content = resp.choices[0].message.content.strip()
                # Extract just the JSON object by finding the outermost braces
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                
                extracted = json.loads(content)
                if isinstance(extracted, list) and len(extracted) == 1:
                    extracted = extracted[0]  # Some models wrap the output in a list
                
                # Clean up any whitespace in keys
                extracted = {k.strip(): v for k, v in extracted.items()}
                
                if not merge_input:
                    return extracted
                
                # Check for field conflicts
                conflicts = set(data.keys()) & set(extracted.keys())
                if conflicts:
                    import warnings
                    warnings.warn(f"The following fields will be overwritten: {conflicts}")
                
                # Merge while giving priority to extracted fields
                return {**data, **extracted}

            except Exception as e:
                last_error = e
                if attempt < retries - 1:  # Don't sleep on the last attempt
                    time.sleep(1)  # Add a small delay between retries
                continue
        
        if suppress_errors:
            return None
        else:
            raise RuntimeError(f"LLM API call failed after {retries} attempts. Last error: {last_error}")

    def sculpt_batch(
        self,
        data_list: List[Dict[str, Any]],
        n_workers: int = 1,
        show_progress: bool = True,
        merge_input: bool = True,
        retries: int = 3,
        suppress_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Processes multiple data items using the LLM, with optional parallelization.

        Args:
            data_list: List of data dictionaries to process
            n_workers: Number of parallel workers (default: 1). If > 1, enables parallel processing
            show_progress: Whether to show progress bar (default: True)
            merge_input: If True, merges input data with extracted fields (default: True)
            retries: Number of times to retry failed attempts (default: 1)
        """
        from tqdm import tqdm
        from functools import partial

        if hasattr(data_list, "to_dict"):
            data_list = data_list.to_dict("records")
        # Create a partial function with fixed merge_input parameter
        sculpt_with_merge = partial(self.sculpt, merge_input=merge_input, retries=retries, suppress_errors=suppress_errors)

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
        return [result for result in results if result is not None]
    
    async def sculpt_async(self, data: Dict[str, Any], merge_input: bool = True, retries: int = 3) -> Dict[str, Any]:
        """Processes a single data item using the LLM asynchronously."""
        return await sync_to_async(self.sculpt)(data, merge_input, retries)
    
    async def sculpt_batch_async(self, data_list: List[Dict[str, Any]], n_workers: int = 100, show_progress: bool = True, merge_input: bool = True, retries: int = 3) -> List[Dict[str, Any]]:
        """Processes multiple data items using the LLM asynchronously."""
        return await sync_to_async(self.sculpt_batch)(data_list, n_workers, show_progress, merge_input, retries)