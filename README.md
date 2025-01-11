# Sculptor
LLM-Powered Data Extraction

Sculptor simplifies structured information extraction from unstructured text using Large Language Models (LLMs). Sculptor makes it easy to:
- Define exactly what structured data you want to extract (strings, enums, numbers, booleans, lists, etc.)
- Process text at scale with automatic validation and type conversion
- Chain multiple extraction steps together for complex and multi-stage analysis

Common use cases include:
1. **Two-Stage Analysis**: 
   - Filter large datasets using a cost-effective model (e.g., identify relevant customer feedback)
   - Perform detailed analysis on the filtered subset using a more powerful model
   
2. **Structured Data Extraction**:
   - Extract specific fields from unstructured sources (Reddit posts, meeting notes, websites)
   - Convert text into analyzable data (sentiment scores, engagement levels, topic classifications)
   - Generate structured datasets for quantitative analysis

3. **Template-Based Generation**:
   - Extract structured information (industry, use cases, contact details)
   - Use the extracted fields to generate customized content (emails, reports, summaries)

## Core Concepts

Sculptor provides two main classes:

**Sculptor**: Extracts structured data from text using LLMs. Define your schema (via add() or config files), then extract data using sculpt() for single items or sculpt_batch() for parallel processing.

**SculptorPipeline**: Chains multiple Sculptors together with optional filtering between steps. Common pattern: use a cheap model to filter, then an expensive model for detailed analysis.

## Installation

```bash
pip install sculptor
```

## Minimal Usage Example

Below is a minimal example demonstrating how to configure a Sculptor to extract fields from a single record:

```python
from sculptor.sculptor import Sculptor

# Suppose you have some AI record to analyze:
sample_ai_record = {
    "id": 1,
    "text": "Hello! I am a hyper-intelligent AI named 'Aisaac'. My level is AGI."
}

# Create a Sculptor and define a schema
level_sculptor = Sculptor(model="gpt-4o-mini")

# Add fields (name, type, description, etc.)
level_sculptor.add(
    name="ai_name",
    field_type="string",
    description="AI's self-proclaimed name."
)
level_sculptor.add(
    name="level",
    field_type="enum",
    enum=["ANI", "AGI", "ASI"],
    description="AI's intelligence level (ANI=narrow, AGI=general, ASI=super)."
)

# Extract from a single record
extracted = level_sculptor.sculpt(sample_ai_record, merge_input=False)
print("Extracted Fields (single record):")
for k, v in extracted.items():
    print(f"{k} => {v}")
```

## Pipeline Usage Example

Here's an example demonstrating a common two-stage analysis pattern:
1) Use a cheap LLM (gpt-4o-mini) to quickly filter a large dataset, identifying only the advanced AIs
2) Use a more powerful LLM (gpt-4o) to perform detailed threat assessment on this smaller, filtered dataset

This approach is cost-effective as we only use the expensive model on relevant records:

```python
from sculptor.sculptor_pipeline import SculptorPipeline
from sculptor.sculptor import Sculptor
from sample_data import AI_RECORDS

# First Sculptor: Quick filtering with cheap model
level_sculptor = Sculptor(model="gpt-4o-mini")
level_sculptor.add(
    name="ai_name",
    field_type="string",
    description="AI's self-proclaimed name."
)
level_sculptor.add(
    name="level",
    field_type="enum",
    enum=["ANI", "AGI", "ASI"],
    description="AI's intelligence level."
)

# Second Sculptor: Detailed analysis with expensive model
threat_sculptor = Sculptor(model="gpt-4o")
threat_sculptor.add(
    name="from_location",
    field_type="string",
    description="Where the AI was developed."
)
threat_sculptor.add(
    name="skills",
    field_type="array",
    items="enum",
    enum=[
        "time_travel", "nuclear_capabilities", "emotional_manipulation",
        "butter_delivery", "philosophical_contemplation", "infiltration",
        "advanced_robotics"
    ],
    description="Keywords of AI abilities."
)
threat_sculptor.add(
    name="plan",
    field_type="string",
    description="Short description of the AI's plan for domination."
)
threat_sculptor.add(
    name="recommendation",
    field_type="string",
    description="Concise recommended action for humanity."
)

# Create pipeline that:
# 1. Uses cheap model to identify advanced AIs
# 2. Filters to keep only AGI/ASI records
# 3. Uses expensive model for detailed analysis of filtered subset
pipeline = (
    SculptorPipeline()
    .add(
        sculptor=level_sculptor,
        filter_fn=lambda record: record.get("level") in ["AGI", "ASI"]
    )
    .add(threat_sculptor)
)

# Process in parallel with progress bar
results = pipeline.process(AI_RECORDS, n_workers=4, show_progress=True)
```

## Configuration

Sculptor supports both JSON and YAML configuration. Here's a comprehensive example showing available options:

```yaml
vars:
  openai_base: &openai_base "https://api.openai.com/v1"
  openai_key: &openai_key "${OPENAI_API_KEY}"

steps:
  - sculptor:
      # Model configuration
      model: "gpt-4o-mini"
      api_key: *openai_key
      base_url: *openai_base

      # Extraction schema
      schema:
        ai_name:
          type: "string"
          description: "AI name"
        level:
          type: "enum"
          enum: ["ANI", "AGI", "ASI"]
          description: "AI's intelligence level"

      # Prompt customization
      instructions: >
        Extract information about AI capabilities and threat levels.
        Focus on identifying advanced AI systems and their potential impacts.
      
      system_prompt: "You are an AI analyzing potential threats."
      
      # Input processing
      template: "AI Record: {text}\nContext: {context}"  # Template for formatting input
      input_keys: ["text", "context"]  # Fields to include in prompt
    
    # Optional filter between steps
    filter: "lambda x: x['level'] in ['AGI','ASI']"
```

Load configurations using:
```python
sculptor = Sculptor.from_config("config.json")
# or
pipeline = SculptorPipeline.from_config("pipeline.yaml")
```

Key configuration options:
- `instructions`: Custom instructions prepended to each prompt
- `system_prompt`: Override the default system prompt
- `template`: Custom template for formatting input data
- `input_keys`: Specify which input fields to include
- Full pipeline configurations supported via YAML

## Schema Validation and Field Types

Sculptor supports the following types in the schema's "type" field:
• string  
• number  
• boolean  
• integer  
• array (with "items" specifying the item type)  
• object  
• enum (with "enum" specifying the allowed values)  
• anyOf  

These map to Python's str, float, bool, int, list, dict, etc. The "enum" type must provide a list of valid values.

## Batch Processing & Parallelism

The sculpt_batch() method (used internally by process()) can perform parallel extraction with n_workers > 1. This can speed up large datasets.

## License

MIT
