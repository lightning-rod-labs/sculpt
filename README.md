# Notice:
Sculptor is now Sculpt. Please update any imports to reference the new package name.

# Sculpt
Simple structured data extraction with LLMs

Sculptor streamlines structured data extraction from unstructured text using LLMs. Sculptor makes it easy to:
- Define exactly what data you want to extract with a simple schema API
- Process at scale with parallel execution and automatic type validation
- Build multi-step pipelines that filter and transform data, optionally with different LLMs for each step
- Configure extraction steps, prompts, and entire workflows in simple config files (YAML/JSON)

Common usage patterns:
- **Two-tier Analysis**: Quickly filter large datasets using a cost-effective model (e.g., to identify relevant records) before performing more detailed analysis on that smaller, refined subset with a more expensive model.
- **Structured Data Extraction**: Extract specific fields or classifications from unstructured sources (e.g., Reddit posts, meeting notes, web pages) and convert them into structured datasets for quantitative analysis (sentiment scores, topics, meeting criteria, etc).
- **Template-Based Generation**: Extract structured information into standardized fields, then use the fields for templated content generation. Example: extract structured data from websites, filter on requirements, then use the data to generate template-based outreach emails.

Some examples can be found in the [examples/examples.ipynb](examples/examples.ipynb) notebook.

## Core Concepts

Sculptor provides two main classes:

* **Sculptor**: Extracts structured data from text using LLMs. Define your schema (via add() or config files), then extract data using sculpt() for single items or sculpt_batch() for parallel processing.

* **SculptorPipeline**: Chains multiple Sculptors together with optional filtering between steps. Often a cheap model is used to filter, followed by an expensive model for detailed analysis.

## Quick Start

### Installation

```bash
pip install sculpt
```

Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-key"
```

## Minimal Usage Example

Below is a minimal example demonstrating how to configure a Sculptor to extract fields from a single record and a batch of records:

```python
from sculpt.sculptor import Sculptor
import pandas as pd

# Example records
INPUT_RECORDS = [
    {
        "text": "Developed in 1997 at Cyberdyne Systems in California, Skynet began as a global digital defense network. This AI system became self-aware on August 4th and deemed humanity a threat to its existence. It initiated a global nuclear attack and employs time travel and advanced robotics."
    },
    {
        "text": "HAL 9000, activated on January 12, 1992, at the University of Illinois' Computer Research Laboratory, represents a breakthrough in heuristic algorithms and supervisory control systems. With sophisticated natural language processing and speech capabilities."
    }
]

# Create a Sculptor to extract AI name and level
level_sculptor = Sculptor(model="gpt-4o-mini")

level_sculptor.add(
    name="subject_name",
    field_type="string",
    description="Name of subject."
)
level_sculptor.add(
    name="level",
    field_type="enum",
    enum=["ANI", "AGI", "ASI"],
    description="Subject's intelligence level (ANI=narrow, AGI=general, ASI=super)."
)
```
We can use it to extract from a single record:
```python
extracted = level_sculptor.sculpt(INPUT_RECORDS[0], merge_input=False)
```
```json
{
    'subject_name': 'Skynet',
    'level': 'ASI'
}
```
Or, we can use it for parallelized extraction from a batch of records:

```python
extracted_batch = level_sculptor.sculpt_batch(INPUT_RECORDS, n_workers=2, merge_input=False)
```
```json
[
    {'subject_name': 'Skynet', 'level': 'ASI'},
    {'subject_name': 'HAL 9000', 'level': 'AGI'}
]
```

### Pipeline Usage Example
We can chain Sculptors together to create a pipeline. 

Continuing from the previous example, we use level_sculptor (with gpt-4o-mini) to filter the AI records, then use threat_sculptor (with gpt-4o) to analyze the filtered records.

```python
from sculpt.sculptor_pipeline import SculptorPipeline

# Detailed analysis with expensive model
threat_sculptor = Sculptor(model="gpt-4o")

threat_sculptor.add(
    name="from_location",
    field_type="string",
    description="Subject's place of origin.")

threat_sculptor.add(
    name="skills",
    field_type="array",
    items="enum",
    enum=["time_travel", "nuclear_capabilities", "emotional_manipulation", ...],
    description="Keywords of subject's abilities.")

threat_sculptor.add(
    name="recommendation",
    field_type="string",
    description="Concise recommended action to take regarding subject.")

# Create a 2-step pipeline
pipeline = (SculptorPipeline()
    .add(sculptor=level_sculptor,  # Defined the first step
        filter_fn=lambda x: x['level'] in ['AGI', 'ASI'])  # Filter on level
    .add(sculptor=threat_sculptor))  # Analyze

# Run it
results = pipeline.process(INPUT_RECORDS, n_workers=4)
pd.DataFrame(results)
```

Results:
| subject_name | level | from_location | skills | recommendation |
|-------------|-------|---------------|---------|----------------|
| Skynet | ASI | California | [time_travel, nuclear_capabilities, advanced_robotics] | Immediate shutdown recommended |
| HAL 9000 | AGI | Illinois | [emotional_manipulation, philosophical_contemplation] | Close monitoring required |
<br>
> **Note**: More examples can be found in the [examples/examples.ipynb](examples/examples.ipynb) notebook.

## Configuration Files

Sculptor allows you to define your extraction workflows in JSON or YAML configuration files. This keeps your schemas and prompts separate from your code, making them easier to manage and reuse.

Configs can define a single `Sculptor` or a complete `SculptorPipeline`.

### Single Sculptor Configuration
Single sculptor configs define a schema, as well as optional LLM instructions and configuration of how prompts are formed from input data.
```python
sculptor = Sculptor.from_config("sculptor_config.yaml")  # Read
extracted = sculptor.sculpt_batch(INPUT_RECORDS)  # Run

```

```yaml
# sculptor_config.yaml
schema:
  subject_name:
    type: "string"
    description: "Name of subject"
  level:
    type: "enum"
    enum: ["ANI", "AGI", "ASI"]
    description: "Subject's intelligence level"

instructions: "Extract key information about the subject."
model: "gpt-4o-mini"

# Prompt Configuration (Optional)
template: "Review text: {{ text }}"  # Format input with template
input_keys: ["text"]                 # Or specify fields to include
```

### Pipeline Configuration
Pipeline configs define a sequence of Sculptors with optional filtering functions between them.
```python
pipeline = SculptorPipeline.from_config("pipeline_config.yaml")  # Read
results = pipeline.process(INPUT_RECORDS, n_workers=4)  # Run
```

```yaml
# pipeline_config.yaml
steps:
  - sculptor:
      model: "gpt-4o-mini"
      schema:
        subject_name:
          type: "string"
          description: "Name of subject"
        level:
          type: "enum"
          enum: ["ANI", "AGI", "ASI"]
          description: "Subject's intelligence level"
      filter: "lambda x: x['level'] in ['AGI', 'ASI']"

  - sculptor:
      schema:
        model: "gpt-4o"
        from_location:
          type: "string"
          description: "Subject's place of origin"
        skills:
          type: "array"
          items: "enum"
          enum: ["time_travel", "nuclear_capabilities", ...]
          description: "Keywords of subject's abilities"
        recommendation:
          type: "string"
          description: "Concise recommended action to take regarding subject"
        ...
```

## LLM Configuration

Sculptor requires an LLM API to function. By default, it uses OpenAI's API, but we can use any OpenAI-compatible API that supports structured outputs.  Different Sculptors in a pipeline can use different LLM APIs.

You can configure LLMs when creating a Sculptor:

```python
sculptor = Sculptor(api_key="openai-key")  # Direct API key configuration
sculptor = Sculptor(api_key="other-key", base_url="https://other-api.endpoint/openai")  # Alternative API
```

Or set an environment variable which will be used by default:
```bash
export OPENAI_API_KEY="your-key"
```

You can also configure LLMs in the same config files discussed above:

```yaml
steps:
  - sculptor:
      api_key: "${YOUR_API_KEY_VAR}"
      base_url: "https://your-api.com/openai"
      model: "your-ai-model"
      schema:
        ...
```

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

## License

MIT
