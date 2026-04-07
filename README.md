# T2P-Bench: A Unified Benchmark for Text-to-Text Privatization

A comprehensive benchmark for text-to-text privatization methods. We provide a full pipeline covering **dataset construction (Module 1)**, **LLM-based data annotation (Module 2)**, and **local privacy detector inference (Module 3)**.

## Project Structure

```
T2P-Bench/
├── codes
│   ├── generate_dataset.py           # Module 1: ICL dataset construction
│   ├── icl_annotate.py               # Module 2: LLM API annotation
│   ├── utils.py                      # Shared utilities
│   ├── requirements.txt              
│   └── .env.example 				  # API configuration template
├── models/
│   └── qwen7b_v2/                    # PrivICL-7B model weights
└── datasets/
    ├── DetectorTrainingTest/         # Privacy detector Training/Test datasets
    │   ├── train/                    # Training split
    │   └── test/                     # Test split
    ├── DownstreamNLP/ 				  # Downstream NLP task datasets
    └── InContextDemonstration/       # In-Context Demonstration datasets
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (for Module 1 & 2)

```bash
cp .env.example .env
# Edit .env and fill in your API credentials
```

---

## Module 1: Dataset Construction

**Script:** `generate_dataset.py`

Generates synthetic privacy-annotated text data through a three-step pipeline:

1. **Virtual Person Generation** — Randomly samples privacy attributes (name, phone, address, medical history, etc.) and prompts an LLM to generate a virtual person profile.
2. **Scene-based Text Generation** — Places the virtual person into a domain-specific scene (e.g., clinical dialogue, financial report) to produce realistic text.
3. **Privacy Entity Annotation** — Extracts and labels all privacy entities from the generated text.

**Supported domains:** `medical`, `financial`, `legal`, `daily` (with 5 sub-scenes: food ordering, school life, work life, leisure, family relations).

### Usage

```python
from generate_dataset import generate_data
from utils import write_jsonl_to_file

samples = []
for i in range(100):
    try:
        sample = generate_data("daily")
        samples.append(sample)
        print(f"Generated sample {i + 1}")
    except Exception as e:
        print(f"Error: {e}")

write_jsonl_to_file(samples, "generated_daily.jsonl")
```

### Output Format

Each generated sample is a JSON object:

```json
{
  "text": "During the consultation, Dr. Smith asked Maria Johnson...",
  "field": "medical",
  "scene": "Clinical Dialogue",
  "privacy_info": {"name": "Maria Johnson", "phone number": "555-0142", ...},
  "privacy_entity": [
    {"entity type": "PERSON", "identifier type": "DIRECT", "text": "Maria Johnson"},
    {"entity type": "CODE", "identifier type": "DIRECT", "text": "555-0142"}
  ]
}
```

---

## Module 2: LLM API-based Annotation

**Script:** `icl_annotate.py`

Annotates raw text datasets for privacy entities using LLM APIs with **in-context learning (ICL)** and **majority voting aggregation**:

1. **ICL Prompting** — Constructs few-shot prompts using examples from the generated dataset (Module 1).
2. **Multi-pass Inference** — Runs `k` independent detection passes per input (default `k=5`).
3. **Majority Voting** — Aggregates results across passes; retains entities that appear in ≥ `thresh` fraction of passes (default `thresh=0.2`), with entity types and identifier types determined by majority vote.

### Usage

```bash
python icl_annotate.py \
  --gen_daily  path/to/generated_daily.jsonl \
  --gen_med    path/to/generated_medical.jsonl \
  --gen_fin    path/to/generated_financial.jsonl \
  --input_daily path/to/daily_data.jsonl \
  --input_med   path/to/medical_data.jsonl \
  --input_fin   path/to/financial_data.jsonl \
  --output_dir  path/to/output/
```

## Module 3: Local Model Inference via vLLM

Deploy PrivICL-7B model locally using [vLLM](https://github.com/vllm-project/vllm), which provides an OpenAI-compatible API server out of the box.

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Start the vLLM Server

```bash
vllm serve path/to/models/PrivICL_7B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096
```

The server exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`.

### 3. Use with T2P-Bench Scripts

Since vLLM serves an OpenAI-compatible API, you can directly use `icl_annotate.py` or any OpenAI client by pointing the environment variables to the local server:

```bash
# In your .env file:
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8000/v1
DET_MODEL=path/to/models/qwen7b_v2
```

Then run `icl_annotate.py` as usual — it will call the local model instead of a remote API.

### 4. Standalone Inference

You can also call the local model directly via the OpenAI Python client or `curl`:

**Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="path/to/models/PrivICL_7B",
    messages=[
        {"role": "user", "content": "..."}
    ],
)
print(response.choices[0].message.content)
```

## Entity Type Taxonomy

| Type       | Description                              | Examples                                           |
| ---------- | ---------------------------------------- | -------------------------------------------------- |
| `PERSON`   | Names or references identifying a person | Full names, nicknames, titles, kinship roles       |
| `CODE`     | Identification numbers and codes         | Phone numbers, SSN, credit cards, patient IDs      |
| `LOC`      | Geographic locations                     | Addresses, cities, GPS coordinates, postal codes   |
| `ORG`      | Organizations                            | Companies, hospitals, schools, government agencies |
| `DEM`      | Demographic/personal attributes          | Age, gender, occupation, hobbies, education        |
| `DATETIME` | Time expressions                         | Dates, durations, timestamps                       |
| `QUANTITY` | Measurements or amounts                  | Salary, prices, percentages                        |
| `MED`      | Medical information                      | Diagnoses, medications, symptoms                   |
| `MISC`     | Other personal information               | Email addresses, web accounts                      |

### Identifier Types

| Type       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `DIRECT`   | Uniquely identifies a person (e.g., full name, phone number, full address) |
| `INDIRECT` | Requires additional context to identify (e.g., job title, age, partial name) |

---

## License

Apache License 2.0