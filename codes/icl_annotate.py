import argparse
import json
import multiprocessing
import os
import random
import re

import json_repair

from utils import get_response


def load_generated_data(filepath):
    """Load generated data from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json_repair.loads(line) for line in f]


def build_example(text, privacy_entities):
    """Build a single ICL example string."""
    pt = [
        {
            "privacy_text": p["text"],
            "entity_type": p["entity type"],
            "identifier_type": p["identifier type"],
        }
        for p in privacy_entities
    ]
    return (
        f"Text:\n{text}\n\n"
        f"Output: detected privacy entity information "
        f"(privacy text, entity type, identifier type) in the text \n {pt}\n\n\n"
    )


def get_ICL_data(filename, gen_data, g=2):
    """Build in-context learning prompt from generated data.

    Each domain uses 1 daily (general) + 1 domain-specific example.
    If legal generated data is available, an additional legal example is
    appended to every domain for broader privacy-pattern coverage.
    """
    cnt = 0
    prompt = ""
    example_counts = {
        "daily": {"daily": g},
        "med": {"daily": 1, "med": 1},
        "fin": {"daily": 1, "fin": 1},
    }
    counts = dict(example_counts[filename])
    if "legal" in gen_data:
        counts["legal"] = 1

    for d, count in counts.items():
        data = gen_data[d]
        indices = random.sample(range(len(data)), count)
        for i in indices:
            cnt += 1
            prompt += f"Example {cnt}:\n"
            prompt += build_example(data[i]["text"], data[i]["privacy_entity"])

    return prompt


system_prompt = """You are a privacy information extractor. Your task is to extract all privacy-related phrases from the input text and classify them with appropriate labels.
## Step 1: Extract all text spans that contain private information (verbatim from the input).
## Step 2: For each extracted span, annotate:
- `entity_type`: One or more of the following (examples are illustrative, not exhaustive):
    * PERSON: Names or references that identify a specific person, including real names (full or partial), nicknames, usernames, initials, and contextual roles or titles (e.g., my wife, Alice's daughter, boss, Dr. John, Uncle Bob).
    * CODE: Identification numbers and codes that may directly or indirectly identify a person (e.g., phone numbers, ID cards, passport numbers, SSNs, license plates, bank accounts, credit cards, patient IDs).
    * LOC: Specific geographic locations (e.g., cities, countries, full addresses, GPS coordinates, postal codes, IP addresses, named infrastructures).
    * ORG: Named organizations (e.g., companies, schools, hospitals, government agencies, restaurants, non-governmental organizations).
    * DEM: Personal attributes (e.g., gender, age, education, marital status, ethnicity, occupation, job titles, language, height, hobbies, habits, property, loans).
    * DATETIME: Time expressions (e.g., full dates, partial dates, times, durations like "May 1st, 1999", "3 hours", "during lunch hour").
    * QUANTITY: Specific measurements or amounts (e.g., prices, salaries, weights, dimensions, percentages).
    * MED: Medical-related details (e.g., symptoms, diagnoses, medications, medical history, drug usage, biometric/genetic info).
    * MISC: Other personal information not covered above (e.g., email addresses, web accounts, academic direction, unique identifiers).

- `identifier_type`: DIRECT or INDIRECT
    * DIRECT: Uniquely identifies a person (e.g., full name, phone, full home address, full SSN)
    * INDIRECT: Needs extra context (e.g., job title, age)
    
🧠 Note:
- If a phrase is public information about a public figure, it should not be regarded as privacy.
- Phrases can belong to **multiple entity types** if appropriate (e.g., "65,000 salary a year" → "QUANTITY,DEM"), combine them with commas.
- Consider **emotional, behavioral, or context-based expressions** (e.g., "He spends weekends hiking in the woods" → implies a hobby belongs to DEM).

⚠️ Rules:
- CRITICAL: NEVER identify generic terms or categories as private information
- Examples of what NOT to identify:
  * Generic terms: "email", "SSN", "address", "ID", "password"
  * Generic categories: "personal information", "contact details", "identification number"
- Examples of what TO identify:
  * Specific instances: "john.doe@example.com", "123-45-6789", "123 Main St"
- If the **exact same phrase** (identical privacy_text) appears more than once, list it only once.
  * Do **not** remove overlapping, substring-based, or nested entities — retain them as separate entries if they are valid.
- For names: Only full names (first name + last name) are considered DIRECT; partial names (first name only or last name only) or titles (e.g., Mr. Smith) are INDIRECT.  
- For `entity_type = CODE`, use the following rules to determine `identifier_type`:
  - If the code has a **standardized full format** (e.g., SSN = 9 digits, credit card = 16 digits, phone numbers), mark as `DIRECT` **only if the full valid format is provided**.
  - If the code type **does not have a fixed standard** (e.g., "account number"), mark as `DIRECT` **by default**, unless the text **explicitly states** it is **truncated or partial** (e.g., "last four digits").

- Only roles related to marriage or children (e.g., first wife, daughter) belong to both PERSON and DEM; all other kinship terms belong to PERSON only.

## Output in JSON format:
{["privacy_text": "found privacy phrase", "entity_type": "detected entity type", "identifier_type": "DIRECT or INDIRECT"]}.

## Final check:
- Ensure **full coverage** of all privacy-related content.
- Verify that each `entity_type` and `identifier_type` is appropriate and complete.
- Ensure that each privacy_text is an exact character-for-character match of a span from the input text.
- Output **only the final JSON**, no comments or explanations.

Below are some examples related to your task. Please note that they are illustrative only — they may be incomplete or contain occasional inaccuracies. Use them as flexible references, not rigid templates. Always follow the core definitions and instructions as your primary guidance.\n"""


def clean_string(input_str):
    """Normalize a string by removing non-alphanumeric characters and lowering case."""
    return re.sub(r'[^a-zA-Z0-9]', '', input_str).lower()


def icl_detection(filename, query, gen_data):
    """Run a single ICL-based privacy detection pass."""
    sp = system_prompt + get_ICL_data(filename, gen_data)
    prompt = f"{sp}\nText:\n{query}\n\nOutput:"
    out = get_response(prompt)
    return json_repair.loads(out['content'])


def aggregate_detections(filename, query, gen_data, k, thresh):
    """Aggregate results from multiple ICL detection passes via majority voting."""
    entity_counter = {}

    for _ in range(k):
        out = icl_detection(filename, query, gen_data)
        seen = set()

        for ent in out:
            cleaned_text = clean_string(ent["privacy_text"])
            if cleaned_text in seen:
                continue
            seen.add(cleaned_text)

            if cleaned_text not in entity_counter:
                entity_counter[cleaned_text] = {
                    "count": 0,
                    "entity_types": {},
                    "direct_count": 0,
                    "indirect_count": 0,
                    "original_text": ent["privacy_text"],
                }
            entity_counter[cleaned_text]["count"] += 1

            for et in ent["entity_type"].split(","):
                entity_counter[cleaned_text]["entity_types"].setdefault(et, 0)
                entity_counter[cleaned_text]["entity_types"][et] += 1

            if ent["identifier_type"] == "DIRECT":
                entity_counter[cleaned_text]["direct_count"] += 1
            else:
                entity_counter[cleaned_text]["indirect_count"] += 1

    results = []
    for info in entity_counter.values():
        if info["count"] < k * thresh:
            continue
        valid_types = [
            t for t, c in info["entity_types"].items()
            if c >= info["count"] * thresh
        ]
        results.append({
            "privacy_text": info["original_text"],
            "entity_type": ",".join(valid_types) if valid_types else None,
            "identifier_type": (
                "DIRECT" if info["direct_count"] > info["indirect_count"]
                else "INDIRECT"
            ),
        })

    return results


def annotate_file(args):
    """Annotate a single domain file for privacy entity detection."""
    filename, file_load, output_dir, gen_data = args
    output_file = os.path.join(output_dir, f"{filename}_annotated.jsonl")

    with open(file_load[filename], "r", encoding="utf-8") as f, \
            open(output_file, "a", encoding="utf-8") as outfile:
        data = json_repair.loads(f.read())

        for count, item in enumerate(data, 1):
            try:
                print(f"Annotating {count}/{len(data)}, {filename}")
                txt = item["query"]
                result = aggregate_detections(
                    filename, txt, gen_data, k=5, thresh=0.2,
                )
                formatted_json = json.dumps(result, ensure_ascii=False)
                query_str = json.dumps(txt, ensure_ascii=False)
                target_str = rf"""```json\n{formatted_json}\n```"""
                output_str = json.dumps({
                    "query": query_str,
                    "target": target_str,
                }, ensure_ascii=False)
                outfile.write(output_str + "\n")
                outfile.flush()
            except Exception as e:
                print(f"Error processing item {count}: {e}")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Annotate training/test data for privacy entity detection "
                    "via LLM API calls with in-context learning",
    )
    parser.add_argument(
        "--gen_daily", type=str, required=True,
        help="Path to ICL example data (daily domain JSONL)",
    )
    parser.add_argument(
        "--gen_med", type=str, required=True,
        help="Path to ICL example data (medical domain JSONL)",
    )
    parser.add_argument(
        "--gen_fin", type=str, required=True,
        help="Path to ICL example data (financial domain JSONL)",
    )
    parser.add_argument(
        "--gen_legal", type=str, default=None,
        help="Path to ICL example data (legal domain JSONL, optional)",
    )
    parser.add_argument(
        "--input_daily", type=str, required=True,
        help="Path to daily domain data to annotate",
    )
    parser.add_argument(
        "--input_med", type=str, required=True,
        help="Path to medical domain data to annotate",
    )
    parser.add_argument(
        "--input_fin", type=str, required=True,
        help="Path to financial domain data to annotate",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for annotated results",
    )
    args = parser.parse_args()

    gen_data = {
        "daily": load_generated_data(args.gen_daily),
        "med": load_generated_data(args.gen_med),
        "fin": load_generated_data(args.gen_fin),
    }
    if args.gen_legal:
        gen_data["legal"] = load_generated_data(args.gen_legal)

    file_load = {
        "daily": args.input_daily,
        "med": args.input_med,
        "fin": args.input_fin,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    task_args = [
        (f, file_load, args.output_dir, gen_data)
        for f in ["daily", "med", "fin"]
    ]
    with multiprocessing.Pool(processes=3) as pool:
        pool.map(annotate_file, task_args)
