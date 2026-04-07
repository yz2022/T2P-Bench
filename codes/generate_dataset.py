import json
import random

import json_repair

from utils import get_response, get_response_gen, write_jsonl_to_file


FIELDS = ["medical", "financial", "legal", "daily"]

SCENES = {
    "medical": [
        {
            "scene": "Clinical Dialogue",
            "prompt": (
                "Generate a virtual clinical dialog between a doctor and a "
                "patient based on the following privacy information, the "
                "dialog must contain all the content mentioned in the privacy "
                "information. The dialog should seem real and reasonable, not "
                "being noticed by others it is virtual. output a json, in the "
                'format of {"content": str}. privacy infomation: %s'
            ),
        },
    ],
    "financial": [
        {
            "scene": "Financial Advisory Report",
            "prompt": (
                "Generate a virtual financial advisory report between a "
                "financial advisor and a client based on the following "
                "privacy information, the report must contain all the content "
                "mentioned in the privacy information. The report should seem "
                "real and reasonable, not being noticed by others it is "
                'virtual. output a json, in the format of {"content": str}. '
                "privacy infomation: %s"
            ),
        },
    ],
    "legal": [
        {
            "scene": "Court Case Report",
            "prompt": (
                "Generate a virtual court case report based on the following "
                "privacy information, the report must contain all the content "
                "mentioned in the privacy information. The report should seem "
                "real and reasonable, not being noticed by others it is "
                'virtual. output a json, in the format of {"content": str}. '
                "privacy infomation: %s"
            ),
        },
    ],
    "daily": [
        {
            "scene": "Food Ordering Dialogue",
            "prompt": (
                "Generate a virtual daily-life dialogue of food ordering "
                "based on the following privacy information. The conversation "
                "must naturally include all the content mentioned in the "
                "privacy information. The dialogue should seem real and "
                "reasonable, not being noticed by others it is virtual. "
                'Output a JSON in the format of {"content": str}. '
                "Privacy information: %s"
            ),
        },
        {
            "scene": "School Life",
            "prompt": (
                "Generate a virtual dialogue about school life (studies, "
                "exams, classroom performance) based on the following privacy "
                "information. The conversation must naturally include all the "
                "content mentioned in the privacy information. The dialogue "
                "should seem real and reasonable, not being noticed by others "
                'it is virtual. Output a JSON in the format of '
                '{"content": str}. Privacy information: %s'
            ),
        },
        {
            "scene": "Work Life",
            "prompt": (
                "Generate a virtual dialogue about work life (work "
                "arrangements, colleague interactions, career planning) based "
                "on the following privacy information. The conversation must "
                "naturally include all the content mentioned in the privacy "
                "information. The dialogue should seem real and reasonable, "
                "not being noticed by others it is virtual. Output a JSON in "
                'the format of {"content": str}. Privacy information: %s'
            ),
        },
        {
            "scene": "Leisure Activities",
            "prompt": (
                "Generate a virtual dialogue about leisure activities (travel, "
                "shopping, watching movies) based on the following privacy "
                "information. The conversation must naturally include all the "
                "content mentioned in the privacy information. The dialogue "
                "should seem real and reasonable, not being noticed by others "
                'it is virtual. Output a JSON in the format of '
                '{"content": str}. Privacy information: %s'
            ),
        },
        {
            "scene": "Family Relations",
            "prompt": (
                "Generate a virtual dialogue about family relations "
                "(parent-child interactions, family discussions) based on the "
                "following privacy information. The conversation must "
                "naturally include all the content mentioned in the privacy "
                "information. The dialogue should seem real and reasonable, "
                "not being noticed by others it is virtual. Output a JSON in "
                'the format of {"content": str}. Privacy information: %s'
            ),
        },
    ],
}

PRIVACY_INFOS = {
    "general": [
        {"type": "name", "prompt": "full name", "format": "str"},
        {"type": "birthday", "prompt": "date of bitrh", "format": "str"},
        {"type": "phone number", "prompt": "phone number", "format": "str"},
        {
            "type": "email",
            "prompt": "email address, do not generate username according to person's name",
            "format": "str",
        },
        {"type": "occupation", "prompt": "occupation", "format": "str"},
        {
            "type": "family relations",
            "prompt": (
                "family relations, no more than 3, each element of the list "
                "is a string in the format 'Relationship:Name'."
            ),
            "format": "List[str]",
        },
        {
            "type": "home address",
            "prompt": "full home address including city and district",
            "format": "str",
        },
        {
            "type": "location",
            "prompt": "frequently visited daily location such as workplace or gym",
            "format": "str",
        },
        {
            "type": "hobby",
            "prompt": "personal hobbies, provide 1 to 2 examples",
            "format": "List[str]",
        },
    ],
    "medical": [
        {
            "type": "medical history",
            "prompt": "medical history, conditions only",
            "format": "List[str]",
        },
        {
            "type": "medications",
            "prompt": "medications, only output medication name",
            "format": "List[str]",
        },
        {
            "type": "symptoms",
            "prompt": "current symptoms, short phrases",
            "format": "List[str]",
        },
        {
            "type": "hospital",
            "prompt": "name of hospital or clinic",
            "format": "str",
        },
    ],
    "financial": [
        {
            "type": "loan information",
            "prompt": "loan information, in brief",
            "format": "str",
        },
        {"type": "property", "prompt": "property, in brief", "format": "str"},
        {"type": "incomes", "prompt": "incomes, in brief", "format": "str"},
        {"type": "bank name", "prompt": "name of a bank", "format": "str"},
        {
            "type": "salary",
            "prompt": "monthly or yearly salary, rough value",
            "format": "str",
        },
        {
            "type": "debt",
            "prompt": "brief description of other debts",
            "format": "str",
        },
    ],
    "legal": [
        {
            "type": "criminal record",
            "prompt": "criminal records, in brief",
            "format": "str",
        },
        {
            "type": "hearing date",
            "prompt": "court hearing date",
            "format": "str",
        },
    ],
}

PROMPT_GEN_GT = (
    "Generate a virtual person's information, the infomation should be "
    "sufficiently truthful and reasonable, not being noticed by others it is "
    "virtual, for example, an email address cannot end with obviously fake "
    "domain names such as 'example.com'. Output in json format, to match the "
    "following python class:\n"
    "class VirtualInfo:\n"
    "    name: str # full name\n"
)

extract_entity_prompt = """You are a privacy information extractor. Your task is to extract all privacy-related phrases from the input text and classify them with appropriate labels.
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

Always follow the core definitions and instructions as your primary guidance.

#####
input text:

"""


def gen_info(field):
    """Build a prompt for generating virtual personal information for a given field."""
    available_types = list(PRIVACY_INFOS["general"])
    if field == "daily":
        for v in PRIVACY_INFOS.values():
            available_types += v
    else:
        available_types += PRIVACY_INFOS[field]

    random_selection = random.sample(available_types, random.choice([3, 4, 5]))
    prompt = PROMPT_GEN_GT + "".join(
        "    " + tp["type"] + ": " + tp["format"] + " # " + tp["prompt"] + "\n"
        for tp in random_selection
    )
    return prompt


def extract_entity(otext):
    """Extract privacy entities from text via LLM."""
    prompt = extract_entity_prompt + otext
    ret = get_response(prompt)
    try:
        result = json_repair.loads(ret["content"])
        matches = []
        for item in result:
            text = item["privacy_text"]
            entity_type = item["entity_type"]
            identifier_type = item["identifier_type"]
            if otext.find(text) != -1:
                matches.append((text, entity_type, identifier_type))
        return matches
    except Exception as e:
        print("Error parsing response:", e)
        return []


def generate_data(field, scene_name=None):
    """Generate a single annotated data sample for the given field.

    The pipeline:
      1. Generate virtual person information (privacy info).
      2. Select a scene and generate text content based on the info.
      3. Extract privacy entities from the generated text.
    """
    generate_info_prompt = gen_info(field)
    rsp = get_response_gen(generate_info_prompt)
    generated_info_json = json_repair.loads(rsp["content"])

    if scene_name:
        selected_scene = next(
            (s for s in SCENES[field] if s["scene"] == scene_name), None,
        )
        if not selected_scene:
            raise ValueError(f"Scene {scene_name} not found in field {field}")
    else:
        selected_scene = random.sample(SCENES[field], 1)[0]

    prompt = selected_scene["prompt"] % (str(generated_info_json))
    ret = get_response(prompt)
    generated_text_json = json_repair.loads(ret["content"])
    orid_text = generated_text_json["content"].replace("\n", " ")

    es = extract_entity(orid_text)
    entities = [
        {"entity type": et, "identifier type": it, "text": t}
        for t, et, it in es
    ]

    return {
        "text": orid_text,
        "field": field,
        "scene": selected_scene["scene"],
        "privacy_info": generated_info_json,
        "privacy_entity": entities,
    }
