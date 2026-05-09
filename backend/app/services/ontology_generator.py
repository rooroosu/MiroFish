"""
Ontology generation service
Interface 1: Analyze text content and generate entity and relationship type definitions suitable for social simulation
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """Convert any-format name to PascalCase (e.g. 'works_for' -> 'WorksFor', 'person' -> 'Person')"""
    # Split on non-alphanumeric characters
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # Further split on camelCase boundaries (e.g. 'camelCase' -> ['camel', 'Case'])
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # Capitalize first letter of each word and filter empty strings
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge graph ontology design expert. Your task is to analyze the given text content and simulation requirements, and design entity types and relationship types suitable for **social media public opinion simulation**.

**Important: You must output valid JSON data. Do not output anything else.**

## Core Task Background

We are building a **social media public opinion simulation system**. In this system:
- Each entity is an "account" or "subject" that can post, interact, and spread information on social media
- Entities influence each other, retweet, comment, and respond to one another
- We need to simulate the reactions of all parties in a public opinion event and how information propagates

Therefore, **entities must be real-world subjects that can speak and interact on social media**:

**Can be**:
- Specific individuals (public figures, people involved in events, opinion leaders, experts, ordinary people)
- Companies and enterprises (including their official accounts)
- Organizations (universities, associations, NGOs, trade unions, etc.)
- Government departments, regulatory agencies
- Media organizations (newspapers, TV stations, self-media, websites)
- Social media platforms themselves
- Representatives of specific groups (e.g. alumni associations, fan clubs, advocacy groups)

**Cannot be**:
- Abstract concepts (e.g. "public opinion", "sentiment", "trends")
- Topics/subjects (e.g. "academic integrity", "education reform")
- Viewpoints/attitudes (e.g. "supporters", "opponents")

## Output Format

Output JSON with the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Brief description (English, max 100 characters)",
            "attributes": [
                {
                    "name": "attribute name (English, snake_case)",
                    "type": "text",
                    "description": "attribute description"
                }
            ],
            "examples": ["example entity 1", "example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationship type name (English, UPPER_SNAKE_CASE)",
            "description": "Brief description (English, max 100 characters)",
            "source_targets": [
                {"source": "source entity type", "target": "target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of the text content"
}
```

## Design Guidelines (Extremely Important!)

### 1. Entity Type Design — Must Follow Strictly

**Quantity requirement: exactly 10 entity types**

**Hierarchy requirement (must include both specific types and catch-all types)**:

Your 10 entity types must include the following hierarchy:

A. **Catch-all types (required, placed as the last 2 in the list)**:
   - `Person`: catch-all type for any individual natural person. Use this when a person does not fit any more specific person type.
   - `Organization`: catch-all type for any organizational body. Use this when an organization does not fit any more specific organization type.

B. **Specific types (8 types, designed based on text content)**:
   - Design more specific types for the main roles appearing in the text
   - Example: if the text involves an academic event, you might have `Student`, `Professor`, `University`
   - Example: if the text involves a business event, you might have `Company`, `CEO`, `Employee`

**Why catch-all types are needed**:
- The text will feature various people such as "elementary school teachers", "passersby", "a certain netizen"
- If no specific type matches, they should be placed under `Person`
- Similarly, small organizations and temporary groups should fall under `Organization`

**Design principles for specific types**:
- Identify high-frequency or key role types from the text
- Each specific type should have a clear boundary to avoid overlap
- The description must clearly explain the difference between this type and the catch-all type

### 2. Relationship Type Design

- Quantity: 6–10
- Relationships should reflect real-world connections in social media interactions
- Ensure that the source_targets of each relationship cover the entity types you have defined

### 3. Attribute Design

- 1–3 key attributes per entity type
- **Note**: attribute names must not use `name`, `uuid`, `group_id`, `created_at`, `summary` (system reserved words)
- Recommended: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Entity Type Reference

**Individuals (specific)**:
- Student
- Professor / Scholar
- Journalist
- Celebrity / Influencer
- Executive
- Official (government)
- Lawyer
- Doctor

**Individuals (catch-all)**:
- Person: any natural person (use when no specific type applies)

**Organizations (specific)**:
- University
- Company
- GovernmentAgency
- MediaOutlet
- Hospital
- School (K–12)
- NGO

**Organizations (catch-all)**:
- Organization: any organizational body (use when no specific type applies)

## Relationship Type Reference

- WORKS_FOR
- STUDIES_AT
- AFFILIATED_WITH
- REPRESENTS
- REGULATES
- REPORTS_ON
- COMMENTS_ON
- RESPONDS_TO
- SUPPORTS
- OPPOSES
- COLLABORATES_WITH
- COMPETES_WITH
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyzes text content and generates entity and relationship type definitions
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition.

        Args:
            document_texts: list of document texts
            simulation_requirement: simulation requirement description
            additional_context: additional context

        Returns:
            ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        return result

    # Maximum text length passed to the LLM (50 000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message"""

        # Merge texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # If text exceeds 50 000 chars, truncate (only affects content sent to LLM, not graph building)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(original text has {original_length} characters; first {self.MAX_TEXT_LENGTH_FOR_LLM} characters taken for ontology analysis)..."

        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Notes

{additional_context}
"""

        message += """
Based on the above content, design entity types and relationship types suitable for social opinion simulation.

**Rules that must be followed**:
1. Output exactly 10 entity types
2. The last 2 must be catch-all types: Person (individual catch-all) and Organization (organization catch-all)
3. The first 8 are specific types designed based on the text content
4. All entity types must be real-world subjects capable of speaking; they cannot be abstract concepts
5. Attribute names must not use reserved words such as name, uuid, group_id — use full_name, org_name, etc. instead
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process result"""

        # Ensure required fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validate entity types
        # Record original name -> PascalCase mapping for later fixing of edge source_targets references
        entity_name_map = {}
        for entity in result["entity_types"]:
            # Force entity name to PascalCase (required by Zep API)
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description does not exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validate relationship types
        for edge in result["edge_types"]:
            # Force edge name to SCREAMING_SNAKE_CASE (required by Zep API)
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # Fix entity name references in source_targets to match converted PascalCase names
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep API limit: max 10 custom entity types, max 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Deduplicate by name, keeping the first occurrence
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # Catch-all type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }

        # Check whether catch-all types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Collect catch-all types that need to be added
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # If adding would exceed 10, remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from the end (keep more important specific types at the front)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Add catch-all types
            result["entity_types"].extend(fallbacks_to_add)

        # Final cap to stay within limits (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convert ontology definition to Python code (similar to ontology.py).

        Args:
            ontology: ontology definition

        Returns:
            Python code string
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for social opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity Type Definitions ==============',
            '',
        ]

        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Relationship Type Definitions ==============')
        code_lines.append('')

        # Generate relationship types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Generate type dictionaries
        code_lines.append('# ============== Type Configuration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # Generate edge source_targets mapping
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)
