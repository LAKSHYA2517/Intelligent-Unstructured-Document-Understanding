from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import logging
import os
import re

SKIP_ELEMENT_TYPES = {
    "page_break",
    "header", 
    "footer",
    "page_number",
    "caption",      # Too short, no claims
    "list_item",    # Handle separately with parent context
}

MIN_CONTENT_LENGTH_FOR_EXTRACTION = 120

EXTRACTION_WORTH_PATTERNS = [
    r'\$[\d,]+',           # Dollar amounts
    r'\d+\.?\d*\s*%',      # Percentages  
    r'\b(revenue|profit|loss|margin|growth|decline)\b',
    r'\b(Q[1-4]\s*20\d\d|FY\s*20\d\d)\b',  # Time periods
    r'\b(increased|decreased|grew|fell|rose|dropped)\b',
    r'\b(million|billion|thousand)\b',
    r'\b(risk|opportunity|threat|challenge)\b',
    r'\b(acquired|merged|launched|announced)\b',
]

def is_worth_extracting(element) -> bool:
    if element.element_type in SKIP_ELEMENT_TYPES:
        return False
    if element.element_type == "image":
        return False
        
    content = element.content.strip()
    if len(content) < 20:
        return False
        
    return True

class NodeType(Enum):
    DOCUMENT = "DOCUMENT"
    SECTION = "SECTION"  
    ELEMENT = "ELEMENT"
    CHUNK = "CHUNK"
    ENTITY = "ENTITY"
    CLAIM = "CLAIM"           
    METRIC = "METRIC"         
    EVENT = "EVENT"           
    COMMUNITY = "COMMUNITY"   

class EdgeType(Enum):
    CONTAINS = "CONTAINS"             
    NEXT_SEQUENTIAL = "NEXT_SEQ"      
    PART_OF = "PART_OF"              
    MENTIONS = "MENTIONS"             
    ASSERTS = "ASSERTS"              
    SUPPORTS = "SUPPORTS"            
    CONTRADICTS = "CONTRADICTS"      
    CAUSED_BY = "CAUSED_BY"         
    TEMPORALLY_BEFORE = "BEFORE"     
    QUANTIFIES = "QUANTIFIES"        
    COMPARED_TO = "COMPARED_TO"     
    DEFINED_IN = "DEFINED_IN"        
    SAME_ENTITY_AS = "SAME_ENTITY"   
    UPDATES = "UPDATES"              
    CONFLICTS_WITH_DOC = "CONFLICTS" 

@dataclass
class ClaimNode:
    claim_id: str
    text: str                          
    source_element_id: str            
    claim_type: str                    
    confidence: float                  
    temporal_scope: Optional[str]      
    entities_involved: List[str]       
    numerical_values: List[Dict[str, Any]]       
    
@dataclass
class MetricNode:
    metric_id: str
    entity_id: str                    
    metric_name: str                  
    value: float
    unit: str                         
    period: str                       
    source_element_id: str
    
@dataclass  
class EventNode:
    event_id: str
    description: str
    event_date: Optional[str]         
    event_type: str                   
    entities_involved: List[str]
    source_element_id: str

@dataclass
class DocumentElement:
    element_id: str
    element_type: str
    content: str
    page_number: Optional[int] = None
    parent_id: Optional[str] = None

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims and numerical metrics 
from this document element.

Element type: {element_type}
Content: {content}

Return JSON with this structure:
{{
  "claims": [
    {{
      "text": "the exact claim as a declarative sentence",
      "claim_type": "statistical|causal|comparative|definitional|predictive",
      "temporal_scope": "Q3 2024 or null",
      "entities_involved": ["entity names"],
      "numerical_values": [{{"value": 10.2, "unit": "billion USD"}}]
    }}
  ],
  "metrics": [
    {{
      "entity": "entity name",
      "metric_name": "revenue|gross_margin|etc",
      "value": 10.2,
      "unit": "USD_billion|percent|count|etc",
      "period": "Q3_2024|FY2023|etc"
    }}
  ],
  "events": [
    {{
      "description": "what happened",
      "event_date": "2024-09-30 or null",
      "event_type": "earnings_release|acquisition|etc",
      "entities_involved": ["entity names"]
    }}
  ]
}}"""
