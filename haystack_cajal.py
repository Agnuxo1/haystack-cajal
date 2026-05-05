"""
Haystack 2.x Custom Component for CAJAL
Scientific Paper Generator Integration
"""

from typing import List, Dict, Any, Optional
from haystack import component, default_from_dict, default_to_dict
import logging

logger = logging.getLogger(__name__)

@component
class CAJALPaperGenerator:
    """
    Generates scientific papers using CAJAL local LLM.
    
    Input:
        - topic: Research topic description
        - sections: List of sections to generate ["abstract", "intro", ...]
    
    Output:
        - paper: Dict with all sections
        - word_count: Total words
        - generation_time: Seconds elapsed
    """
    
    def __init__(self, model: str = "Agnuxo/CAJAL-4B-P2PCLAW", device: str = "cuda"):
        self.model = model
        self.device = device
    
    @component.output_types(paper=Dict, word_count=int, generation_time=float)
    def run(self, topic: str, sections: List[str] = None):
        if sections is None:
            sections = ["abstract", "introduction", "methodology", "results", "conclusion"]
        
        import time
        start = time.time()
        
        # Paper generation logic (placeholder for actual CAJAL integration)
        paper = {}
        total_words = 0
        
        for section in sections:
            # This would call the actual CAJAL model
            content = f"Generated {section} for: {topic}"
            paper[section] = content
            total_words += len(content.split())
        
        elapsed = time.time() - start
        
        return {
            "paper": paper,
            "word_count": total_words,
            "generation_time": elapsed
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, model=self.model, device=self.device)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAJALPaperGenerator":
        return default_from_dict(cls, data)


@component
class CAJALPaperReviewer:
    """
    AI Tribunal peer review for generated papers.
    
    Input:
        - paper: Dict with paper sections
    
    Output:
        - scores: Dict with 9-dimension scores
        - paper: Paper with scores embedded
        - recommendation: "accept", "revise", or "reject"
    """
    
    def __init__(self, experts: int = 9):
        self.experts = experts
    
    @component.output_types(scores=Dict, paper=Dict, recommendation=str)
    def run(self, paper: Dict[str, Any]):
        # AI Tribunal scoring (placeholder)
        scores = {
            "novelty": 7.2,
            "methodology": 8.1,
            "clarity": 6.8,
            "significance": 7.5,
            "originality": 7.0,
            "technical_correctness": 8.3,
            "reproducibility": 6.5,
            "literature_review": 7.8,
            "ethical_compliance": 9.0
        }
        
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 8.0:
            recommendation = "accept"
        elif avg_score >= 6.0:
            recommendation = "revise"
        else:
            recommendation = "reject"
        
        paper_with_scores = {**paper, "_tribunal_scores": scores}
        
        return {
            "scores": scores,
            "paper": paper_with_scores,
            "recommendation": recommendation
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, experts=self.experts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAJALPaperReviewer":
        return default_from_dict(cls, data)
