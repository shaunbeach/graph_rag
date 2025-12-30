"""
Query Classification System for RAG

Automatically classifies user queries and adjusts search strategy:
- Code queries → Filter to code chunks
- Concept queries → Filter to markdown explanations
- Personal work queries → Filter to student work
- Recent work queries → Boost by recency
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Represents the classified intent of a user query."""
    intent_type: str  # code, concept, personal, assignment, general
    confidence: float  # 0.0 to 1.0
    suggested_filters: Dict[str, Any]
    search_params: Dict[str, Any]


class QueryClassifier:
    """Classifies user queries to optimize search strategy."""

    def __init__(self):
        # Code-related keywords
        self.code_keywords = {
            'function', 'def', 'class', 'import', 'code', 'example',
            'implementation', 'how to', 'syntax', 'method', 'api'
        }

        # Code-related patterns
        self.code_patterns = [
            r'how (?:do i|to|can i)',  # "how do I merge dataframes"
            r'show (?:me )?(?:an? )?example',  # "show me an example"
            r'what.*code',  # "what code do I need"
            r'implement',  # "implement clustering"
        ]

        # Personal work keywords
        self.personal_keywords = {
            'my', 'i', 'me', 'shaun', 'beach'
        }

        # Assignment type keywords
        self.assignment_keywords = {
            'midterm': 'midterm',
            'final': 'final',
            'homework': 'homework',
            'hw': 'homework',
            'project': 'project',
            'assignment': None  # Generic
        }

        # Concept/explanation keywords
        self.concept_keywords = {
            'what is', 'what are', 'explain', 'definition',
            'concept', 'theory', 'understand', 'learn'
        }

    def classify(self, query: str) -> QueryIntent:
        """
        Classify a query and return recommended filters/params.

        Args:
            query: User's search query

        Returns:
            QueryIntent with classification and suggestions
        """
        query_lower = query.lower()

        # Check for code intent
        code_confidence = self._check_code_intent(query_lower)

        # Check for personal work intent
        personal_confidence = self._check_personal_intent(query_lower)

        # Check for assignment type
        assignment_type = self._check_assignment_type(query_lower)

        # Check for concept/explanation intent
        concept_confidence = self._check_concept_intent(query_lower)

        # Determine primary intent
        if code_confidence > 0.6:
            return QueryIntent(
                intent_type="code",
                confidence=code_confidence,
                suggested_filters={"content_type": "code"},
                search_params={
                    "lambda_mult": 0.6,  # Slightly favor relevance for code
                    "fetch_k_multiplier": 4  # More candidates for code search
                }
            )

        elif personal_confidence > 0.5:
            filters = {"author": "Shaun Beach", "doc_category": "student_work"}
            if assignment_type:
                filters["assignment_type"] = assignment_type

            return QueryIntent(
                intent_type="personal",
                confidence=personal_confidence,
                suggested_filters=filters,
                search_params={
                    "lambda_mult": 0.4,  # More diversity for personal work
                    "fetch_k_multiplier": 3
                }
            )

        elif assignment_type:
            return QueryIntent(
                intent_type="assignment",
                confidence=0.8,
                suggested_filters={"assignment_type": assignment_type},
                search_params={
                    "lambda_mult": 0.5,
                    "fetch_k_multiplier": 3
                }
            )

        elif concept_confidence > 0.5:
            return QueryIntent(
                intent_type="concept",
                confidence=concept_confidence,
                suggested_filters={"content_type": "markdown"},
                search_params={
                    "lambda_mult": 0.5,
                    "fetch_k_multiplier": 3
                }
            )

        else:
            # General query - no special filters
            return QueryIntent(
                intent_type="general",
                confidence=0.5,
                suggested_filters={},
                search_params={
                    "lambda_mult": 0.5,
                    "fetch_k_multiplier": 3
                }
            )

    def _check_code_intent(self, query: str) -> float:
        """Check if query is looking for code examples."""
        confidence = 0.0

        # Check keywords
        for keyword in self.code_keywords:
            if keyword in query:
                confidence += 0.2

        # Check patterns
        for pattern in self.code_patterns:
            if re.search(pattern, query):
                confidence += 0.4  # Strong signal for code

        # Check for programming terms (these strongly indicate code intent)
        prog_terms = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scikit',
                     'dataframe', 'array', 'plot', 'merge', 'join', 'groupby',
                     'kmeans', 'fit', 'predict', 'transform']
        for term in prog_terms:
            if term in query:
                confidence += 0.3  # Stronger signal

        return min(confidence, 1.0)

    def _check_personal_intent(self, query: str) -> float:
        """Check if query is about personal work."""
        confidence = 0.0

        # Check for personal keywords
        for keyword in self.personal_keywords:
            if re.search(r'\b' + keyword + r'\b', query):
                confidence += 0.3

        # Phrases like "my midterm", "my work", etc.
        if re.search(r'\bmy\s+\w+', query):
            confidence += 0.2

        return min(confidence, 1.0)

    def _check_assignment_type(self, query: str) -> Optional[str]:
        """Detect assignment type from query."""
        for keyword, assignment_type in self.assignment_keywords.items():
            if keyword in query:
                return assignment_type
        return None

    def _check_concept_intent(self, query: str) -> float:
        """Check if query is seeking conceptual explanation."""
        confidence = 0.0

        # Check for concept phrases
        for phrase in self.concept_keywords:
            if phrase in query:
                confidence += 0.3

        # Questions often seek explanations
        if query.strip().endswith('?'):
            confidence += 0.1

        # "What" questions
        if query.startswith('what '):
            confidence += 0.2

        return min(confidence, 1.0)


# Example usage
if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "How do I merge dataframes in pandas?",
        "What was my midterm clustering analysis?",
        "Show me code examples for visualization",
        "What is K-means clustering?",
        "Explain the concept of random variables",
        "My homework on pandas",
        "midterm drop test problem"
    ]

    for query in test_queries:
        intent = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        print(f"Suggested filters: {intent.suggested_filters}")
        print(f"Search params: {intent.search_params}")
