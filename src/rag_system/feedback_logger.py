"""
Feedback Logging System for RAG

Tracks query performance, user feedback, and enables continuous improvement:
- Logs all queries and retrieved results
- Records user feedback (thumbs up/down, ratings)
- Analyzes patterns in successful/failed retrievals
- Provides insights for improving chunking and re-ranking
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class QueryLog:
    """Represents a single query execution log."""
    query_id: str
    timestamp: str
    query_text: str
    intent_type: str
    intent_confidence: float
    filters_applied: Dict[str, Any]
    search_params: Dict[str, Any]
    num_results: int
    results_summary: List[Dict[str, str]]  # [{source, content_preview, score}]
    user_feedback: Optional[str] = None  # 'positive', 'negative', 'neutral'
    feedback_comment: Optional[str] = None
    response_time_ms: Optional[float] = None


class FeedbackLogger:
    """Manages logging and analysis of RAG query feedback."""

    def __init__(self, log_dir: Path = None):
        """Initialize feedback logger.

        Args:
            log_dir: Directory to store log files (default: workspace/logs)
        """
        if log_dir is None:
            from . import DEFAULT_WORKSPACE
            log_dir = DEFAULT_WORKSPACE / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.query_log_file = self.log_dir / "queries.jsonl"
        self.feedback_log_file = self.log_dir / "feedback.jsonl"
        self.analytics_file = self.log_dir / "analytics.json"

    def log_query(
        self,
        query_text: str,
        intent_type: str,
        intent_confidence: float,
        filters_applied: Dict[str, Any],
        search_params: Dict[str, Any],
        results: List[Any],
        response_time_ms: Optional[float] = None
    ) -> str:
        """Log a query execution.

        Args:
            query_text: User's search query
            intent_type: Classified intent (code, concept, personal, etc.)
            intent_confidence: Classification confidence
            filters_applied: Metadata filters used
            search_params: Search parameters (lambda_mult, fetch_k)
            results: Retrieved documents
            response_time_ms: Query response time in milliseconds

        Returns:
            query_id: Unique identifier for this query
        """
        # Generate unique query ID
        timestamp = datetime.now()
        query_id = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        # Summarize results
        results_summary = []
        for i, doc in enumerate(results):
            summary = {
                "rank": i + 1,
                "source": doc.metadata.get("source_file", "unknown"),
                "content_type": doc.metadata.get("content_type", "unknown"),
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
            }
            results_summary.append(summary)

        # Create log entry
        log_entry = QueryLog(
            query_id=query_id,
            timestamp=timestamp.isoformat(),
            query_text=query_text,
            intent_type=intent_type,
            intent_confidence=intent_confidence,
            filters_applied=filters_applied,
            search_params=search_params,
            num_results=len(results),
            results_summary=results_summary,
            response_time_ms=response_time_ms
        )

        # Append to log file
        self._append_jsonl(self.query_log_file, asdict(log_entry))

        return query_id

    def add_feedback(
        self,
        query_id: str,
        feedback: str,
        comment: Optional[str] = None
    ):
        """Add user feedback to a logged query.

        Args:
            query_id: Query ID to add feedback to
            feedback: 'positive', 'negative', or 'neutral'
            comment: Optional text comment
        """
        feedback_entry = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "comment": comment
        }

        self._append_jsonl(self.feedback_log_file, feedback_entry)

        # Update the original query log
        self._update_query_feedback(query_id, feedback, comment)

    def _update_query_feedback(self, query_id: str, feedback: str, comment: Optional[str]):
        """Update the feedback field in the query log."""
        if not self.query_log_file.exists():
            return

        # Read all logs
        logs = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                log = json.loads(line)
                if log['query_id'] == query_id:
                    log['user_feedback'] = feedback
                    log['feedback_comment'] = comment
                logs.append(log)

        # Write back
        with open(self.query_log_file, 'w', encoding='utf-8') as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')

    def get_analytics(self) -> Dict[str, Any]:
        """Generate analytics from logged queries.

        Returns:
            Dictionary with analytics:
            - Total queries
            - Intent distribution
            - Average confidence
            - Feedback distribution
            - Common filters
            - Performance metrics
        """
        if not self.query_log_file.exists():
            return {
                "total_queries": 0,
                "message": "No queries logged yet"
            }

        # Read all logs
        logs = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs.append(json.loads(line))

        # Calculate analytics
        total_queries = len(logs)

        # Intent distribution
        intent_counts = {}
        intent_confidences = {}
        for log in logs:
            intent = log['intent_type']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if intent not in intent_confidences:
                intent_confidences[intent] = []
            intent_confidences[intent].append(log['intent_confidence'])

        # Average confidence per intent
        avg_confidence = {
            intent: sum(confs) / len(confs)
            for intent, confs in intent_confidences.items()
        }

        # Feedback distribution
        feedback_counts = {"positive": 0, "negative": 0, "neutral": 0, "no_feedback": 0}
        for log in logs:
            fb = log.get('user_feedback')
            if fb:
                feedback_counts[fb] = feedback_counts.get(fb, 0) + 1
            else:
                feedback_counts['no_feedback'] += 1

        # Common filters
        filter_usage = {}
        for log in logs:
            for key in log['filters_applied'].keys():
                filter_usage[key] = filter_usage.get(key, 0) + 1

        # Performance metrics
        response_times = [log['response_time_ms'] for log in logs if log.get('response_time_ms')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None

        analytics = {
            "total_queries": total_queries,
            "intent_distribution": intent_counts,
            "average_confidence_by_intent": avg_confidence,
            "feedback_distribution": feedback_counts,
            "filter_usage": filter_usage,
            "average_response_time_ms": avg_response_time,
            "queries_with_feedback": sum(v for k, v in feedback_counts.items() if k != 'no_feedback'),
            "feedback_rate": (sum(v for k, v in feedback_counts.items() if k != 'no_feedback') / total_queries * 100) if total_queries > 0 else 0
        }

        # Save to file
        with open(self.analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)

        return analytics

    def get_low_confidence_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get queries with low classification confidence.

        These might need better classification rules or manual review.

        Args:
            threshold: Confidence threshold (default: 0.5)

        Returns:
            List of low-confidence query logs
        """
        if not self.query_log_file.exists():
            return []

        low_confidence = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                log = json.loads(line)
                if log['intent_confidence'] <= threshold:
                    low_confidence.append(log)

        return low_confidence

    def get_negative_feedback_queries(self) -> List[Dict[str, Any]]:
        """Get queries with negative user feedback.

        These indicate retrieval failures that need investigation.

        Returns:
            List of queries with negative feedback
        """
        if not self.query_log_file.exists():
            return []

        negative_queries = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                log = json.loads(line)
                if log.get('user_feedback') == 'negative':
                    negative_queries.append(log)

        return negative_queries

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent queries.

        Args:
            limit: Number of recent queries to return

        Returns:
            List of recent query logs
        """
        if not self.query_log_file.exists():
            return []

        logs = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs.append(json.loads(line))

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)

        return logs[:limit]

    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Append a JSON object as a line to a JSONL file."""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


# Example usage
if __name__ == "__main__":
    logger = FeedbackLogger()

    # Simulate logging a query
    from dataclasses import dataclass

    @dataclass
    class MockDoc:
        page_content: str
        metadata: Dict[str, Any]

    mock_results = [
        MockDoc(
            page_content="Sample code for merging dataframes in pandas",
            metadata={"source_file": "week_05.ipynb", "content_type": "code"}
        ),
        MockDoc(
            page_content="Explanation of merge operations",
            metadata={"source_file": "lecture_notes.ipynb", "content_type": "markdown"}
        )
    ]

    query_id = logger.log_query(
        query_text="How do I merge dataframes?",
        intent_type="code",
        intent_confidence=0.95,
        filters_applied={"content_type": "code"},
        search_params={"lambda_mult": 0.6, "fetch_k_multiplier": 4},
        results=mock_results,
        response_time_ms=245.3
    )

    print(f"Logged query: {query_id}")

    # Add feedback
    logger.add_feedback(query_id, "positive", "Great results!")

    # Get analytics
    analytics = logger.get_analytics()
    print(f"\nAnalytics:")
    print(json.dumps(analytics, indent=2))
