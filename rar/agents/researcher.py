"""
Researcher Agent - Information retrieval and evidence collection.
"""

from typing import Optional

from .base import BaseAgent, AgentState
from ..protocols.acl import Performative
from ..tracing.tracer import Tracer


class ResearcherAgent(BaseAgent):
    """
    Researcher agent responsible for:
    - Searching local corpus for relevant information
    - Collecting and organizing evidence
    - Providing citations and evidence anchors
    """

    role = "researcher"
    description = "Information retrieval and evidence collection"

    def __init__(
        self,
        agent_id: str = "researcher",
        tracer: Optional[Tracer] = None,
        tools: dict = None
    ):
        super().__init__(agent_id, tracer, tools)

    def process(self, task: dict) -> dict:
        """
        Process a research task.

        Args:
            task: {
                "id": str,
                "type": "research",
                "description": str,
                "query": str
            }

        Returns:
            {
                "success": bool,
                "output": str,      # Summary of findings
                "evidence": list,   # Evidence anchors
                "raw_results": list # Raw search results
            }
        """
        self.activate()

        # Update beliefs
        self.update_beliefs({
            "task_id": task.get("id", ""),
            "query": task.get("query", task.get("description", ""))
        }, reason="Research task received")

        # Set desires
        self.set_desires([f"Find relevant information for: {task.get('description', '')}"])

        query = task.get("query", task.get("description", ""))

        # Check if search tool is available
        if "local_search" not in self.tools:
            self.deactivate()
            return {
                "success": False,
                "output": "",
                "error": "Search tool not available",
                "evidence": [],
                "raw_results": []
            }

        # Perform search
        search_result = self.use_tool("local_search", query=query, top_k=5)

        if not search_result.success:
            self.deactivate()
            return {
                "success": False,
                "output": "",
                "error": search_result.error,
                "evidence": [],
                "raw_results": []
            }

        # Process results
        raw_results = search_result.output
        evidence_anchors = search_result.evidence_anchors

        # Synthesize findings
        findings = self._synthesize_findings(query, raw_results)

        # Update beliefs with findings
        self.update_beliefs({
            "findings": findings,
            "evidence_count": len(evidence_anchors)
        }, reason="Research completed")

        # Send result message
        self.send_message(
            receiver="planner",
            performative=Performative.INFORM,
            content={
                "task_id": task.get("id", ""),
                "findings": findings,
                "evidence_count": len(evidence_anchors)
            },
            evidence=evidence_anchors
        )

        self.result = {
            "success": True,
            "output": findings,
            "evidence": evidence_anchors,
            "raw_results": raw_results
        }

        self.deactivate()
        return self.result

    def _synthesize_findings(self, query: str, results: list) -> str:
        """Synthesize search results into a coherent summary."""
        if not results:
            return f"No relevant information found for: {query}"

        parts = [f"Research findings for: {query}\n\n"]

        for i, result in enumerate(results, 1):
            doc_id = result.get("doc_id", "unknown")
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            score = result.get("score", 0)

            parts.append(f"[{i}] {title} (relevance: {score:.2f})\n")
            # Extract key sentences
            sentences = snippet.split(". ")
            key_sentences = sentences[:3]  # Take first 3 sentences
            parts.append("   " + ". ".join(key_sentences) + ".\n\n")

        parts.append(f"\nTotal: {len(results)} relevant passages found.")

        return "".join(parts)

    def extract_key_facts(self, results: list) -> list:
        """Extract key facts from search results."""
        facts = []
        for result in results:
            snippet = result.get("snippet", "")
            # Simple fact extraction - sentences with numbers or key phrases
            sentences = snippet.split(". ")
            for sent in sentences:
                # Keep sentences with numbers, percentages, or key indicators
                if any(c.isdigit() for c in sent) or "%" in sent:
                    facts.append({
                        "fact": sent.strip(),
                        "source": result.get("doc_id", "unknown"),
                        "paragraph": result.get("paragraph", 0)
                    })
        return facts[:10]  # Return top 10 facts
