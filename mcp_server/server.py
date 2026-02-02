#!/usr/bin/env python3
"""CustomKB MCP Server - Provides KB-specific search tools for Claude Code.

This server exposes each knowledgebase as a separate search tool,
enabling targeted research across specialized knowledge domains.
"""

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Initialize FastMCP server
mcp = FastMCP("customkb")

# Base directory for knowledgebases
VECTORDBS = os.environ.get("VECTORDBS", "/var/lib/vectordbs")

# KB metadata: (short_description, paragraph_description)
KB_METADATA = {
    "appliedanthropology": (
        "Applied anthropology expertise covering human evolution, cultural development, and secular dharma perspectives.",
        "A comprehensive knowledgebase focused on applied anthropology and secular dharma interpretations. Covers evolutionary biology, human behavioral evolution, cultural sociology, and the human condition."
    ),
    "jakartapost": (
        "Jakarta Post archive (1994-2005) for Indonesian politics, society, and journalism research.",
        "An archival knowledgebase containing The Jakarta Post newspaper content from 1994 to 2005. Specialized for research on Indonesian politics, society, and historical events during this period."
    ),
    "okusiassociates": (
        "Indonesian PMA company setup and corporate services expertise for foreign direct investment since 1997.",
        "Professional knowledgebase for Okusi Associates, an Indonesian corporate support services firm. Specializes in PMA (Foreign Direct Investment) company setup, corporate law, taxation, accounting, and statutory audits."
    ),
    "okusimail": (
        "AI email assistant for Okusi Associates handling business inquiries about Indonesian corporate services.",
        "An AI-powered email response system for Okusi Associates. Handles business inquiries about PMA company setup, corporate law, taxation, accounting, audits, work permits, and regulatory compliance in Indonesia."
    ),
    "okusiresearch": (
        "Research specialist for Indonesian direct investment, corporate law, taxation, and PMA company management.",
        "A specialized research knowledgebase focusing on Indonesian direct investment and corporate support services. Covers legal, accountancy, audit, and business aspects for PMA companies."
    ),
    "ollama": (
        "Elite programming and AI systems engineering expertise specializing in Ollama configuration and operation.",
        "A technical knowledgebase for Ollama, focusing on configuration, operation, and AI systems engineering. Provides expert guidance on setting up and optimizing Ollama installations."
    ),
    "openai-docs": (
        "OpenAI API documentation and usage guides for developers.",
        "Official OpenAI documentation covering API usage, models, best practices, and integration patterns for AI application development."
    ),
    "peraturan.go.id": (
        "Indonesian law and regulations assistant with expertise in legal documentation and regulatory compliance.",
        "A comprehensive legal knowledgebase containing Indonesian laws and regulations from peraturan.go.id. Functions as a specialized legal assistant providing clear, accurate answers about Indonesian law."
    ),
    "prosocial.world": (
        "Psychology-philosophy insights into human motivations enriched by modern scientific perspectives.",
        "A knowledgebase combining psychology and philosophy to explore human motivations and behavior. Provides nuanced insights enriched by modern philosophical, scientific, and psychological perspectives."
    ),
    "seculardharma": (
        "Secular dharma philosophy exploring ethical living paths through science, anthropology, and modern thought.",
        "A philosophical knowledgebase focused on secular interpretations of dharma and ethical living. Explores dharma as diverse paths adopted by individuals and groups."
    ),
    "smi": (
        "SMI knowledgebase for specialized domain research.",
        "Specialized domain knowledgebase for SMI-related research and documentation."
    ),
    "uv": (
        "Full-stack programming and AI systems engineering expertise for advanced technical solutions.",
        "A technical knowledgebase designed for experienced full-stack programmers and AI systems engineers. Provides comprehensive guidance on software development and AI integration."
    ),
    "wayang.net": (
        "Indonesian anthropology specialization focusing on wayang culture and traditional arts.",
        "An anthropological knowledgebase specializing in Indonesian culture, particularly wayang (traditional puppet theater) and related arts."
    ),
}


async def run_customkb_search(
    kb: str,
    query: str,
    top_k: int = 50,
    output_format: str = "markdown"
) -> str:
    """Execute customkb query with context-only mode.

    Args:
        kb: Knowledgebase name
        query: Search query string
        top_k: Number of results to return
        output_format: Output format (xml, json, markdown, plain)

    Returns:
        Search results in specified format
    """
    cmd = [
        "/ai/scripts/customkb/customkb",
        "query",
        kb,
        query,
        "-c",  # context-only (no AI response)
        "-k", str(top_k),
        "-f", output_format,
        "-q"  # quiet mode
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "VECTORDBS": VECTORDBS}
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            return stdout.decode("utf-8")
        else:
            error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
            return f"Error querying {kb}: {error_msg}"
    except (OSError, RuntimeError, ValueError) as e:
        return f"Error executing search: {str(e)}"


def get_available_kbs() -> list[str]:
    """Get list of available knowledgebases."""
    available = []
    vectordbs_path = Path(VECTORDBS)

    if vectordbs_path.exists():
        for item in vectordbs_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                cfg_file = item / f"{item.name}.cfg"
                if cfg_file.exists():
                    available.append(item.name)

    return sorted(available)


# ============================================================================
# Utility Tools
# ============================================================================

@mcp.tool(
    name="list_knowledgebases",
    annotations={
        "title": "List Available Knowledgebases",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def list_knowledgebases() -> str:
    """List all available knowledgebases with their descriptions.

    Returns a table of KB names and short descriptions to help
    identify which knowledgebase to query for a given topic.
    """
    available = get_available_kbs()

    lines = ["# Available Knowledgebases\n"]
    lines.append("| KB Name | Description |")
    lines.append("|---------|-------------|")

    for kb in available:
        short_desc, _ = KB_METADATA.get(kb, ("No description available", ""))
        lines.append(f"| `{kb}` | {short_desc} |")

    lines.append(f"\n*Total: {len(available)} knowledgebases*")
    return "\n".join(lines)


@mcp.tool(
    name="get_kb_info",
    annotations={
        "title": "Get Knowledgebase Details",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_kb_info(
    knowledgebase: str = Field(..., description="Name of the knowledgebase")
) -> str:
    """Get detailed information about a specific knowledgebase.

    Returns the full description, availability status, and usage hints.
    """
    available = get_available_kbs()

    if knowledgebase not in available:
        return f"Error: Knowledgebase '{knowledgebase}' not found.\n\nAvailable: {', '.join(available)}"

    short_desc, full_desc = KB_METADATA.get(
        knowledgebase,
        ("No description available", "No detailed description available.")
    )

    return f"""# {knowledgebase}

**Short Description:**
{short_desc}

**Full Description:**
{full_desc}

**Status:** Available

**Usage:**
```
search_{knowledgebase}(query="your search query", top_k=50)
```
"""


# ============================================================================
# KB-Specific Search Tools
# ============================================================================

@mcp.tool(
    name="search_appliedanthropology",
    annotations={
        "title": "Search Applied Anthropology KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_appliedanthropology(
    query: str = Field(..., description="Search query about anthropology, evolution, or dharma"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search applied anthropology knowledge covering human evolution, cultural development, and secular dharma."""
    return await run_customkb_search("appliedanthropology", query, top_k, output_format)


@mcp.tool(
    name="search_jakartapost",
    annotations={
        "title": "Search Jakarta Post Archive KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_jakartapost(
    query: str = Field(..., description="Search query about Indonesian history, politics 1994-2005"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Jakarta Post archive (1994-2005) for Indonesian politics, society, and historical events."""
    return await run_customkb_search("jakartapost", query, top_k, output_format)


@mcp.tool(
    name="search_okusiassociates",
    annotations={
        "title": "Search Okusi Associates KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_okusiassociates(
    query: str = Field(..., description="Search query about PMA companies, Indonesian corporate services"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Indonesian PMA company setup, corporate law, taxation, and business compliance knowledge."""
    return await run_customkb_search("okusiassociates", query, top_k, output_format)


@mcp.tool(
    name="search_okusimail",
    annotations={
        "title": "Search Okusi Mail KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_okusimail(
    query: str = Field(..., description="Search query about Okusi business inquiries and responses"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Okusi Associates email correspondence for business inquiry patterns and responses."""
    return await run_customkb_search("okusimail", query, top_k, output_format)


@mcp.tool(
    name="search_okusiresearch",
    annotations={
        "title": "Search Okusi Research KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_okusiresearch(
    query: str = Field(..., description="Search query about Indonesian investment research"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Indonesian direct investment research, corporate law, and PMA management knowledge."""
    return await run_customkb_search("okusiresearch", query, top_k, output_format)


@mcp.tool(
    name="search_ollama",
    annotations={
        "title": "Search Ollama KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_ollama(
    query: str = Field(..., description="Search query about Ollama configuration and usage"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Ollama configuration, operation, and AI systems engineering knowledge."""
    return await run_customkb_search("ollama", query, top_k, output_format)


@mcp.tool(
    name="search_openai_docs",
    annotations={
        "title": "Search OpenAI Docs KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_openai_docs(
    query: str = Field(..., description="Search query about OpenAI API and models"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search OpenAI API documentation, models, and integration patterns."""
    return await run_customkb_search("openai-docs", query, top_k, output_format)


@mcp.tool(
    name="search_peraturan",
    annotations={
        "title": "Search Indonesian Regulations KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_peraturan(
    query: str = Field(..., description="Search query about Indonesian laws and regulations"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Indonesian laws and regulations from peraturan.go.id for legal and compliance research."""
    return await run_customkb_search("peraturan.go.id", query, top_k, output_format)


@mcp.tool(
    name="search_prosocial",
    annotations={
        "title": "Search Prosocial World KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_prosocial(
    query: str = Field(..., description="Search query about psychology, philosophy, human behavior"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search psychology-philosophy insights into human motivations and behavior."""
    return await run_customkb_search("prosocial.world", query, top_k, output_format)


@mcp.tool(
    name="search_seculardharma",
    annotations={
        "title": "Search Secular Dharma KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_seculardharma(
    query: str = Field(..., description="Search query about secular dharma, ethics, philosophy"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search secular dharma philosophy, ethical living, and modern interpretations of dharma."""
    return await run_customkb_search("seculardharma", query, top_k, output_format)


@mcp.tool(
    name="search_smi",
    annotations={
        "title": "Search SMI KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_smi(
    query: str = Field(..., description="Search query for SMI domain"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search SMI knowledgebase for specialized domain research."""
    return await run_customkb_search("smi", query, top_k, output_format)


@mcp.tool(
    name="search_uv",
    annotations={
        "title": "Search UV Technical KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_uv(
    query: str = Field(..., description="Search query about programming, AI systems, software development"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search full-stack programming, AI systems engineering, and technical solutions."""
    return await run_customkb_search("uv", query, top_k, output_format)


@mcp.tool(
    name="search_wayang",
    annotations={
        "title": "Search Wayang Culture KB",
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def search_wayang(
    query: str = Field(..., description="Search query about wayang, Indonesian culture, traditional arts"),
    top_k: int = Field(50, description="Number of results to return"),
    output_format: str = Field("markdown", description="Output format: xml, json, markdown, plain")
) -> str:
    """Search Indonesian wayang culture, traditional puppet theater, and cultural anthropology."""
    return await run_customkb_search("wayang.net", query, top_k, output_format)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
