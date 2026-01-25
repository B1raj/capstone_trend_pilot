"""
Generate mock trending technology topics for LinkedIn post generation.
"""

import random
from typing import List, Dict


def get_mock_trends() -> List[Dict[str, str]]:
    """
    Generate a list of mock trending technology topics.

    Returns:
        List of dictionaries containing topic, description, and relevance keywords.
    """
    all_trends = [
        {
            "topic": "AI Agents and Autonomous Systems",
            "description": "The rise of AI agents that can perform complex tasks autonomously, from coding assistants to customer service bots. LangChain, AutoGPT, and similar frameworks are enabling new possibilities.",
            "relevance_keywords": ["artificial intelligence", "machine learning", "automation", "llm", "langchain", "gpt", "claude", "ai agents", "autonomous systems"]
        },
        {
            "topic": "Retrieval Augmented Generation (RAG)",
            "description": "RAG systems combine large language models with external knowledge bases to provide accurate, up-to-date responses. Critical for enterprise AI applications.",
            "relevance_keywords": ["rag", "vector database", "embeddings", "llm", "ai", "machine learning", "knowledge base", "semantic search"]
        },
        {
            "topic": "Platform Engineering and Developer Experience",
            "description": "Organizations are investing in platform teams to improve developer productivity through self-service infrastructure, golden paths, and internal developer platforms.",
            "relevance_keywords": ["devops", "platform engineering", "developer experience", "infrastructure", "kubernetes", "ci/cd", "internal tools"]
        },
        {
            "topic": "FinOps and Cloud Cost Optimization",
            "description": "As cloud spending grows, organizations are adopting FinOps practices to optimize costs while maintaining performance and innovation velocity.",
            "relevance_keywords": ["cloud", "aws", "azure", "gcp", "finops", "cost optimization", "cloud economics", "infrastructure"]
        },
        {
            "topic": "Zero Trust Security Architecture",
            "description": "The shift from perimeter-based security to zero trust models where every access request is verified, regardless of location.",
            "relevance_keywords": ["cybersecurity", "security", "zero trust", "authentication", "identity", "access management", "network security"]
        },
        {
            "topic": "Rust for Systems Programming",
            "description": "Rust adoption is accelerating for systems programming, WebAssembly, and performance-critical applications due to its memory safety guarantees.",
            "relevance_keywords": ["rust", "systems programming", "webassembly", "performance", "memory safety", "programming languages"]
        },
        {
            "topic": "Real-time Data Streaming and Event-Driven Architecture",
            "description": "Organizations are moving from batch processing to real-time data streaming using Kafka, Flink, and event-driven architectures.",
            "relevance_keywords": ["kafka", "streaming", "event-driven", "real-time", "data engineering", "apache flink", "microservices"]
        },
        {
            "topic": "Observability 2.0: OpenTelemetry and Beyond",
            "description": "Modern observability goes beyond monitoring with distributed tracing, metrics, and logs unified through OpenTelemetry standards.",
            "relevance_keywords": ["observability", "monitoring", "opentelemetry", "distributed tracing", "metrics", "logging", "devops"]
        },
        {
            "topic": "AI-Powered Code Generation and GitHub Copilot",
            "description": "AI coding assistants are transforming software development, with tools like GitHub Copilot, Amazon CodeWhisperer, and Cursor raising developer productivity.",
            "relevance_keywords": ["github copilot", "ai", "code generation", "developer tools", "productivity", "software development", "llm"]
        },
        {
            "topic": "Web3 and Decentralized Applications",
            "description": "Despite market volatility, web3 technologies continue to evolve with new use cases in DeFi, NFTs, and decentralized identity.",
            "relevance_keywords": ["web3", "blockchain", "ethereum", "smart contracts", "defi", "nft", "decentralization"]
        },
        {
            "topic": "Edge Computing and 5G Integration",
            "description": "The combination of edge computing and 5G networks enables low-latency applications for IoT, autonomous vehicles, and AR/VR.",
            "relevance_keywords": ["edge computing", "5g", "iot", "low latency", "distributed systems", "networking"]
        },
        {
            "topic": "MLOps and ML Model Governance",
            "description": "As ML models move to production, MLOps practices ensure reliable deployment, monitoring, and governance of machine learning systems.",
            "relevance_keywords": ["mlops", "machine learning", "ai", "model deployment", "ml engineering", "data science", "governance"]
        },
        {
            "topic": "Privacy-Preserving Machine Learning",
            "description": "Techniques like federated learning and differential privacy enable ML models to learn from sensitive data without compromising privacy.",
            "relevance_keywords": ["privacy", "machine learning", "federated learning", "differential privacy", "data protection", "security"]
        },
        {
            "topic": "Developer Productivity Metrics and DORA",
            "description": "Engineering organizations are adopting DORA metrics and similar frameworks to measure and improve software delivery performance.",
            "relevance_keywords": ["dora", "metrics", "productivity", "software engineering", "devops", "performance", "agile"]
        },
        {
            "topic": "Quantum Computing for Enterprise",
            "description": "While still early, quantum computing is showing promise for optimization problems, cryptography, and drug discovery.",
            "relevance_keywords": ["quantum computing", "quantum algorithms", "optimization", "cryptography", "emerging technology"]
        }
    ]

    # Shuffle and return 8-10 random trends
    random.shuffle(all_trends)
    num_trends = random.randint(8, 10)
    return all_trends[:num_trends]


def get_trend_by_topic(topic: str) -> Dict[str, str]:
    """
    Get a specific trend by topic name.

    Args:
        topic: The topic name to search for

    Returns:
        Dictionary containing the trend information, or None if not found
    """
    all_trends = get_mock_trends()
    for trend in all_trends:
        if trend["topic"].lower() == topic.lower():
            return trend
    return None
