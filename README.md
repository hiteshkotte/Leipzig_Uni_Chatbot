<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiWi AI Tutor - Educational Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: #f5f5f5; padding: 20px; text-align: center; }
        .section { margin: 40px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .architecture img { max-width: 100%; height: auto; }
        .tech-stack { columns: 3; list-style-type: none; padding: 0; }
        .tech-stack li { break-inside: avoid; margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
        .evaluation-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .evaluation-table th, .evaluation-table td { border: 1px solid #ddd; padding: 10px; text-align: center; }
    </style>
</head>
<body>
    <header>
        <h1>BiWi AI Tutor: Scalable Educational Chatbot</h1>
        <p>Powered by Retrieval-Augmented Generation (RAG) and LLMs</p>
    </header>

    <div class="section">
        <h2>Project Overview</h2>
        <p>The BiWi AI Tutor is an advanced educational chatbot designed to provide scalable, personalized learning support for higher education students. Developed as part of the tech4compKI initiative funded by the German Federal Ministry of Education and Research, this system leverages large language models (LLMs) and retrieval-augmented generation to deliver context-aware responses to student queries.</p>
        <p>Key objectives:</p>
        <ul>
            <li>Provide 24/7 academic support</li>
            <li>Deliver contextually relevant answers using course materials</li>
            <li>Support self-regulated learning through interactive feedback</li>
        </ul>
    </div>

    <div class="section">
        <h2>Technical Architecture</h2>
        <div class="architecture">
            <h3>1. Learning Material Indexing & Retrieval</h3>
            <!-- Insert Figure 3 image here -->
            <p><strong>Figure 3: Indexing and Retrieval Process</strong></p>
            <p>This process involves:</p>
            <ol>
                <li>Course material parsing using Llama Parser</li>
                <li>Text chunking into 1024-token segments</li>
                <li>Vector embeddings generation with OpenAI's text-embedding-3-large</li>
                <li>Hybrid retrieval using semantic search (Vector Index) and keyword search (BM25 Index)</li>
                <li>Context reranking using Cohere's reranker model</li>
            </ol>

            <h3>2. Chatbot Interaction Flow</h3>
            <!-- Insert Figure 4 image here -->
            <p><strong>Figure 4: Chatbot Interaction Flow</strong></p>
            <p>Key components:</p>
            <ul>
                <li>Conversation history retrieval</li>
                <li>Tool selection using LangChain's Function Calling Agent</li>
                <li>Context-aware response generation</li>
                <li>Dynamic multi-turn interaction handling</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Technology Stack</h2>
        <ul class="tech-stack">
            <li><strong>LLM Engine:</strong> OpenAI GPT-3.5-turbo (Fine-tuned for educational domain)</li>
            <li><strong>Retrieval System:</strong> LangChain with hybrid Vector + BM25 indexes</li>
            <li><strong>Reranking:</strong> Cohere reranker-v3.0 model</li>
            <li><strong>Text Parsing:</strong> LlamaIndex with Llama Parser module</li>
            <li><strong>Embeddings:</strong> OpenAI text-embedding-3-large</li>
            <li><strong>Observability:</strong> LangSmith for interaction tracking</li>
            <li><strong>Deployment:</strong> Dockerized container with FastAPI backend</li>
            <li><strong>Frontend:</strong> React-based chat interface</li>
            <li><strong>Database:</strong> PostgreSQL for conversation history</li>
            <li><strong>Monitoring:</strong> Prometheus + Grafana</li>
        </ul>
    </div>

    <div class="section">
        <h2>Performance Evaluation</h2>
        <p>The system was evaluated using a dataset of 60 questions across three categories:</p>
        <table class="evaluation-table">
            <tr>
                <th>Category</th>
                <th>Human Evaluation Accuracy</th>
                <th>GPT-4 Evaluation Accuracy</th>
            </tr>
            <tr>
                <td>Lecture Content</td>
                <td>80%</td>
                <td>85%</td>
            </tr>
            <tr>
                <td>Seminar Material</td>
                <td>75%</td>
                <td>75%</td>
            </tr>
            <tr>
                <td>Organizational Queries</td>
                <td>85%</td>
                <td>85%</td>
            </tr>
        </table>
        <p>Reranker implementation improved organizational query accuracy to 100% in subsequent tests</p>
    </div>

    <div class="section">
        <h2>Key Innovations</h2>
        <ul>
            <li>Hybrid retrieval system combining semantic and keyword search</li>
            <li>Dynamic context reranking for improved relevance</li>
            <li>Multi-turn conversation handling with LangChain agents</li>
            <li>Integration with existing LMS infrastructure</li>
            <li>Adaptive response generation based on conversation history</li>
        </ul>
    </div>

    <div class="section">
        <h2>Future Development</h2>
        <ul>
            <li>Integration of emotional intelligence components</li>
            <li>Personalized learning pathway generation</li>
            <li>Multi-language support for international courses</li>
            <li>Advanced bias detection and mitigation</li>
            <li>Hybrid model deployment with on-premise options</li>
        </ul>
    </div>

    <footer>
        <p>Developed by the tech4compKI team | Funded by BMBF (Grant No. 16DHB2206)</p>
    </footer>
</body>
</html>
