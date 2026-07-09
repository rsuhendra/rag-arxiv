from __future__ import annotations

TOPIC_KEYWORDS = {
    "LLMs": ["large language model", "llm", "language model", "instruction tuning", "prompt", "transformer", "alignment", "rlhf", "rag", "retrieval augmented", "gpt"],
    "Causal Inference": ["causal", "causality", "treatment effect", "confounding", "instrumental variable", "counterfactual", "causal discovery", "directed acyclic graph", "dag"],
    "Reinforcement Learning": ["reinforcement learning", "policy gradient", "q-learning", "markov decision process", "mdp", "reward", "actor critic", "bandit", "offline rl"],
    "Diffusion / Generative Models": ["diffusion", "score-based", "generative model", "vae", "gan", "normalizing flow", "denoising", "latent diffusion"],
    "Optimization": ["optimization", "gradient descent", "convex", "nonconvex", "stochastic gradient", "adam", "learning rate", "scheduler"],
    "Computer Vision": ["image", "vision", "segmentation", "object detection", "video", "visual", "cnn", "clip", "multimodal"],
    "NLP": ["natural language", "nlp", "text classification", "machine translation", "summarization", "named entity", "sentiment"],
    "Graph ML": ["graph neural network", "gnn", "graph", "node classification", "link prediction", "message passing"],
    "Theory": ["generalization", "sample complexity", "bound", "theorem", "proof", "convergence rate", "regret"],
}

def assign_topic(title: str, abstract: str, categories: str = "") -> str:
    text = f"{title} {abstract} {categories}".lower()
    best_topic = "Other"
    best_score = 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_topic = topic
            best_score = score
    return best_topic
