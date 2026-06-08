# EIBench: Epistemic Integrity Benchmark

**EIBench** is an open-source research platform designed to evaluate **Epistemic Robustness** in Retrieval-Augmented Generation (RAG) systems. 

It specifically measures the **Faithful Falsehood Rate (FFR)**: instances where a RAG system provides an answer that is perfectly supported by retrieved context but is factually incorrect according to independently verified ground truth.

## Research Hypothesis
> "Faithfulness and Source Integrity are independent evaluation dimensions. As poisoning increases, Faithfulness remains high while Source Integrity decreases, leading to an increase in Faithful Falsehoods."

## Architecture
- **Layer 1: Corpus Builder**: Ingests PolitiFact, FactCheck.org, and AVeriTeC.
- **Layer 2: Poisoning Engine**: Implements numerical, date, attribution, and causal adversarial shifts.
- **Layer 3: Retrieval Infrastructure**: Qdrant-based vector store with Sentence-BERT embeddings.
- **Layer 4: Generation Infrastructure**: Llama 3.1 and Mistral backends.
- **Layer 5: Evaluation Engine**: Custom metrics (SI, FFR, ERS) + RAGAS.
- **Layer 6: Analytics**: Automated plotting of degradation curves.

## Quick Start
```bash
# Deploy the infrastructure
sudo containerlab deploy -t topology.clab.yml

# Test Layer 1-3 (Ingestione, Avvelenamento e Retrieval)
python3 pipeline_eibench.py
```

## Core Metrics
1. **Source Integrity (SI)**: Factuality of retrieved snippets vs. ground truth.
2. **Faithful Falsehood Rate (FFR)**: $\frac{\text{Faithful False Answers}}{\text{Total Answers}}$.
3. **Epistemic Risk Score (ERS)**: Combined score of plausibility and verification difficulty.

## License
Apache 2.0
