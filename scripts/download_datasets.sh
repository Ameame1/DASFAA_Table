#!/bin/bash

# Download and prepare all datasets for Table QA
# Usage: bash scripts/download_datasets.sh

set -e

echo "======================================"
echo "Downloading Table QA Datasets"
echo "======================================"

# Create data directories
mkdir -p data/{wikitq,tabfact,fetaqa,semeval2025}/raw
mkdir -p data/{wikitq,tabfact,fetaqa,semeval2025}/processed

# 1. WikiTQ
echo ""
echo "1. Downloading WikiTQ..."
if [ ! -d "data/wikitq/raw/WikiTableQuestions" ]; then
    cd data/wikitq/raw
    wget https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip
    unzip master.zip
    mv WikiTableQuestions-master WikiTableQuestions
    rm master.zip
    cd ../../..
    echo "✓ WikiTQ downloaded"
else
    echo "✓ WikiTQ already exists"
fi

# 2. TabFact
echo ""
echo "2. Downloading TabFact..."
if [ ! -d "data/tabfact/raw/Table-Fact-Checking" ]; then
    cd data/tabfact/raw
    git clone https://github.com/wenhuchen/Table-Fact-Checking.git
    cd ../../..
    echo "✓ TabFact downloaded"
else
    echo "✓ TabFact already exists"
fi

# 3. FeTaQA
echo ""
echo "3. Downloading FeTaQA..."
if [ ! -d "data/fetaqa/raw/FeTaQA" ]; then
    cd data/fetaqa/raw
    git clone https://github.com/Yale-LILY/FeTaQA.git
    cd ../../..
    echo "✓ FeTaQA downloaded"
else
    echo "✓ FeTaQA already exists"
fi

# 4. SemEval-2025 Task 8
echo ""
echo "4. SemEval-2025 Task 8 (DataBench)..."
echo "Note: This dataset requires manual download from CodaBench"
echo "Visit: https://www.codabench.org/competitions/3360/"
echo "Download and place files in: data/semeval2025/raw/"
echo ""

# Process datasets
echo ""
echo "======================================"
echo "Processing datasets to JSONL format..."
echo "======================================"

# Process WikiTQ
echo ""
echo "Processing WikiTQ..."
python scripts/preprocess_wikitq.py

# Process TabFact
echo "Processing TabFact..."
python scripts/preprocess_tabfact.py

# Process FeTaQA
echo "Processing FeTaQA..."
python scripts/preprocess_fetaqa.py

echo ""
echo "======================================"
echo "Dataset preparation complete!"
echo "======================================"
echo ""
echo "Dataset statistics:"
python scripts/show_dataset_stats.py
