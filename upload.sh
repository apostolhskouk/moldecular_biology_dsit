#!/bin/bash

# --- Configuration ---
HF_USER="ApostolosK"
# The name of the repository you want to create/use on the Hub
REPO_NAME="chemflow-assets"

# Construct the full repository ID
REPO_ID="$HF_USER/$REPO_NAME"

echo "--------------------------------------------------"
echo "Preparing to upload files to Hugging Face Hub"
echo "Repository: $REPO_ID"
echo "--------------------------------------------------"

# 1. Create the repository on the Hub if it doesn't exist.
echo "[Step 1/3] Ensuring repository '$REPO_ID' exists..."
huggingface-cli repo create $REPO_ID --type dataset --exist-ok
echo "Repository is ready."
echo ""

# 2. Define the list of files and directories to upload.
FILES_TO_UPLOAD=(
  "ChemFlow/experiments/success_rate/drd2_1000.csv"
  "ChemFlow/experiments/success_rate/drd2_1000.txt"
  "ChemFlow/experiments/success_rate/gsk3b_1000.csv"
  "ChemFlow/experiments/success_rate/gsk3b_1000.txt"
  "ChemFlow/experiments/success_rate/jnk3_1000.csv"
  "ChemFlow/experiments/success_rate/jnk3_1000.txt"
  "ChemFlow/experiments/success_rate/plogp_1000.csv"
  "ChemFlow/experiments/success_rate/plogp_1000.txt"
  "ChemFlow/experiments/success_rate/qed_1000.csv"
  "ChemFlow/experiments/success_rate/qed_1000.txt"
  "ChemFlow/experiments/success_rate/sa_1000.csv"
  "ChemFlow/experiments/success_rate/sa_1000.txt"
  "pdb_files/conf.txt"
)

DIRS_TO_UPLOAD=(
  "ChemFlow/data"
  "ChemFlow/checkpoints"
  "ChemFlow/extend/optimization_results_molgen_truncated_100steps"
)

# 3. Upload the files and directories
echo "[Step 2/3] Uploading individual files..."
for FILE in "${FILES_TO_UPLOAD[@]}"; do
  if [ -f "$FILE" ]; then
    echo "Uploading file: $FILE"
    huggingface-cli upload $REPO_ID "$FILE" "$FILE" --commit-message "Upload file $FILE"
  else
    echo "Warning: File '$FILE' not found. Skipping."
  fi
done
echo "File uploads complete."
echo ""

echo "[Step 3/3] Uploading directories..."
for DIR in "${DIRS_TO_UPLOAD[@]}"; do
  if [ -d "$DIR" ]; then
    echo "Uploading directory: $DIR"
    huggingface-cli upload $REPO_ID "$DIR/" "$DIR" --commit-message "Upload directory $DIR"
  else
    echo "Warning: Directory '$DIR' not found. Skipping."
  fi
done
echo "Directory uploads complete."
echo ""

echo "--------------------------------------------------"
echo "✅ All operations finished successfully!"
echo "You can view your files at: https://huggingface.co/datasets/$REPO_ID/tree/main"
echo "--------------------------------------------------"