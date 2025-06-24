#!/bin/bash

# --- Configuration ---
# Your Hugging Face username
HF_USER="ApostolosK"
# The name of the repository you want to create/use on the Hub
REPO_NAME="chemflow-assets"
# --- End of Configuration ---

# Construct the full repository ID
REPO_ID="$HF_USER/$REPO_NAME"

echo "--------------------------------------------------"
echo "Preparing to upload files to Hugging Face Hub"
echo "Repository: $REPO_ID"
echo "--------------------------------------------------"

# 1. Create the repository on the Hub if it doesn't exist.
# The `--exist-ok` flag handles the check for you.
# We'll create it as a 'dataset' type, which is suitable for storing assets.
echo "[Step 1/3] Ensuring repository '$REPO_ID' exists..."
huggingface-cli repo create $REPO_ID --type dataset --exist-ok
echo "Repository is ready."
echo ""

# 2. Define the list of files and directories to upload.
# The format is "path/to/local/file_or_dir"
# For directories, add a trailing slash to the local path to upload its content.
FILES_TO_UPLOAD=(
  "ChemFlow/data/processed/zmc.smi"
  "ChemFlow/data/processed/zmc_data.pt"
  "ChemFlow/data/processed/zinc250k.smi"
)

DIRS_TO_UPLOAD=(
  "ChemFlow/data/interim/props"
  "ChemFlow/checkpoints/neural_ode"
  "ChemFlow/checkpoints/prop_predictor"
  "ChemFlow/checkpoints/vae"
)

# 3. Upload the files and directories
echo "[Step 2/3] Uploading individual files..."
for FILE in "${FILES_TO_UPLOAD[@]}"; do
  if [ -f "$FILE" ]; then
    echo "Uploading file: $FILE"
    # The first argument is the local path, the second is the path inside the repo.
    # Making them identical preserves the structure.
    huggingface-cli upload $REPO_ID "$FILE" "$FILE"
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
    # The first argument is the local path, the second is the path inside the repo.
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