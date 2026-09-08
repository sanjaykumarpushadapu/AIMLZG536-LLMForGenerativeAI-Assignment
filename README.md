# AIMLZG536-LLMForGenerativeAI-Assignment

## Assignment 1A Colab Runtime

The team will run Assignment 1A on a **Google Colab T4 GPU**. Google Drive is optional during
development: `Assignment1A.ipynb` runs from the notebook folder with Drive disabled, and Drive
can be enabled only when persistent checkpoints are needed.

1. In Colab, select **Runtime → Change runtime type → T4 GPU**.
2. Clone the repository or upload the `Ass-1` folder, then change the working directory to
   `Ass-1` before running `Assignment1A.ipynb`.
3. Confirm that `raw_pdfs/` contains the Type 2 Diabetes source PDFs. The assignment reference
   PDFs are not training-corpus documents.
4. Confirm `torch.cuda.is_available()` and record the GPU/runtime versions in the notebook.
5. Before CPT or QLoRA, check `torch.cuda.is_bf16_supported()`. The assignment specifies BF16
   for T4, but if the runtime cannot execute it reliably, agree an FP16 fallback before
   producing graded results.

For persistent storage, set `USE_GOOGLE_DRIVE = True` in the first notebook cell, mount Drive,
and use `MyDrive/AIMLZG536-Ass-1`. With Drive disabled, checkpoints are stored under the local
`persistent/` folder and will be lost when the Colab runtime disconnects. See
`Ass-1/SPRINT-PLAN.md` for ownership, handoffs, and the complete execution sequence.
