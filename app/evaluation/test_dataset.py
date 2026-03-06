"""
Hand-crafted test dataset for evaluating the healthcare RAG system.

WHY HAND-CRAFTED?
In production evaluation, you'd have domain experts (clinicians) create test
sets. For development, we create a small set that covers different question
types and difficulty levels. Each entry has:

- question: What the user asks
- ground_truth: The correct answer (from the source document)
- ground_truth_context: The specific text from the document that contains
  the answer. Ragas uses this to check if the retriever found the right chunks.

The test set should cover:
1. Simple factual lookups (dosage, threshold)
2. Multi-part questions (multiple facts needed)
3. Questions that require warnings/contraindications
4. Edge cases (question not fully answerable from context)
"""

TEST_DATASET = [
    {
        "question": "What are the diagnostic criteria for Type 2 Diabetes?",
        "ground_truth": (
            "Diagnosis of T2DM is based on: Fasting Plasma Glucose >= 126 mg/dL, "
            "2-hour Plasma Glucose >= 200 mg/dL during OGTT, HbA1c >= 6.5%, or "
            "Random Plasma Glucose >= 200 mg/dL with classic symptoms."
        ),
        "ground_truth_context": (
            "Diagnosis of T2DM should be based on one or more of the following "
            "laboratory criteria: Fasting Plasma Glucose (FPG) greater than or "
            "equal to 126 mg/dL (7.0 mmol/L) after at least 8 hours of fasting. "
            "2-hour Plasma Glucose greater than or equal to 200 mg/dL (11.1 mmol/L) "
            "during an oral glucose tolerance test (OGTT) using a 75g glucose load. "
            "HbA1c greater than or equal to 6.5% (48 mmol/mol). Random Plasma "
            "Glucose greater than or equal to 200 mg/dL (11.1 mmol/L) in a patient "
            "with classic symptoms of hyperglycemia."
        ),
    },
    {
        "question": "What is the recommended first-line drug for Type 2 Diabetes and what is the starting dose?",
        "ground_truth": (
            "Metformin is the recommended first-line agent. Starting dose is "
            "500mg once daily with meals, titrating to a maximum of 2000mg daily "
            "in divided doses over 4-8 weeks."
        ),
        "ground_truth_context": (
            "Metformin remains the recommended first-line pharmacological agent "
            "for T2DM unless contraindicated. Initiate at 500mg once daily with "
            "meals, titrating to a maximum of 2000mg daily in divided doses over "
            "4-8 weeks as tolerated."
        ),
    },
    {
        "question": "When should metformin be temporarily stopped?",
        "ground_truth": (
            "Metformin should be temporarily discontinued 48 hours before and "
            "after administration of iodinated contrast media due to the risk "
            "of contrast-induced nephropathy and subsequent lactic acidosis."
        ),
        "ground_truth_context": (
            "Metformin should be temporarily discontinued 48 hours before and "
            "after administration of iodinated contrast media due to the risk "
            "of contrast-induced nephropathy and subsequent lactic acidosis."
        ),
    },
    {
        "question": "What is the HbA1c target for elderly patients with diabetes?",
        "ground_truth": (
            "For older adults (age 65 and above) or patients with significant "
            "comorbidities, the HbA1c target is less than 8.0% (64 mmol/mol)."
        ),
        "ground_truth_context": (
            "Older adults (age 65 and above) or patients with significant "
            "comorbidities: HbA1c less than 8.0% (64 mmol/mol)"
        ),
    },
    {
        "question": "What drugs are recommended for diabetic patients with heart failure?",
        "ground_truth": (
            "SGLT2 inhibitors are preferred for patients with heart failure "
            "(HFrEF or HFpEF), based on evidence from DAPA-HF and "
            "EMPEROR-Reduced trials."
        ),
        "ground_truth_context": (
            "For patients with heart failure (HFrEF or HFpEF): "
            "SGLT2 inhibitors — preferred based on evidence from DAPA-HF "
            "and EMPEROR-Reduced trials"
        ),
    },
    {
        "question": "How should painful diabetic neuropathy be treated?",
        "ground_truth": (
            "First-line treatments include pregabalin (75-300mg daily), "
            "duloxetine (60-120mg daily), or gabapentin (900-3600mg daily "
            "in divided doses)."
        ),
        "ground_truth_context": (
            "First-line treatments for painful diabetic neuropathy include "
            "pregabalin (75-300mg daily), duloxetine (60-120mg daily), or "
            "gabapentin (900-3600mg daily in divided doses)."
        ),
    },
]
