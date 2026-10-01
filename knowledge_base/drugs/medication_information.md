# Medication information policy for the prototype

Medication records in the project's CSV dataset can be used as structured model-training data and retrieval context. The dataset may contain drug names, dosage forms, typical dosage text, quantity fields, storage notes, use cases, regions, and mission types.

The RAG layer should treat the dataset as project data rather than authoritative prescribing guidance. Any dosage or medication instruction displayed to a user should be clearly labelled as dataset/model output and should be reviewed against authoritative clinical references before real-world use.
