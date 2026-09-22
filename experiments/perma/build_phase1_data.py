"""PERMA reader prompt shared by RPMem training and evaluation."""

def build_prompt(question: str, options: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(options))]
    option_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
    return (
        "Answer the multiple-choice question using the conversation memory.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Answer with the letter only ({labels[0]}-{labels[-1]})."
    )
