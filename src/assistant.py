"""AI Liver Health Assistant.

Uses the model result + XAI output to explain things in plain language.
  * If ANTHROPIC_API_KEY is set (and `anthropic` is installed) -> LLM answers.
  * Otherwise -> a built-in rule-based assistant, so the app works offline.
"""
import os

DISCLAIMER = ("This is an AI-based screening estimate for educational purposes only. "
              "It is not a medical diagnosis. Please consult a qualified healthcare professional.")

GLOSSARY = {
    "alt": "ALT (alanine aminotransferase) is an enzyme found mostly in liver cells. It leaks into the blood when liver cells are irritated or damaged.",
    "ast": "AST (aspartate aminotransferase) is an enzyme in the liver, heart and muscles. High levels together with ALT can point to liver cell injury.",
    "bilirubin": "Bilirubin is a yellow pigment made when red blood cells break down. The liver clears it; high levels can cause jaundice. 'Direct' bilirubin is the form already processed by the liver.",
    "alkaline": "Alkaline phosphatase (ALP) is an enzyme in the liver and bile ducts (and bone). Raised levels can suggest bile-duct problems.",
    "alp": "Alkaline phosphatase (ALP) is an enzyme in the liver and bile ducts (and bone). Raised levels can suggest bile-duct problems.",
    "albumin": "Albumin is a protein made by the liver. Low levels can mean the liver is not producing enough protein.",
    "protein": "Total proteins measure albumin plus globulins in the blood; the liver makes most of them.",
    "ratio": "The albumin/globulin ratio compares the two main blood protein groups. A low ratio can be associated with liver or kidney problems.",
    "pca": "PCA (Principal Component Analysis) compresses correlated lab values into a few uncorrelated 'components' before the classifier sees them.",
}


def _summary(result: dict) -> str:
    p = result["probability"]
    level = "higher" if p >= 0.5 else "lower"
    
    top = result["local"].head(3)
    
    drivers = []
    
    for r in top.itertuples():
        if abs(r.contribution) < 0.005:
            effect = "minimal measurable effect"
        elif r.contribution > 0:
            effect = "raises the estimate"
        else:
            effect = "lowers the estimate"
        drivers.append(f"{r.label} ({effect})")
    
    abn = [f"{x['name']} is {x['status']} ({x['value']:g} {x['unit']})"
           for x in result["flags"] if x["status"] != "normal"]
    text = (f"The model ({result['model']}) estimates a **{level} likelihood of liver disease** "
            f"(probability about {p:.0%}).\n\n"
            f"The inputs that influenced this result most were: {', '.join(drivers)}.\n\n")
    text += ("Values outside the typical reference range: " + "; ".join(abn) + ".\n\n") if abn else \
            "All entered lab values are within typical reference ranges.\n\n"
    return text + DISCLAIMER


def rule_based_answer(question: str, result: dict) -> str:
    q = question.lower()
    for key, txt in GLOSSARY.items():
        if key in q:
            return txt + "\n\n" + DISCLAIMER
    if any(w in q for w in ("why", "reason", "because", "influence", "driver")):
        
        top = result["local"].head(4)
        
        lines = []
        
        for r in top.itertuples():
            if abs(r.contribution) < 0.005:
                effect = "minimal measurable effect"
            elif r.contribution > 0:
                effect = "pushed toward liver disease"
            else:
                effect = "pushed away from liver disease"

            lines.append(
        f"- **{r.label}** = {r.value:g}: {effect} "
        f"({r.contribution:+.1%} probability)"
            )
        return ("These features had the biggest effect on this prediction:\n" + "\n".join(lines) +
                "\n\n" + DISCLAIMER)
    if any(w in q for w in ("accurate", "reliable", "trust", "sure")):
        return ("The model is a statistical estimate trained on past patient records and can be wrong, "
                "especially for unusual patients. Treat it as a screening aid only.\n\n" + DISCLAIMER)
    if any(w in q for w in ("do", "next", "should", "treat", "cure", "diet")):
        return ("I can't give treatment advice. A sensible next step is to share your lab reports with a "
                "doctor, who can interpret them together with your symptoms and history.\n\n" + DISCLAIMER)
    return _summary(result)


def _llm_answer(question: str, result: dict):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        ctx = (f"Model: {result['model']}\nPredicted probability of liver disease: {result['probability']:.1%}\n"
               f"Top feature contributions (positive = toward disease):\n"
               + result["local"].head(6)[["label", "value", "contribution"]].to_string(index=False)
               + "\nReference-range flags:\n"
               + "\n".join(f"{x['name']}: {x['value']:g} {x['unit']} ({x['status']})" for x in result["flags"]))
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=600,
            system=("You are a careful liver-health explainer inside a student ML project. Explain the model "
                    "result in simple language using ONLY the context given. Never diagnose, never recommend "
                    "treatment or medication, and always remind the user to consult a doctor."),
            messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
        )
        return msg.content[0].text
    except Exception:
        return None  # fall back to rule-based


def answer(question: str, result: dict) -> str:
    return _llm_answer(question, result) or rule_based_answer(question, result)


def build_report(inputs: dict, result: dict) -> str:
    lines = ["# Liver Disease Prediction Report", "", "## Patient input"]
    for k, v in inputs.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Prediction",
              f"- Model: {result['model']}",
              f"- Result: {'Higher likelihood of liver disease' if result['probability'] >= 0.5 else 'Lower likelihood of liver disease'}",
              f"- Estimated probability: {result['probability']:.1%}", "",
              "## Important features (this patient)"]
    for r in result["local"].head(5).itertuples():
        lines.append(f"- {r.label} = {r.value:g} → {r.contribution:+.1%}")
    lines += ["", "## Reference-range check"]
    for x in result["flags"]:
        lines.append(f"- {x['name']}: {x['value']:g} {x['unit']} — {x['status']}")
    lines += ["", "## AI summary", _summary(result), "",
              "## Recommendation", "Please consult a healthcare professional for interpretation and next steps."]
    return "\n".join(lines)
