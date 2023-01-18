# sentence-classifier via Streamlit
classify sentence type from emails

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mynameisbram-sentence-classifier-app-6v2po6.streamlit.app)


## Code snippet
```json

curl --location --request POST 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "Please let me know if we can schedule an introductory meeting to walk you through our capabilities and leading case studies."
}'

```
## output
```
{
    "confidence": 98.88,
    "prediction": "call_to_action",
    "preds": {
        "call_to_action": 0.9888,
        "credibility_statement": 0.1456,
        "greeting": 0.1357,
        "intention_statement": 0.7619,
        "intro": 0.3152,
        "problem_statement": 0.0534,
        "sign_off": 0.3479,
        "value_prop": 0.0018,
        "warm_up": 0.1175
    },
    "sentence": "Please let me know if we can schedule an introductory meeting to walk you through our capabilities and leading case studies.",
    "subconf": 87.58,
    "subpred": "meeting_cta",
    "subpreds": {
        "contact_cta": 0.4306,
        "feedback_cta": 0.8451,
        "information_cta": 0.8311,
        "intro_cta": 0.2292,
        "meeting_cta": 0.8758,
        "need_validation_cta": 0.1588,
        "rejection_cta": 0.5459,
        "response_cta": 0.8285,
        "webinar_cta": 0.0667
    }
}

```
