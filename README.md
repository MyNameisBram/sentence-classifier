# sentence-classifier via Streamlit
classify sentence type from emails

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mynameisbram-sentence-classifier-app-6v2po6.streamlit.app)


## Code snippet
```

curl --location --request POST 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "Please let me know if we can schedule an introductory meeting to walk you through our capabilities and leading case studies."
}'

```
