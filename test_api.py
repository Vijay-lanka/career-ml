import requests
import json

URL = "http://127.0.0.1:8000/predict"

test_data = {
    "experience": 3,
    "education": "B.Tech",
    "skills": ["Python", "Machine Learning", "SQL"],
    "resume_text": "Experienced Python developer with machine learning and SQL projects"
}

response = requests.post(URL, json=test_data, timeout=10)

print("Status Code:", response.status_code)

data = response.json()
print(json.dumps(data, indent=2))

print("\nTop 3 Career Predictions:")
for item in data["top_3_recommendations"]:
    print(f"- {item['career']} : {item['confidence']*100:.2f}%")
