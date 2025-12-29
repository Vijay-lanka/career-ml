import requests
from tabulate import tabulate
import csv

URL = "http://127.0.0.1:8000/predict"

test_cases = [
    {"experience": 3, "skills": ["Python", "ML", "SQL"], "education": "B.Tech"},
    {"experience": 5, "skills": ["Java", "Spring"], "education": "M.Tech"},
    {"experience": 1, "skills": ["HTML", "CSS", "JS"], "education": "B.Sc"},
]

table = []
csv_rows = []

for i, test in enumerate(test_cases, start=1):
    response = requests.post(URL, json=test)
    prediction = response.json()["recommendations"]  # ✅ FIX

    formatted = ", ".join(
        [f"{career}: {prob:.4f}" for career, prob in prediction.items()]
    )

    table.append([i, test, formatted])

    csv_rows.append([
        i,
        test["experience"],
        test["education"],
        ", ".join(test["skills"]),
        *prediction.values()
    ])

print(tabulate(table, headers=["Test #", "Input", "Predictions"], tablefmt="grid"))

# Save CSV
with open("prediction_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Test_No", "Experience", "Education", "Skills"] + list(prediction.keys()))
    writer.writerows(csv_rows)

print("\n✅ Results saved to 'prediction_results.csv'")
