import requests

response = requests.post(
    "http://localhost:5000/predict-risk-by-name",
    json={
        "product_name": "Nike Running Shoes"
    },
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.json())
