curl -X POST http://127.0.0.1:8001/analyze/single \
-H "Content-Type: application/json" \
-d '{
    "stock_code": "000019",
    "start_date": "20240101",
    "end_date": "20250101",
    "days": 60
}'