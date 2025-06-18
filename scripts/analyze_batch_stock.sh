curl -X POST http://127.0.0.1:8001/analyze/batch \
-H "Content-Type: application/json" \
-d '{
    "stock_codes": ["000001", "600001"],
    "start_date": "20230101",
    "end_date": "20240101",
    "days": 60
}'