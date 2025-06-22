curl -X POST http://127.0.0.1:8001/stocks/list \
-H "Content-Type: application/json" \
-d '{
    "exchange": "SH",
    "refresh": false
}'