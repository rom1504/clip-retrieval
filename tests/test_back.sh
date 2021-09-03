echo '{"example_index": "/tmp/my_index"}' > indices_paths.json
clip-retrieval back --port 1234 --indices-paths indices_paths.json &
FOO_PID=$!
sleep 10
curl -d '{"text":"cat", "modality":"image", "num_images": 10, "indice_name": "example_index"}' -H "Content-Type: application/json" -X POST http://localhost:1234/knn-service
kill $FOO_PID
