# --model meta-llama/Llama-3.3-70B-Instruct
--model meta-llama/Llama-3.1-8B-Instruct
--log-level info
--enable-metrics
--log-requests
--tensor-parallel-size 4
--pipeline-parallel-size 1
--chunked-prefill-size 4096
--kv-cache-dtype fp8_e5m2
