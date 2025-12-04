# --model meta-llama/Llama-3.3-70B-Instruct
--model meta-llama/Llama-3.1-8B-Instruct
--log-level info
--enable-metrics
--log-requests
--tensor-parallel-size 4
--pipeline-parallel-size 1
--data-parallel-size 1
--chunked-prefill-size 8192