
## How this was built
Built the plan with free copilot credit - luna 6 was chosen by 'auto' in plan mode, local.
Manually switched to Radiance qwen3.8 flash next to build the plan. 
Astonishingly fast. 1-7k prompt processing, 150-200 t/s Feels significantly faster than Luna.

## Purpose
I want a way to repeatably test the latency and throughput of a triton server with more than one model being served.  The goal is to iterate a bit on batch configurations with regard to size, wait duration, etc. and see what sort of throughput we can achieve on an RTX3080 which is a good stand in for Ampre GPU in EC2 level of performance.

I would like to get a non batched vs a batched configuration comparison in place to prove out the hypothesis thatr we are spending 10x what we need on GPU's at the moment.

## TODO things to test out
Adjust values in the config.pbtxt files from the starting values here:
- max_batch_size: 32
- preferred_batch_size [2, 4, 8, 16, 24, 32]
- max_queue_delay_microseconds: 2000

Adjust the docker run command for triton to include the sagemaker wrapper vs triton directly

Adjust which models are being served to include a larger embedding model than miniLM-L6-v2

Try a single model instead of two at a time.

Hold the priority 1 model at 100 requests / second while increasing the lower priority model and verify that the high priority model can maintain latency while lower priority suffers increased latency.

Test instance_group.count - is that more copies of the model loaded in vram? Does it help throughput more than trying to catch more requests in a longer queue delay?