| Model                     | Dataset           | Prompt Mode | AUROC (95% CI)        | AUPRC  | Mean p_vuln | P-C (%) | P-V (%) | FPR@90% Rec | FPR@95% Rec | Top-1 (%) |
|---------------------------|-------------------|-------------|-----------------------|--------|-------------|---------|---------|-------------|-------------|-----------|
| Qwen2.5-Coder-7B-Instruct | primevul_paired   | zero_shot   | 0.5105 (0.490, 0.531) | 0.5096 | 0.3233      | 3.0     | 16.2    | 0.8687      | 0.9293      | 100.0     |
| Qwen2.5-Coder-7B-Instruct | primevul_paired   | 2_shot      | 0.5214 (0.500, 0.547) | 0.5167 | 0.2778      | 3.1     | 14.4    | 0.8660      | 0.9588      | 100.0     |
| Qwen2.5-Coder-7B-Instruct | primevul_unpaired | zero_shot   | 0.5179 (0.431, 0.598) | 0.4988 | 0.2839      | —       | —       | 0.8900      | 0.9300      | 100.0     |
| Qwen2.5-Coder-7B-Instruct | primevul_unpaired | 2_shot      | 0.5637 (0.482, 0.643) | 0.5338 | 0.1731      | —       | —       | 0.8081      | 0.8788      | 100.0     |
| Qwen2.5-Coder-7B-Instruct | juliet            | zero_shot   | 0.9751 (0.952, 0.994) | 0.9810 | 0.6605      | 71.0    | 27.0    | 0.0300      | 0.0700      | 100.0     |
| Qwen2.5-Coder-7B-Instruct | juliet            | 2_shot      | 0.9696 (0.942, 0.992) | 0.9761 | 0.5188      | 87.9    | 8.1     | 0.0303      | 0.0909      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | primevul_paired   | zero_shot   | 0.5115 (0.490, 0.534) | 0.5097 | 0.3822      | 3.0     | 10.1    | 0.8990      | 0.9394      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | primevul_paired   | 2_shot      | 0.5026 (0.484, 0.521) | 0.5106 | 0.4815      | 7.2     | 39.2    | 0.8969      | 0.9794      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | primevul_unpaired | zero_shot   | 0.6923 (0.618, 0.766) | 0.6543 | 0.3368      | —       | —       | 0.5800      | 0.7400      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | primevul_unpaired | 2_shot      | 0.6693 (0.592, 0.745) | 0.6657 | 0.3787      | —       | —       | 0.6768      | 0.8283      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | juliet            | zero_shot   | 0.9473 (0.914, 0.979) | 0.9506 | 0.5446      | 72.0    | 23.0    | 0.0800      | 0.1400      | 100.0     |
| Qwen2.5-Coder-3B-Instruct | juliet            | 2_shot      | 0.8918 (0.851, 0.929) | 0.9023 | 0.5120      | 65.7    | 21.2    | 0.2929      | 0.6465      | 100.0     |
| Llama-3.1-8B-Instruct     | primevul_paired   | zero_shot   | 0.4934 (0.469, 0.517) | 0.5088 | 0.5382      | 7.1     | 54.5    | 0.9091      | 0.9394      | 100.0     |
| Llama-3.1-8B-Instruct     | primevul_paired   | 2_shot      | 0.5027 (0.482, 0.523) | 0.5117 | 0.7037      | 2.0     | 79.6    | 0.9082      | 0.9388      | 100.0     |
| Llama-3.1-8B-Instruct     | primevul_unpaired | zero_shot   | 0.6517 (0.568, 0.724) | 0.6113 | 0.4962      | —       | —       | 0.7100      | 0.7700      | 100.0     |
| Llama-3.1-8B-Instruct     | primevul_unpaired | 2_shot      | 0.6554 (0.579, 0.733) | 0.6324 | 0.4799      | —       | —       | 0.8283      | 0.8485      | 100.0     |
| Llama-3.1-8B-Instruct     | juliet            | zero_shot   | 0.9410 (0.903, 0.973) | 0.9486 | 0.4722      | 80.0    | 9.0     | 0.1000      | 0.1900      | 100.0     |
| Llama-3.1-8B-Instruct     | juliet            | 2_shot      | 0.9327 (0.895, 0.968) | 0.9380 | 0.5871      | 73.7    | 20.2    | 0.1515      | 0.4040      | 100.0     |
| Phi-3-mini-4k-instruct    | primevul_paired   | zero_shot   | 0.4855 (0.452, 0.514) | 0.5189 | 0.6258      | 4.1     | 69.1    | 0.9072      | 0.9175      | 100.0     |
| Phi-3-mini-4k-instruct    | primevul_paired   | 2_shot      | 0.5131 (0.476, 0.551) | 0.5264 | 0.2222      | 2.1     | 2.1     | 0.8947      | 0.9789      | 100.0     |
| Phi-3-mini-4k-instruct    | primevul_unpaired | zero_shot   | 0.5431 (0.458, 0.631) | 0.5244 | 0.5503      | —       | —       | 0.8750      | 0.9271      | 100.0     |
| Phi-3-mini-4k-instruct    | primevul_unpaired | 2_shot      | 0.2538 (0.187, 0.324) | 0.3359 | 0.2712      | —       | —       | 0.9368      | 0.9684      | 100.0     |
| Phi-3-mini-4k-instruct    | juliet            | zero_shot   | 0.8965 (0.855, 0.935) | 0.8871 | 0.7085      | 31.0    | 69.0    | 0.2800      | 0.3900      | 100.0     |
| Phi-3-mini-4k-instruct    | juliet            | 2_shot      | 0.8077 (0.750, 0.867) | 0.8534 | 0.7002      | 17.2    | 74.7    | 0.8081      | 0.9192      | 100.0     |
