# DeepSeek-V4.1-Flash, preset c1, against SGLang

| prompt | prompt tokens | top-1 vs SGLang | KL vs SGLang | SGLang noise floor KL |
|---|---|---|---|---|
| chat0 | 21 | 0.969 | 0.0394 | 0.0569 |
| chat1 | 54 | 0.961 | 0.0047 | 0.0029 |
| chat2 | 33 | 1.000 | 0.0014 | 0.0013 |
| chat3 | 24 | 1.000 | 0.0021 | 0.0061 |
| chat4 | 22 | 0.944 | 0.0485 | 0.1007 |
| chat5 | 26 | 0.977 | 0.0116 | 0.0083 |
| chat6 | 24 | 0.969 | 0.0234 | 0.0070 |
| chat7 | 13 | 0.914 | 0.0586 | 0.0517 |
| chat8 | 16 | 1.000 | 0.0065 | 0.0031 |
| chat9 | 32 | 0.984 | 0.0035 | 0.0041 |
| raw0 | 8 | 0.977 | 0.0041 | 0.0097 |
| raw1 | 8 | 1.000 | 0.0007 | 0.0007 |
| long0 | 3780 | 0.984 | 0.0112 | 0.0088 |
| long1 | 14265 | 0.961 | 0.0184 | 0.0112 |
| long2 | 59741 | 0.969 | 0.0131 | 0.0077 |
| **all, 1638 positions** | | **0.976** | **0.0147** | **0.0131** |

| check | result |
|---|---|
| fresh prefill, prompts of 300 to 2816 tokens, 2 runs each | deterministic at every length, first-token KL vs SGLang 0.00-0.32 |
| two identical runs of the 15 prompts | top-1 agreement 0.998, mean KL 0.001 |
| every cubin loaded on a B300, entry functions resolved | 107 of 107 |
| `cargo test --release` | all pass |
