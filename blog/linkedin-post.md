# LinkedIn post (teaser for the article)

I trained my own Jev-style classification model. Total cost: $0.

Together AI published a great walkthrough last month: fine-tune a 4B model on ~38k public examples and you get a fast, cheap classifier that takes a piece of state + predefined questions and returns a strict answer — a letter, a boolean, a score. Their version: $17 to train on their platform, plus hourly serving costs.

I wanted the same thing, but running entirely on hardware I own. So I did it on a Mac Mini M4 with 48GB of unified memory, using Apple's MLX framework:

→ Same public dataset mix (MultiNLI, BoolQ, Banking77, AG News, SST-5, policy + routing rules)
→ Same JSON schema and system prompt as the original
→ LoRA fine-tune of Qwen3.5-4B — LoRA because a full fine-tune needs 40–80GB+ of accelerator memory, and for a rigid output format it's within a rounding error of the same quality anyway
→ Trained overnight (vs 25 minutes on an H100 — that's the trade)
→ Served locally behind an OpenAI-compatible endpoint, temperature 0, 8 max tokens

Now it answers classification questions in under a second, for free, forever, and no data ever leaves my network.

The honest trade-offs (in the full write-up):
• You need ≥32GB unified memory — a 4GB-VRAM laptop GPU can't do this
• It's an overnight job, not a coffee break
• If you just want to explore, Together hosts the model serverless for pennies

But if you already own an Apple Silicon Mac: the cloud was never required.

Full write-up with every command in the comments. 👇

#MachineLearning #LLM #FineTuning #LocalAI #AppleSilicon #OpenSource
