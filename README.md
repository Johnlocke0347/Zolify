# Zolify - Secure AI Training Platform

**Docker + Groth16 ZK Proofs + Fixed Evaluation.** Train anywhere, verify everywhere.

## Core Technology
- **Docker Security**: Exact reproducible + isolated environment
- **Groth16 Proofs**: Math proof dataset → model weights  
- **Fixed Evaluation**: Pre-defined test set → deterministic score
- **One-Click Verification**: Fastest browser verification of any proof as compared to its peers

## How It Works
```bash
# 1. Post job with fixed eval
zolify job post --dataset imdb --model llama --min-score 0.85 --reward 5000_ZOL

# 2. Nodes run Docker training
docker pull ghcr.io/Johnlocke0347/zolify:latest
docker run zolify/train --job ABC123


## $ZOL Token (Devnet LIVE)
**DevNet live**: `C9SjNmZRX1hVc1bRV2DquA7sNpktwMz1qnVuSzUdF1oW`

## Get Involved
- Docker nodes: Run anywhere
- Jobs waitlist: discord.gg/zolify  
- Verify proofs: Browser verifier live

**github.com/Johnlocke0347/zolify**

# 3. Submit proof + eval score
zolify proof submit --job ABC123 --proof 0xabc... --eval-score 0.87
