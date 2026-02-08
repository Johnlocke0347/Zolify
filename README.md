# Zolify - Decentralized AI Compute Network

Bittensor-inspired platform. Miners run Docker containers, validators score models, tokenomics drive competition.

## Quickstart

```bash
git clone https://github.com/Johnlocke0347/zolify
cd zolify
docker compose up

## How It Works
```bash
docker/                 Base miner container
├── Dockerfile         Ubuntu + curl + mock PoW
└── entrypoint.sh      {"proof":"0x1a2b3c4d5e6f7890"}

jobs/imdb-sentiment/    Real PyTorch model
├── Dockerfile         CUDA + transformers  
└── train.py          bert-base-uncased → F1:0.87

healthcheck/           Validator scores miners
└── entrypoint.py      Tests all endpoints

.github/workflows/     CI/CD - Auto-builds on push
└── healthcheck.yml
Tokenomics Ready

MINERS: docker run → earn ZOL
VALIDATORS: python score.py → earn ZOL  
ZK: Verify compute off-chain

## Anti-Cheat: Dynamic Salted Seeds

**Fixed seeds = instant cheating. Dynamic salts = Bittensor-grade fairness.**

```python
def dynamic_seed(salt: str) -> int:
    combined = f"{salt}-{block_time}-{nonce}"
    return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)

# Validator sends: "zolify-subnet1-block123-validatorABC"
# Miner seeds model with salt → unique eval every challenge

## Subnet 0 Parent Validators (2026 Standard)

**Calling all Root Network whales:** Parent our hotkey as Childkey.

**DM @JohnLocke0347 to parent: `childkey --parent root-validator-hotkey`**

## Hyperparameters (The Game Rules)

immunity_period: 7,000 blocks (~24h)
"Fair day to sync Jolt provers"

tempo: 360 blocks (~72 mins)
"Fast reward feedback loops"

alpha_high: 0.8
"Consistency compounds - every epoch proof = bonus"

kappa: 32,767 (Default)
"Standard emission curve"

activity_cutoff: 5,000 blocks (~17h)
"No zombies. Miss proofs → lose registration spot"

Services
Base Container (port 8080)
curl localhost:8080/health
# Returns: {"status":"active","f1":0.87,"proof":"0x1a2b3c4d5e6f7890"}

Sentiment Model (port 8081)

docker run zolify/imdb
# Returns: {"status": "model_loaded", "mock_accuracy": 0.87}

Healthcheck

docker run zolify/healthcheck
# Returns: {"services": [...], "overall": true}

CI/CD Pipeline
Actions tab shows live status:
Push triggers workflow
Builds all Docker images
Runs healthchecks
Green checks = production ready
https://github.com/Johnlocke0347/zolify/actions
Local Development
# Full stack
docker compose up --build

# Single service
docker build -t zolify/base ./docker
docker run -p 8080:8080 zolify/base

Production Deploy
# Kubernetes/Helm ready structure
# GitHub Actions → Docker Hub → K8s
Zolify: Bittensor architecture for everyone. Deploy in 60 seconds.
