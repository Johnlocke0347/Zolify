# Zolify: Decentralized Jolt-Native Compute Network

![Miner Jobs Status](https://github.com/Johnlocke0347/zolify/actions/workflows/healthcheck.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Bittensor: Finney](https://img.shields.io/badge/Subnet-Finney-blue)

Zolify is a Bittensor-integrated subnet leveraging **Jolt/Lasso** for zero-knowledge auditability of AI compute. Miners execute high-performance training jobs while Validators verify execution traces in <120ms.

---

## Quickstart

Deploy a production-ready miner in 60 seconds:

```bash
git clone [https://github.com/Johnlocke0347/zolify](https://github.com/Johnlocke0347/zolify)
cd zolify
docker compose up --build
Architecture
The repository is structured for high-availability CI/CD:

docker/: Base miner environment (Ubuntu 22.04 + Non-root security).

jobs/imdb-sentiment/: Jolt-wrapped PyTorch models using distilbert-base-uncased.

healthcheck/: Automated service mesh validation.

.github/workflows/: Continuous Integration via GitHub Actions.
Anti-Cheat: Dynamic Salted Seeds
Zolify uses Bittensor-grade dynamic salting to prevent pre-computed "look-up" cheating. Every challenge requires a unique execution trace.
Code #
def dynamic_seed(salt: str) -> int:
    combined = f"{salt}-{int(time.time()//60)}"
    return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)
Hyperparameters (Finney Mainnet)
Parameter            Value         Description
Immunity Period      7,000         Blocks(~24h) for initial synchronization.
Tempo                 360          Fast reward loops (72 mins).
Alpha High          52,4280.8      scaling for consistency bonuses.
Activity Cutoff     5,000          Missing proofs leads to deregistration.

Services & API
Base Miner (Port 8080)
Returns real-time status and mock proof for connectivity checks.
curl localhost:8080/health
Sentiment Model (Port 8081)
Executes Jolt-audited training passes.
# Returns: {"status": "success", "zk_proof": "0x...", "f1": 0.87}
Validator Healthcheck
Comprehensive check of the local service mesh.
docker run zolify/healthcheck
Join the Network
We are currently recruiting the first 10 miners and root-network whales for Childkey parenting.

Twitter: @AliHaider139481

Parenting Command: btcli subnets childkey --parent <root-validator-hotkey>

