HEL-FedRec: A General and Lightweight Privacy-preserving Federated Recommendation against Inference Attack
A lightweight privacy-preserving framework for federated recommendation using additive homomorphic encryption, achieving SOTA privacy protection with only ~1/3 communication time.
      
________________________________________
📋 Abstract
Federated recommendation (FedRec) enables collaborative model training without sharing raw user data. However, recent studies reveal that an honest-but-curious server can still infer sensitive user attributes from uploaded gradients, posing severe privacy risks. Existing privacy-preserving methods face three key challenges: (1) limited scenarios — hard to generalize across diverse FedRec architectures; (2) high communication costs — cryptography-based methods introduce substantial overhead; (3) difficulty in balance — obfuscation-based methods trade off recommendation accuracy for privacy.
HEL-FedRec addresses these challenges through three core innovations: - 🔐 Transferability: Integrates additive homomorphic encryption with federated averaging, compatible with general and personalized FedRecs (FedGMF, FedMLP, FedNeuMF, PFedRec) - ⚡ Lightweightness: A novel client selection strategy based on a value function (dataset size / loss reduction) reduces communication time to ~1/3 - 🎯 Efficiency: Encrypted gradients are directly aggregated without perturbation noise, preserving original optimization and recommendation performance
Extensive experiments on 5 real-world datasets demonstrate that HEL-FedRec achieves SOTA privacy protection (inference F1 score = 0) while maintaining competitive recommendation performance.
Keywords: Federated Recommendation | Homomorphic Encryption | Privacy-Preserving | Inference Attack | Lightweight
________________________________________
🏗️ Framework Overview
System Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEDERATED RECOMMENDATION SYSTEM                  │
│                                                                          │
│   ┌─────────────┐                    ┌─────────────────────────────┐      │
│   │  Client C₁  │                    │   Honest-but-Curious Server  │      │
│   │             │                    │                              │      │
│   │ ┌─────────┐ │    Enc(Θ₁, pk)   │  ┌───────────────────────┐  │      │
│   │ │User Emb │ │ ────────────────▶│  │      Aggregator       │  │      │
│   │ └─────────┘ │   (TLS/SSL)      │  │  ┌─────────────────┐  │  │      │
│   │ ┌─────────┐ │                    │  │ │ Client Selection │  │  │      │
│   │ │Item Emb │ │                    │  │ │  (Value Function)│  │  │      │
│   │ └─────────┘ │                    │  │ └─────────────────┘  │  │      │
│   │ ┌─────────┐ │                    │  │  ┌─────────────────┐  │  │      │
│   │ │Basic    │ │                    │  │ │ Add(M₁, M₂,...) │  │  │      │
│   │ │Recomm.  │ │                    │  │ │  (HE Aggregation)│  │  │      │
│   │ │   Q₁    │ │                    │  │ └─────────────────┘  │  │      │
│   │ └─────────┘ │                    │  │         │            │  │      │
│   │      │      │                    │  │         ▼            │  │      │
│   │   Loss Lc₁  │                    │  │    Mₐ (Global Enc)   │  │      │
│   │      │      │                    │  └───────────────────────┘  │      │
│   │   Value v₁  │◀───────────────────│         │                    │      │
│   │ (D₁/ΔLoss)  │   Enc(Mₐ, pk)      │         │ Enc(Mₐ, pk)        │      │
│   │      │      │   (TLS/SSL)        │         │ (TLS/SSL)          │      │
│   │  Dec(Mₐ,sk) │◀───────────────────┘         │                    │      │
│   │      │      │                              │                    │      │
│   │   Wₐ (New   │                              │                    │      │
│   │   Weights)  │                              │                    │      │
│   └─────────────┘                              │                    │      │
│         .                                      │                    │      │
│         .         ┌─────────────┐              │                    │      │
│         .         │  Client Cₖ  │              │                    │      │
│                   │  (Same Flow) │              │                    │      │
│                   └─────────────┘              │                    │      │
│                                                │                    │      │
│   Legend:  🔒 = Encryption    🔓 = Decryption   🗝️ = Key Management │      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
Key Modules
1️⃣ Additive Homomorphic Encryption (Paillier)
Client Side:                           Server Side:
┌─────────────────────┐               ┌─────────────────────┐
│  Local Weights Θ    │               │  Encrypted Params M │
│       │             │               │  ┌───────────────┐  │
│       ▼             │               │  │ M₁/(ς·k)      │  │
│  ┌─────────────┐    │   Upload      │  │ M₂/(ς·k)      │  │
│  │  Packing    │    │ ──────────▶   │  │ ...           │  │
│  │  [prec-bit] │    │   Enc(Θ,pk)   │  │               │  │
│  └─────────────┘    │               │  │ Add(·) ──▶ Mₐ │  │
│       │             │               │  └───────────────┘  │
│       ▼             │               │         │           │
│  ┌─────────────┐    │               │         ▼           │
│  │ Enc(Θ, pk)  │────┘               │    Global Enc Mₐ    │
│  │   ──▶ M     │                    │                     │
│  └─────────────┘                    └─────────────────────┘
       │                                        │
       │         ┌─────────────┐                │
       └────────▶│  Dec(Mₐ,sk) │◀───────────────┘
                 │   ──▶ Wₐ    │
                 └─────────────┘
Key Properties: - CPA-Secure: Server cannot infer plaintext from ciphertext - Additive Homomorphism: Enc(a) · Enc(b) = Enc(a + b) — enables aggregation on encrypted data - No Third Party: Clients generate keys locally, no trusted key distributor needed
2️⃣ Client Selection Strategy
Value Function:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Round 1:    vₖ¹ = Dₖ / Lcₖ¹                            │
│                                                          │
│   Round t>1:  vₖᵗ = Dₖ / (Lcₖᵗ⁻¹ - Lcₖᵗ)                  │
│                                                          │
│   Where:                                                 │
│   • Dₖ = dataset size (number of user ratings)           │
│   • Lcₖ = local training loss                            │
│   • ΔLoss = loss reduction (training progress)           │
│                                                          │
│   Selection: Top ς% clients by value vₖ                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
Design Rationale: A client with large dataset but small loss reduction may be overfitting; a client with small dataset but large loss reduction is making good progress. The value function balances both factors.
________________________________________
📊 Experimental Results
RQ1: Privacy Protection Effectiveness
HEL-FedRec achieves complete defense against attribute inference attacks (F1 score = 0) across all FedRec scenarios:
Attack Method	FedGMF	HEL-FedGMF	FedMLP	HEL-FedMLP	FedNeuMF	HEL-FedNeuMF	PFedRec	HEL-PFedRec
RA	0.514	0.514	0.576	0.576	0.565	0.565	0.509	0.509
KNN	0.659	0.000	0.537	0.000	0.724	0.000	0.612	0.000
AIA	0.678	0.000	0.729	0.000	0.643	0.000	0.711	0.000
Results on ML-1M dataset. AIA = Attribute Inference Attack (3-layer DNN). F1=0 indicates complete defense.
RQ2: Recommendation Performance vs Privacy Trade-off
HEL-FedRec maintains stable recommendation performance while achieving full privacy protection:
Dataset	Metric	FedGMF	HEL-FedGMF	FedMLP	HEL-FedMLP	FedNeuMF	HEL-FedNeuMF	PFedRec	HEL-PFedRec
ML-1M	HR@10	0.628	0.627	0.605	0.604	0.642	0.641	0.729	0.724
	NDCG@10	0.365	0.364	0.352	0.351	0.393	0.392	0.442	0.439
ML-100K	HR@10	0.658	0.657	0.491	0.490	0.671	0.670	0.722	0.718
	NDCG@10	0.395	0.394	0.291	0.290	0.402	0.401	0.421	0.418
Lastfm 2K	HR@10	0.686	0.685	0.632	0.631	0.695	0.694	0.804	0.799
	NDCG@10	0.532	0.531	0.456	0.455	0.496	0.495	0.701	0.697
Foursquare NY	HR@10	0.378	0.377	0.341	0.340	0.392	0.391	0.421	0.417
	NDCG@10	0.261	0.260	0.254	0.253	0.286	0.285	0.345	0.342
Amazon	HR@10	0.551	0.550	0.534	0.533	0.577	0.576	0.664	0.660
	NDCG@10	0.349	0.348	0.342	0.341	0.369	0.368	0.413	0.410
Performance gap < 1% compared to original FedRecs — encryption introduces no accuracy loss.
RQ3: Communication Efficiency
Client Selection reduces communication time by ~50%:
Dataset	Clients	Original Time (s)	HEL-FedRec Time (s)	Reduction
ML-1M	110	~300	~240	~20%
ML-100K	110	~290	~148	~48.9%
Lastfm 2K	110	~295	~150	~49.2%
Foursquare NY	110	~298	~152	~49.0%
Amazon	110	~297	~151	~49.2%
Encryption Scheme Comparison (MovieLens-1M, FedNeuMF):
Scheme	Encryption (s)	Decryption (s)	HR@10	NDCG@10
Paillier	~12,000	~7,500	0.642	0.393
BFV	~4,800	~600	0.124	0.000
CKKS	~11,800	~6,600	0.558	0.307
Paillier chosen for optimal balance: supports exact arithmetic (unlike BFV) and full precision (unlike CKKS).
RQ4: Transferability Across FedRecs
HEL-FedRec is model-agnostic — applicable to: - General FedRecs: FedGMF, FedMLP, FedNeuMF - Personalized FedRecs: PFedRec - All achieve F1=0 against inference attacks with <1% recommendation performance drop.
RQ5: Parameter Sensitivity
Dimension d	HR@10	NDCG@10	Client Ratio ς	HR@10	NDCG@10
16	0.18	0.10	0.2	0.20	0.12
32	0.49	0.27	0.4	0.52	0.38
64	0.62	0.38	0.6	0.65	0.49
128	0.64	0.39	0.8	0.70	0.50
Stable performance for d ≥ 64 and ς ≥ 0.6. Default: d=128, ς=0.8.
________________________________________
🚀 Quick Start
Requirements
•	Python >= 3.8
•	PyTorch >= 1.12
•	NumPy, SciPy
•	phe (Paillier encryption library)
Installation
git clone https://github.com/Honerlaaco/HEL-FedRec.git
cd HEL-FedRec
pip install -r requirements.txt
Data Preparation
Download datasets from MovieLens, Last.fm, Foursquare, and Amazon, then place them in the data/ directory.
python scripts/preprocess.py --dataset movielens-1m
Training
# Train HEL-FedNeuMF on MovieLens-1M
python train.py   --model fedneumf   --dataset movielens-1m   --embedding_dim 128   --clients 50   --select_ratio 0.8   --local_epochs 10   --global_epochs 30   --lr 0.0001   --use_encryption   --encryption_scheme paillier
Evaluation
# Recommendation performance
python evaluate.py   --model fedneumf   --dataset movielens-1m   --checkpoint checkpoints/best_model.pth

# Privacy attack simulation
python attack.py   --attack aia   --dataset movielens-1m   --checkpoint checkpoints/best_model.pth
________________________________________
📁 Project Structure
HEL-FedRec/
├── configs/                    # Configuration files
│   ├── train.yaml
│   ├── model/
│   │   ├── fedgmf.yaml
│   │   ├── fedmlp.yaml
│   │   ├── fedneumf.yaml
│   │   └── pfedrec.yaml
│   └── encryption.yaml
├── data/                       # Dataset directory
│   ├── movielens-1m/
│   ├── movielens-100k/
│   ├── lastfm-2k/
│   ├── foursquare-ny/
│   └── amazon/
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── fedgmf.py              # Generalized Matrix Factorization
│   ├── fedmlp.py              # Multi-Layer Perceptron
│   ├── fedneumf.py            # Neural Matrix Factorization
│   └── pfedrec.py             # Personalized FedRec
├── encryption/                 # Homomorphic encryption module
│   ├── __init__.py
│   ├── paillier.py            # Paillier scheme
│   ├── bfv.py                 # BFV scheme (baseline)
│   ├── ckks.py                # CKKS scheme (baseline)
│   └── packing.py             # Ciphertext packing
├── federated/                  # Federated learning components
│   ├── __init__.py
│   ├── client.py              # Client logic
│   ├── server.py              # Server logic
│   ├── aggregator.py          # HE-based aggregation
│   └── selector.py            # Client selection strategy
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── metrics.py             # HR@10, NDCG@10, F1
│   ├── data_loader.py         # Data loading
│   └── visualization.py       # Plotting tools
├── attacks/                    # Attack simulations
│   ├── __init__.py
│   ├── random_attack.py       # RA baseline
│   ├── knn_attack.py          # KNN attack
│   └── aia_attack.py          # Attribute inference attack
├── scripts/                    # Helper scripts
│   ├── preprocess.py
│   └── download_data.py
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── attack.py                   # Privacy attack simulation
├── requirements.txt
├── LICENSE
└── README.md
________________________________________
🔬 Reproducing Paper Results
Privacy Protection (RQ1)
# Run all attack methods on all FedRec variants
bash scripts/run_privacy_experiments.sh
Recommendation Performance (RQ2)
# Compare with baselines
bash scripts/run_recommendation_experiments.sh
Communication Cost (RQ3)
# Measure communication time with/without client selection
python experiments/communication_cost.py --clients 5,10,20,50,80,110
Encryption Scheme Comparison
python experiments/encryption_comparison.py   --schemes paillier,bfv,ckks   --dataset movielens-1m   --model fedneumf
________________________________________
📖 Citation
If you find this work useful, please cite:
@article{bai2024helfedrec,
  title={A General and Lightweight Privacy-preserving Federated Recommendation against Inference Attack},
  author={Bai, Yang and Liu, Cun and Chen, Jinyin and Lv, Mingqi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
________________________________________
📬 Contact
•	📧 Email: [your-email@example.com]
•	🐛 Issues: GitHub Issues
•	💬 Discussions: GitHub Discussions
________________________________________
📄 License
This project is licensed under the MIT License.
________________________________________
🙏 Acknowledgments
•	MovieLens datasets provided by GroupLens
•	Paillier implementation based on python-paillier
•	FedRec baselines inspired by Neural Collaborative Filtering
________________________________________
⭐ Star this repo if you find it helpful!
