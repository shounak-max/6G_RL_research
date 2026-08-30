"""
agents/marl/gat_crl/ — Cooperative Decentralized Multi-Agent RL with GAT and Selective Policy Sharing.
"""

from agents.marl.gat_crl.gat_encoder import GATEncoder
from agents.marl.gat_crl.ntn_similarity import NeuralTensorNetworkSimilarity
from agents.marl.gat_crl.selective_sharing import SelectivePolicySharing, CommunicationMetrics

__all__ = [
    "GATEncoder",
    "NeuralTensorNetworkSimilarity",
    "SelectivePolicySharing",
    "CommunicationMetrics",
]
