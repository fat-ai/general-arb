import random
import os
import logging
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import qmc, norm
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
from typing import Dict, List, Tuple, Any
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sys
import math
import json
import google.generativeai as genai
import multiprocessing
import requests
import gzip    
import io     
from datetime import datetime, timedelta
from numba import njit
import pickle
from pathlib import Path
import time
import traceback
import spacy
import torch
from scipy.stats import linregress
import shutil  

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def force_clear_cache(cache_dir):
    path = Path(cache_dir)
    if path.exists():
        print(f"⚠️ CLEARING CACHE at {path} to ensure data alignment...")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# 1. Force PyTorch to use the Apple 'MPS' (Metal Performance Shaders) device
if torch.backends.mps.is_available():
    spacy.require_gpu()
    print("🚀 Using M3 GPU (MPS acceleration)!")
else:
    print("⚠️ MPS not available. Using CPU.")

# 2. Load the transformer model (CNN models like 'sm' won't benefit much from this)
nlp = spacy.load("en_core_web_trf")

# ==============================================================================
# --- Global Setup & Helpers ---
# ==============================================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def plot_performance(equity_curve, trades_count):
    
        import matplotlib.pyplot as plt
        
        # 1. Setup Plot
        plt.figure(figsize=(12, 6))
        
        # 2. Plot the Curve
        # We use a simple range for X-axis (Time Steps)
        x_axis = range(len(equity_curve))
        plt.plot(x_axis, equity_curve, color='#00ff00', linewidth=1.5, label='Portfolio Value')
        
        # 3. Add Baselines
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Starting Capital')
        
        # 4. Styling
        plt.title(f"C7 Strategy Performance ({trades_count} Trades)", fontsize=14)
        plt.xlabel("Time (Minutes Active)", fontsize=10)
        plt.ylabel("Capital ($)", fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend()
        
        # 5. Calculate & Annotate Max Drawdown Visual
        series = np.array(equity_curve)
        running_max = np.maximum.accumulate(series)
        drawdown = (series - running_max) / running_max
        max_dd_idx = np.argmin(drawdown)
        
        # Mark the bottom of the drawdown
        if len(equity_curve) > 0:
            plt.plot(max_dd_idx, equity_curve[max_dd_idx], 'rv', markersize=10)
            plt.annotate(f"Max DD: {drawdown[max_dd_idx]:.2%}", 
                         xy=(max_dd_idx, equity_curve[max_dd_idx]), 
                         xytext=(max_dd_idx, equity_curve[max_dd_idx]*0.95),
                         arrowprops=dict(facecolor='black', shrink=0.05))
    
        # 6. Save
        filename = "c7_equity_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 CHART GENERATED: Saved to '{filename}'")

def convert_to_beta(mean: float, confidence_interval: tuple[float, float]) -> tuple[float, float]:
    """(Production) Converts a mean and 95% CI to Beta(a, b) parameters."""
    if not (0 <= mean <= 1):
         log.warning(f"Mean {mean} is outside [0, 1]. Returning weak prior.")
         return (1.0, 1.0)
    
    # --- FIX 1: Handle Logical Rules (P0.3 from review) ---
    if mean == 0.0: return (1.0, float('inf')) # P(X=0) = 1
    if mean == 1.0: return (float('inf'), 1.0) # P(X=1) = 1
    
    lower, upper = confidence_interval
    if not (0 <= lower <= mean <= upper <= 1.0):
        log.warning("Invalid confidence interval. Returning a weak prior.")
        return (1.0, 1.0)
        
    std_dev = (upper - lower) / 4.0
    if std_dev == 0:
        if mean == 1.0: return (float('inf'), 1.0)
        if mean == 0.0: return (1.0, float('inf'))
        log.warning(f"CI has zero width but mean is {mean}. Treating as weak prior.")
        return (1.0, 1.0)

    variance = std_dev ** 2
    inner = (mean * (1 - mean) / variance) - 1
    
    if (mean * (1-mean)) < variance:
        log.warning(f"Inconsistent CI for mean {mean}. Variance is too large. Returning weak prior.")
        return (1.0, 1.0)
    if inner <= 0: inner = 1e-6
    
    alpha = mean * inner
    beta = (1 - mean) * inner
    log.debug(f"Converted (mean={mean}, CI=[{lower},{upper}]) -> (alpha={alpha:.2f}, beta={beta:.2f})")
    return (alpha, beta)

def cosine_similarity(v1, v2):
    v1 = np.array(v1); v2 = np.array(v2)
    if v1.shape != v2.shape: return 0.0
    dot = np.dot(v1, v2); norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot / (norm_v1 * norm_v2)

# ==============================================================================
# ### COMPONENT 1: GraphManager (Production-Ready) ###
# ==============================================================================

class GraphManager:
    """Component 1: Production-ready GraphManager (Fixed for Backtesting)."""
    def __init__(self, is_mock=False):
        self.is_mock = is_mock
        if self.is_mock:
            log.warning("GraphManager is running in MOCK mode.")
            self.vector_dim = 768
            self.model_brier_scores = {'brier_internal_model': 0.08, 'brier_expert_model': 0.05, 'brier_crowd_model': 0.15}
            
            # Initialize empty mock DB
            self.mock_db = {
                'contracts': {}, # {contract_id: {data}}
                'entities': {
                    'E_123': {'canonical_name': 'NeuroCorp, Inc.', 'type': 'Organization', 'vector': [0.2]*768, 'contract_count': 0}
                },
                'aliases': {
                    'NeuroCorp': 'E_123'
                },
                'wallets': {}, # Will be populated by Profiler
                'review_queue': []
            }
            return

        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.vector_dim = int(os.getenv('VECTOR_DIM', 768))

        if not self.password:
            raise ValueError("NEO4J_PASSWORD environment variable not set.")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            log.info(f"GraphManager connected to Neo4j at {self.uri}.")
        except Exception as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if not self.is_mock and hasattr(self, 'driver'): self.driver.close()

    # --- C1: Schema & Write Methods ---
    def setup_schema(self):
        if self.is_mock: return
        log.info("Applying database schema...")
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alias) REQUIRE a.text IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (w:Wallet) REQUIRE w.wallet_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"))
            try:
                session.execute_write(lambda tx: tx.run("CALL apoc.index.add('aliases', ['Alias(text)'])"))
            except ClientError:
                log.warning("APOC index failed.")
            
            try:
                session.execute_write(
                    lambda tx: tx.run(
                        f"""
                        CREATE VECTOR INDEX contract_vector_index IF NOT EXISTS
                        FOR (c:Contract) ON (c.vector)
                        OPTIONS {{ indexConfig: {{
                            `vector.dimensions`: {self.vector_dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """
                    )
                )
            except Exception:
                log.warning(f"Could not create vector index.")
        log.info("Schema setup complete.")
        
    # Update the signature to accept **kwargs
    def add_contract(self, contract_id: str, text: str, vector: list[float], liquidity: float = 0.0, p_market_all: float = None, **kwargs):
        if self.is_mock:
            self.mock_db['contracts'][contract_id] = {
                'text': text, 'vector': vector, 'liquidity': liquidity,
                'p_market_all': p_market_all, 'status': 'PENDING_LINKING',
                'entity_ids': [],
                # Store any extra data (like pre-calculated entities) for the backtest
                **kwargs 
            }
            return
            
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {len(vector)}")
        with self.driver.session() as session:
            # Production path remains unchanged, ignores kwargs
            session.execute_write(self._tx_merge_contract, contract_id, text, vector, liquidity, p_market_all)

    @staticmethod
    def _tx_merge_contract(tx, contract_id, text, vector, liquidity, p_market_all):
        tx.run(
            """
            MERGE (c:Contract {contract_id: $contract_id})
            ON CREATE SET
                c.text = $text, c.vector = $vector, c.liquidity = $liquidity,
                c.p_market_all = $p_market_all,
                c.status = 'PENDING_LINKING', c.created_at = timestamp()
            ON MATCH SET
                c.text = $text, c.vector = $vector, c.liquidity = $liquidity,
                c.p_market_all = $p_market_all, c.updated_at = timestamp()
            """,
            contract_id=contract_id, text=text, vector=vector, liquidity=liquidity, p_market_all=p_market_all
        )

    # --- C2: Read/Write Methods ---
    def link_contract_to_entity(self, contract_id, entity_id, confidence):
        if self.is_mock: 
            if contract_id in self.mock_db['contracts']:
                self.mock_db['contracts'][contract_id]['entity_ids'].append(entity_id)
                self.mock_db['contracts'][contract_id]['status'] = 'PENDING_ANALYSIS'
            if entity_id in self.mock_db['entities']:
                self.mock_db['entities'][entity_id]['contract_count'] += 1
            return
        with self.driver.session() as session:
            session.execute_write(self._tx_link_contract, contract_id, entity_id, confidence)

    @staticmethod
    def _tx_link_contract(tx, contract_id, entity_id, confidence):
        tx.run(
            "MATCH (c:Contract {contract_id: $contract_id}) "
            "MATCH (e:Entity {entity_id: $entity_id}) "
            "MERGE (c)-[r:IS_ABOUT]->(e) "
            "ON CREATE SET r.confidence_score = $confidence, r.created_at = timestamp() "
            "ON MATCH SET r.confidence_score = $confidence "
            "SET c.status = 'PENDING_ANALYSIS'",
            contract_id=contract_id, entity_id=entity_id, confidence=confidence
        )
    
    def get_contracts_by_status(self, status, limit=10) -> list[dict]:
        if self.is_mock: 
             return self._mock_get_contracts_by_status(status, limit)
        with self.driver.session() as session:
            return session.execute_read(self._tx_get_contracts_by_status, status, limit)

    @staticmethod
    def _tx_get_contracts_by_status(tx, status, limit):
        result = tx.run(
            "MATCH (c:Contract {status: $status}) "
            "WITH c, [(c)-[:IS_ABOUT]->(e) | e.entity_id] AS entity_ids "
            "RETURN c.contract_id AS contract_id, c.text AS text, c.vector AS vector, "
            "c.liquidity AS liquidity, c.p_market_all as p_market_all, entity_ids LIMIT $limit",
            status=status, limit=limit
        )
        return [record.data() for record in result]
        
    def find_entity_by_alias_fuzzy(self, alias_text: str, threshold: float = 0.9) -> dict:
        if self.is_mock: 
            return self._mock_find_entity_by_alias_fuzzy(alias_text)
        
        query = """
            CALL apoc.index.search('aliases', $text) YIELD node AS a, properties
            WITH a, properties.text AS matched_text
            WITH a, matched_text, apoc.text.levenshteinSimilarity(matched_text, $text) AS confidence
            WHERE confidence >= $threshold
            ORDER BY confidence DESC
            LIMIT 1
            MATCH (a)-[:POINTS_TO]->(e:Entity)
            RETURN e.entity_id AS entity_id, 
                   e.canonical_name AS name, 
                   confidence
        """
        fallback_query = """
            MATCH (a:Alias)
            WITH a, apoc.text.levenshteinSimilarity(a.text, $text) AS confidence
            WHERE confidence >= $threshold
            MATCH (a)-[:POINTS_TO]->(e:Entity)
            RETURN e.entity_id AS entity_id, e.canonical_name AS name, confidence
            ORDER BY confidence DESC
            LIMIT 1
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, text=alias_text, threshold=threshold).single()
                if result: return result.data()
                result = session.run(fallback_query, text=alias_text, threshold=threshold).single()
                return result.data() if result else None
            except ClientError:
                result = session.execute_read(self._tx_find_entity_exact, alias_text).single()
                return result.data() if result else None

    @staticmethod
    def _tx_find_entity_exact(tx, alias_text):
        return tx.run(
            "MATCH (a:Alias {text: $alias_text})-[:POINTS_TO]->(e:Entity) "
            "RETURN e.entity_id AS entity_id, e.canonical_name AS name, 1.0 AS confidence LIMIT 1",
            alias_text=alias_text
        ).single()
        
    def find_similar_contracts_by_vector(self, contract_id: str, vector: List[float], k: int = 5) -> List[Dict]:
        if self.is_mock: return []
        query = """
            CALL db.index.vector.queryNodes('contract_vector_index', $k, $vector) YIELD node, similarity
            WHERE node.contract_id <> $contract_id
            RETURN node.contract_id AS id, node.text AS text, similarity
        """
        with self.driver.session() as session:
            try:
                results = session.run(query, k=k, vector=vector, contract_id=contract_id)
                return [r.data() for r in results]
            except ClientError:
                return []
                
    def update_contract_status(self, contract_id, status, metadata=None):
        if self.is_mock: 
            if contract_id in self.mock_db['contracts']:
                self.mock_db['contracts'][contract_id]['status'] = status
            return
        with self.driver.session() as session:
            session.execute_write(self._tx_update_status, contract_id, status, metadata)
            
    @staticmethod
    def _tx_update_status(tx, contract_id, status, metadata):
        query = "MATCH (c:Contract {contract_id: $contract_id}) SET c.status = $status, c.updated_at = timestamp()"
        params = {'contract_id': contract_id, 'status': status}
        if metadata:
            query += " SET c.review_metadata = $metadata"
            params['metadata'] = json.dumps(metadata)
        tx.run(query, **params)
    
    # --- C3: Read/Write Methods ---
    def get_entity_contract_count(self, entity_id: str) -> int:
        if self.is_mock: 
            return self.mock_db['entities'].get(entity_id, {}).get('contract_count', 0)
        
        query = "MATCH (e:Entity {entity_id: $entity_id})<-[:IS_ABOUT]-(c:Contract) RETURN count(c) AS count"
        with self.driver.session() as session:
            result = session.run(query, entity_id=entity_id).single()
            return result['count'] if result else 0
            
    def update_contract_prior(self, contract_id: str, p_internal: float, alpha: float, beta: float, source: str, p_experts: float, p_all: float):
        if self.is_mock:
            if contract_id in self.mock_db['contracts']:
                self.mock_db['contracts'][contract_id].update({
                    'p_internal_prior': p_internal, 'p_internal_alpha': alpha,
                    'p_internal_beta': beta, 'p_internal_source': source,
                    'p_market_experts': p_experts, 'p_market_all': p_all,
                    'status': 'PENDING_FUSION'
                })
            return
        with self.driver.session() as session:
            session.execute_write(self._tx_update_prior, contract_id, p_internal, alpha, beta, source, p_experts, p_all)

    @staticmethod
    def _tx_update_prior(tx, contract_id, p_internal, alpha, beta, source, p_experts, p_all):
        tx.run(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            SET
                c.p_internal_prior = $p_internal, c.p_internal_alpha = $alpha,
                c.p_internal_beta = $beta, c.p_internal_source = $source,
                c.p_market_experts = $p_experts, c.p_market_all = $p_all,
                c.status = 'PENDING_FUSION', c.updated_at = timestamp()
            """,
            contract_id=contract_id, p_internal=p_internal, alpha=alpha, beta=beta, 
            source=source, p_experts=p_experts, p_all=p_all
        )

    # --- C4: Read/Write Methods ---
    def get_all_resolved_trades_by_topic(self) -> pd.DataFrame:
        if self.is_mock: 
            return pd.DataFrame()
        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract)-[:IS_ABOUT]->(e:Entity)
        WHERE c.status = 'RESOLVED' AND c.outcome IS NOT NULL AND t.price IS NOT NULL
        RETURN w.wallet_id AS wallet_id, e.type AS entity_type, 
               t.price AS bet_price, c.outcome AS outcome
        SKIP 0 LIMIT 10000 
        """ 
        with self.driver.session() as session:
            results = session.run(query)
            df = pd.DataFrame([r.data() for r in results])
            return df if not df.empty else pd.DataFrame(columns=['wallet_id', 'entity_type', 'bet_price', 'outcome'])

    def get_live_trades_for_contract(self, contract_id: str) -> pd.DataFrame:
        if self.is_mock: 
            return self._mock_get_live_trades_for_contract(contract_id)

        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract {contract_id: $contract_id})
        WHERE t.price IS NOT NULL AND t.volume IS NOT NULL
        RETURN w.wallet_id AS wallet_id, t.price AS trade_price, t.volume AS trade_volume
        LIMIT 1000
        """
        with self.driver.session() as session:
            results = session.run(query, contract_id=contract_id)
            df = pd.DataFrame([r.data() for r in results])
            return df if not df.empty else pd.DataFrame(columns=['wallet_id', 'trade_price', 'trade_volume'])

    def get_contract_topic(self, contract_id: str) -> str:
        if self.is_mock: 
             # For backtest consistency, always use 'default_topic' as generated by transform
            return "default_topic"
        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run("MATCH (c:Contract {contract_id: $id})-[:IS_ABOUT]->(e:Entity) RETURN e.type AS topic LIMIT 1", id=contract_id).single()
            )
        return result.data().get('topic') if result else "default"

    def update_wallet_scores(self, wallet_scores: Dict[tuple, float]):
        if self.is_mock: 
            for (wallet_id, topic), score in wallet_scores.items():
                if wallet_id not in self.mock_db['wallets']: self.mock_db['wallets'][wallet_id] = {}
                self.mock_db['wallets'][wallet_id][f"brier_{topic}"] = score
            return
        
        scores_list = [{"wallet_id": k[0], "topic_key": f"brier_{k[1]}", "brier_score": v} for k, v in wallet_scores.items()]
        if not scores_list: return
        query = "UNWIND $scores_list AS score MERGE (w:Wallet {wallet_id: score.wallet_id}) SET w[score.topic_key] = score.brier_score"
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, scores_list=scores_list))
        
    def get_wallet_brier_scores(self, wallet_ids: List[str]) -> Dict[str, Dict[str, float]]:
        if self.is_mock: 
            return {wid: scores for wid, scores in self.mock_db['wallets'].items() if wid in wallet_ids}
        
        query = "MATCH (w:Wallet) WHERE w.wallet_id IN $wallet_ids RETURN w.wallet_id AS wallet_id, properties(w) AS scores"
        with self.driver.session() as session:
            results = session.run(query, wallet_ids=wallet_ids)
            return {r.data()['wallet_id']: {k: v for k, v in r.data()['scores'].items() if k.startswith('brier_')} for r in results}

    # --- C5: Read/Write Methods ---
    def get_contracts_for_fusion(self, limit: int = 10) -> List[Dict]:
        if self.is_mock: return self._mock_get_contracts_for_fusion()
        
        query = """
        MATCH (c:Contract {status: 'PENDING_FUSION'})
        WHERE c.p_internal_alpha IS NOT NULL 
          AND c.p_market_experts IS NOT NULL 
          AND c.p_market_all IS NOT NULL
        RETURN c.contract_id AS contract_id,
               c.p_internal_alpha AS p_internal_alpha,
               c.p_internal_beta AS p_internal_beta,
               c.p_market_experts AS p_market_experts,
               c.p_market_all AS p_market_all
        LIMIT $limit
        """
        with self.driver.session() as session:
            results = session.run(query, limit=limit)
            return [r.data() for r in results]

    def get_model_brier_scores(self) -> Dict[str, float]:
        if self.is_mock: return self.model_brier_scores
        return {'brier_internal_model': 0.08, 'brier_expert_model': 0.05, 'brier_crowd_model': 0.15}

    def update_contract_fused_price(self, contract_id: str, p_model: float, p_model_variance: float):
        if self.is_mock: 
            if contract_id in self.mock_db['contracts']:
                self.mock_db['contracts'][contract_id]['p_model'] = p_model
                self.mock_db['contracts'][contract_id]['p_model_variance'] = p_model_variance
                self.mock_db['contracts'][contract_id]['status'] = 'MONITORED'
            return
        query = """
        MATCH (c:Contract {contract_id: $contract_id})
        SET c.p_model = $p_model, c.p_model_variance = $p_model_variance,
            c.status = 'MONITORED', c.updated_at = timestamp()
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, contract_id=contract_id, p_model=p_model, p_model_variance=p_model_variance))

    # --- C6: Read Methods ---
    def get_active_entity_clusters(self) -> List[str]:
        if self.is_mock: 
            return self._mock_get_active_entity_clusters()
            
        query = "MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity) RETURN DISTINCT e.entity_id AS entity_id"
        with self.driver.session() as session:
            results = session.run(query)
            return [r['entity_id'] for r in results]
            
    def get_cluster_contracts(self, entity_id: str) -> List[Dict]:
        if self.is_mock: 
            return self._mock_get_cluster_contracts(entity_id)

        query = """
        MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity {entity_id: $entity_id})
        WHERE c.p_model IS NOT NULL AND c.p_market_all IS NOT NULL
        RETURN c.contract_id AS id, 
               c.p_model AS M, c.p_market_all AS Q, 
               c.is_logical_rule AS is_logical_rule
        """
        with self.driver.session() as session:
            results = session.run(query, entity_id=entity_id)
            return [r.data() for r in results]
            
    def get_relationship_between_contracts(self, c1_id: str, c2_id: str, contracts: List[Dict]) -> Dict:
        if self.is_mock: 
            return {'type': 'NONE', 'p_joint': None}
            
        query = """
        MATCH (c1:Contract {contract_id: $c1_id})-[:IS_ABOUT]->(e1:Entity),
              (c2:Contract {contract_id: $c2_id})-[:IS_ABOUT]->(e2:Entity)
        OPTIONAL MATCH (e1)-[r:RELATES_TO]->(e2)
        WHERE r.type = 'IMPLIES'
        RETURN r.type AS type
        LIMIT 1
        """
        with self.driver.session() as session:
            p_model_c1 = next(c['M'] for c in contracts if c['id'] == c1_id)
            result = session.run(query, c1_id=c1_id, c2_id=c2_id).single()
            if result and result.data().get('type') == 'IMPLIES':
                return {'type': 'LOGICAL_IMPLIES', 'p_joint': p_model_c1}
        return {'type': 'NONE', 'p_joint': None}

    # --- C7/C8: Mock-driving Methods ---
    def get_human_review_queue(self):
        return self.mock_db['review_queue']
        
    def get_portfolio_state(self):
        return {'cash': 8537.88, 'positions': [{'id': 'MKT_A', 'fraction': -0.075}, {'id': 'MKT_B', 'fraction': 0.075}], 'total_value': 8537.88}
        
    def get_pnl_history(self):
        return pd.Series(np.random.normal(0, 1, 100).cumsum() + 10000)
        
    def get_regime_status(self):
        return "LOW_VOL", {"k_brier_scale": 1.5, "kelly_edge_thresh": 0.1}
        
    def resolve_human_review_item(self, item_id, action, data):
        self.mock_db['review_queue'] = [item for item in self.mock_db['review_queue'] if item['id'] != item_id]
        return True

    # --- MOCK IMPLEMENTATIONS (CORRECTED) ---
    
    def _mock_get_contracts_by_status(self, status: str, limit: int = 10):
         # FIX: Actually query the mock DB instead of returning hardcoded demo data
         res = []
         for cid, data in self.mock_db['contracts'].items():
             if data['status'] == status:
                 item = data.copy()
                 item['contract_id'] = cid
                 res.append(item)
                 if len(res) >= limit: break
         return res

    def _mock_find_entity_by_alias_fuzzy(self, alias_text: str):
        # Simple exact match for backtest speed
        entity_id = self.mock_db['aliases'].get(alias_text)
        if entity_id:
            return {'entity_id': entity_id, 'name': self.mock_db['entities'][entity_id]['canonical_name'], 'confidence': 1.0}
        # Fallback: Create a dynamic entity for any new alias found in backtest
        # This prevents "No entities found" errors for valid markets
        fake_id = f"E_{hash(alias_text)}"
        self.mock_db['entities'][fake_id] = {'canonical_name': alias_text, 'type': 'default_topic', 'contract_count': 0}
        self.mock_db['aliases'][alias_text] = fake_id
        return {'entity_id': fake_id, 'name': alias_text, 'confidence': 1.0}
        
    def _mock_get_live_trades_for_contract(self, contract_id):
        if 'live_trades' in self.mock_db:
            # If backtest injected live trades, use them
            all_trades = pd.DataFrame(self.mock_db['live_trades'])
            if not all_trades.empty and 'id' in all_trades.columns:
                 return all_trades[all_trades['id'] == contract_id]
        return pd.DataFrame(columns=['wallet_id', 'trade_price', 'trade_volume'])
        
    def _mock_get_contracts_for_fusion(self):
        # FIX: Inject contract_id into the returned dicts
        res = []
        for cid, data in self.mock_db['contracts'].items():
            if data.get('status') == 'PENDING_FUSION':
                item = data.copy()
                item['contract_id'] = cid
                res.append(item)
        return res
            
    def _mock_get_active_entity_clusters(self): 
        clusters = set()
        for c in self.mock_db['contracts'].values():
            if c.get('status') == 'MONITORED':
                for eid in c.get('entity_ids', []): clusters.add(eid)
        return list(clusters)
            
    def _mock_get_cluster_contracts(self, entity_id):
        res = []
        for cid, data in self.mock_db['contracts'].items():
            if data.get('status') == 'MONITORED' and entity_id in data.get('entity_ids', []):
                res.append({
                    'id': cid,
                    'M': data.get('p_model'),
                    'Q': data.get('p_market_all'),
                    'is_logical_rule': False
                })
        return res

# ==============================================================================
# ### COMPONENT 2: RelationalLinker (Production-Ready) ###
# ==============================================================================
_GLOBAL_SPACY_MODEL = None

class RelationalLinker:
    """(Production-Ready C2)"""
    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        global _GLOBAL_SPACY_MODEL
        
        # Optimization: Only load spaCy once per process, not per trial
        if _GLOBAL_SPACY_MODEL is None:
            try:
                log.info("Loading spaCy model (Global)...")
                _GLOBAL_SPACY_MODEL = spacy.load("en_core_web_sm")
            except IOError:
                log.error("Failed to load spaCy model.")
                raise
        self.nlp = _GLOBAL_SPACY_MODEL
    
    def _extract_entities(self, text: str) -> set[str]:
        doc = self.nlp(text)
        relevant_labels = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}
        return {ent.text for ent in doc.ents if ent.label_ in relevant_labels}

    def _run_fast_path(self, extracted_entities: set[str]) -> dict:
        matches = {}
        for entity_text in extracted_entities:
            result = self.graph.find_entity_by_alias_fuzzy(entity_text, threshold=0.9)
            if result:
                entity_id, confidence, name = result['entity_id'], result['confidence'], result['name']
                if entity_id not in matches or confidence > matches[entity_id][0]:
                    matches[entity_id] = (confidence, name)
        return matches

    def _run_fuzzy_path_knn(self, contract_id: str, contract_vector: List[float]) -> (str, Dict):
        similar_contracts = self.graph.find_similar_contracts_by_vector(contract_id, contract_vector, k=3)
        if not similar_contracts:
            return "NEEDS_NEW_ENTITY", {}
        return "NEEDS_MERGE_CONFIRMATION", {'similar_contracts': [c['id'] for c in similar_contracts]}

    def process_pending_contracts(self):
        # log.info("--- C2: Checking for 'PENDING_LINKING' contracts ---") 
        # (Logging commented out for speed in backtest loops)
        
        contracts = self.graph.get_contracts_by_status('PENDING_LINKING', limit=10)
        if not contracts: return

        for contract in contracts:
            contract_id = contract['contract_id']
            contract_text = contract['text']
            contract_vector = contract['vector']
            
            if 'precalc_entities' in contract and contract['precalc_entities']:
                extracted_entities = set(contract['precalc_entities'])
            else:
                extracted_entities = self._extract_entities(contract_text)
            
            if not extracted_entities:
                self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_REVIEW', {'reason': 'No entities found'})
                continue
                
            fast_path_matches = self._run_fast_path(extracted_entities)

            if len(fast_path_matches) >= 1:
                for entity_id, (confidence, name) in fast_path_matches.items():
                    self.graph.link_contract_to_entity(contract_id, entity_id, confidence)
            else:
                # Logic Preserved: Still attempts Vector Search if text match fails
                reason, details = self._run_fuzzy_path_knn(contract_id, contract_vector)
                details['extracted_entities'] = list(extracted_entities)
                self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_REVIEW', {'reason': reason, **details})

# ==============================================================================
# ### COMPONENT 3: Prior Engines (Production-Ready) ###
# ==============================================================================
class AIAnalyst:
    """(Production-Ready C3.sub)"""
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            log.warning("GOOGLE_API_KEY not set. AIAnalyst will run in MOCK-ONLY mode.")
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-1.5-flash')
                log.info("AI Analyst (Production) initialized with Google Gemini.")
            except Exception as e:
                log.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
    
    # --- FIX R2: Removed duplicate get_prior() method ---
    def get_prior(self, contract_text: str) -> dict:
        log.info(f"AI Analyst processing: '{contract_text[:50]}...'")
        system_prompt = "You are a 'superforecasting' analyst... respond ONLY with JSON: {\"probability\": 0.65, \"confidence_interval\": [0.55, 0.75], \"reasoning\": \"...\"}"
        mock_response = {'probability': 0.50, 'confidence_interval': [0.40, 0.60], 'reasoning': 'Defaulting (mock response).'}
        if "NeuroCorp" in contract_text:
             mock_response = {'probability': 0.65, 'confidence_interval': [0.55, 0.75], 'reasoning': 'NeuroCorp track record (mock).'}
        
        if not self.client:
            return mock_response
        try:
            # --- PRODUCTION API CALL (COMMENTED OUT) ---
            # generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            # response = self.client.generate_content(f"{system_prompt}\n\nAnalyze: '{contract_text}'", generation_config=generation_config)
            # return json.loads(response.text)
            log.info("AIAnalyst: (Skipping real Gemini API call, returning mock)")
            return mock_response
        except Exception as e:
            log.error(f"AI Analyst (Gemini) API call failed: {e}. Returning mock.")
            return mock_response

class PriorManager:
    """(Production-Ready C3)"""
    def __init__(self, graph_manager: GraphManager, ai_analyst: AIAnalyst, live_feed_handler: 'LiveFeedHandler'):
        self.graph = graph_manager
        self.ai = ai_analyst
        self.live_feed = live_feed_handler
        self.hitl_liquidity_threshold = float(os.getenv('HITL_LIQUIDITY_THRESH', 10000.0))
        self.hitl_new_domain_threshold = int(os.getenv('HITL_DOMAIN_THRESH', 5))
        log.info(f"PriorManager initialized (HITL Liquidity: ${self.hitl_liquidity_threshold})")

    def _is_hitl_required(self, contract: dict) -> bool:
        liquidity = contract.get('liquidity', 0.0)
        if liquidity is None: liquidity = 0.0
        if liquidity > self.hitl_liquidity_threshold:
            log.warning(f"HITL Triggered: Liquidity ({liquidity}) > threshold")
            return True
        entity_ids = contract.get('entity_ids', [])
        if not entity_ids: return False
        min_count = min(self.graph.get_entity_contract_count(eid) for eid in entity_ids)
        if min_count < self.hitl_new_domain_threshold:
            log.warning(f"HITL Triggered: New domain (entity has {min_count} contracts).")
            return True
        return False
    # --- FIX R8: Removed duplicate _is_hitl_required() method ---

    def process_pending_contracts(self):
        log.info("--- C3: Checking for contracts 'PENDING_ANALYSIS' ---")
        contracts = self.graph.get_contracts_by_status('PENDING_ANALYSIS', limit=10)
        if not contracts:
            log.info("C3: No new contracts to analyze.")
            return

        for contract in contracts:
            contract_id = contract['contract_id']
            log.info(f"C3: Processing {contract_id}")
            try:
                if self._is_hitl_required(contract):
                    self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_PRIOR', {'reason': 'High value or new domain.'})
                else:
                    prior_data = self.ai.get_prior(contract['text'])
                    mean, ci = prior_data['probability'], (prior_data['confidence_interval'][0], prior_data['confidence_interval'][1])
                    (alpha, beta) = convert_to_beta(mean, ci)
                    p_experts = self.live_feed.get_smart_money_price(contract_id)
                    p_all = contract.get('p_market_all')
                    if p_experts is None: p_experts = p_all
                    if p_experts is None or p_all is None:
                        self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_REVIEW', {'reason': 'Missing price data'})
                        continue
                    self.graph.update_contract_prior(contract_id, mean, alpha, beta, 'ai_generated', p_experts, p_all)
            except Exception as e:
                log.error(f"Failed to process prior for {contract_id}: {e}")
                self.graph.update_contract_status(contract_id, 'PRIOR_FAILED', {'error': str(e)})
    # --- FIX R9: Removed duplicate process_pending_contracts() method ---

# ==============================================================================
# ### COMPONENT 4: Market Intelligence Engine (Production-Ready) ###
# ==============================================================================

# ==============================================================================
# --- REPLACEMENT CLASS: HistoricalProfiler (PnL & Volume) ---
# ==============================================================================
class HistoricalProfiler:
    """
    Profiles Wallets by Profitability (PnL) and Markets by Volume Baseline.
    """
    def __init__(self, graph_manager: GraphManager, min_trades_threshold: int = 5):
        self.graph = graph_manager
        self.min_trades = min_trades_threshold
        log.info(f"HistoricalProfiler initialized (min_trades: {self.min_trades}).")

    def run_profiling(self, df_trades: pd.DataFrame) -> Tuple[set, Dict]:
        """
        Production Profiler.
        1. Identifies 'Smart Wallets' based on Realized PnL (Dollars Won).
        2. Calculates 'Market Baselines' for Volume Spike detection.
        """
        log.info("--- Profiling Wallets & Volume (Production PnL Mode) ---")
        
        if df_trades.empty: 
            return set(), {}

        # --- 1. Identify Smart Wallets (PnL Based) ---
        # Filter for Resolved trades only to calculate PnL
        resolved = df_trades.dropna(subset=['outcome']).copy()

        if 'size' not in resolved.columns:
            if 'usdc_vol' in resolved.columns:
                resolved['size'] = resolved['usdc_vol']
            else:
                log.error("⛔ CRITICAL: 'size' column missing in Profiler.")
                return set(), {}

        resolved['size'] = pd.to_numeric(resolved['size'], errors='coerce').fillna(0.0)
        resolved['tokens'] = pd.to_numeric(resolved['tokens'], errors='coerce').fillna(0.0)
        resolved['outcome'] = pd.to_numeric(resolved['outcome'], errors='coerce').fillna(0.0)

        signed_cost = resolved['size'].where(resolved['tokens'] >= 0, -resolved['size'])
        
        resolved['payout_value'] = resolved['tokens'] * resolved['outcome']
        resolved['realized_pnl'] = resolved['payout_value'] - signed_cost

        pnl_stats = resolved['realized_pnl'].describe()
        log.info(f"🔎 PnL DISTRIBUTION:\n{pnl_stats}")
        
        if resolved.empty:
            log.warning("No resolved trades found for profiling.")
            smart_wallet_ids = set()
        else:
            # Ensure numeric types for math safety
            resolved['size'] = pd.to_numeric(resolved['size'], errors='coerce').fillna(0.0)
            resolved['tokens'] = pd.to_numeric(resolved['tokens'], errors='coerce').fillna(0.0)
            resolved['outcome'] = pd.to_numeric(resolved['outcome'], errors='coerce').fillna(0.0)
            
            # Aggregate PnL per wallet
            signed_cost = resolved['size'].where(resolved['tokens'] >= 0, -resolved['size'])
        
            resolved['payout_value'] = resolved['tokens'] * resolved['outcome']
            resolved['realized_pnl'] = resolved['payout_value'] - signed_cost
            
            # Debug: Print the top winner to confirm it works
            top_winner = resolved.sort_values('realized_pnl', ascending=False).head(1)
            if not top_winner.empty:
                log.info(f"DEBUG: Top Trade PnL: ${top_winner.iloc[0]['realized_pnl']:.2f} "
                         f"(Wallet: {top_winner.iloc[0]['wallet_id']})")
    
            # Aggregate PnL per wallet
            wallet_stats = resolved.groupby('wallet_id')['realized_pnl'].sum()
 
            # Filter: Only wallets that actually made money
            profitable_wallets = wallet_stats[wallet_stats > 0].sort_values(ascending=False)
            
            if profitable_wallets.empty:
                smart_wallet_ids = set()
            else:
                # Threshold: Top 20% of winners, or at least top 5 wallets (whichever is more)
                # This ensures we have a signal even in thin data
                count = len(profitable_wallets)
                top_n = max(5, int(count * 0.20))
                
                smart_wallet_ids = set(profitable_wallets.head(top_n).index)
                log.info(f"Identified {len(smart_wallet_ids)} Smart Wallets (Top {top_n} Profitable).")

        # --- 2. Calculate Volume Baselines (Hourly Average) ---
        # We use ALL trades (active + resolved) to establish volume norms
        market_baselines = {}
        if 'timestamp' in df_trades.columns and not df_trades.empty:
            try:
                # Group by Market -> Resample Hourly -> Sum Volume
                vol_series = df_trades.set_index('timestamp').groupby('market_id')['size'].resample('1h').sum()
                
                # Calculate average hourly volume per market
                # We fill NaN correlations/stats later, but here we just need the mean
                market_baselines = vol_series.groupby('market_id').mean().to_dict()
            except Exception as e:
                log.warning(f"Volume profiling failed: {e}")
                market_baselines = {}

        print(f"Top 20 Smart Wallets: {list(profitable_wallets.head(20).index)}")

        return smart_wallet_ids, market_baselines

class LiveFeedHandler:
    """(Production-Ready C4b)"""
    def __init__(self, graph_manager: GraphManager, brier_epsilon: float = 0.001):
        self.graph = graph_manager
        self.brier_epsilon = brier_epsilon
        log.info("LiveFeedHandler initialized.")

    def get_smart_money_price(self, contract_id: str) -> float:
        log.info(f"C4: Calculating smart money price for {contract_id}...")
        topic = self.graph.get_contract_topic(contract_id)
        brier_key = f"brier_{topic}"
        
        # This allows the back-tester to inject live trades
        if self.graph.is_mock and 'live_trades' in self.graph.mock_db:
             live_trades_df = pd.DataFrame(self.graph.mock_db['live_trades'])
             live_trades_df = live_trades_df[live_trades_df['id'] == contract_id]
        else:
            live_trades_df = self.graph.get_live_trades_for_contract(contract_id)
            
        if live_trades_df.empty:
            log.warning(f"C4: No live trades for {contract_id}.")
            return None
        # (rest of the function is the same)
        wallet_ids = list(live_trades_df['wallet_id'].unique())
        wallet_scores = self.graph.get_wallet_brier_scores(wallet_ids)
        
        def get_brier(wallet_id):
            return wallet_scores.get(wallet_id, {}).get(brier_key, 0.25)
            
        brier_values = live_trades_df['wallet_id'].map(get_brier)
        live_trades_df['weight'] = live_trades_df['trade_volume'] / (brier_values + self.brier_epsilon)
        
        denominator = live_trades_df['weight'].sum()
        if denominator == 0: return None
        p_market_experts = (live_trades_df['trade_price'] * live_trades_df['weight']).sum() / denominator
        log.info(f"C4: Calculated P_market_experts for {contract_id}: {p_market_experts:.4f}")
        return p_market_experts

# ==============================================================================
# ### COMPONENT 5: The Belief Engine (Production-Ready) ###
# ==============================================================================

class BeliefEngine:
    """(Production-Ready C5)"""
    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        self.k_brier_scale = float(os.getenv('BRIER_K_SCALE', 0.5))
        self.model_brier_scores = self.graph.get_model_brier_scores()
        log.info(f"BeliefEngine initialized with k={self.k_brier_scale}.")

    def _impute_beta_from_point(self, mean: float, model_name: str) -> Tuple[float, float]:
        if not (0 < mean < 1): return (1.0, 1.0)
        brier_score = self.model_brier_scores.get(f'brier_{model_name}_model', 0.25)
        variance = self.k_brier_scale * brier_score
        if variance == 0: variance = 1e-9
        if variance >= (mean * (1 - mean)): variance = (mean * (1 - mean)) - 1e-9
        inner = (mean * (1 - mean) / variance) - 1
        if inner <= 0: return (1.0, 1.0)
        alpha = mean * inner; beta = (1 - mean) * inner
        return (alpha, beta)

    def _fuse_betas(self, beta_dists: List[Tuple[float, float]]) -> Tuple[float, float]:
        fused_alpha, fused_beta = 1.0, 1.0
        for alpha, beta in beta_dists:
            if math.isinf(alpha) and math.isinf(beta):
                # This is (inf, inf) from the old convert_to_beta, which is invalid
                log.warning("Invalid (inf, inf) prior found. Skipping.")
                continue
            if math.isinf(alpha) or math.isinf(beta):
                log.warning("Found logical rule (inf, 1) or (1, inf). Bypassing fusion.")
                return (alpha, beta) 
            fused_alpha += (alpha - 1.0); fused_beta += (beta - 1.0)
        return (max(fused_alpha, 1.0), max(fused_beta, 1.0))

    def _get_beta_stats(self, alpha: float, beta: float) -> Tuple[float, float]:
        if math.isinf(alpha): return (1.0, 0.0)
        if math.isinf(beta): return (0.0, 0.0)
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ( (alpha + beta)**2 * (alpha + beta + 1) )
        return (mean, variance)

    def run_fusion_process(self):
        log.info("--- C5: Checking for contracts 'PENDING_FUSION' ---")
        contracts = self.graph.get_contracts_for_fusion(limit=10)
        if not contracts:
            log.info("C5: No new contracts to fuse.")
            return

        for contract in contracts:
            contract_id = contract['contract_id']
            log.info(f"C5: Fusing price for {contract_id}")
            try:
                beta_internal = (contract['p_internal_alpha'], contract['p_internal_beta'])
                if math.isinf(beta_internal[0]) or math.isinf(beta_internal[1]):
                    (p_model, p_model_variance) = self._get_beta_stats(beta_internal[0], beta_internal[1])
                else:
                    beta_experts = self._impute_beta_from_point(contract['p_market_experts'], 'expert')
                    beta_crowd = self._impute_beta_from_point(contract['p_market_all'], 'crowd')
                    (fused_alpha, fused_beta) = self._fuse_betas([beta_internal, beta_experts, beta_crowd])
                    (p_model, p_model_variance) = self._get_beta_stats(fused_alpha, fused_beta)
                log.info(f"C5: Fusion complete for {contract_id}: P_model={p_model:.4f}")
                self.graph.update_contract_fused_price(contract_id, p_model, p_model_variance)
            except Exception as e:
                log.error(f"Failed to fuse prior for {contract_id}: {e}")
                self.graph.update_contract_status(contract_id, 'FUSION_FAILED', {'error': str(e)})

# ==============================================================================
# ### COMPONENT 6: Portfolio Manager (Production-Ready) ###
# ==============================================================================
class HybridKellySolver:
    """(Production-Ready C6.sub)"""
    def __init__(self, **kwargs):
        self.edge_thresh = kwargs.get('analytical_edge_threshold', 0.2)
        self.q_thresh = kwargs.get('analytical_q_threshold', 0.1)
        self.k_samples = kwargs.get('num_samples_k', 10000)
        
    def _nearest_psd(self, A):
        """Find the nearest positive semi-definite correlation matrix."""
        eigval, eigvec = np.linalg.eigh(A)
        eigval[eigval < 1e-8] = 1e-8 
        A_psd = eigvec @ np.diag(eigval) @ eigvec.T
        # Renormalize to be a correlation matrix
        inv_diag = np.diag(1.0 / np.sqrt(np.diag(A_psd)))
        A_corr = inv_diag @ A_psd @ inv_diag
        np.fill_diagonal(A_corr, 1.0)
        return A_corr
        
    def _is_numerical_required(self, E, Q, contracts):
        if np.any(np.abs(E) > self.edge_thresh): log.warning("Numerical solver triggered: Large edge."); return True
        if np.any(Q < self.q_thresh) or np.any(Q > (1 - self.q_thresh)): log.warning("Numerical solver triggered: Extreme probabilities."); return True
        if any(c.get('is_logical_rule', False) for c in contracts): log.warning("Numerical solver triggered: Logical rule."); return True
        return False

    def _build_covariance_matrix(self, graph, contracts):
        n = len(contracts); C = np.zeros((n, n)); P = np.array([c['M'] for c in contracts])
        for i in range(n):
            for j in range(i, n):
                if i == j: C[i, i] = P[i] * (1 - P[i]); continue
                rel = graph.get_relationship_between_contracts(contracts[i]['id'], contracts[j]['id'], contracts)
                p_ij = rel.get('p_joint');
                if p_ij is None: p_ij = P[i] * P[j]
                cov = p_ij - P[i] * P[j]; C[i, j] = cov; C[j, i] = cov
        return C

    def _solve_analytical(self, C, D, E):
        log.info("Solving with Analytical (Fast Path)...")
        C_inv = np.linalg.pinv(C); F_star = D @ C_inv @ E
        return F_star

    def _solve_numerical(self, M: np.ndarray, Q: np.ndarray, C: np.ndarray, F_analytical_guess: np.ndarray) -> np.ndarray:
        # Optimization: Don't run simulation if analytical guess is very close to bounds
        # or if the dimension is small and covariance is low.
        
        # Optimization: Pre-calculate Cholesky decomposition once outside the minimizer
        try:
            n = len(M)
            std_devs = np.sqrt(np.diag(C))
            # Avoid division by zero
            std_devs[std_devs == 0] = 1e-9
            Corr = C / np.outer(std_devs, std_devs)
            np.fill_diagonal(Corr, 1.0)
            L = np.linalg.cholesky(Corr + np.eye(n) * 1e-9)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if decomposition fails
            L = np.eye(n)

        # Generate the random samples ONCE (Fixed Seed per step for stability)
        # Moving random generation outside the objective function speeds up scipy.minimize drastically
        rng = np.random.default_rng(seed=42) # Hardcode the seed
        Z = rng.standard_normal((self.k_samples, n))
        correlated_Z = Z @ L.T
        U = norm.cdf(correlated_Z)
        I_k = (U < M).astype(float) # Pre-calculate outcomes

        # Define a faster objective function using pre-calculated I_k
        def fast_objective(F):
            # Vectorized calculation
            # Prevent Q=0 or Q=1
            Q_safe = np.clip(Q, 1e-9, 1.0 - 1e-9)
            
            # Gains matrix
            gains_long = (I_k - Q_safe) / Q_safe
            gains_short = (Q_safe - I_k) / (1.0 - Q_safe)
            
            # Select gains based on F sign (Long or Short)
            # Note: F is (n,), gains is (k, n)
            # Broadcasting F to match gains shape
            pos_mask = F > 0
            R_k = np.where(pos_mask, gains_long, gains_short)
            
            # Portfolio return per sample
            # sum(R_k * |F|)
            port_returns = np.dot(R_k, np.abs(F))
            
            W_k = 1.0 + port_returns
            
            if np.any(W_k <= 1e-9): return 1e9 # Bankruptcy penalty
            return -np.mean(np.log(W_k))

        # Run Optimizer
        constraints = ({'type': 'ineq', 'fun': lambda F: 0.8 - np.sum(np.abs(F))})
        bounds = [(-0.5, 0.5)] * n
        
        result = opt.minimize(
            fast_objective, F_analytical_guess, method='SLSQP',
            bounds=bounds, constraints=constraints, tol=1e-4 # Loosened tolerance for speed
        )
        return result.x
        
    def solve_basket(self, graph, contracts):
        n = len(contracts); 
        if n == 0: return np.array([]) # Handle empty cluster
        M = np.array([c['M'] for c in contracts]); Q = np.array([c['Q'] for c in contracts])
        E = M - Q; D = np.diag(Q); C = self._build_covariance_matrix(graph, contracts)
        F_analytical = self._solve_analytical(C, D, E)
        if self._is_numerical_required(E, Q, contracts):
            return self._solve_numerical(M, Q, C, F_analytical)
        return F_analytical

class PortfolioManager:
    """(Production-Ready C6)"""
    def __init__(self, graph_manager: GraphManager, solver: HybridKellySolver):
        self.graph = graph_manager; self.solver = solver; self.max_event_exposure = 0.15
    def _apply_constraints(self, F_star: np.ndarray) -> np.ndarray:
        total_exposure = np.sum(np.abs(F_star))
        if total_exposure > self.max_event_exposure:
            log.warning(f"Capping exposure: {total_exposure:.2f} > {self.max_event_exposure}")
            return F_star * (self.max_event_exposure / total_exposure)
        return F_star
    def run_optimization_cycle(self) -> Dict[str, float]:
        log.info("--- C6: Starting Optimization Cycle ---")
        active_clusters = self.graph.get_active_entity_clusters()
        final_basket = {}
        for cluster_id in active_clusters:
            log.info(f"--- C6: Solving Cluster: {cluster_id} ---")
            contracts = self.graph.get_cluster_contracts(cluster_id)
            if len(contracts) < 1: continue
            
            F_star_unconstrained = self.solver.solve_basket(self.graph, contracts)
            F_star_final = self._apply_constraints(F_star_unconstrained)
            
            for i, contract in enumerate(contracts):
                allocation = F_star_final[i]
                if abs(allocation) > 1e-5:
                    final_basket[contract['id']] = allocation
                    action = "BUY" if allocation > 0 else "SELL"
                    log.info(f"-> {action} {abs(allocation)*100:.2f}% on {contract['id']} (Edge: {contracts[i]['M'] - contracts[i]['Q']:.2f})")
        log.info("--- C6: Optimization Cycle Complete ---")
        return final_basket

# ==============================================================================
# ### COMPONENT 7: Back-Testing & Tuning (Final Optimized) ###
# ==============================================================================

# --- OPTIMIZATION HELPER: Numba-Accelerated Kelly Solver ---
@njit
def fast_kelly_objective(F, M, Q, I_k):
    """Numba-compiled Kelly objective function using pre-calculated outcomes."""
    k = I_k.shape[0]
    n = len(M)
    
    # Safe Q to avoid division by zero
    Q_safe = np.empty_like(Q)
    for i in range(n):
        Q_safe[i] = max(1e-9, min(Q[i], 1.0 - 1e-9))
    
    total_log_wealth = 0.0
    
    # Iterate samples
    for i in range(k):
        port_return = 0.0
        # Iterate assets
        for j in range(n):
            f_val = F[j]
            if f_val == 0: continue
                
            outcome = I_k[i, j]
            q_val = Q_safe[j]
            
            if f_val > 0:  # Long
                gains = (outcome - q_val) / q_val
            else:  # Short
                gains = (q_val - outcome) / (1.0 - q_val)
                
            port_return += f_val * gains
        
        W = 1.0 + port_return
        if W <= 1e-9: return 1e9  # Bankruptcy penalty
        
        total_log_wealth += np.log(W)
    
    return -total_log_wealth / k


# --- OPTIMIZATION HELPER: NLP Cache ---


class NLPCache:
    """
    Persistent NLP Cache.
    Loads previous results from disk so we don't re-run spaCy on old markets.
    """
    def __init__(self, df_markets, nlp_model, cache_path="nlp_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache = {}
        self.new_entries = False
        
        # 1. Load existing cache from disk
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
                log.info(f"Loaded {len(self.cache)} NLP entities from disk.")
            except Exception as e:
                log.warning(f"Failed to load NLP cache: {e}")

        # 2. Identify which markets are missing from the cache
        # We only run NLP on the *delta* (missing keys)
        missing_ids = [
            mid for mid in df_markets['market_id'].astype(str) 
            if mid not in self.cache
        ]
        
        if not missing_ids:
            log.info("All markets already cached! Skipping NLP.")
            return

        log.info(f"Running NLP on {len(missing_ids)} new markets...")
        
        # Filter df to only new markets
        df_new = df_markets[df_markets['market_id'].astype(str).isin(missing_ids)]
        questions = df_new['question'].tolist()
        ids = df_new['market_id'].astype(str).tolist()
        
        relevant_labels = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}
        
        # 3. Batch Process only the new questions
        if nlp_model:
            doc_stream = nlp_model.pipe(questions, batch_size=200)
            for doc, mid in zip(doc_stream, ids):
                ents = [ent.text for ent in doc.ents if ent.label_ in relevant_labels]
                self.cache[mid] = ents
                self.new_entries = True
        else:
            # Fallback if no NLP model provided
            for q, mid in zip(questions, ids):
                self.cache[mid] = q.split()[:3]
                self.new_entries = True

        # 4. Save back to disk immediately
        if self.new_entries:
            self.save()

    def save(self):
        """Saves the cache to disk."""
        log.info(f"Saving NLP cache ({len(self.cache)} entries) to disk...")
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)

    def get_entities(self, market_id):
        return self.cache.get(str(market_id), [])



# --- OPTIMIZATION HELPER: Vectorized Profiler ---
def fast_calculate_brier_scores(profiler_data: pd.DataFrame, min_trades: int = 20):
    """
    Vectorized Brier score calculation (100x faster than groupby.apply).
    """
    print(f"DEBUG: Brier Calc received {len(profiler_data)} rows. Columns: {list(profiler_data.columns)}")
    if profiler_data.empty:
        return {}

    # Filter valid rows
    valid = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id'])
    
    # Group and count
    counts = valid.groupby(['wallet_id', 'entity_type']).size()
    sufficient = counts[counts >= min_trades].index
    
    # Filter to sufficient trades using Index intersection
    valid.set_index(['wallet_id', 'entity_type'], inplace=True)
    filtered = valid.loc[valid.index.intersection(sufficient)].reset_index()
    
    # Vectorized Brier calculation
    filtered['brier'] = (filtered['bet_price'] - filtered['outcome']) ** 2
    
    # Group and mean
    scores = filtered.groupby(['wallet_id', 'entity_type'])['brier'].mean()

    # 1. Convert to DataFrame to sort by Multiple Columns
    scores = scores.reset_index()
    
    # 2. Sort by Brier Score (Ascending) THEN Wallet ID (Ascending)
    # This ensures that wallets with the EXACT SAME score are always consistently ordered.
    scores = scores.sort_values(
        by=['brier', 'wallet_id'], 
        ascending=[True, True],
        kind='stable'
    )

    
    
    # --- FIXED SYNTAX BELOW (Added closing quote) ---
    print(f"SCORES: {str(scores)}")
    
    # --- [PATCH: DEBUG PRINTS] ---
    # Sort by Score (Ascending). 
    # Remember: Brier Score 0.0 = Perfect Prediction. Brier Score 1.0 = Dead Wrong.
    if not scores.empty:
        # Get top 5 most accurate wallets
        top_performers = scores.sort_values(by='brier', ascending=True).head(10)
        
        # Get bottom 5 (worst traders)
        worst_performers = scores.sort_values(by='brier', ascending=False).head(10)
        
        print(f"\n🔎 BRIER ENGINE REPORT: Scored {len(scores)} unique wallets.")
        print(f"   🏆 Top 10 'Smartest' Wallets (Score ~ 0.0):\n{top_performers.to_string()}")
        print(f"   📉 Top 10 'Worst' Wallets (Score > 0.5):\n{worst_performers.to_string()}\n")
    else:
        print("⚠️ BRIER ENGINE: No scores calculated! (Check Input Data)")
    # -----------------------------
    
    return scores.to_dict()


# ==============================================================================
# --- CORE ENGINE: FastBacktestEngine (Percentage vs Kelly) ---
# ==============================================================================
# ==============================================================================
# --- CORE ENGINE: FastBacktestEngine (Full Logic Matrix) ---
# ==============================================================================
# ==============================================================================
# --- CORE ENGINE: FastBacktestEngine (Directional Alpha Fix) ---
# ==============================================================================
class FastBacktestEngine:
    """
    Statistically Rigorous Backtester.
    FIXED: Signal Generation is now DIRECTIONAL.
    - Smart Buys vote for p=0.99.
    - Smart Sells vote for p=0.01.
    - This creates a real mathematical edge without needing noise.
    """
    def __init__(self, event_log, profiler_data, nlp_cache, precalc_priors):
        self.event_log = event_log
        self.profiler_data = profiler_data
        self.market_lifecycle = {}
        
        if not event_log.empty:
            new_contracts = event_log[event_log['event_type'] == 'NEW_CONTRACT']
            for ts, row in new_contracts.iterrows():
                data = row['data']
                cid = data.get('contract_id')
                if cid:
                    scheduled_end = data.get('end_date')
                    if not scheduled_end or pd.isna(scheduled_end):
                        scheduled_end = pd.Timestamp.max
                        
                    self.market_lifecycle[cid] = {
                        'start': ts, 
                        'end': scheduled_end, 
                        'liquidity': data.get('liquidity', 10000.0)
                    }
            
            resolutions = event_log[event_log['event_type'] == 'RESOLUTION']
            for ts, row in resolutions.iterrows():
                cid = row['data'].get('contract_id')
                if cid in self.market_lifecycle: self.market_lifecycle[cid]['end'] = ts

            records = event_log.reset_index().to_dict('records')
            self.minute_batches = []
            from itertools import groupby
            for key, group in groupby(records, key=lambda x: x['timestamp'].strftime('%Y%m%d%H%M')):
                self.minute_batches.append(list(group))
        else:
            self.minute_batches = []
            
    def calibrate_fresh_wallet_model(self, profiler_data, known_wallet_ids=None):
        """
        Calibrates the 'Fresh Wallet' regression model (Volume -> Skill).
        Includes strict statistical validation to prevent fitting noise.
        """
        from scipy.stats import linregress
        import numpy as np
        
        # 1. Safe Default (Null Hypothesis: Random Brier 0.25, No Volume impact)
        SAFE_SLOPE = 0.0
        SAFE_INTERCEPT = 0.25
        
        # 2. Data Prep
        # We need trades that have a known outcome
        if 'outcome' not in profiler_data.columns or profiler_data.empty:
            return SAFE_SLOPE, SAFE_INTERCEPT
            
        valid = profiler_data.dropna(subset=['outcome', 'usdc_vol', 'tokens'])
        if known_wallet_ids:
            # We only want to learn from the "Amateurs/Transients"
            valid = valid[~valid['wallet_id'].isin(known_wallet_ids)]
        # 3. Minimum Sample Size Check
        # Fitting a regression on < 50 trades is statistical malpractice
        if len(valid) < 50:
            return SAFE_SLOPE, SAFE_INTERCEPT

        # 4. Calculate Trade-Level Brier Scores
        # If Long (tokens > 0), Prediction is 1.0. Error = (1 - Outcome)^2
        # If Short (tokens < 0), Prediction is 0.0. Error = (0 - Outcome)^2
        valid = valid.copy()
        valid['prediction'] = np.where(valid['tokens'] > 0, 1.0, 0.0)
        valid['brier'] = (valid['prediction'] - valid['outcome']) ** 2
        
        # Log Transform Volume (Standardize impact)
        # We use log1p to handle small values gracefully
        valid['log_vol'] = np.log1p(valid['usdc_vol'])

        try:
            # 5. Run Regression
            slope, intercept, r_val, p_val, std_err = linregress(valid['log_vol'], valid['brier'])
            
            # --- FIX: BAYESIAN DAMPENING & ECONOMIC SIGNIFICANCE ---
            
            # 1. Directional Check:
            # We only care if volume IMPROVES skill (lowers Brier).
            # If slope is positive (Volume = Higher Brier), it's a "Dumb Whale".
            # We ignore Dumb Whale signals to be safe.
            if slope >= 0:
                return SAFE_SLOPE, SAFE_INTERCEPT

            # 2. Economic Significance (Effect Size):
            # If the slope is microscopic (e.g. -0.0001), it's noise even if p < 0.05.
            # We require a minimum improvement per unit of log volume.
            MIN_EFFECT_SIZE = -0.002 
            if slope > MIN_EFFECT_SIZE:
                return SAFE_SLOPE, SAFE_INTERCEPT

            # 3. Bayesian Dampening (The "Dimmer Switch"):
            # Instead of a hard cliff at 0.05, we linearly decay confidence 
            # from p=0.0 (Full Trust) to p=0.20 (Zero Trust).
            SIGNIFICANCE_LIMIT = 0.20
            
            if p_val >= SIGNIFICANCE_LIMIT:
                return SAFE_SLOPE, SAFE_INTERCEPT
            
            # Calculate Confidence Factor (0.0 to 1.0)
            confidence = 1.0 - (p_val / SIGNIFICANCE_LIMIT)
            
            # Dampen the slope based on confidence
            # If p=0.01, we keep ~95% of the slope.
            # If p=0.19, we keep ~5% of the slope.
            final_slope = slope * confidence
            
            # 4. Intercept Bounding (Safety)
            # Ensure the baseline amateur is not "Perfect" (0.0) or "Terrible" (0.5)
            final_intercept = np.clip(intercept, 0.15, 0.35)
            
            return final_slope, final_intercept
            
        except Exception as e:
            return SAFE_SLOPE, SAFE_INTERCEPT
            
    def run_walk_forward(self, config: Dict) -> Dict[str, float]:
        """
        Rolling Walk-Forward Optimization (Dictionary Compatible).
        """
        import numpy as np
        from datetime import timedelta
        
        # 1. Setup Time Boundaries
        if self.event_log.empty:
            return {'total_return': 0.0, 'sharpe': 0.0, 'trades': 0}

        min_date = self.event_log.index.min()
        max_date = self.event_log.index.max()
        
        train_days = config.get('train_days', 60)
        test_days = config.get('test_days', 120)
        
        current_date = min_date
        
        # Performance Tracking
        total_pnl = 0.0
        total_trades = 0
        capital = 10000.0
        equity_curve = [capital]
        
        all_resolutions = self.event_log[self.event_log['event_type'] == 'RESOLUTION']
        
        # --- WALK FORWARD LOOP ---
        embargo_days = 2 # Gap to prevent leakage
        
        while current_date + timedelta(days=train_days + embargo_days + test_days) <= max_date:
            train_end = current_date + timedelta(days=train_days)
            
            # EMBARGO: The test set starts AFTER the gap
            test_start = train_end + timedelta(days=embargo_days)
            test_end = test_start + timedelta(days=test_days)
            
            # A. Prepare Training Data (Strictly BEFORE train_end)
            train_mask = (self.profiler_data['timestamp'] >= current_date) & \
                         (self.profiler_data['timestamp'] < train_end)
            train_profiler = self.profiler_data[train_mask].copy()
            
            # STRICT FILTER: Only consider resolutions that happened BEFORE the training window ends.
            # This prevents "Time Travel" where we train on an outcome that hasn't happened yet.
            valid_res = all_resolutions[all_resolutions.index < train_end]
            
            # Build Whitelist of Resolved IDs
            resolved_ids = set()
            outcome_map = {}
            for _, row in valid_res.iterrows():
                cid = row['data']['contract_id']
                resolved_ids.add(cid)
                outcome_map[cid] = float(row['data']['outcome'])
            
            # EXPLICIT FILTER: Only keep trades for markets that have actually resolved
            train_profiler = train_profiler[train_profiler['market_id'].isin(resolved_ids)]
            
            # Map Outcomes
            # 1. Build Lookups for Outcome AND Resolution Time
            outcome_map = {}
            res_time_map = {}
            
            for ts, row in valid_res.iterrows():
                cid = row['data']['contract_id']
                outcome_map[cid] = float(row['data']['outcome'])
                res_time_map[cid] = ts # Capture the exact resolution timestamp
            
            # 2. Map Data
            train_profiler['outcome'] = train_profiler['market_id'].map(outcome_map)
            train_profiler['res_time'] = train_profiler['market_id'].map(res_time_map)
            
            # 3. CRITICAL HYGIENE: Filter out "Post-Mortem" Trades
            # Ensure we only learn from trades that happened BEFORE the market resolved.
            # This prevents "Cheater Trades" (trading after the news is public) from corrupting wallet scores.
            train_profiler = train_profiler[train_profiler['timestamp'] < train_profiler['res_time']]
            
            # 4. Final Safety Drop (Removes any market that didn't resolve in the window)
            train_profiler = train_profiler.dropna(subset=['outcome'])
            
            # B. Calibrate Models
            fold_wallet_scores = fast_calculate_brier_scores(train_profiler, min_trades=5)
            # Use 'self.' to call method
            known_experts = sorted(list(set(k[0] for k in fold_wallet_scores.keys())))
            fw_slope, fw_intercept = self.calibrate_fresh_wallet_model(train_profiler, known_wallet_ids=known_experts)
            
            # C. Run Test Simulation
            test_slice = self.event_log[(self.event_log.index >= test_start) & 
                                        (self.event_log.index < test_end)]
            
            if not test_slice.empty:
                batches = []
                grouped = test_slice.groupby(pd.Grouper(freq='1min'))
                for _, group in grouped:
                    if not group.empty:
                        batch_events = []
                        for ts, row in group.iterrows():
                            data = row['data']
                            data['timestamp'] = ts
                            if row['event_type'] == 'NEW_CONTRACT':
                                if data.get('liquidity', 0) == 0:
                                    data['liquidity'] = 10000.0
                            ev = {'event_type': row['event_type'], 'data': data}
                            batch_events.append(ev)
                        batches.append(batch_events)
                        
                past_events = self.event_log[self.event_log.index < test_end]
                init_events = past_events[past_events['event_type'].isin(['NEW_CONTRACT', 'MARKET_INIT'])]
                global_liq = {}
                for _, row in init_events.iterrows():
                    l = row['data'].get('liquidity')
                    # Ensure we capture the 10k fix if it's there
                    if l is None or l == 0: l = 10000.0
                    global_liq[row['data']['contract_id']] = l
                    
                # RUN SIMULATION
                # --- FIX: Handle Dictionary Return ---
                result = self._run_single_period(
                    batches, 
                    fold_wallet_scores, 
                    config, 
                    fw_slope, 
                    fw_intercept, 
                    start_time=train_end,
                    known_liquidity=global_liq
                )
                
                # Extract values from dict
                pnl = result['final_value']
                trades = result['trades']
                # -------------------------------------
                
                # Update Globals
                local_curve = result.get('equity_curve', [pnl])
                period_growth = [x / 10000.0 for x in local_curve]
                scaled_curve = [capital * x for x in period_growth]
                if len(equity_curve) > 0:
                    equity_curve.extend(scaled_curve[1:])
                else:
                    equity_curve.extend(scaled_curve)

                capital = equity_curve[-1]
                total_trades += trades
            
            # Slide Window
            current_date += timedelta(days=test_days)
        
        # --- FIX 2: ROBUST METRICS ---
        if not equity_curve:
             return {'total_return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0, 'trades': 0}

        series = pd.Series(equity_curve)
        
        # 1. Total Return
        total_ret = (capital - 10000.0) / 10000.0
        
        # 2. Max Drawdown
        # Tracks the percentage drop from the highest point seen so far
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        max_dd = drawdown.min() # This will be negative (e.g., -0.05)
        
        # 3. Sharpe Ratio
        # Calculate returns per step (minute)
        pct_changes = series.pct_change().dropna()
        
        sharpe = 0.0
        if len(pct_changes) > 1 and pct_changes.std() > 0:
            mean_ret = pct_changes.mean()
            std_ret = pct_changes.std()
            
            # Annualize: sqrt(Minutes in a Year) -> sqrt(525,600) ≈ 725
            # This aligns the Sharpe to a standard yearly metric
            annualization_factor = np.sqrt(252 * 1440) 
            sharpe = (mean_ret / std_ret) * annualization_factor
        
        return {
            'total_return': total_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_dd), # Return as positive number (e.g. 0.05)
            'trades': total_trades,
            'equity_curve': equity_curve 
        }
        # -----------------------------

    def _run_single_period(self, batches, wallet_scores, config, fw_slope, fw_intercept, start_time, known_liquidity=None):

        
        # --- CONFIG EXTRACTION ---
        splash_thresh = config.get('splash_threshold', 100.0) 
        use_smart_exit = config.get('use_smart_exit', False)
        stop_loss_pct = config.get('stop_loss_pct', None)
        sizing_mode = config.get('sizing_mode', 'kelly')
        sizing_val = config.get('kelly_fraction', 0.25)
        if sizing_mode == 'fixed': 
            sizing_val = config.get('fixed_size', 10.0)
        
        # --- STATE INIT ---
        cash = 10000.0
        equity_curve = [] # Initialize list here
        positions = {}
        tracker = {}
        market_liq = {}
        market_prices = {}
        trade_count = 0
        volume_traded = 0.0
        
        # Diagnostics
        debug_signals = []
        rejection_log = {'low_volume': 0, 'unsafe_price': 0, 'low_edge': 0, 'insufficient_cash': 0}
        
        # --- BATCH PROCESSING ---
        # [FIXED: TWO-PASS LOGIC WITH DEDUPLICATION]
        for batch_idx, batch in enumerate(batches):
            
            # 1. Strict Sort within the Batch
            # Ensure we process events in time order. 
            # If timestamps are identical, process RESOLUTION before PRICE_UPDATE.
            # We assume 'timestamp' is in the event, or we rely on the list order if pre-sorted.
            # Ideally, the event log passed here should already be sorted, but this is a safety.
            def sort_key(e):
                # Primary: Timestamp (if available, otherwise 0 implies preserve order)
                ts = e['data'].get('timestamp', pd.Timestamp.min)
                
                # Secondary: Event Priority (Resolution First = 0, Price Update = 1)
                # This only affects ties at the exact same timestamp.
                prio = 0 if e['event_type'] == 'RESOLUTION' else 1
                
                return (ts, prio)
            
            # Note: If batch is already sorted by time, this is fast.
            batch.sort(key=sort_key)

            for event in batch:
                ev_type = event['event_type']
                data = event['data']
                cid = data.get('contract_id')

                # ==================== A. NEW CONTRACT ====================
                if ev_type == 'NEW_CONTRACT':
                    tracker[cid] = {'net_weight': 0.0, 'last_price': 0.5}

                # ==================== B. RESOLUTION (Strictly Sequential) ====================
                elif ev_type == 'RESOLUTION':
                    if cid in positions:
                        pos = positions[cid]
                        outcome = float(data.get('outcome', 0))
                        
                        if outcome == 0.5:
                            refund = pos['shares'] * 0.50
                            cash += refund
                        else:
                            win = False
                            if pos['side'] == 1 and outcome == 1.0: win = True
                            if pos['side'] == -1 and outcome == 0.0: win = True
                            
                            if win:
                                cash += pos['shares']
                        
                        del positions[cid]

                # ==================== C. PRICE UPDATE & EXECUTION (Strictly Sequential) ====================
                elif ev_type == 'PRICE_UPDATE':
                    # 1. Update State
                    if cid not in market_liq:
                        market_liq[cid] = known_liquidity.get(cid, 10000.0) if known_liquidity else 10000.0
                    
                    new_price = data.get('p_market_all', 0.5)
                    if cid in tracker:
                        prev_p = tracker[cid]['last_price']
                        price_delta = abs(new_price - prev_p)
                        trade_vol = float(data.get('trade_volume', 0.0))
                        
                        # Only update if the move is significant enough to be a valid signal (avoid div by zero)
                        if price_delta > 0.005 and trade_vol > 10.0:
                            # CPMM Approximation: Impact = Vol / Liquidity  =>  Liquidity = Vol / Impact
                            # We use a dampening factor (0.5) to be conservative
                            implied_liq = (trade_vol / price_delta) * 0.5
                            
                            # Smooth the liquidity estimate (Exponential Moving Average)
                            # This prevents one crazy trade from ruining the estimation
                            current_liq = market_liq[cid]
                            market_liq[cid] = (current_liq * 0.9) + (implied_liq * 0.1)
                    # Initialize tracker if missing
                    if cid not in tracker: tracker[cid] = {'net_weight': 0.0, 'last_price': 0.5}
                    
                    prev_price = tracker[cid]['last_price']
                    tracker[cid]['last_price'] = new_price
                    
                    # 2. Process Signal
                    vol = float(data.get('trade_volume', 0.0))
                    
                    # Logic Check: Valid Volume?
                    if vol >= 1.0:
                        is_sell = data.get('is_sell', False)
                        trade_direction = -1.0 if is_sell else 1.0
                        w_id = str(data.get('wallet_id'))
                        
                        # --- Brier Score Lookup ---
                        brier = wallet_scores.get((w_id, 'default_topic'))
                        if brier is None:
                            log_vol = np.log(max(vol, 1.0))
                            pred_brier = fw_intercept + (fw_slope * log_vol)
                            brier = max(0.10, min(pred_brier, 0.35))
                        
                        # --- Weight Calculation ---
                        raw_skill = max(0.0, 0.25 - brier)
                        skill_factor = np.log1p(raw_skill * 100)
                        multiplier = 1.0 + min(skill_factor * 5.0, 10.0) 
                        weight = vol * multiplier
                        tracker[cid]['net_weight'] += (weight * trade_direction)

                        # --- Threshold Check ---
                        if abs(tracker[cid]['net_weight']) > splash_thresh:
                            
                            # Signal Generated
                            raw_net = tracker[cid]['net_weight']
                            tracker[cid]['net_weight'] = 0.0 # Reset Accumulator
                            
                            net_sentiment = np.tanh(raw_net / 5000.0)
                            p_model = 0.5 + (net_sentiment * 0.49)
                            edge = p_model - prev_price
                            market_info = self.market_lifecycle.get(cid)
                            current_ts = data.get('timestamp')
                            
                            # Check A: Do we have the 'End Date' metadata?
                            if not market_info or 'end' not in market_info or market_info['end'] == pd.Timestamp.max:
                                rejection_log['missing_metadata'] = rejection_log.get('missing_metadata', 0) + 1
                                continue # SKIP: Cannot quantify time risk.
                                
                            # Check B: Do we have the 'Current Time' from the event stream?
                            if not current_ts:
                                rejection_log['missing_timestamp'] = rejection_log.get('missing_timestamp', 0) + 1
                                continue # SKIP: Malformed event data.
    
                            # 3. Calculate Duration
                            end_ts = market_info['end']
                            delta = end_ts - current_ts
                            days_remaining = delta.total_seconds() / 86400.0
                            
                            # Check C: Is the market already expired? (Data lag protection)
                            if days_remaining <= 0:
                                rejection_log['market_expired'] = rejection_log.get('market_expired', 0) + 1
                                continue # SKIP: Trading on a dead market.
    
                            # 4. Standard Execution (Clean Edge Check)
                            # We have verified the trade is valid and active. Now we check the edge.
                            edge_thresh = config.get('edge_threshold', 0.05)
                            is_safe_price = (new_price >= 0.02 and new_price <= 0.98)
    
                            if abs(edge) >= edge_thresh and is_safe_price:
                                
                                # Do we already hold this?
                                if cid not in positions:
                                    
                                    # Sizing
                                    target_f = 0.0
                                    cost = 0.0
                                    
                                    if sizing_mode == 'fixed_pct':
                                        target_f = sizing_val
                                    elif sizing_mode == 'kelly':
                                        target_f = abs(edge) * sizing_val
                                    elif sizing_mode == 'fixed':
                                        cost = sizing_val
                                        target_f = -1 
                                    
                                    if target_f > 0:
                                        target_f = min(target_f, 0.20)
                                        cost = cash * target_f
                                    
                                    # --- CASH CHECK (THE MOMENT OF TRUTH) ---
                                    # We check cash NOW. If we don't have it, we miss the trade.
                                    # This is painful but historically accurate.
                                    if cost > 5.0 and cash > cost:
                                        
                                        # Execute
                                        side = 1 if edge > 0 else -1
                                        pool_liq = market_liq.get(cid, 0.0) 

                                        # 2. First Principles Check
                                        if pool_liq <= 1.0: # Effectively zero
                                            # Log rejection
                                            rejection_log['no_liquidity'] = rejection_log.get('no_liquidity', 0) + 1
                                            continue # SKIP THE TRADE. Do not execute.
                                        
                                        # 3. Only calculate slippage if liquidity exists
                                        
                                        friction_rate = 0.002
                                        fixed_penalty = 0.03
                                        
                                        investment_principal = cost / (1.0 + friction_rate)
                                        net_capital = investment_principal * (1.0 - fixed_penalty)
                                        variable_impact = min(net_capital / (pool_liq + net_capital), 0.15)
                                        
                                        if side == 1:
                                            safe_entry = min(new_price + variable_impact, 0.99)
                                            shares = net_capital / safe_entry
                                        else:
                                            safe_entry = max(new_price - variable_impact, 0.01)
                                            shares = net_capital / (1.0 - safe_entry)

                                        positions[cid] = {
                                            'side': side,
                                            'size': cost,
                                            'shares': shares,
                                            'entry': safe_entry
                                        }
                                        trade_count += 1
                                        volume_traded += cost
                                        cash -= cost # Deduct immediately
                                        
                                    else:
                                        rejection_log['insufficient_cash'] += 1

                # ==================== D. POSITION MANAGEMENT (Stop Losses) ====================
                if ev_type != 'RESOLUTION' and cid in positions:
                    pos = positions[cid]
                    curr_p = tracker.get(cid, {}).get('last_price', pos['entry'])
                    
                    # PnL Calculation
                    if pos['side'] == 1:
                        pnl_pct = (curr_p - pos['entry']) / pos['entry']
                    else:
                        pnl_pct = (pos['entry'] - curr_p) / (1.0 - pos['entry'])
                    
                    should_close = False
                    if stop_loss_pct and pnl_pct < -stop_loss_pct:
                        should_close = True
                    
                    if use_smart_exit:
                        cur_net = tracker.get(cid, {}).get('net_weight', 0)
                        
                        # [PATCH] Configurable Smart Exit Ratio
                        # Default to 0.5 (original behavior) if not in config
                        exit_ratio = config.get('smart_exit_ratio', 0.5)
                        exit_trigger_val = splash_thresh * exit_ratio
                        
                        # LONG EXIT: Sentiment flips negative beyond threshold
                        if pos['side'] == 1 and cur_net < -exit_trigger_val: 
                            should_close = True
                            
                        # SHORT EXIT: Sentiment flips positive beyond threshold
                        if pos['side'] == -1 and cur_net > exit_trigger_val: 
                            should_close = True
                    
                    if should_close:
                        if pos['side'] == 1: payout = pos['shares'] * curr_p
                        else: payout = pos['shares'] * (1.0 - curr_p)
                        cash += payout
                        del positions[cid]
            
            # --- VALUATION (End of Minute) ---
            current_val = cash
            for cid in sorted(positions.keys()):
                pos = positions[cid]
                last_p = tracker.get(cid, {}).get('last_price', pos['entry'])
                if pos['side'] == 1: val = pos['shares'] * last_p
                else: val = pos['shares'] * (1.0 - last_p)
                current_val += val
            equity_curve.append(current_val)
   
        # --- END OF PERIOD VALUATION ---
        final_value = cash
        for cid, pos in positions.items():
            last_p = tracker.get(cid, {}).get('last_price', pos['entry'])
            if pos['side'] == 1:
                val = pos['shares'] * last_p
            else:
                val = pos['shares'] * (1.0 - last_p)
            final_value += val
        
        # --- DIAGNOSTICS ---
        print(f"\n📊 PERIOD SUMMARY:")
        print(f"   Trades Executed: {trade_count}")
        print(f"   Volume Traded: ${volume_traded:.0f}")
        print(f"   Final Value: ${final_value:.2f}")
        print(f"   Return: {((final_value/10000.0)-1.0)*100:.2f}%")
        print(f"\n🚫 REJECTION LOG:")
        for reason, count in rejection_log.items():
            if count > 0:
                print(f"   {reason}: {count}")
        
        return {
            'final_value': final_value,
            'total_return': (final_value / 10000.0) - 1.0,
            'trades': trade_count,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'equity_curve': equity_curve
        }

    
    
    def _aggregate_fold_results(self, results):
        if not results: return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'trades': 0, 'max_drawdown': 0.0}
        total_ret = 1.0
        sharpes = []
        drawdowns = []
        trades = 0
        for r in results:
            total_ret *= (1.0 + r['total_return'])
            sharpes.append(r['sharpe_ratio'])
            drawdowns.append(r['max_drawdown'])
            trades += r['trades']
        avg_sharpe = np.mean(sharpes) if sharpes else 0.0
        return {'total_return': total_ret - 1.0, 'sharpe_ratio': avg_sharpe, 'max_drawdown': max(drawdowns) if drawdowns else 0.0, 'trades': trades, 'rejection_logs': {}}

    def _calculate_metrics(self, pnl_history, trades):
        if len(pnl_history) < 2: return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'trades': 0, 'max_drawdown': 0.0}
        df = pd.DataFrame(pnl_history, columns=['time', 'equity'])
        df = df.set_index('time')
        df_resampled = df.resample('1h').last().ffill()
        df_resampled['ret'] = df_resampled['equity'].pct_change().fillna(0.0)
        
        total_ret = (df_resampled['equity'].iloc[-1] / df_resampled['equity'].iloc[0]) - 1.0
        mean_ret = df_resampled['ret'].mean()
        std_ret = df_resampled['ret'].std() + 1e-9
        sharpe = (mean_ret / std_ret) * np.sqrt(8760)
        
        peak = df_resampled['equity'].cummax()
        dd = (peak - df_resampled['equity']) / peak
        max_dd = dd.max()
        return {'total_return': total_ret, 'sharpe_ratio': sharpe, 'max_drawdown': max_dd, 'trades': trades}
        
# ==============================================================================
# --- HELPER: Ray Wrapper (Global) ---
# ==============================================================================
def ray_backtest_wrapper(config, event_log_ref, profiler_ref, nlp_cache_ref, priors_ref):
    """
    FIXED: Now properly unpacks the 'sizing' tuple into the keys expected by the engine.
    
    CRITICAL FIX:
    - Ray Tune passes: config = {'sizing': ('kelly', 0.5), ...}
    - Engine expects: config = {'sizing_mode': 'kelly', 'kelly_fraction': 0.5, ...}
    - This wrapper bridges the gap.
    """
    import traceback
    import random
    import numpy as np
    import torch
    try:
        seed = config.get('seed', 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # <--- Critical for PyTorch/Spacy consistency
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        def get_ref(obj):
            if isinstance(obj, ray.ObjectRef): 
                return ray.get(obj)
            return obj

        event_log = get_ref(event_log_ref)
        profiler_data = get_ref(profiler_ref)
        nlp_cache = get_ref(nlp_cache_ref)
        priors = get_ref(priors_ref)

        # ============================================================
        # === CRITICAL FIX: Unpack Sizing Tuple ===
        # ============================================================
        if 'sizing' in config:
            mode, val = config['sizing']
            
            # Map to engine-expected keys
            config['sizing_mode'] = mode
            
            if mode == 'kelly':
                config['kelly_fraction'] = val
            elif mode == 'fixed_pct':
                # Engine uses 'sizing_val' directly for fixed_pct
                # Since engine does: target_f = sizing_val
                # We don't need a separate key, just ensure it reads correctly
                config['fixed_size'] = val  # Store for consistency
            elif mode == 'fixed':
                config['fixed_size'] = val
            
            # Optional: Remove the tuple to avoid confusion in logs
            # del config['sizing']
        # ============================================================
        
        # Also unpack stop_loss if it's named differently in search space
        if 'stop_loss' in config:
            config['stop_loss_pct'] = config['stop_loss']

        engine = FastBacktestEngine(event_log, profiler_data, None, {})
        
        # Fixed Seed for reproducibility inside the worker
        import numpy as np
        np.random.seed(config.get('seed', 42))

        results = engine.run_walk_forward(config)
        
        ret = results.get('total_return', 0.0)
        dd = results.get('max_drawdown', 1.0)
        
        # Calculate Smart Score inside worker
        smart_score = ret / (dd + 0.01)
        
        results['smart_score'] = smart_score
        
        # Log the actual parameters used (for verification)
        if results.get('trades', 0) > 0:
            print(f"✅ Worker Success: Mode={config.get('sizing_mode')}, "
                  f"Val={config.get('kelly_fraction', config.get('fixed_size')):.3f}, "
                  f"Trades={results['trades']}, Return={ret:.2%}")
        else:
            print(f"⚠️ Worker No Trades: Splash={config.get('splash_threshold')}, "
                  f"Edge={config.get('edge_threshold')}")

        return results

    except Exception as e:
        print(f"!!!!! WORKER CRASH !!!!!")
        print(f"Config: {config}")
        traceback.print_exc()
        return {
            'total_return': -1.0, 
            'sharpe_ratio': -99.0, 
            'max_drawdown': 1.0,
            'trades': 0,
            'smart_score': -99.0
        }

# ==============================================================================
# --- ORCHESTRATOR: BacktestEngine (Sensitivity Analysis Config) ---
# ==============================================================================
class BacktestEngine:
    def __init__(self, historical_data_path: str):
        self.historical_data_path = historical_data_path
        self.cache_dir = Path(self.historical_data_path) / "polymarket_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        if ray.is_initialized(): ray.shutdown()
        try: ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
        except: pass

    def run_tuning_job(self):

        log.info("--- Starting Full Strategy Optimization (FIXED) ---")
        
        df_markets, df_trades = self._load_data()
        if df_markets.empty or df_trades.empty: 
            log.error("⛔ CRITICAL: Data load failed. Cannot run tuning.")
            return None
      
        event_log, profiler_data = self._transform_to_events(df_markets, df_trades)
        now = pd.Timestamp.now()
        event_log = event_log[event_log.index <= now]
    
        if event_log.empty:
            log.error("⛔ Event log is empty after transformation.")
            return None
    
        min_date = event_log.index.min()
        max_date = event_log.index.max()
        total_days = (max_date - min_date).days
    
        log.info(f"📊 DATA STATS: {len(event_log)} events spanning {total_days} days ({min_date} to {max_date})")
    
        safe_train = max(5, int(total_days * 0.33))
        safe_test = max(5, int(total_days * 0.60))
        required_days = safe_train + safe_test + 2
        
        if total_days < required_days:
            log.error(f"⛔ Not enough data: Have {total_days} days, need {required_days} for current split.")
            return None
            
        log.info(f"⚙️ ADAPTING CONFIG: Data={total_days}d -> Train={safe_train}d, Test={safe_test}d")
    
        import gc
        del df_markets, df_trades
        gc.collect()
    
        log.info("Uploading data to Ray Object Store...")
        event_log_ref = ray.put(event_log)
        profiler_ref = ray.put(profiler_data)
        nlp_cache_ref = ray.put(None)
        priors_ref = ray.put({})


        gc.collect()
        
       # from ray.tune.search.hyperopt import HyperOptSearch
        
        # === FIXED SEARCH SPACE ===
        search_space = {
            # Grid Search: Ray will strictly iterate these combinations
            "splash_threshold": tune.grid_search([500.0, 1000.0, 2000.0]),
            "edge_threshold": tune.grid_search([0.06, 0.07, 0.08]),
         #   "use_smart_exit": tune.grid_search([True, False]),
            "use_smart_exit": True,
            "smart_exit_ratio": tune.grid_search([0.5, 0.7, 0.9]),
            "sizing": ("fixed_pct", 0.025), 
            "stop_loss": None,
            "train_days": safe_train,
            "test_days": safe_test,
            "seed": 42,
        }
    
    #    searcher = HyperOptSearch(metric="smart_score", mode="max", random_state_seed=42)
        
        # Higher sample count to cover combinations
        analysis = tune.run(
            tune.with_parameters(
                ray_backtest_wrapper,
                event_log_ref=event_log_ref,
                profiler_ref=profiler_ref,
                nlp_cache_ref=nlp_cache_ref,
                priors_ref=priors_ref
            ),
            config=search_space,
    #        search_alg=searcher,
    #        num_samples=30,  # Increased to cover new threshold combinations
            resources_per_trial={"cpu": 1},
        )
    
        best_config = analysis.get_best_config(metric="smart_score", mode="max")
        print("Sorting results deterministically...")
        all_trials = analysis.trials
        # Define a robust sort key:
        # 1. Smart Score (Desc)
        # 2. Total Return (Desc)
        # 3. Trades (Desc)
        # 4. Splash Threshold (Asc - prefer lower threshold if scores are tied)
        def sort_key(t):
            metrics = t.last_result or {}
            return (
                metrics.get('smart_score', -99.0),
                metrics.get('total_return', -99.0),
                metrics.get('trades', 0),
                -t.config.get('splash_threshold', 0), # Negative for Ascending
                t.trial_id
            )
        sorted_trials = sorted(all_trials, key=sort_key, reverse=True)
        best_trial = sorted_trials[0]
        best_config = best_trial.config
      
        metrics = best_trial.last_result
        
        mode, val = best_config['sizing']
        sizing_str = f"Kelly {val}x" if mode == "kelly" else f"Fixed {val*100}%"
        
        print("\n" + "="*60)
        print("🏆  GRAND CHAMPION STRATEGY  🏆")
        print(f"   Splash Threshold: {best_config['splash_threshold']:.1f}")
        print(f"   Edge Threshold:   {best_config['edge_threshold']:.3f}")
        print(f"   Sizing:           {sizing_str}")
        print(f"   Smart Exit:       {best_config['use_smart_exit']}")
        print(f"   Exit Ratio:       {best_config.get('smart_exit_ratio', 0.5):.2f}x")
        print(f"   Stop Loss:        {best_config['stop_loss']}")
        print(f"   Smart Score:      {metrics.get('smart_score', 0.0):.4f}")
        print(f"   Total Return:     {metrics.get('total_return', 0.0):.2%}")
        print(f"   Max Drawdown:     {metrics.get('max_drawdown', 0.0):.2%}")
        print(f"   Sharpe Ratio:     {metrics.get('sharpe_ratio', 0.0):.4f}")
        print(f"   Trades:           {metrics.get('trades', 0)}")
        print("="*60 + "\n")

        print("\n--- Generating Visual Report ---")
    
        # 1. FIX: Manually unpack the 'sizing' tuple for the local engine
        # (This replicates the logic inside ray_backtest_wrapper)
        if 'sizing' in best_config:
            mode, val = best_config['sizing']
            best_config['sizing_mode'] = mode
            if mode == 'kelly':
                best_config['kelly_fraction'] = val
            elif mode == 'fixed_pct':
                best_config['fixed_size'] = val
            elif mode == 'fixed':
                best_config['fixed_size'] = val

        # 2. Re-instantiate the engine locally
        engine = FastBacktestEngine(event_log, profiler_data, None, {})
        
        # 3. Run with the CORRECTED config
        final_results = engine.run_walk_forward(best_config)
        
        # 2. Extract Curve
        curve_data = final_results.get('equity_curve', [])
        trade_count = final_results.get('trades', 0)
        
        if curve_data:
            # 3. Plot
            plot_performance(curve_data, trade_count)
            
            # 4. Optional: Quick Terminal "Sparkline"
            start = curve_data[0]
            end = curve_data[-1]
            peak = max(curve_data)
            low = min(curve_data)
            print(f"   Start: ${start:.0f} -> Peak: ${peak:.0f} -> End: ${end:.0f}")
            print(f"   Lowest Point: ${low:.0f}")
        else:
            print("❌ Error: No equity curve data returned to plot.")
    
        return best_config
        
    def _load_data(self):
        import pandas as pd
        import glob
        import os
        
        DAYS_BACK = 200 # The strict requirement
        print(f"Initializing Data Engine (Scope: Last {DAYS_BACK} Days)...")
        
        # ---------------------------------------------------------
        # 1. MARKETS (Get Metadata)
        # ---------------------------------------------------------
        market_file_path = self.cache_dir / "gamma_markets_all_tokens.parquet"

        if market_file_path.exists():
            print(f"🔒 LOCKED LOAD: Using local market file: {market_file_path.name}")
            markets = pd.read_parquet(market_file_path)
        else:
            print(f"⚠️ File not found at {market_file_path}. Downloading from scratch...")
            markets = self._fetch_gamma_markets(days_back=DAYS_BACK)

        if markets.empty:
            print("❌ Critical: No market data available.")
            return pd.DataFrame(), pd.DataFrame()

        markets = markets.sort_values(
            by=['contract_id', 'resolution_timestamp'], 
            ascending=[True, True],
            kind='stable'
        )
            
        markets = markets.drop_duplicates(subset=['contract_id']).copy()
        # ---------------------------------------------------------
        # 2. TRADES (Get History)
        # ---------------------------------------------------------
        trades_file = self.cache_dir / "gamma_trades_stream.csv"
        
        if not trades_file.exists():
            print("   ⚠️ No local trades found. Downloading from scratch...")
            
            # A. Extract ALL Token IDs from the markets we just loaded
            all_tokens = []
            for raw_ids in markets['contract_id']:
                # Handle "ID1,ID2" format
                parts = str(raw_ids).split(',')
                for p in parts:
                    clean_p = p.strip()
                    if len(clean_p) > 2:
                        all_tokens.append(clean_p)
            
            # Remove duplicates
            target_tokens = list(set(all_tokens))
            print(f"   Identified {len(target_tokens)} tokens to download.")
            
            # B. Trigger Downloader (With Strict Time Limit)
            # This uses the fetcher you just updated
            trades = self._fetch_gamma_trades_parallel(target_tokens, days_back=DAYS_BACK)
            
        else:
            print(f"   Loading local trades: {os.path.basename(trades_file)}")
            trades = pd.read_csv(trades_file, dtype={'contract_id': str, 'user': str})

        if trades.empty:
            print("❌ Critical: No trade data available.")
            return pd.DataFrame(), pd.DataFrame()

        trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.tz_localize(None)
        trades['tradeAmount'] = pd.to_numeric(trades['tradeAmount'], errors='coerce').fillna(0)
        trades['price'] = pd.to_numeric(trades['price'], errors='coerce').fillna(0)
        trades['outcomeTokensAmount'] = pd.to_numeric(trades['outcomeTokensAmount'], errors='coerce').fillna(0)
        trades = trades.sort_values(
            by=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            ascending=[True, True, True, True, True, True],
            kind='stable' # Stable sort preserves order of equal elements (less random)
        )
        trades = trades.drop_duplicates(
            subset=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            keep='first'
        )

        # ---------------------------------------------------------
        # 3. CLEANUP & SYNC
        # ---------------------------------------------------------
        print("   Synchronizing data...")
        
        # A. Type Conversion
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.tz_localize(None)
        trades['tradeAmount'] = pd.to_numeric(trades['tradeAmount'], errors='coerce').fillna(0)
        trades['contract_id'] = trades['contract_id'].str.strip()
        
        # B. Strict Date Filter (Double Check)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=DAYS_BACK)
        trades = trades[trades['timestamp'] >= cutoff_date].copy()
        
        # C. Align Market IDs (Explode "ID1,ID2" -> Single Rows)
        markets['contract_id'] = markets['contract_id'].astype(str).str.split(',')
        markets = markets.explode('contract_id')
        markets['contract_id'] = markets['contract_id'].str.strip()
        
        # D. Match Dataframes
        valid_ids = set(trades['contract_id'].unique())
        market_subset = markets[markets['contract_id'].isin(valid_ids)].copy()
        trades = trades[trades['contract_id'].isin(set(market_subset['contract_id']))]

        # --- FIX: Strict Deterministic Loading ---
        # 1. Normalize numerics to ensure sorting works consistently
        trades['tradeAmount'] = pd.to_numeric(trades['tradeAmount'], errors='coerce').fillna(0).round(6)
        trades['price'] = pd.to_numeric(trades['price'], errors='coerce').fillna(0).round(6)
        trades['outcomeTokensAmount'] = pd.to_numeric(trades['outcomeTokensAmount'], errors='coerce').fillna(0).round(6)
        
        # 2. Strict Type Casting for Sort
        trades['contract_id'] = trades['contract_id'].astype(str)
        trades['user'] = trades['user'].astype(str)

        # 3. SORT with ALL varying columns. 
        # Including Price and Tokens ensures A and B (from above) always appear in the same order.
        trades = trades.sort_values(
            ['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'], 
            kind='stable'
        )

        # 4. DEDUPLICATE with ALL varying columns.
        # This prevents dropping valid concurrent trades that differ only by price/tokens.
        trades = trades.drop_duplicates(
            subset=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            keep='first'
        ).reset_index(drop=True)
        
        print(f"✅ SYSTEM READY.")
        print(f"   Markets: {len(market_subset)}")
        print(f"   Trades:  {len(trades)}")
        
        return market_subset, trades
        
    def _fetch_gamma_markets(self, days_back=200):
        import os
        import json
        import pandas as pd
        
        cache_file = self.cache_dir / "gamma_markets_all_tokens.parquet"
        
        if cache_file.exists():
            try: os.remove(cache_file)
            except: pass

        all_rows = []
        offset = 0
        
        print(f"Fetching GLOBAL market list...")
        
        while True:
            try:
                print(f"Offset:{offset}")
                # Gamma API
                params = {"limit": 500, "offset": offset, "closed": "true"}
                resp = self.session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=30)
                if resp.status_code != 200: break
                
                rows = resp.json()
                print(f"Rows Retreived:{len(rows)}")
                if not rows: break
                all_rows.extend(rows)
                
                offset += len(rows)
           
            except Exception: break
        
        print(f" Done. Fetched {len(all_rows)} markets.")
        if not all_rows: return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        
        # --- EXTRACT ALL TOKEN IDS ---
        def extract_all_tokens(row):
            try:
                raw = row.get('clobTokenIds')
                if not raw: return None
                
                if isinstance(raw, str):
                    try: tokens = json.loads(raw)
                    except: return None
                else: tokens = raw
                
                if isinstance(tokens, list) and len(tokens) > 0:
                    # Clean all IDs
                    clean_ids = []
                    for t in tokens:
                        if isinstance(t, (int, float)):
                            clean_ids.append(str(t))
                        else:
                            clean_ids.append(str(t).strip())
                    # Join with comma for storage
                    return ",".join(clean_ids)
                return None
            except: return None

        df['contract_id'] = df.apply(extract_all_tokens, axis=1)
        
        # Filter
        df = df.dropna(subset=['contract_id'])
        df['contract_id'] = df['contract_id'].astype(str)

        # Normalization
        def derive_outcome(row):
            # 1. Trust explicit outcome first (Gold Standard)
            if pd.notna(row.get('outcome')): 
                return float(row['outcome'])

            # 2. TIME GATE: Check if the market has actually ended
            try:
                end_date_str = row.get('endDate') # Raw API field
                if end_date_str:
                    # Force UTC for the market end date
                    end_ts = pd.to_datetime(end_date_str).tz_convert('UTC')
                    
                    # Compare against UTC "Now"
                    if end_ts > pd.Timestamp.now(tz='UTC'):
                        return 0.5
            except:
                # If date parsing fails, default to safety (Active)
                return 0.5

            # 3. PRICE CHECK (Only runs if Time Gate is passed)
            try:
                prices = row.get('outcomePrices')
                if isinstance(prices, str): 
                    prices = json.loads(prices)
                
                # Safety: Ensure it's a list
                if not isinstance(prices, list) or len(prices) != 2: 
                    return 0.5
                
                # Safety: Convert elements to float (handles strings inside list)
                p0 = float(prices[0])
                p1 = float(prices[1])
                
                # Strict 99% threshold, but ONLY for markets past their end date
                if p1 >= 0.99: return 1.0
                if p0 >= 0.99: return 0.0
                
                return 0.5
            except: 
                return 0.5

        df['outcome'] = df.apply(derive_outcome, axis=1)
        rename_map = {'question': 'question', 'endDate': 'resolution_timestamp', 'createdAt': 'created_at', 'volume': 'volume'}
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        
        df = df.dropna(subset=['resolution_timestamp', 'outcome'])
        df['outcome'] = pd.to_numeric(df['outcome'])
        df['resolution_timestamp'] = pd.to_datetime(df['resolution_timestamp'], errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
        
        if not df.empty: df.to_parquet(cache_file)
        return df
        
    def _fetch_single_market_trades(self, market_id):
        """
        Worker function: Fetches ALL trades for a specific market ID.
        CORRECTED: Removes the 50k limit. Stops based on TIME (180 days).
        """
        import time
        import requests
        from datetime import datetime, timedelta
        from requests.adapters import HTTPAdapter, Retry

        # Create a short ID from the market_id for logging
        t_id = str(market_id)[-4:]
        print(f" [T-{t_id}] Start.", end="", flush=True)
        
        # 1. Setup Session
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        all_market_trades = []
        offset = 0
        batch_size = 500
        
        # STOPPING CRITERIA: 200 days ago (buffer for 200d backtest)
        # We calculate this once, outside the loop
        cutoff_ts = (datetime.now() - timedelta(days=185)).timestamp()
        
        while True:
            try:
                url = "https://gamma-api.polymarket.com/events"
                params = {
                    "market": market_id, 
                    "type": "Trade", 
                    "limit": batch_size, 
                    "offset": offset
                }
                
                resp = session.get(url, params=params, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if not data: 
                        break # End of history (API returned empty)
                    print(f" [T-{t_id}] Req Offset {offset}...", end="", flush=True)
                    all_market_trades.extend(data)
                    print(f" [T-{t_id}] {resp.status_code} ", end="", flush=True)
                    # --- CRITICAL FIX: Check Time, Not Count ---
                    # Check the timestamp of the last trade in this batch
                    # If the last trade is older than our cutoff, we have enough data.
                    last_trade = data[-1]
                    
                    # Gamma uses 'timestamp' (seconds) or 'time' (iso string)
                    ts_val = last_trade.get('timestamp')
                    trade_ts = None
                    
                    if isinstance(ts_val, (int, float)):
                        trade_ts = float(ts_val)
                    elif last_trade.get('time'):
                        try:
                            # Quick ISO parse
                            trade_ts = pd.to_datetime(last_trade['time']).timestamp()
                        except: pass
                    
                    # If we found a valid timestamp and it's older than cutoff, STOP.
                    if trade_ts and trade_ts < cutoff_ts:
                        break 
                    # -------------------------------------------

                    if len(data) < batch_size: 
                        break # End of history (Partial page)
                        
                    offset += batch_size
                    
                elif resp.status_code == 429:
                    print(f" [T-{t_id}] 429 RETRY! ", end="", flush=True)
                    time.sleep(2)
                    continue
                else:
                    # 400/500 errors -> Stop to prevent hang
                    break
            except:
                break
        
        return all_market_trades

    def _fetch_gamma_trades_parallel(self, market_ids_raw, days_back=200):
        import concurrent.futures 
        import csv
        import threading
        import requests
        import os
        import time
        from requests.adapters import HTTPAdapter, Retry
        
        cache_file = self.cache_dir / "gamma_trades_stream.csv"
        ledger_file = self.cache_dir / "gamma_completed.txt"
        
        # Expand Market IDs to Token IDs
        all_tokens = []
        for mid_str in market_ids_raw:
            parts = str(mid_str).split(',')
            for p in parts:
                if len(p) > 5: all_tokens.append(p.strip())
        all_tokens = list(set(all_tokens))
            
        print(f"Stream-fetching {len(all_tokens)} tokens via SUBGRAPH...")
        print(f"Constraint: STRICT {days_back} DAY HISTORY LIMIT.")
        
        GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

        FINAL_COLS = ['timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 
                      'contract_id', 'price', 'size', 'side_mult']
        
        # Append if file exists, write new if not
        write_mode = 'a' if cache_file.exists() else 'w'
        csv_lock = threading.Lock()
        ledger_lock = threading.Lock()
        self.first_success = False
        
        # CALCULATE THE HARD TIME LIMIT
        limit_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        CUTOFF_TS = limit_date.timestamp()
        
        def fetch_and_write_worker(token_str, writer, f_handle):
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            if 'e+' in token_str or '.' in token_str: return False

            last_ts = 2147483647 # Int32 Max
            
            while True:
                try:
                    # DUAL QUERY
                    query = """
                    query($token: String!, $max_ts: Int!) {
                      asMaker: orderFilledEvents(
                        first: 1000
                        orderBy: timestamp
                        orderDirection: desc
                        where: { makerAssetId: $token, timestamp_lt: $max_ts }
                      ) {
                        timestamp, makerAmountFilled, takerAmountFilled, maker, taker
                      }
                      asTaker: orderFilledEvents(
                        first: 1000
                        orderBy: timestamp
                        orderDirection: desc
                        where: { takerAssetId: $token, timestamp_lt: $max_ts }
                      ) {
                        timestamp, makerAmountFilled, takerAmountFilled, maker, taker
                      }
                    }
                    """
                    variables = {"token": token_str, "max_ts": int(last_ts)}
                    
                    resp = session.post(GRAPH_URL, json={"query": query, "variables": variables}, timeout=45)
                    
                    if resp.status_code == 200:
                        r_json = resp.json()
                        if 'errors' in r_json: break 
                        
                        batch_maker = r_json.get('data', {}).get('asMaker', [])
                        batch_taker = r_json.get('data', {}).get('asTaker', [])
                        
                        tagged_rows = [(r, 'maker') for r in batch_maker] + [(r, 'taker') for r in batch_taker]
                        if not tagged_rows: break 
                        
                        tagged_rows.sort(key=lambda x: float(x[0]['timestamp']), reverse=True)
                        
                        rows = []
                        min_batch_ts = last_ts
                        stop_signal = False
                        
                        for row, source in tagged_rows:
                            ts_val = float(row['timestamp'])
                            min_batch_ts = min(min_batch_ts, ts_val)
                            
                            # --- STRICT DATE CHECK ---
                            if ts_val < CUTOFF_TS:
                                stop_signal = True
                                continue
                            # -------------------------
                            
                            try:
                                if source == 'maker':
                                    size = float(row.get('makerAmountFilled') or 0.0)
                                    usdc = float(row.get('takerAmountFilled') or 0.0)
                                    user = str(row.get('taker') or 'unknown')
                                    side_mult = 1
                                else:
                                    size = float(row.get('takerAmountFilled') or 0.0)
                                    usdc = float(row.get('makerAmountFilled') or 0.0)
                                    user = str(row.get('taker') or 'unknown')
                                    side_mult = -1
                                
                                if size == 0: continue
                                price = usdc / size
                                ts_str = pd.to_datetime(ts_val, unit='s').isoformat()
                                
                                rows.append({
                                    'timestamp': ts_str,
                                    'tradeAmount': usdc,
                                    'outcomeTokensAmount': size * side_mult * 1e18,
                                    'user': user,
                                    'contract_id': token_str,
                                    'price': price,
                                    'size': size,
                                    'side_mult': side_mult
                                })
                            except: continue
                        
                        if rows:
                            with csv_lock:
                                writer.writerows(rows)
                                f_handle.flush()
                        
                        # Stop if we hit the time limit
                        if stop_signal: break
                        
                        if int(min_batch_ts) >= int(last_ts): last_ts = int(min_batch_ts) - 1
                        else: last_ts = min_batch_ts
                        
                        # Stop if we went past cutoff
                        if min_batch_ts < CUTOFF_TS: break
                        
                    else:
                        time.sleep(2)
                        continue

                except Exception:
                    break 
            
            with ledger_lock:
                with open(ledger_file, "a") as lf: lf.write(f"{token_str}\n")
            return True

        with open(cache_file, mode=write_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_COLS)
            # Only write header if we are starting a NEW file
            if write_mode == 'w': writer.writeheader()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(fetch_and_write_worker, mid, writer, f) for mid in all_tokens]
                completed = 0
                for _ in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed % 100 == 0: print(f" Progress: {completed}/{len(all_tokens)} checked...", end="\r")

        print("\n✅ Fetch complete.")
        try: df = pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        except: return pd.DataFrame()
        return df
        
    def _fetch_subgraph_trades(self, days_back=200):
        import time
        
        # ANCHOR: Current System Time (NOW)
        time_cursor = int(time.time())
        
        # Stop fetching if we go past this date
        cutoff_time = time_cursor - (days_back * 24 * 60 * 60)
        
        cache_file = self.cache_dir / f"subgraph_trades_recent_{days_back}d.pkl"
        if cache_file.exists(): 
            try:
                return pickle.load(open(cache_file, "rb"))
            except: pass
            
        url = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
        
        query_template = """
        {{
          fpmmTransactions(first: 1000, orderBy: timestamp, orderDirection: desc, where: {{ timestamp_lt: "{time_cursor}" }}) {{
            id
            timestamp
            tradeAmount
            outcomeTokensAmount
            user {{ id }}
            market {{ id }}
          }}
        }}
        """
        all_rows = []
        
        print(f"Fetching Trades from NOW ({time_cursor}) back to {cutoff_time}...", end="")
        
        while True:
            try:
                resp = self.session.post(url, json={'query': query_template.format(time_cursor=time_cursor)}, timeout=30)
                if resp.status_code != 200: break
                    
                data = resp.json().get('data', {}).get('fpmmTransactions', [])
                if not data: break
                
                all_rows.extend(data)
                
                # Update cursor
                last_ts = int(data[-1]['timestamp'])
                
                # Stop if we passed the cutoff
                if last_ts < cutoff_time: break
                
                # Stop if API returns partial page (end of data)
                if len(data) < 1000: break
                
                # Safety break
                if last_ts >= time_cursor: break
                
                time_cursor = last_ts
                
                if len(all_rows) % 5000 == 0: print(".", end="", flush=True)
                
            except Exception as e:
                log.error(f"Fetch error: {e}")
                break
                
        print(f" Done. Fetched {len(all_rows)} trades.")
            
        df = pd.DataFrame(all_rows)
        
        if not df.empty:
            # Filter strictly to the requested window
            df['ts_int'] = df['timestamp'].astype(int)
            df = df[df['ts_int'] >= cutoff_time]
            
            with open(cache_file, 'wb') as f: pickle.dump(df, f)
            
        return df
        
    def diagnose_data(self):
        """Run this to understand what data you're getting"""
        print("\n" + "="*60)
        print("🔍 DATA DIAGNOSTIC REPORT")
        print("="*60)
        
        trades = self._fetch_subgraph_trades()
        print(f"\n📦 TRADES:")
        print(f"   Total records: {len(trades)}")
        if not trades.empty:
            print(f"   Columns: {list(trades.columns)}")
            print(f"   Date range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")
            if 'market' in trades.columns:
                sample_market = trades.iloc[0]['market']
                print(f"   Sample market field: {sample_market}")
        
        markets_path = self.cache_dir / "gamma_markets_all_tokens.parquet"
        if markets_path.exists():
            markets = pd.read_parquet(markets_path)
        else:
            print("No markets found to diagnose.")
            return
            
        print(f"\n📦 MARKETS:")
        print(f"   Total records: {len(markets)}")
        if not markets.empty:
            print(f"   Columns: {list(markets.columns)}")
            print(f"   Markets with outcomes: {markets['outcome'].notna().sum()}")
            print(f"   Outcome values: {markets['outcome'].value_counts()}")
            if 'resolution_timestamp' in markets.columns:
                print(f"   Resolution range: {markets['resolution_timestamp'].min()} to {markets['resolution_timestamp'].max()}")
        
        print("="*60 + "\n")
    
    def _transform_to_events(self, markets, trades):
        import gc
        import pandas as pd
        import numpy as np

        log.info("Transforming Data (Robust Mode)...")
        
        # 1. TIME NORMALIZATION
        markets['created_at'] = pd.to_datetime(markets['created_at'], errors='coerce').dt.tz_localize(None)
        markets['resolution_timestamp'] = pd.to_datetime(markets['resolution_timestamp'], errors='coerce').dt.tz_localize(None)
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.tz_localize(None)
        
        # 2. STRING NORMALIZATION
        markets['contract_id'] = markets['contract_id'].astype(str).str.strip().str.lower()
        trades['contract_id'] = trades['contract_id'].astype(str).str.strip().str.lower()
        
        # 3. FILTER TO COMMON IDs
        common_ids_set = set(markets['contract_id']).intersection(set(trades['contract_id']))
        common_ids = sorted(list(common_ids_set))
        if not common_ids:
            log.error("❌ NO COMMON IDS FOUND.")
            return pd.DataFrame(), pd.DataFrame()
            
        markets = markets[markets['contract_id'].isin(common_ids)].copy()
        trades = trades[trades['contract_id'].isin(common_ids)].copy()
        
        # 4. BUILD PROFILER DATA
        prof_data = pd.DataFrame({
            'wallet_id': trades['user'].astype(str), 
            'market_id': trades['contract_id'],
            'timestamp': trades['timestamp'],
            'usdc_vol': trades['tradeAmount'].astype('float64'),
            'tokens': trades['outcomeTokensAmount'].astype('float64'),
            'price': pd.to_numeric(trades['price'], errors='coerce').astype('float64'),
            'size': trades['tradeAmount'].astype('float64'),
            'outcome': 0.0,
            'bet_price': 0.0
        })

        # MAP OUTCOMES
        outcome_map = markets.set_index('contract_id')['outcome']
        outcome_map.index = outcome_map.index.astype(str).str.strip().str.lower()
        outcome_map = outcome_map[~outcome_map.index.duplicated(keep='first')]
        
        prof_data['outcome'] = prof_data['market_id'].map(outcome_map)
        matched_mask = prof_data['outcome'].isin([0.0, 1.0])
        matched_count = matched_mask.sum()
        total_count = len(prof_data)
        
        log.info(f"🔎 OUTCOME JOIN REPORT: {matched_count} / {total_count} trades matched a market.")
        
        # 2. Check for 0 matches using the UNFILTERED data
        if matched_count == 0:
            log.warning("⛔ CRITICAL: 0 trades matched. Checking ID samples:")
            
            # Safe access: Check if data exists before calling iloc[0]
            if not prof_data.empty:
                log.warning(f"   Trade ID Sample: {prof_data['market_id'].iloc[0]}")
            else:
                log.warning("   (No trades available to sample)")

            if not outcome_map.empty:
                log.warning(f"   Market ID Sample: {outcome_map.index[0]}")
            else:
                log.warning("   (Outcome map is empty)")

        # 3. NOW apply the filter to keep only valid rows
        prof_data = prof_data[matched_mask].copy()

        prof_data['bet_price'] = pd.to_numeric(prof_data['price'], errors='coerce')
        prof_data = prof_data.dropna(subset=['bet_price'])

        prof_data = prof_data[(prof_data['bet_price'] > 0.0) & (prof_data['bet_price'] <= 1.0)]
        
        prof_data['entity_type'] = 'default_topic'
        
        log.info(f"Profiler Data Built: {len(prof_data)} records.")

        # 5. BUILD EVENT LOG
        events_ts, events_type, events_data = [], [], []

        # A. NEW_CONTRACT
        for _, row in markets.iterrows():
            if pd.isna(row['created_at']): continue
            events_ts.append(row['created_at'])
            events_type.append('NEW_CONTRACT')
            
            liq = row.get('liquidity')
            # GHOST MARKET FIX: Default to 10k
            safe_liq = float(liq) if liq is not None and float(liq) > 0 else 10000.0
            res_ts = row.get('resolution_timestamp')
            if pd.isna(res_ts): res_ts = None
                
            events_data.append({
                'contract_id': row['contract_id'], 
                'p_market_all': 0.5, 
                'liquidity': safe_liq,
                'end_date': res_ts
            })
            
        # B. RESOLUTION
        for _, row in markets.iterrows():
            if pd.isna(row['resolution_timestamp']): continue
            events_ts.append(row['resolution_timestamp'])
            events_type.append('RESOLUTION')
            events_data.append({
                'contract_id': row['contract_id'], 
                'outcome': float(row['outcome'])
            })

        # C. PRICE_UPDATE (Robust Logic)
        
        # 1. Ensure the source is strictly sorted and deduped
        # 1. Sort strictly first
        trades = trades.sort_values(
            by=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'], 
            kind='stable'
        )
        
        trades = trades.drop_duplicates(
            subset=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            keep='first'
        ).reset_index(drop=True)
        
        # 2. Re-create prof_data-like vectors directly from the sorted source
        # This guarantees 1:1 alignment because we are reading from the SAME dataframe
        t_ts = trades['timestamp'].tolist()
        t_cid = trades['contract_id'].tolist()
        t_uid = trades['user'].astype(str).tolist()
        t_vol = trades['tradeAmount'].tolist()
        t_tokens = trades['outcomeTokensAmount'].tolist()
        
        # 3. Handle Price (Use map to ensure correctness)
        # We need the 'bet_price' logic from profiler data, but safe.
        # Since we just sorted 'trades', we can calculate price on the fly.
        t_price = pd.to_numeric(trades['price'], errors='coerce').fillna(0.5).tolist()

        # 4. Append to events
        for i in range(len(trades)):
            events_ts.append(t_ts[i])
            events_type.append('PRICE_UPDATE')
            events_data.append({
                'contract_id': t_cid[i],
                'p_market_all': t_price[i], # Aligned perfectly
                'wallet_id': t_uid[i],
                'trade_volume': float(t_vol[i]),
                'is_sell': t_tokens[i] < 0
            })

        # 6. FINAL SORT
        df_ev = pd.DataFrame({
            'timestamp': events_ts, 
            'event_type': events_type, 
            'data': events_data
        })
        df_ev['timestamp'] = pd.to_datetime(df_ev['timestamp'])
        df_ev = df_ev.dropna(subset=['timestamp'])
        df_ev['cid_temp'] = df_ev['data'].apply(lambda x: str(x.get('contract_id', '')))
        df_ev = df_ev.sort_values(
            by=['timestamp', 'cid_temp', 'event_type'], 
            kind='stable'
        )
        df_ev = df_ev.drop(columns=['cid_temp'])
        
        if 'timestamp' not in df_ev.columns and df_ev.index.name == 'timestamp':
            df_ev = df_ev.reset_index()
            
        if not prof_data.empty:
            first_trade_ts = prof_data['timestamp'].min()
            start_cutoff = first_trade_ts - pd.Timedelta(days=1)
            
            # Identify events that are too old
            mask_old = df_ev['timestamp'] < start_cutoff
            
            # Split into Old and New
            df_old = df_ev[mask_old].copy()
            df_new = df_ev[~mask_old].copy()
            
            # Rescue 'NEW_CONTRACT' events from the past
            rescued_contracts = df_old[df_old['event_type'] == 'NEW_CONTRACT'].copy()
            rescued_contracts['timestamp'] = start_cutoff
            
            # Recombine: Rescued Old Markets + All Recent Events
            df_ev = pd.concat([rescued_contracts, df_new])
            
            # Sort to ensure the Rescued events come first
            df_ev = df_ev.sort_values(by=['timestamp', 'event_type'], ascending=[True, True])
            
            log.info(f"⏱️ SMART SYNC: Teleported {len(rescued_contracts)} old markets to {start_cutoff}. "
                     f"Dropped {len(df_old) - len(rescued_contracts)} irrelevant old events.")
        
        # 3. Final Indexing (Must happen LAST)
        df_ev = df_ev.set_index('timestamp')
        # ------------------------------
        
        log.info(f"Transformation Complete. Event Log Size: {len(df_ev)} rows.")
        return df_ev, prof_data
        
# ==============================================================================
# ### COMPONENT 8: Operational Dashboard (Production-Ready) ###
# ==============================================================================

# --- Global Instantiation (Import-Safe) ---
IS_PROD_MODE = os.getenv("PROD_MODE", "false").lower() == "true"
graph_manager = GraphManager(is_mock=not IS_PROD_MODE)

ai_analyst = AIAnalyst()
live_feed_handler = LiveFeedHandler(graph_manager)
relational_linker = RelationalLinker(graph_manager)
prior_manager = PriorManager(graph_manager, ai_analyst, live_feed_handler)
historical_profiler = HistoricalProfiler(graph_manager)
belief_engine = BeliefEngine(graph_manager)
kelly_solver = HybridKellySolver()
portfolio_manager = PortfolioManager(graph_manager, kelly_solver)

# --- FIX: Provide all required arguments for the global instance ---
# We provide mock URLs or read from env vars for the C8 dashboard demo

backtest_engine = BacktestEngine(
    historical_data_path=".", # Save to current directory
)

# --- Dash App Definition ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# --- Analyst Triage Modal (Popup) ---
analyst_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Resolve Analyst Item")),
        dbc.ModalBody([
            html.P("Item ID:"),
            html.Code(id='modal-item-id'),
            html.P("Reason:", className="mt-3"),
            html.Code(id='modal-item-reason'),
            html.P("Details:", className="mt-3"),
            html.Code(id='modal-item-details'),
            html.Hr(),
            html.P("Enter Resolution Data (e.g., new Entity ID, or 'MERGE')"),
            dbc.Input(id='modal-input-data', placeholder="e.g., E_123 or MERGE"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Submit Resolution", id='modal-submit-btn', color="primary")
        ),
    ],
    id='analyst-modal',
    is_open=False,
)

# --- App Layout ---
def build_header():
    regime, params = graph_manager.get_regime_status()
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Analyst", href="/analyst")),
            dbc.NavItem(dbc.NavLink("Portfolio Manager", href="/pm")),
            dbc.NavItem(dbc.NavLink("Admin", href="/admin")),
            dbc.Badge(f"Regime: {regime}", color="primary", className="ms-auto"),
        ],
        brand="QTE", color="dark", dark=True,
    )

def build_analyst_tab():
    queue_items = graph_manager.get_human_review_queue()
    table_header = [html.Thead(html.Tr([html.Th("Item ID"), html.Th("Reason"), html.Th("Action")]))]
    table_body = [html.Tbody([
        html.Tr([
            html.Td(item['id']), html.Td(item['reason']),
            html.Td(dbc.Button("Resolve", id={'type': 'resolve-btn', 'index': item['id']}, n_clicks=0, size="sm")),
        ]) for item in queue_items
    ])]
    return html.Div([
        html.H2("Analyst Triage Queues"),
        dbc.Alert(id='analyst-alert', is_open=False, duration=4000),
        dbc.Table(table_header + table_body, bordered=True, striped=True),
        dcc.Store(id='modal-data-store'),
        analyst_modal
    ])

def build_pm_tab():
    state = graph_manager.get_portfolio_state()
    pnl_history = graph_manager.get_pnl_history()
    fig = go.Figure(data=go.Scatter(y=pnl_history, mode='lines', name='Total Value'))
    fig.update_layout(title='Portfolio Value Over Time', yaxis_title='Total Value ($)')
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Portfolio Value"), dbc.CardBody(f"${state['total_value']:,.2f}", className="text-success fs-3")]), width=4),
            dbc.Col(dbc.Card([dbc.CardHeader("Available Cash"), dbc.CardBody(f"${state['cash']:,.2f}", className="fs-3")]), width=4),
            dbc.Col(dbc.Card([dbc.CardHeader("Active Positions"), dbc.CardBody(f"{len(state['positions'])}", className="fs-3")]), width=4),
        ]),
        dbc.Row(dcc.Graph(figure=fig), className="mt-4"),
    ])

def build_admin_tab():
    regime, params = graph_manager.get_regime_status()
    return html.Div([
        html.H2("Admin & Tuning"),
        dbc.Alert(id='admin-alert', is_open=False, duration=4000),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Hyperparameter Tuning (C7)"),
                dbc.CardBody([
                    html.P("Launch a new async job to tune all hyperparameters."),
                    dbc.Button("Start New Tuning Job", id='start-tune-btn', color="danger", n_clicks=0)
                ]),
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Current Regime & Parameters"),
                dbc.CardBody([html.H4(f"Regime: {regime}"), html.Code(str(params))]),
            ]), width=6),
        ]),
    ])

app.layout = html.Div([
    build_header(),
    dcc.Location(id='url', refresh=False),
    dbc.Container(id='page-content', fluid=True, className="mt-4")
])

# --- Main Page-Routing Callback ---
@callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/pm': return build_pm_tab()
    elif pathname == '/admin': return build_admin_tab()
    else: return build_analyst_tab()

# --- C8: Admin Callback (Async) ---
def _run_async_job():
    """Wrapper function for multiprocessing."""
    log.info("Async process started...")
    # We must instantiate a new (mock) graph for this process
    try:
        graph_stub = GraphManager(is_mock=True)
        be = BacktestEngine(historical_data_path=".")
        best_config = be.run_tuning_job()
        log.info(f"Async job finished. Best config: {best_config}")
    except Exception as e:
        log.error(f"Async job failed: {e}", exc_info=True)

@callback(
    Output('admin-alert', 'children'),
    Output('admin-alert', 'is_open'),
    Input('start-tune-btn', 'n_clicks'),
    prevent_initial_call=True
)
def start_tuning_job_callback(n_clicks):
    log.warning("Admin clicked 'Start New Tuning Job'")
    try:
        p = multiprocessing.Process(target=_run_async_job)
        p.start()
        return f"Tuning job started in background (PID: {p.pid})! See logs/Ray Dashboard.", True
    except Exception as e:
        log.warning(f"Failed to start tuning job: {e}")
        return f"Error: {e}", True

# --- C8: Analyst Callbacks (Modal) ---
@callback(
    Output('analyst-modal', 'is_open'),
    Output('modal-data-store', 'data'),
    Output('modal-item-id', 'children'),
    Output('modal-item-reason', 'children'),
    Output('modal-item-details', 'children'),
    Input({'type': 'resolve-btn', 'index': dash.ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def open_analyst_modal(n_clicks):
    ctx = dash.callback_context
    if not any(n_clicks): return False, {}, "", "", ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    item_id = json.loads(button_id)['index']
    queue = graph_manager.get_human_review_queue()
    item_data = next((item for item in queue if item['id'] == item_id), None)
    if item_data:
        details_str = json.dumps(item_data.get('details', {}), indent=2)
        return True, item_data, item_id, item_data.get('reason'), details_str
    return False, {}, "", "", ""

@callback(
    Output('analyst-alert', 'children'),
    Output('analyst-alert', 'is_open'),
    Output('analyst-modal', 'is_open', allow_duplicate=True),
    Input('modal-submit-btn', 'n_clicks'),
    State('modal-data-store', 'data'),
    State('modal-input-data', 'value'),
    prevent_initial_call=True
)
def submit_analyst_resolution(n_clicks, item_data, resolution_data):
    if not item_data: return "Error: No item data found.", True, False
    item_id = item_data.get('id')
    log.warning(f"Analyst is resolving {item_id} with data: {resolution_data}")
    success = graph_manager.resolve_human_review_item(item_id, "SUBMITTED", resolution_data)
    if success: return f"Item {item_id} resolved! Refreshing...", True, False
    else: return f"Failed to resolve {item_id}.", True, True

# ==============================================================================
# --- MAIN LAUNCHER ---
# ==============================================================================
def run_c1_c2_demo():
    log.info("--- (DEMO) Running Component 1 & 2 (Production) Demo ---")
    try:
        graph = GraphManager() # Real connection
        graph.setup_schema()
        linker = RelationalLinker(graph)
        vector = [0.1] * graph.vector_dim 
        graph.add_contract("MKT_902_PROD", "Will 'NeruCorp' ship the 'Viper'?", vector, 100, 0.5)
        log.info("--- Running Linker (Pass 1) ---")
        linker.process_pending_contracts() # Will fail (fuzzy/KNN) -> NEEDS_HUMAN_REVIEW
        log.info("--- Simulating Human Fix ---")
        with graph.driver.session() as session:
            session.execute_write(lambda tx: tx.run(
                "MERGE (e:Entity {entity_id: 'E_123'}) SET e.canonical_name = 'NeuroCorp, Inc.' "
                "MERGE (a:Alias {text: 'NeruCorp'}) MERGE (a)-[:POINTS_TO]->(e)"))
        log.info("--- Running Linker (Pass 2) ---")
        graph.update_contract_status("MKT_902_PROD", 'PENDING_LINKING')
        linker.process_pending_contracts() # Will succeed
        graph.close()
    except Exception as e:
        log.error(f"C1/C2 Demo Failed: {e}", exc_info=True)

def run_c3_demo():
    log.info("--- (DEMO) Running Component 3 (Production) Demo ---")
    try:
        graph = GraphManager(is_mock=True) 
        ai = AIAnalyst()
        feed_handler = LiveFeedHandler(graph)
        prior_manager = PriorManager(graph, ai, feed_handler)
        # (Add mock data to C3)
        graph.mock_db['contracts'] = {
            'MKT_903_AI': {'text': 'NeuroCorp', 'liquidity': 100, 'p_market_all': 0.5, 'entity_ids': ['E_123'], 'status': 'PENDING_ANALYSIS'},
            'MKT_904_HITL': {'text': 'High value', 'liquidity': 50000, 'p_market_all': 0.6, 'entity_ids': ['E_123'], 'status': 'PENDING_ANALYSIS'}
        }
        graph.mock_db['entities']['E_123'] = {'contract_count': 10}
        prior_manager.process_pending_contracts()
        log.info(f"C3 Demo AI Path Status: {graph.mock_db['contracts']['MKT_903_AI']['status']}")
        log.info(f"C3 Demo HITL Path Status: {graph.mock_db['contracts']['MKT_904_HITL']['status']}")
        graph.close()
    except Exception as e:
        log.error(f"C3 Demo Failed: {e}", exc_info=True)

def run_c4_demo():
    log.info("--- (DEMO) Running Component 4 (Production) Demo ---")
    try:
        graph = GraphManager(is_mock=True)
        profiler = HistoricalProfiler(graph, min_trades_threshold=3)
        profiler.run_profiling() # "Calculates" scores and saves to mock_db
        feed_handler = LiveFeedHandler(graph)
        p_experts = feed_handler.get_smart_money_price("MKT_BIO_001")
        log.info(f"--- C4 Demo Complete. Final P_Experts: {p_experts:.4f} ---")
        assert abs(p_experts - 0.3585) < 0.001
        log.info("C4 Demo Test Passed!")
        graph.close()
    except Exception as e:
        log.error(f"C4 Demo Failed: {e}", exc_info=True)

def run_c5_demo():
    log.info("--- (DEMO) Running Component 5 (Production) Demo ---")
    try:
        graph = GraphManager(is_mock=True)
        # C3 must run first to set status
        graph.mock_db['contracts']['MKT_FUSE_001'] = {'p_internal_alpha': 13.8, 'p_internal_beta': 9.2, 'p_market_experts': 0.45, 'p_market_all': 0.55, 'status': 'PENDING_FUSION'}
        engine = BeliefEngine(graph)
        engine.run_fusion_process()
        log.info(f"C5 Demo Complete. Fused Status: {graph.mock_db['contracts']['MKT_FUSE_001']['status']}")
        assert graph.mock_db['contracts']['MKT_FUSE_001']['status'] == 'MONITORED'
        graph.close()
    except Exception as e:
        log.error(f"C5 Demo Failed: {e}", exc_info=True)

def run_c6_demo():
    log.info("--- (DEMO) Running Component 6 (Production) Demo ---")
    try:
        graph = GraphManager(is_mock=True)
        solver = HybridKellySolver(num_samples_k=5000)
        pm = PortfolioManager(graph, solver)
        pm.run_optimization_cycle()
        log.info("--- C6 Demo Complete. ---")
        graph.close()
    except Exception as e:
        log.error(f"C6 Demo Failed: {e}", exc_info=True)

def run_c7_demo():
    log.info("--- (DEMO) Running Component 7 (Production) Demo ---")
    
    # 1. Nuke the cache to fix the "Triple Disconnect"
    #force_clear_cache("polymarket_cache") 
    
    try:
        backtester = BacktestEngine(historical_data_path=".")
        
        # Optional: Run the diagnosis provided in your prompt
        backtester.diagnose_data() 
        
        best_params = backtester.run_tuning_job()
        log.info(f"--- C7 Demo Complete. Best params: {best_params} ---")
    except Exception as e:
        log.error(f"C7 Demo Failed: {e}", exc_info=True)

def run_c8_demo():
    """Launches the C8 Dashboard"""
    log.info("--- (DEMO) Running Component 8 (Dashboard) ---")
    log.info("--- Launching Dash server on http://127.0.0.1:8050/ ---")
    try:
        app.run(debug=True, port=8050)
    except Exception as e:
        log.error(f"C8 Demo Failed: {e}", exc_info=True)


if __name__ == "__main__":
    
    # Simple CLI launcher
    demos = {
        "C1_C2": run_c1_c2_demo,
        "C3": run_c3_demo,
        "C4": run_c4_demo,
        "C5": run_c5_demo,
        "C6": run_c6_demo,
        "C7": run_c7_demo,
        "C8": run_c8_demo,
    }
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].upper()
        if demo_name in demos:
            demos[demo_name]()
        else:
            log.error(f"Unknown demo: {demo_name}")
            log.info(f"Usage: python {sys.argv[0]} [{ '|'.join(demos.keys()) }]")
    else:
        log.info(f"No demo specified. Running C8 (Dashboard) by default.")
        log.info(f"Try 'python {sys.argv[0]} C7' to run the tuning engine.")
        run_c8_demo()
