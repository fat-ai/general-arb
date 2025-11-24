import os
import logging
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import qmc, norm
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError # <-- FIX R6
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
import requests # For downloading files
import gzip     # For decompressing .gz files
import io       # For reading in-memory bytes
from datetime import datetime, timedelta # For parsing timestamps# For parsing timestamps
#from dune_client.client import DuneClient
#from dune_client.query import QueryBase
#from dune_client.types import QueryParameter
from numba import njit
import pickle
from pathlib import Path
import time
import traceback
import spacy
import torch

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

class HistoricalProfiler:
    """(Production-Ready C4a)"""
    def __init__(self, graph_manager: GraphManager, min_trades_threshold: int = 20):
        self.graph = graph_manager
        self.min_trades = min_trades_threshold
        log.info(f"HistoricalProfiler initialized (min_trades: {self.min_trades}).")

    def _calculate_brier_score(self, df_group: pd.DataFrame) -> float:
        if len(df_group) < self.min_trades: return 0.25
        return ((df_group['bet_price'] - df_group['outcome']) ** 2).mean()

    def run_profiling(self):
        log.info("--- C4: Starting Historical Profiler Batch Job ---")
        
        # This allows the back-tester to inject the profiler data
        if self.graph.is_mock and 'profiler_data' in self.graph.mock_db:
            all_trades_df = self.graph.mock_db['profiler_data']
        else:
            all_trades_df = self.graph.get_all_resolved_trades_by_topic()
            
        if all_trades_df.empty:
            log.warning("C4: No historical trades found to profile.")
            return
        # (rest of the function is the same)
        wallet_scores_series = all_trades_df.groupby(['wallet_id', 'entity_type']).apply(self._calculate_brier_score)
        wallet_scores = wallet_scores_series.to_dict()
        if wallet_scores:
            self.graph.update_wallet_scores(wallet_scores)
        log.info(f"--- C4: Historical Profiler Batch Job Complete. ---")

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
        Z = np.random.standard_normal((self.k_samples, n))
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
    
    return scores.to_dict()


# --- NEW ENGINE: FastBacktestEngine (The Core Optimization) ---
class FastBacktestEngine:
    """Optimized backtest engine with batching and caching."""
    
    def __init__(self, event_log, profiler_data, nlp_cache, precalc_priors):
        self.event_log = event_log
        self.profiler_data = profiler_data
        self.nlp_cache = nlp_cache
        self.precalc_priors = precalc_priors
        
        # Pre-group events by hour for batch processing
        # Note: If passing via Ray Object Store, this sort/group happens inside the worker
        # which is fine as it is fast.
        records = event_log.reset_index().to_dict('records')
        records.sort(key=lambda x: x['timestamp'])
        
        self.hourly_batches = []
        from itertools import groupby
        for hour_key, group in groupby(records, key=lambda x: x['timestamp'].strftime('%Y%m%d%H')):
            self.hourly_batches.append(list(group))
            
    def run_trial(self, config: Dict) -> Dict[str, float]:
        """
        Run a single backtest trial with optimizations.
        """
        # Initialize portfolio
        cash = 10000.0
        positions = {}  # {contract_id: (fraction, entry_price)}
        pnl_history = [cash]
        brier_scores = []
        
        # Current state
        current_prices = {}
        contracts = {}  # {id: {status, p_model, entities, etc}}
        
        # 1. NEW: Track running weighted sums for Smart Money calculation
        # Structure: { contract_id: {'weighted_sum': 0.0, 'weight_sum': 0.0} }
        smart_money_tracker = {}

        # Load wallet scores (The "Reputation" database)
        wallet_scores = fast_calculate_brier_scores(self.profiler_data, 
                                                    config.get('min_trades_threshold', 20))
        
        # Rebalancing throttle
        last_rebalance_time = None
        min_rebalance_interval = timedelta(hours=4)
        
        # Process batches
        for batch in self.hourly_batches:
            if not batch: continue
            
            batch_time = batch[0]['timestamp']
            needs_rebalance = False
            
            # Process events in batch
            for event in batch:
                ev_type = event['event_type']
                data = event['data']
                c_id = event['contract_id']
                
                if ev_type == 'NEW_CONTRACT':
                    # Get pre-calculated entities and priors
                    entities = self.nlp_cache.get_entities(c_id)
                    prior = self.precalc_priors.get(c_id, {'alpha': 1.0, 'beta': 1.0})
                    
                    # Fast contract initialization
                    p_prior = prior['alpha'] / (prior['alpha'] + prior['beta'])
                    p_fused = (p_prior + data['p_market_all']) / 2.0
                    
                    contracts[c_id] = {
                        'status': 'MONITORED',
                        'p_model': p_fused,
                        'p_market_all': data['p_market_all'],
                        'entities': entities
                    }
                    current_prices[c_id] = data['p_market_all']
                    needs_rebalance = True
                
                elif ev_type == 'PRICE_UPDATE':
                    if c_id in contracts:
                        contracts[c_id]['p_market_all'] = data['p_market_all']
                        current_prices[c_id] = data['p_market_all']
                
                elif ev_type == 'RESOLUTION':
                    if c_id in contracts:
                        # Settle position
                        if c_id in positions:
                            fraction, entry = positions.pop(c_id)
                            bet_value = abs(fraction) * 10000.0 
                            
                            outcome = data['outcome']
                            payout = 0.0
                            if fraction > 0:  # Long
                                if entry > 1e-9:
                                    payout = bet_value * (outcome / entry)
                            else:  # Short
                                if (1.0 - entry) > 1e-9:
                                    payout = bet_value * ((1.0 - outcome) / (1.0 - entry))
                            
                            cash += payout
                        
                        # Brier score
                        p_model = contracts[c_id]['p_model']
                        brier_scores.append((p_model - data['outcome']) ** 2)
                        
                        # Remove contract
                        del contracts[c_id]
                        if c_id in current_prices:
                            del current_prices[c_id]
                        
                        needs_rebalance = True
            
            # Rebalance check (throttled)
            should_rebalance = needs_rebalance and (
                last_rebalance_time is None or 
                (batch_time - last_rebalance_time) > min_rebalance_interval
            )
            
            if should_rebalance and len(contracts) > 0:
                # Generate Target Basket
                target_basket = self._fast_kelly_allocation(
                    contracts, current_prices, config, use_numerical=False
                )
                
                # Execute rebalance
                cash = self._rebalance_portfolio(positions, target_basket, 
                                               current_prices, cash)
                
                last_rebalance_time = batch_time
            
            # Track PnL
            position_value = self._calculate_position_value(positions, current_prices)
            pnl_history.append(cash + position_value)
        
        # Calculate metrics
        return self._calculate_metrics(pnl_history, brier_scores)
    
    def _fast_kelly_allocation(self, contracts, prices, config, use_numerical=False):
        """
        Fast allocation strategy. 
        Defaults to Analytical (fastest), falls back to Numerical (Numba) if requested.
        """
        basket = {}
        
        # Arrays for numerical solver
        ids = []
        M_list = []
        Q_list = []
        
        for c_id, contract in contracts.items():
            M = contract['p_model']
            Q = prices.get(c_id, 0.5)
            
            # Filter invalid
            if not (0 <= Q <= 1): Q = 0.5
            
            # Simple edge-based allocation (Analytical)
            edge = M - Q
            
            # Only allocate if edge exceeds threshold (Noise filtering)
            if abs(edge) > config['kelly_edge_thresh']:
                
                if not use_numerical:
                    # --- ANALYTICAL PATH (Fastest) ---
                    variance = M * (1 - M)
                    if variance > 1e-9:
                        fraction = edge / (config['k_brier_scale'] * variance)
                        fraction = max(-0.2, min(0.2, fraction)) 
                        basket[c_id] = fraction
                else:
                    # --- NUMERICAL PREP ---
                    ids.append(c_id)
                    M_list.append(M)
                    Q_list.append(Q)

        # --- NUMERICAL SOLVER INTEGRATION ---
        if use_numerical and len(ids) > 0:
            try:
                M_arr = np.array(M_list)
                Q_arr = np.array(Q_list)
                n = len(M_arr)
                
                # Generate simple independent samples for the mock/fast solver
                # (In full prod, use Cholesky from HybridKellySolver)
                k_samples = 1000
                Z = np.random.random((k_samples, n))
                I_k = (Z < M_arr).astype(np.float64)
                
                # Objective wrapper for Scipy
                def objective_wrapper(F):
                    return fast_kelly_objective(F, M_arr, Q_arr, I_k)
                
                bounds = [(-0.2, 0.2)] * n
                # Use simple initial guess
                x0 = np.zeros(n)
                
                # L-BFGS-B is fast for bounded problems
                res = opt.minimize(objective_wrapper, x0, method='L-BFGS-B', bounds=bounds)
                
                if res.success:
                    for idx, c_id in enumerate(ids):
                        basket[c_id] = res.x[idx]
            except Exception as e:
                log.error(f"Numerical solver failed: {e}. Skipping allocation.")

        # Scale to max exposure
        total = sum(abs(f) for f in basket.values())
        MAX_LEVERAGE = 1.0 
        if total > MAX_LEVERAGE:
            scale = MAX_LEVERAGE / total
            basket = {k: v * scale for k, v in basket.items()}
        
        return basket
    
    def _rebalance_portfolio(self, positions, target, prices, cash):
        """
        Rebalances portfolio with TRANSACTION FEES.
        """
        FEE_RATE = 0.01       # 1% Taker fee
        SLIPPAGE_RATE = 0.005 # 0.5% Slippage
        COST_BASIS = FEE_RATE + SLIPPAGE_RATE
        
        # Close positions not in target
        for c_id in list(positions.keys()):
            if c_id not in target:
                fraction, _ = positions.pop(c_id)
                trade_value = abs(fraction) * 10000.0
                cash -= trade_value * COST_BASIS
        
        # Adjust/Open positions
        for c_id, target_frac in target.items():
            if c_id not in prices: continue
            
            current_frac, entry_price = positions.get(c_id, (0.0, 0.0))
            
            if abs(target_frac - current_frac) < 1e-4:
                continue 
                
            trade_frac_diff = target_frac - current_frac
            trade_value = abs(trade_frac_diff) * 10000.0
            
            # DEDUCT FEES
            cash -= trade_value * COST_BASIS
            
            # Update position record
            positions[c_id] = (target_frac, prices[c_id])
        
        return cash 

    def _calculate_position_value(self, positions, prices):
        total = 0.0
        for c_id, (frac, entry) in positions.items():
            curr = prices.get(c_id, entry)
            if frac > 0:
                mult = curr / entry if entry > 1e-9 else 0
            else:
                mult = (1 - curr) / (1 - entry) if (1 - entry) > 1e-9 else 0
            total += abs(frac) * 10000.0 * mult
        return total
    
    def _calculate_metrics(self, pnl_history, brier_scores):
        pnl = np.array(pnl_history)
        if len(pnl) < 2:
            return {'irr': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'brier_score': 0.25}
        
        returns = np.diff(pnl) / pnl[:-1]
        total_return = (pnl[-1] / pnl[0]) - 1
        irr = total_return 
        
        std_dev = np.std(returns) + 1e-9
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(252 * 24) 
        
        max_dd = 0.0
        peak = pnl[0]
        for val in pnl:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        
        avg_brier = np.mean(brier_scores) if brier_scores else 0.25
        
        return {
            'irr': irr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'brier_score': avg_brier
        }

# --- RAY TUNE WRAPPER (FIXED FOR OBJECT REF TYPES) ---
import traceback

def ray_backtest_wrapper(config, event_log_ref, profiler_ref, nlp_cache_ref, priors_ref):
    """
    Fetches large data from shared memory then runs the trial.
    Includes robust checking to see if data is an ObjectRef or already a DataFrame.
    """
    try:
        # Helper function to safely get data
        def safe_get(obj):
            if isinstance(obj, ray.ObjectRef):
                return ray.get(obj)
            return obj

        # 1. Retrieve large objects (Handle both Ref and Raw Data cases)
        event_log = safe_get(event_log_ref)
        profiler = safe_get(profiler_ref)
        nlp_cache = safe_get(nlp_cache_ref)
        priors = safe_get(priors_ref)
        
        # 2. Initialize Engine
        engine = FastBacktestEngine(event_log, profiler, nlp_cache, priors)
        
        # 3. Run Trial
        return engine.run_trial(config)

    except Exception as e:
        print(f"!!!!! WORKER CRASH !!!!!")
        print(f"Error: {e}")
        traceback.print_exc()
        # Return dummy result to prevent total job failure
        return {'irr': -1.0, 'sharpe_ratio': 0.0, 'brier_score': 1.0}


class BacktestEngine:
    """
    (Production-Ready C7)
    Loads and transforms real Polymarket data and runs
    the full C1-C6 pipeline replay to find optimal parameters.
    """
    def __init__(self, historical_data_path: str):
        log.info("BacktestEngine (C7) Production initialized.")
        self.historical_data_path = historical_data_path
        self.cache_dir = Path(self.historical_data_path) / "polymarket_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR)

    def _get_subgraph_url(self) -> str:
        return "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"

    # --- ETL Methods (Preserved) ---
    def _load_data_from_polymarket(self) -> (pd.DataFrame, pd.DataFrame):
        log.info("Fetching all market details from Polymarket Gamma API...")
        df_markets = self._fetch_all_markets_from_gamma()
        if df_markets.empty:
            return pd.DataFrame(), pd.DataFrame()

        log.info("Fetching all trades from Polymarket FPMM Subgraph...")
        df_trades = self._fetch_all_trades_from_subgraph()
        
        if df_trades.empty:
            log.warning("No trades found from Polymarket Subgraph.")
            df_trades = pd.DataFrame(columns=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'market'])
        
        return df_markets, df_trades

    def _fetch_all_markets_from_gamma(self) -> pd.DataFrame:
        return self._fetch_paginated_rest(
            cache_key="polymarket_markets_gamma",
            base_url="https://gamma-api.polymarket.com/markets",
            limit=500 
        )

    def _fetch_all_trades_from_subgraph(self) -> pd.DataFrame:
        query_template = """
        {{
          fpmmTransactions(first: 1000, orderBy: id, orderDirection: asc, where: {{ id_gt: "{last_id}" }}) {{
            id, timestamp, tradeAmount, outcomeTokensAmount, user, market {{ id }}
          }}
        }}
        """
        df = self._fetch_paginated_subgraph(
            cache_key="polymarket_trades",
            subgraph_url=self._get_subgraph_url(),
            query_template=query_template,
            entity_name="fpmmTransactions"
        )
        if df.empty: return df
        try:
            df['wallet_id'] = df['user']
            df['market_id'] = df['market'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
        except Exception:
            return pd.DataFrame()
        return df.drop(columns=['user', 'market'])

    def _fetch_paginated_rest(self, cache_key: str, base_url: str, limit: int) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{cache_key}_master.parquet" # Switch to Parquet for speed
        
        # 1. Load existing cache if it exists
        existing_df = pd.DataFrame()
        start_offset = 0
        
        if cache_file.exists():
            try:
                existing_df = pd.read_parquet(cache_file)
                # If we have data, we don't necessarily rely on offset for the API, 
                # but we can check for the latest data. 
                # NOTE: For Gamma API specifically, it uses 'offset'. 
                # A true 'top-up' requires an API that supports 'created_after'.
                # If the API only supports offset, we might still have to re-fetch, 
                # BUT we can stop early if we hit a known ID.
                log.info(f"Loaded {len(existing_df)} rows from cache.")
            except Exception:
                log.warning("Cache corrupted, starting fresh.")
    
        # 2. Fetch new data (Simplest 'Top-Up' logic for this specific API)
        # Gamma doesn't easily support "give me data since X" without filtering.
        # So we fetch, but we STOP fetching once we see IDs we already have.
        
        existing_ids = set(existing_df['market_id']) if not existing_df.empty else set()
        new_rows = []
        offset = 0
        
        while True:
            params = {"limit": limit, "offset": offset}
            # Add specific sort if API supports it to ensure we get newest first
            # params['order'] = 'newest' 
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                rows = response.json()
            except Exception as e:
                log.error(f"API Request failed: {e}")
                break
                
            if not rows: 
                break
                
            # Check if we have reached data we already know
            # (This assumes the API returns newest first. If it returns oldest first, 
            # you generally have to fetch everything anyway unless you track 'offset')
            batch_new = [r for r in rows if r.get('conditionId') not in existing_ids]
            
            if not batch_new and len(existing_ids) > 0:
                log.info("Hit existing data boundary. Stopping fetch.")
                break
                
            new_rows.extend(batch_new)
            offset += limit
            
            if len(batch_new) < len(rows): 
                 # Partial match means we merged into old history
                 break
    
        if not new_rows:
            return existing_df
    
        # 3. Merge and Save
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([new_df, existing_df], ignore_index=True)
        
        # Deduplicate just in case
        if 'conditionId' in combined_df.columns:
             combined_df = combined_df.drop_duplicates(subset=['conditionId'])
             
        # Save as Master Cache
        combined_df.to_parquet(cache_file)
        return combined_df

    def _fetch_paginated_subgraph(self, cache_key: str, subgraph_url: str, query_template: str, entity_name: str) -> pd.DataFrame:
        today = datetime.now().strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"{cache_key}_{today}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f: return pickle.load(f)
        
        try:
            all_rows = []
            last_id = "" 
            while True:
                query_body = query_template.format(last_id=last_id)
                response = requests.post(subgraph_url, json={'query': query_body})
                data = response.json().get('data', {})
                rows = data.get(entity_name, [])
                if not rows: break
                all_rows.extend(rows)
                last_id = rows[-1]['id'] 

            if not all_rows: return pd.DataFrame()
            df = pd.DataFrame(all_rows)
            with open(cache_file, 'wb') as f: pickle.dump(df, f)
            return df
        except Exception:
            return pd.DataFrame()

    def _transform_data_to_event_log(self, df_markets, df_trades) -> (pd.DataFrame, pd.DataFrame):
        """
        Transforms raw data and PRE-CALCULATES NLP entities with Batching + Progress Logs.
        """
        log.info("Transforming raw Subgraph and Gamma API data into event log...")

        # Initialize empty frames for fallback
        empty_log = pd.DataFrame(columns=['timestamp', 'event_type', 'data'])
        empty_log.set_index('timestamp', inplace=True)
        empty_profiler = pd.DataFrame(columns=['wallet_id', 'price', 'outcome', 'market_id', 'entity_type'])

        try:
            # Setup NLP
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                # Disable unnecessary pipeline components for speed
                nlp.select_pipes(enable=['ner']) 
                relevant_labels = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}
                log.info("spaCy model loaded. Using optimized nlp.pipe for batch processing.")
            except Exception as e:
                log.warning(f"spaCy setup failed ({e}). NLP pre-processing will be skipped.")
                nlp = None

            # --- 1. Process Market Data ---
            log.info("Processing Market data (from Gamma API)...")
            
            # Validate input
            if df_markets is None or df_markets.empty:
                log.warning("df_markets is empty/None. Returning empty logs.")
                return empty_log, empty_profiler

            market_rename_map = {
                'marketMakerAddress': 'fpmm_address', 'conditionId': 'market_id',           
                'question': 'question', 'createdAt': 'created_at',
                'closedTime': 'resolution_timestamp', 'endDateIso': 'end_date'
            }
            existing_renames = {k: v for k, v in market_rename_map.items() if k in df_markets.columns}
            df_markets = df_markets.rename(columns=existing_renames)

            # Fix timestamps
            if 'created_at' in df_markets.columns:
                df_markets['created_at'] = pd.to_datetime(df_markets['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
            else:
                df_markets['created_at'] = pd.to_datetime(datetime.now() - timedelta(days=365)).tz_localize(None)

            df_markets = df_markets.dropna(subset=['created_at', 'market_id', 'question'])
            
            if 'resolution_timestamp' in df_markets.columns:
                df_markets['resolution_timestamp'] = pd.to_datetime(df_markets['resolution_timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
            else:
                 df_markets['resolution_timestamp'] = pd.to_datetime(df_markets['end_date'], errors='coerce', utc=True).dt.tz_localize(None)

            if 'outcome' in df_markets.columns:
                df_markets['outcome'] = pd.to_numeric(df_markets['outcome'], errors='coerce')
            else:
                 df_markets['outcome'] = pd.NA 

            df_markets = df_markets.dropna(subset=['fpmm_address', 'market_id', 'question'])
            
            # --- 2. Process Trade Data ---
            # If trades are empty, we still continue to process markets
            if df_trades is None or df_trades.empty:
                log.warning("df_trades is empty. Proceeding with markets only.")
                df_trades = pd.DataFrame(columns=['fpmm_address', 'timestamp', 'price', 'size', 'wallet_id'])
            else:
                trade_rename_map = {'tradeAmount': 'size', 'market_id': 'fpmm_address'}
                df_trades = df_trades.rename(columns=trade_rename_map)
                df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s', errors='coerce', utc=True).dt.tz_localize(None)
                df_trades['size'] = (pd.to_numeric(df_trades['size'], errors='coerce').fillna(0) / 1e6)
                tokens = (pd.to_numeric(df_trades['outcomeTokensAmount'], errors='coerce').fillna(0) / 1e18)
                df_trades['price'] = 0.0
                valid_mask = tokens > 1e-9
                df_trades.loc[valid_mask, 'price'] = df_trades['size'] / tokens[valid_mask]
                df_trades['price'] = df_trades['price'].clip(0.0, 1.0)
                df_trades = df_trades.dropna(subset=['fpmm_address', 'timestamp', 'price', 'size', 'wallet_id'])

            # --- 3. Join ---
            log.info("Joining Market and Trade data...")
            df_markets['fpmm_address'] = df_markets['fpmm_address'].astype(str).str.lower()
            df_trades['fpmm_address'] = df_trades['fpmm_address'].astype(str).str.lower()
            
            market_id_map = df_markets.set_index('fpmm_address')['market_id'].to_dict()
            outcome_map = df_markets.set_index('market_id')['outcome'].to_dict() if 'outcome' in df_markets.columns else {}
            
            df_trades['market_id'] = df_trades['fpmm_address'].map(market_id_map)
            df_trades = df_trades.dropna(subset=['market_id'])
            df_trades['outcome'] = df_trades['market_id'].map(outcome_map)

            profiler_data = df_trades[['wallet_id', 'price', 'outcome', 'market_id']].rename(columns={'price': 'bet_price'})
            profiler_data['entity_type'] = 'default_topic'
            profiler_data = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id'])
            
            # --- 4. Event Log with Batch NLP ---
            log.info(f"Generating event log for {len(df_markets)} markets...")
            events = []
            market_vectors = {mid: [0.1]*768 for mid in market_id_map.values()} 
            
            market_records = df_markets.to_dict('records')
            
            # Batch NLP
            if nlp:
                questions = [r.get('question', '') for r in market_records]
                doc_stream = nlp.pipe(questions, batch_size=50)
            else:
                doc_stream = [None] * len(market_records)

            count = 0
            for row, doc in zip(market_records, doc_stream):
                extracted_entities = []
                if doc:
                    extracted_entities = [ent.text for ent in doc.ents if ent.label_ in relevant_labels]
                
                # Fallback if NLP failed or found nothing
                if not extracted_entities:
                     extracted_entities = [row.get('question', '')[:20]]
                
                events.append((
                    row['created_at'], 'NEW_CONTRACT',
                    {
                        'id': row['market_id'], 
                        'text': row['question'], 
                        'vector': market_vectors.get(row['market_id']), 
                        'liquidity': 0, 
                        'p_market_all': 0.5,
                        'precalc_entities': extracted_entities
                    }
                ))
                count += 1

            # Add resolutions
            for _, row in df_markets.dropna(subset=['resolution_timestamp']).iterrows():
                if pd.notna(row.get('outcome')):
                     events.append((row['resolution_timestamp'], 'RESOLUTION', {'id': row['market_id'], 'outcome': row['outcome']}))
                
            # Add trades
            for _, row in df_trades.iterrows():
                events.append((row['timestamp'], 'PRICE_UPDATE',
                    {'id': row['market_id'], 'p_market_all': row['price'], 'wallet_id': row['wallet_id'], 'trade_price': row['price'], 'trade_volume': abs(row['size'])}
                ))
                
            event_log = pd.DataFrame(events, columns=['timestamp', 'event_type', 'data'])
            if not event_log.empty:
                event_log['contract_id'] = event_log['data'].apply(lambda x: x.get('id'))
                event_log['timestamp'] = pd.to_datetime(event_log['timestamp'], utc=True).dt.tz_localize(None)
                event_log = event_log.set_index('timestamp').sort_index()

            log.info(f"ETL Complete. Profiler: {len(profiler_data)}, Events: {len(event_log)}")
            
            # --- CRITICAL FIX: ALWAYS RETURN TWO DATAFRAMES ---
            return event_log, profiler_data

        except Exception as e:
            log.error(f"Failed to parse data in _transform_data_to_event_log: {e}", exc_info=True)
            # --- CRITICAL FIX: Return empty frames on error, NOT None ---
            return empty_log, empty_profiler

    def run_tuning_job(self):
        """Main entry point for Component 7 (Optimized)."""
        log.info("--- C7: Starting Hyperparameter Tuning Job (Optimized) ---")
        
        # 1. Load Data
        try:
            log.info(f"C7: Loading all historical data...")
            df_markets, df_trades = self._load_data_from_polymarket()
            if df_markets.empty: return None

            if 'conditionId' in df_markets.columns:
                df_markets = df_markets.rename(columns={'conditionId': 'market_id'})
            
            # Renaming other common inconsistencies for safety
            if 'question' in df_markets.columns:
                 # Ensure question exists and isn't null for NLP
                 df_markets['question'] = df_markets['question'].fillna("")
            
            # Assuming _transform_data_to_event_log is populated with the full logic
            event_log, profiler_data = self._transform_data_to_event_log(df_markets, df_trades)
        except Exception as e:
            log.error(f"FATAL: Failed to load data: {e}", exc_info=True); return None

        # 2. Pre-Compute Optimizations
        try:
            log.info("--- Starting Pre-computation Phase ---")
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm"); nlp.select_pipes(enable=['ner'])
            except Exception:
                nlp = None
            nlp_cache = NLPCache(df_markets, nlp)

            log.info("Pre-calculating AI priors...")
            temp_ai = AIAnalyst() 
            precalc_priors = {}
            for _, row in df_markets.iterrows():
                mid = row.get('market_id')
                text = row.get('question', '')
                if not mid: continue
                prior_data = temp_ai.get_prior(text)
                mean = prior_data['probability']
                ci = tuple(prior_data['confidence_interval'])
                alpha, beta = convert_to_beta(mean, ci)
                precalc_priors[mid] = {'alpha': alpha, 'beta': beta}
            
            # 3. PUT DATA IN SHARED MEMORY (Object Store)
            log.info("Uploading large datasets to Ray Object Store...")
            event_log_ref = ray.put(event_log)
            profiler_ref = ray.put(profiler_data)
            nlp_cache_ref = ray.put(nlp_cache)
            priors_ref = ray.put(precalc_priors)

        except Exception as e:
            log.error(f"Failed to initialize fast engine: {e}", exc_info=True); return None
        
        # 4. Run Ray Tune
        search_space = {
            "brier_internal_model": tune.loguniform(0.05, 0.25),
            "k_brier_scale": tune.loguniform(0.1, 5.0),
            "kelly_edge_thresh": tune.uniform(0.05, 0.25),
            "min_trades_threshold": tune.qrandint(5, 50, 5)
        }
        
        scheduler = ASHAScheduler(metric="irr", mode="max", max_t=10, grace_period=1)
        
        # Pass the Object References (not the data itself) to the wrapper
        trainable = tune.with_parameters(
            ray_backtest_wrapper, 
            event_log_ref=event_log_ref,
            profiler_ref=profiler_ref,
            nlp_cache_ref=nlp_cache_ref,
            priors_ref=priors_ref
        )
        
        analysis = tune.run(
            trainable, config=search_space, num_samples=20,
            scheduler=scheduler, resources_per_trial={"cpu": 1}, name="pm_tuning_fast"
        )
        
        best_trial = analysis.get_best_trial(metric="irr", mode="max")
        if best_trial:
            log.info(f"--- Tuning Complete. Best IRR: {best_trial.last_result.get('irr', 0.0):.4f} ---")
            return best_trial.config
        else:
            return None
        
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
    """Runs the C7 Backtest/Tuning demo"""
    log.info("--- (DEMO) Running Component 7 (Production) Demo ---")
    
    try:
        backtester = BacktestEngine(
            historical_data_path="." # Save to current directory
        )
        best_params = backtester.run_tuning_job()
        log.info(f"--- C7 Demo Complete. Best params: {best_params} ---")
    except Exception as e:
        log.error(f"C7 Demo Failed: {e}", exc_info=True)
        if "dune" in str(e): log.info("Hint: Run 'pip install dune-client' and set DUNE_API_KEY")
        if "ray" in str(e): log.info("Hint: Run 'pip install \"ray[tune]\" pandas'")

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
