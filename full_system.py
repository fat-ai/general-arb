import os
import logging
import spacy
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
import pickle
from pathlib import Path
import time

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
    """Component 1: Production-ready GraphManager."""
    def __init__(self, is_mock=False):
        self.is_mock = is_mock
        if self.is_mock:
            log.warning("GraphManager is running in MOCK mode.")
            self.vector_dim = 768
            self.model_brier_scores = {'brier_internal_model': 0.08, 'brier_expert_model': 0.05, 'brier_crowd_model': 0.15}
            
            # --- FIX R1: Initialize mock_db in constructor ---
            self.mock_db = {
                'contracts': {}, # {contract_id: {data}}
                'entities': {
                    'E_123': {'canonical_name': 'NeuroCorp, Inc.', 'type': 'Organization', 'vector': [0.2]*768, 'contract_count': 0}
                },
                'aliases': {
                    'NeuroCorp': 'E_123'
                },
                'wallets': {
                    'Wallet_ABC': {'brier_biotech': 0.0567, 'brier_geopolitics': 0.81},
                    'Wallet_XYZ': {'brier_biotech': 0.49, 'brier_geopolitics': 0.01},
                    'Wallet_CROWD_1': {'brier_biotech': 0.25},
                },
                'review_queue': [
                    {'id': 'MKT_902', 'reason': 'No alias match found', 'details': json.dumps({'entities': ['NeruCorp']})},
                    {'id': 'MKT_905', 'reason': 'NEEDS_MERGE_CONFIRMATION', 'details': json.dumps({'similar_contracts': ['MKT_888']})}
                ]
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
                log.info("APOC text index 'aliases' ensured.")
            except ClientError:
                log.warning("APOC index failed (is APOC plugin installed?).")
            
            # --- FIX R3: Add Vector Index Creation ---
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
                log.info("Contract vector index applied.")
            except Exception:
                log.warning(f"Could not create vector index (requires Neo4j Enterprise/Aura).")
        log.info("Schema setup complete.")
        
    def add_contract(self, contract_id: str, text: str, vector: list[float], liquidity: float = 0.0, p_market_all: float = None):
        if self.is_mock:
            self.mock_db['contracts'][contract_id] = {
                'text': text, 'vector': vector, 'liquidity': liquidity,
                'p_market_all': p_market_all, 'status': 'PENDING_LINKING',
                'entity_ids': []
            }
            return
            
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {len(vector)}")
        with self.driver.session() as session:
            session.execute_write(self._tx_merge_contract, contract_id, text, vector, liquidity, p_market_all)
        log.info(f"Merged Contract: {contract_id}")

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
            self.mock_db['contracts'][contract_id]['entity_ids'].append(entity_id)
            self.mock_db['contracts'][contract_id]['status'] = 'PENDING_ANALYSIS'
            # (Fix R5) Increment contract count for this entity
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
             # --- FIX R5: Use mock_db ---
             res = []
             for cid, data in self.mock_db['contracts'].items():
                 if data['status'] == status:
                     res.append({'contract_id': cid, **data})
                     if len(res) >= limit: break
             return res
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
            # --- FIX R5: Use mock_db ---
            entity_id = self.mock_db['aliases'].get(alias_text)
            if entity_id:
                return {'entity_id': entity_id, 'name': self.mock_db['entities'][entity_id]['canonical_name'], 'confidence': 1.0}
            return None
        
        # --- PRODUCTION-READY C2 LOGIC ---
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
            except ClientError as e:
                log.warning(f"APOC fuzzy query failed: {e}. Falling back to exact match.")
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
        
        # --- PRODUCTION-READY C2 LOGIC ---
        query = """
            CALL db.index.vector.queryNodes('contract_vector_index', $k, $vector) YIELD node, similarity
            WHERE node.contract_id <> $contract_id
            RETURN node.contract_id AS id, node.text AS text, similarity
        """
        with self.driver.session() as session:
            try:
                results = session.run(query, k=k, vector=vector, contract_id=contract_id)
                return [r.data() for r in results]
            except ClientError as e:
                log.warning(f"Vector KNN query failed (is index 'contract_vector_index' built?): {e}")
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
            # --- FIX 2: Use json.dumps for safe serialization (P0.4) ---
            params['metadata'] = json.dumps(metadata)
        tx.run(query, **params)
    
    # --- C3: Read/Write Methods ---
    def get_entity_contract_count(self, entity_id: str) -> int:
        if self.is_mock: 
            # --- FIX R5: Use mock_db ---
            return self.mock_db['entities'].get(entity_id, {}).get('contract_count', 0)
        
        query = "MATCH (e:Entity {entity_id: $entity_id})<-[:IS_ABOUT]-(c:Contract) RETURN count(c) AS count"
        with self.driver.session() as session:
            result = session.run(query, entity_id=entity_id).single()
            return result['count'] if result else 0
            
    def update_contract_prior(self, contract_id: str, p_internal: float, alpha: float, beta: float, source: str, p_experts: float, p_all: float):
        if self.is_mock:
            # --- FIX R5: Use mock_db ---
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
    # --- FIX R7: Removed duplicate update_contract_prior method ---

    # --- C4: Read/Write Methods ---
    def get_all_resolved_trades_by_topic(self) -> pd.DataFrame:
        if self.is_mock: 
            # --- FIX R5: Use mock_db (albeit still hardcoded for this demo) ---
            return pd.DataFrame([
                {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.8, 'outcome': 1.0},
                {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.7, 'outcome': 1.0},
                {'wallet_id': 'Wallet_XYZ', 'entity_type': 'geopolitics', 'bet_price': 0.4, 'outcome': 0.0},
            ])
        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract)-[:IS_ABOUT]->(e:Entity)
        WHERE c.status = 'RESOLVED' AND c.outcome IS NOT NULL AND t.price IS NOT NULL
        RETURN w.wallet_id AS wallet_id, e.type AS entity_type, 
               t.price AS bet_price, c.outcome AS outcome
        SKIP 0 LIMIT 10000 
        """ # <-- P1: Added LIMIT
        with self.driver.session() as session:
            results = session.run(query)
            df = pd.DataFrame([r.data() for r in results])
            return df if not df.empty else pd.DataFrame(columns=['wallet_id', 'entity_type', 'bet_price', 'outcome'])

    def get_live_trades_for_contract(self, contract_id: str) -> pd.DataFrame:
        if self.is_mock: 
            # --- FIX R5: Use mock_db (hardcoded) ---
            return pd.DataFrame([
                {'wallet_id': 'Wallet_ABC', 'trade_price': 0.35, 'trade_volume': 5000},
                {'wallet_id': 'Wallet_CROWD_1', 'trade_price': 0.60, 'trade_volume': 100},
            ])
        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract {contract_id: $contract_id})
        WHERE t.price IS NOT NULL AND t.volume IS NOT NULL
        RETURN w.wallet_id AS wallet_id, t.price AS trade_price, t.volume AS trade_volume
        LIMIT 1000
        """ # <-- P1: Added LIMIT
        with self.driver.session() as session:
            results = session.run(query, contract_id=contract_id)
            df = pd.DataFrame([r.data() for r in results])
            return df if not df.empty else pd.DataFrame(columns=['wallet_id', 'trade_price', 'trade_volume'])

    def get_contract_topic(self, contract_id: str) -> str:
        if self.is_mock: return "biotech"
        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run("MATCH (c:Contract {contract_id: $id})-[:IS_ABOUT]->(e:Entity) RETURN e.type AS topic LIMIT 1", id=contract_id).single()
            )
        return result.data().get('topic') if result else "default"

    def update_wallet_scores(self, wallet_scores: Dict[tuple, float]):
        if self.is_mock: 
            log.info(f"MockGraph: Updating {len(wallet_scores)} wallet scores (STUB)")
            for (wallet_id, topic), score in wallet_scores.items():
                if wallet_id not in self.mock_db['wallets']: self.mock_db['wallets'][wallet_id] = {}
                self.mock_db['wallets'][wallet_id][f"brier_{topic}"] = score
            return
        
        scores_list = [{"wallet_id": k[0], "topic_key": f"brier_{k[1]}", "brier_score": v} for k, v in wallet_scores.items()]
        if not scores_list: return
        query = "UNWIND $scores_list AS score MERGE (w:Wallet {wallet_id: score.wallet_id}) SET w[score.topic_key] = score.brier_score"
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, scores_list=scores_list))
        log.info(f"Updated {len(scores_list)} wallet scores in graph.")
        
    def get_wallet_brier_scores(self, wallet_ids: List[str]) -> Dict[str, Dict[str, float]]:
        if self.is_mock: 
            # --- FIX R5: Use mock_db ---
            return {wid: scores for wid, scores in self.mock_db['wallets'].items() if wid in wallet_ids}
        
        query = "MATCH (w:Wallet) WHERE w.wallet_id IN $wallet_ids RETURN w.wallet_id AS wallet_id, properties(w) AS scores"
        with self.driver.session() as session:
            results = session.run(query, wallet_ids=wallet_ids)
            return {r.data()['wallet_id']: {k: v for k, v in r.data()['scores'].items() if k.startswith('brier_')} for r in results}

    # --- C5: Read/Write Methods ---
    def get_contracts_for_fusion(self, limit: int = 10) -> List[Dict]:
        """Gets all raw data needed for C5 fusion."""
        if self.is_mock: return self._mock_get_contracts_for_fusion() # <-- FIX: Call the correct mock
        
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
            # --- FIX R5: Use mock_db ---
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
            # --- FIX R5: Use mock_db ---
            clusters = set()
            for c in self.mock_db['contracts'].values():
                if c['status'] == 'MONITORED':
                    for eid in c['entity_ids']: clusters.add(eid)
            return list(clusters) if clusters else ["E_DUNE_3"] # Default for demo
            
        query = "MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity) RETURN DISTINCT e.entity_id AS entity_id"
        with self.driver.session() as session:
            results = session.run(query)
            return [r['entity_id'] for r in results]
            
    def get_cluster_contracts(self, entity_id: str) -> List[Dict]:
        if self.is_mock: 
            # --- FIX R5: Use mock_db (default to arb demo) ---
            return [
                {'id': 'MKT_A', 'M': 0.60, 'Q': 0.60, 'is_logical_rule': True},
                {'id': 'MKT_B', 'M': 0.60, 'Q': 0.50, 'is_logical_rule': True}
            ]
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
            # --- FIX R5: Use mock_db (default to arb demo) ---
            if c1_id == 'MKT_A' and c2_id == 'MKT_B':
                p_A = next(c['M'] for c in contracts if c['id'] == 'MKT_A')
                return {'type': 'LOGICAL_IMPLIES', 'p_joint': p_A}
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
                log.debug(f"Found LOGICAL_IMPLIES between {c1_id} and {c2_id}")
                return {'type': 'LOGICAL_IMPLIES', 'p_joint': p_model_c1}
        return {'type': 'NONE', 'p_joint': None}

    # --- C7/C8: Mock-driving Methods ---
    def get_historical_data_for_replay(self, start_date, end_date):
        log.info(f"MockGraph: Fetching historical data from {start_date} to {end_date} (STUB)")
        return pd.DataFrame([
            ('2023-01-01T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_1', 'text': 'NeuroCorp release...', 'vector': [0.1]*768, 'liquidity': 100, 'p_market_all': 0.50}),
            ('2023-01-01T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.51, 'p_market_experts': 0.55}),
            ('2023-01-02T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_2', 'text': 'Dune 3...', 'vector': [0.2]*768, 'liquidity': 50000, 'p_market_all': 0.70}),
            ('2023-01-02T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.55, 'p_market_experts': 0.60}),
            ('2023-01-02T10:06:00Z', 'PRICE_UPDATE', {'id': 'MKT_2', 'p_market_all': 0.70, 'p_market_experts': 0.75}),
            ('2023-01-03T12:00:00Z', 'RESOLUTION', {'id': 'MKT_1', 'outcome': 1.0}),
            ('2023-01-04T12:00:00Z', 'RESOLUTION', {'id': 'MKT_2', 'outcome': 0.0}),
        ], columns=['timestamp', 'event_type', 'data'])
        
    def get_human_review_queue(self):
        log.info("MockGraph: Fetching 'NEEDS_HUMAN_REVIEW' queue...")
        return self.mock_db['review_queue']
        
    def get_portfolio_state(self):
        log.info("MockGraph: Fetching current portfolio...")
        # (Hardcoded for demo)
        return {'cash': 8537.88, 'positions': [{'id': 'MKT_A', 'fraction': -0.075}, {'id': 'MKT_B', 'fraction': 0.075}], 'total_value': 8537.88}
        
    def get_pnl_history(self):
        log.info("MockGraph: Fetching P&L history...")
        return pd.Series(np.random.normal(0, 1, 100).cumsum() + 10000)
        
    def get_regime_status(self):
        log.info("MockGraph: Fetching regime status...")
        return "LOW_VOL", {"k_brier_scale": 1.5, "kelly_edge_thresh": 0.1}
        
    def resolve_human_review_item(self, item_id, action, data):
        log.warning(f"MockGraph: Resolving {item_id} with action '{action}' and data: {data}")
        self.mock_db['review_queue'] = [item for item in self.mock_db['review_queue'] if item['id'] != item_id]
        # (In prod, this would trigger C2 linker to re-run on this contract_id)
        return True

    # --- MOCK IMPLEMENTATIONS (Called if is_mock=True) ---
    
    def _mock_get_contracts_by_status(self, status: str, limit: int = 10):
         if status == 'PENDING_LINKING':
             return [{'contract_id': 'MKT_902_SPACY_DEMO', 'text': "Will 'NeuroCorp' release the 'Viper'?", 'vector': [0.1]*768, 'liquidity': 100, 'p_market_all': 0.5, 'entity_ids': []}]
         if status == 'PENDING_ANALYSIS':
            return [{'contract_id': 'MKT_903', 'text': 'Test contract for NeuroCorp', 'vector': [0.3]*768, 'liquidity': 100, 'p_market_all': 0.5, 'entity_ids': ['E_123']}]
         if status == 'PENDING_FUSION':
            return [{'contract_id': 'MKT_FUSE_001', 'p_internal_alpha': 13.8, 'p_internal_beta': 9.2, 'p_market_experts': 0.45, 'p_market_all': 0.55, 'status': 'PENDING_FUSION'}]
         return []
         
    def _mock_find_entity_by_alias_fuzzy(self, alias_text: str):
        if alias_text == "NeuroCorp": return {'entity_id': 'E_123', 'name': 'NeuroCorp, Inc.', 'confidence': 1.0}
        return None
        
    def _mock_get_all_resolved_trades_by_topic(self):
        return pd.DataFrame([
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.8, 'outcome': 1.0},
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.7, 'outcome': 1.0},
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.2, 'outcome': 0.0},
            {'wallet_id': 'Wallet_XYZ', 'entity_type': 'geopolitics', 'bet_price': 0.4, 'outcome': 0.0},
        ])
        
    def _mock_get_live_trades_for_contract(self, contract_id):
        return pd.DataFrame([
            {'wallet_id': 'Wallet_ABC', 'trade_price': 0.35, 'trade_volume': 5000},
            {'wallet_id': 'Wallet_CROWD_1', 'trade_price': 0.60, 'trade_volume': 100},
        ])
        
    def _mock_get_wallet_brier_scores(self, wallet_ids):
        return { 'Wallet_ABC': {'brier_biotech': 0.0567}, 'Wallet_CROWD_1': {'brier_biotech': 0.25} }
        
    def _mock_get_contracts_for_fusion(self):
        return [c for c in self.mock_db['contracts'].values() if c['status'] == 'PENDING_FUSION']
        
    def _mock_get_model_brier_scores(self): 
        return self.model_brier_scores
        
    def _mock_get_active_entity_clusters(self): 
        clusters = set()
        for c in self.mock_db['contracts'].values():
            if c.get('status') == 'MONITORED':
                for eid in c.get('entity_ids', []): clusters.add(eid)
        return list(clusters) if clusters else ["E_DUNE_3_MOCK"] # Default for demo
            
    def _mock_get_cluster_contracts(self, entity_id):
        if entity_id == "E_DUNE_3_MOCK":
            return [{'id': 'MKT_A', 'M': 0.60, 'Q': 0.60, 'is_logical_rule': True}, {'id': 'MKT_B', 'M': 0.60, 'Q': 0.50, 'is_logical_rule': True}]
        
        res = []
        for cid, data in self.mock_db['contracts'].items():
            if data.get('status') == 'MONITORED' and entity_id in data.get('entity_ids', []):
                res.append({
                    'id': cid,
                    'M': data.get('p_model'),
                    'Q': data.get('p_market_all'),
                    'is_logical_rule': data.get('is_logical_rule', False)
                })
        return res
            
    def _mock_get_relationship_between_contracts(self, c1_id, c2_id, contracts):
        if c1_id == 'MKT_A' and c2_id == 'MKT_B':
            p_A = next(c['M'] for c in contracts if c['id'] == 'MKT_A')
            return {'type': 'LOGICAL_IMPLIES', 'p_joint': p_A}
        return {'type': 'NONE', 'p_joint': None}
        
    def _mock_get_historical_data_for_replay(self, s, e):
        df = pd.DataFrame([
            ('2023-01-01T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_1', 'text': 'NeuroCorp release...', 'vector': [0.1]*768, 'liquidity': 100, 'p_market_all': 0.50}),
            ('2023-01-01T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.51, 'p_market_experts': 0.55}),
            ('2023-01-02T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_2', 'text': 'Dune 3...', 'vector': [0.2]*768, 'liquidity': 50000, 'p_market_all': 0.70}),
            ('2023-01-02T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.55, 'p_market_experts': 0.60}),
            ('2023-01-02T10:06:00Z', 'PRICE_UPDATE', {'id': 'MKT_2', 'p_market_all': 0.70, 'p_market_experts': 0.75}),
            ('2023-01-03T12:00:00Z', 'RESOLUTION', {'id': 'MKT_1', 'outcome': 1.0}),
            ('2023-01-04T12:00:00Z', 'RESOLUTION', {'id': 'MKT_2', 'outcome': 0.0}),
        ], columns=['timestamp', 'event_type', 'data'])
        df['contract_id'] = df['data'].apply(lambda x: x.get('id'))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df
        
    def _mock_get_human_review_queue(self): 
        return self.mock_db['review_queue']
        
    def _mock_get_portfolio_state(self): 
        return {'cash': 8500.0, 'positions': [], 'total_value': 8500.0}
        
    def _mock_get_pnl_history(self): 
        return pd.Series(np.random.normal(0, 1, 100).cumsum() + 10000)
        
    def _mock_get_regime_status(self): 
        return "LOW_VOL", {"k": 1.5, "edge": 0.1}
        
    def _mock_resolve_human_review_item(self, id, action, data): 
        self.mock_db['review_queue'] = [item for item in self.mock_db['review_queue'] if item['id'] != id]
        return True

# ==============================================================================
# ### COMPONENT 2: RelationalLinker (Production-Ready) ###
# ==============================================================================
class RelationalLinker:
    """(Production-Ready C2)"""
    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        model_name = "en_core_web_sm"
        try:
            self.nlp = spacy.load(model_name)
            log.info("spaCy NER model loaded.")
        except IOError:
            log.error(f"Failed to load spaCy model '{model_name}'. Run: python -m spacy download en_core_web_sm")
            raise
    
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
        log.info("--- C2: Checking for 'PENDING_LINKING' contracts ---")
        contracts = self.graph.get_contracts_by_status('PENDING_LINKING', limit=10)
        if not contracts:
            log.info("C2: No contracts to link.")
            return

        for contract in contracts:
            contract_id, contract_text, contract_vector = contract['contract_id'], contract['text'], contract['vector']
            log.info(f"--- C2: Processing Contract: {contract_id} ---")
            extracted_entities = self._extract_entities(contract_text)
            
            if not extracted_entities:
                self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_REVIEW', {'reason': 'No entities found'})
                continue
                
            fast_path_matches = self._run_fast_path(extracted_entities)

            if len(fast_path_matches) >= 1:
                log.info(f"C2: Fast Path success. Linking {len(fast_path_matches)} entities.")
                for entity_id, (confidence, name) in fast_path_matches.items():
                    self.graph.link_contract_to_entity(contract_id, entity_id, confidence)
            else:
                log.info("C2: No Fast Path matches. Escalating to Fuzzy Path (KNN).")
                reason, details = self._run_fuzzy_path_knn(contract_id, contract_vector)
                details['extracted_entities'] = list(extracted_entities)
                self.graph.update_contract_status(contract_id, 'NEEDS_HUMAN_REVIEW', {'reason': reason, **details})
            log.info(f"--- C2: Finished Processing: {contract_id} ---")

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
        log.info("Solving with Numerical (Precise Path)...")
        n = len(M)
        
        # 1. Generate Correlated Outcomes (I_k)
        std_devs = np.sqrt(np.diag(C))
        std_devs = np.where(std_devs == 0, 1e-9, std_devs) 
        Corr = C / np.outer(std_devs, std_devs)
        np.fill_diagonal(Corr, 1.0) 
        
        L = None # Flag to track which path we took
        try:
            # Try the fast Cholesky path first
            Corr_jitter = Corr + np.eye(n) * 1e-9 
            L = np.linalg.cholesky(Corr_jitter)
        except np.linalg.LinAlgError:
            # --- THIS IS THE FIX ---
            # If Cholesky fails (e.g., perfect correlation),
            # fall back to the slower but more robust MVN sampler.
            log.warning("Cov matrix not positive definite. Using MVN sampler (slower).")
            try:
                # We must sample from the COVARIANCE matrix C, not the CORRELATION matrix
                # Add jitter to C to ensure it's positive semi-definite for the sampler
                C_jitter = C + np.eye(n) * 1e-9
                sampler = qmc.MultivariateNormal(mean=np.zeros(n), cov=C_jitter)
                # We use 'Z' here just for variable name consistency
                Z = sampler.random(self.k_samples)
                # We skip Cholesky, so Z is already our correlated samples
                
            except Exception as e:
                log.error(f"FATAL: MVN sampler also failed: {e}. Falling back to independence.")
                L = np.eye(n) # Last resort
            
        if L is not None:
            # Standard (fast) Cholesky path
            sampler = qmc.Sobol(d=n, scramble=True)
            m_power = int(math.ceil(math.log2(self.k_samples)))
            U_unif = sampler.random_base2(m=m_power)[:self.k_samples]
            Z = norm.ppf(U_unif) @ L.T
            
        # Convert correlated standard normals (Z) to uniforms (U)
        U = norm.cdf(Z) 
        
        # Convert uniforms to correlated Bernoulli outcomes
        I_k = (U < M).astype(int) 

        # 2. Define the Objective Function (Unchanged)
        def objective(F: np.ndarray) -> float:
            # Calculate returns for long and short positions
            # We add epsilon to Q to prevent division by zero if Q=0 or Q=1
            Q_safe = np.clip(Q, 1e-9, 1.0 - 1e-9)
            gains_long = (I_k - Q_safe) / Q_safe
            gains_short = (Q_safe - I_k) / (1.0 - Q_safe)
            
            R_k_matrix = np.where(F > 0, gains_long, gains_short)
            portfolio_returns = np.sum(R_k_matrix * np.abs(F), axis=1)
            W_k = 1.0 + portfolio_returns
            
            if np.any(W_k <= 1e-9): # Bankruptcy
                return 1e9 # Massive penalty
            return -np.mean(np.log(W_k)) # Minimize negative log-wealth

        # 3. Run the Optimizer (Unchanged)
        initial_guess = F_analytical_guess
        constraints = ({'type': 'ineq', 'fun': lambda F: 0.8 - np.sum(np.abs(F))})
        bounds = [(-0.5, 0.5)] * n
        
        result = opt.minimize(
            objective, initial_guess, method='SLSQP',
            bounds=bounds, constraints=constraints, options={'ftol': 1e-6, 'disp': False}
        )
        
        if not result.success:
            log.warning(f"Numerical solver failed: {result.message}. Falling back.")
            return initial_guess
        
        log.info(f"Numerical solver converged. Final E[log(W)] = {-result.fun:.6f}")
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
# ### COMPONENT 7: Back-Testing & Tuning (Production-Ready) ###
# ==============================================================================

class BacktestPortfolio:
    """
    (Production-Ready C7.sub) Helper class for C7.
    Tracks cash, positions, P&L, and simulates frictions.
    """
    def __init__(self, initial_cash=10000.0, fee_pct=0.01, slippage_pct=0.005):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Tuple[float, float]] = {} 
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.pnl_history = [initial_cash]
        self.brier_scores = []
        self.start_time = None
        self.end_time = None

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        position_value = 0.0
        for contract_id, (fraction, entry_price) in self.positions.items():
            current_price = current_prices.get(contract_id, entry_price)
            pos_value_mult = 0.0
            if fraction > 0: # Long
                if entry_price > 1e-9: pos_value_mult = current_price / entry_price
            else: # Short
                if (1.0 - entry_price) > 1e-9: pos_value_mult = (1.0 - current_price) / (1.0 - entry_price)
            position_value += (abs(fraction) * self.initial_cash) * pos_value_mult
        return self.cash + position_value

    def rebalance(self, target_basket: Dict[str, float], current_prices: Dict[str, float]):
        all_contracts = set(target_basket.keys()) | set(self.positions.keys())
        for contract_id in all_contracts:
            target_fraction = target_basket.get(contract_id, 0.0)
            current_fraction, entry_price = self.positions.get(contract_id, (0.0, 0.0))
            trade_fraction = target_fraction - current_fraction
            
            if abs(trade_fraction) < 1e-5: continue
            
            trade_value = abs(trade_fraction) * self.initial_cash
            trade_price = current_prices.get(contract_id)
            if trade_price is None:
                log.warning(f"No price for {contract_id} in rebalance. Skipping trade.")
                continue

            fees = trade_value * self.fee_pct
            slippage_cost = trade_value * self.slippage_pct
            self.cash -= (fees + slippage_cost)
            
            if trade_fraction > 0: # BUYING
                self.cash -= trade_value
                if current_fraction >= 0: # Increasing long
                    new_avg_price = ((current_fraction * entry_price) + (trade_fraction * trade_price)) / target_fraction if target_fraction != 0 else trade_price
                    self.positions[contract_id] = (target_fraction, new_avg_price)
                else: # Closing short
                    self.positions[contract_id] = (target_fraction, trade_price) 
            else: # SELLING
                self.cash += trade_value
                if abs(target_fraction) < 1e-5: # Full exit
                    if contract_id in self.positions: del self.positions[contract_id]
                else:
                    if current_fraction == 0.0: # Opening new short
                        self.positions[contract_id] = (target_fraction, trade_price)
                    else: 
                        self.positions[contract_id] = (target_fraction, entry_price)
    
    def handle_resolution(self, contract_id: str, outcome: float, p_model: float, current_prices: Dict):
        if contract_id in self.positions:
            fraction, entry_price = self.positions.pop(contract_id)
            bet_value = abs(fraction) * self.initial_cash
            
            if fraction > 0: # Long
                payout = bet_value * (outcome / entry_price) if entry_price > 1e-9 else 0.0
            else: # Short
                payout = bet_value * ((1.0 - outcome) / (1.0 - entry_price)) if (1.0 - entry_price) > 1e-9 else 0.0
            self.cash += payout
        
        self.pnl_history.append(self.get_total_value(current_prices))
        if p_model is not None:
            self.brier_scores.append((p_model - outcome)**2)

    def get_final_metrics(self) -> Dict[str, float]:
        pnl = np.array(self.pnl_history)
        returns = (pnl[1:] - pnl[:-1]) / pnl[:-1]
        if len(returns) == 0: returns = np.array([0])
        final_pnl = pnl[-1]
        initial_pnl = self.initial_cash
        
        try:
            total_days = (self.end_time.date() - self.start_time.date()).days
            if total_days == 0: total_days = 1
            total_return = (final_pnl / initial_pnl) - 1.0
            irr = ((1.0 + total_return) ** (365.0 / total_days)) - 1.0
        except Exception:
            irr = (final_pnl / initial_pnl) - 1.0
            
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        max_drawdown = np.max((np.maximum.accumulate(pnl) - pnl) / np.maximum.accumulate(pnl)) if len(pnl) > 0 else 0.0
        avg_brier = np.mean(self.brier_scores) if self.brier_scores else 0.25
        
        return {'irr': irr, 'sharpe_ratio': sharpe, 'max_drawdown': max_drawdown, 'brier_score': avg_brier}


class BacktestEngine:
    """
    (Production-Ready C7)
    Loads and transforms real Polymarket data and runs
    the full C1-C6 pipeline replay to find optimal parameters.
    """
    def __init__(self, historical_data_path: str):
        log.info("BacktestEngine (C7) Production initialized.")
        self.historical_data_path = historical_data_path # Useful for caching
        self.cache_dir = Path(self.historical_data_path) / "dune_cache" # <-- ADD THIS
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dune_api_key = os.getenv("DUNE_API_KEY")
        
 #       self.dune_api_key = os.getenv("DUNE_API_KEY")
 #       if not self.dune_api_key:
#            log.error("DUNE_API_KEY environment variable not set. C7 will fail.")
#            self.dune_client = None
#        else:
#            self.dune_client = DuneClient(self.dune_api_key)
            
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR)


    def _load_data_from_dune(self, start_date: str, end_date: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads and transforms Polymarket data from Dune Analytics
        using pre-defined query IDs and daily caching.
        
        Note: The start_date and end_date params are no longer used to *filter*
        Dune queries, but are kept for consistency in the API. The pre-defined
        queries are assumed to fetch all necessary data.
        """
    #    if not self.dune_client:
    #        log.error("Dune client not initialized. Cannot load data.")
    #        return pd.DataFrame(), pd.DataFrame()

        # Define Query IDs from your spec
        MARKET_DETAILS_QUERY_ID = 6175624
        TRADES_QUERY_ID = 6213459
        # --- Query 1: Markets ---
        log.info(f"Fetching market details (Query ID: {MARKET_DETAILS_QUERY_ID})...")
        df_markets = self._get_cached_dune_result(MARKET_DETAILS_QUERY_ID)
        
        if df_markets.empty:
            log.error("Failed to fetch markets from Dune. Aborting.")
            return pd.DataFrame(), pd.DataFrame()

        # --- Query 2: Trades ---
        log.info(f"Fetching all trades (Query ID: {TRADES_QUERY_ID})...")
        df_trades = self._get_cached_dune_result(TRADES_QUERY_ID)
        
        if df_trades.empty:
            log.warning("No trades found from any Dune query.")
            df_trades = pd.DataFrame(columns=['market_id', 'timestamp', 'price', 'size', 'maker_address', 'taker_address', 'outcome'])

        # --- Data Type Coercion (Same as original) ---
        try:
            # --- NEW: Rename columns to match script's expectations ---
            log.info("Renaming columns from Dune to match internal schema...")
            
            # Market Details Mappings
            market_rename_map = {
                'condition_id': 'market_id',
                # 'question' column already matches
                'market_start_time': 'created_at',
                'resolved_on_timestamp': 'resolution_timestamp'
                # 'outcome' column already matches
            }
            df_markets = df_markets.rename(columns=market_rename_map)

            # Check for 'start_price', which was in the old query.
            if 'start_price' not in df_markets.columns:
                log.warning("Market query missing 'start_price'. Defaulting to 0.50.")
                # We can't know the true start price, so we'll use a neutral default.
                # The transform logic expects this column.
                df_markets['start_price'] = 0.50

            if not df_trades.empty:
                # Trades Mappings
                trade_rename_map = {
                    'condition_id': 'market_id',
                    'block_time': 'timestamp',
                    # 'price' column already matches
                    'amount': 'size', # Map 'amount' to 'size'
                    'maker': 'maker_address',
                    'taker': 'taker_address'
                }
                df_trades = df_trades.rename(columns=trade_rename_map)

            # --- Type Coercion (uses new names) ---
            df_markets['resolution_timestamp'] = pd.to_datetime(df_markets['resolution_timestamp'], errors='coerce')
            df_markets['created_at'] = pd.to_datetime(df_markets['created_at'], errors='coerce') # Coerce errors for market_start_time
            df_markets['start_price'] = pd.to_numeric(df_markets['start_price'], errors='coerce')
            df_markets['outcome'] = pd.to_numeric(df_markets['outcome'], errors='coerce')
            
            if not df_trades.empty:
                df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
                df_trades['price'] = pd.to_numeric(df_trades['price'], errors='coerce')
                df_trades['size'] = pd.to_numeric(df_trades['size'], errors='coerce')
                # The old query normalized size: "size" / 1e6 AS "size"
                # Your new query returns 'amount' as 'double'.
                # If this 'amount' column (now 'size') is *not* normalized,
                # you MUST uncomment the line below:
                # df_trades['size'] = df_trades['size'] / 1e6
            
            # Drop any rows where key data failed to parse (using renamed columns)
            df_markets = df_markets.dropna(subset=['market_id', 'question', 'created_at', 'outcome', 'start_price'])
            if not df_trades.empty:
                df_trades = df_trades.dropna(subset=['market_id', 'timestamp', 'price', 'size', 'maker_address', 'taker_address'])
            else:
                # Create empty df with expected columns for _transform_data_to_event_log
                df_trades = pd.DataFrame(columns=['market_id', 'timestamp', 'price', 'size', 'maker_address', 'taker_address', 'outcome'])


        except KeyError as e:
            log.error(f"Column mismatch from Dune query: {e}. Check your query results.")
            log.error("Market columns:" + str(df_markets.columns))
            if not df_trades.empty: log.error("Trades columns:" + str(df_trades.columns))
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            log.error(f"Failed to parse data types from Dune: {e}")
            return pd.DataFrame(), pd.DataFrame()

        log.info(f"Loaded {len(df_markets)} markets and {len(df_trades)} trades from Dune (using cache).")
        return df_markets, df_trades

    def _get_cached_dune_result(self, query_id: int) -> pd.DataFrame:
        """
        Fetches a Dune query result by ID, using a daily cache.
        Cleans up old cache files for this query ID.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"dune_query_{query_id}_{today}.pkl"
        
        # Clean up old cache files for this query_id
        for old_file in self.cache_dir.glob(f"dune_query_{query_id}_*.pkl"):
            if old_file.name != cache_file.name:
                log.info(f"Removing old cache file: {old_file}")
                try:
                    old_file.unlink()
                except OSError as e:
                    log.warning(f"Could not delete old cache file {old_file}: {e}")

        # Check for today's cache file
        if cache_file.exists():
            log.info(f"Loading cached result for query {query_id} from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                log.warning(f"Failed to load cache file {cache_file}: {e}. Refetching.")
        
        # If no cache, fetch from Dune
      #  if not self.dune_client:
      #      log.error("Dune client not initialized. Cannot fetch data.")
      #      return pd.DataFrame()
            
      #  log.info(f"Fetching new result for query {query_id} from Dune...")
        if not self.dune_api_key:
             log.error("Dune API key not set. Cannot fetch data.")
             return pd.DataFrame()

        log.info(f"Fetching new result for query {query_id} from Dune...")
        
        try:
            # Construct the URL and headers as requested
             all_dfs = []
             limit = 100000  # As requested
             offset = 0
             headers = {"x-dune-api-key": self.dune_api_key}
             
             
             base_url = f"https://api.dune.com/api/v1/query/{query_id}/results"

             while True:
                 paginated_url = f"{base_url}?limit={limit}&offset={offset}"
                 log.info(f"Fetching chunk: offset={offset}, limit={limit}")
                 
                 # Make the direct API call
                 response = requests.get(paginated_url, headers=headers)
                 response.raise_for_status() # Raise an exception for bad status codes
                 
                 json_response = response.json()
 
                 # Extract the rows from the JSON response
                 if "result" not in json_response or "rows" not in json_response["result"]:
                     log.warning(f"Dune query {query_id} (offset {offset}) returned unexpected JSON structure.")
                     break # Stop if something is wrong
                 
                 rows = json_response["result"]["rows"]
                 if not rows:
                     log.info("No more rows returned. Pagination complete.")
                     break # This is the exit condition
                
                 all_dfs.append(pd.DataFrame(rows))
                
                 # Increment offset for the next loop
                 offset += limit
 
             if not all_dfs:
                 log.warning(f"No data fetched for query {query_id}.")
                 df = pd.DataFrame()
             else:
                 df = pd.concat(all_dfs, ignore_index=True)
                 log.info(f"Pagination complete. Fetched {len(df)} total rows.")
            
            # Save to cache
             with open(cache_file, 'wb') as f:
                 pickle.dump(df, f)
             log.info(f"Saved new cache file: {cache_file}")
             return df
            
        except Exception as e:
            log.error(f"Failed to process Dune response for {query_id}: {e}", exc_info=True)
            return pd.DataFrame()

    def _transform_data_to_event_log(self, df_markets, df_trades) -> (pd.DataFrame, pd.DataFrame):
        """
        This is the "Transform" phase.
        It creates two crucial DataFrames:
        1. profiler_data: Used to *pre-train* the C4 HistoricalProfiler.
        2. event_log: The time-series log for the C7 replay harness.
        """
        log.info("Transforming raw data into event log...")
        
        # --- 1. Prepare Market Data (for lookups) ---
        df_markets['resolution_timestamp'] = pd.to_datetime(df_markets['resolution_timestamp'], errors='coerce')
        df_markets['created_at'] = pd.to_datetime(df_markets['created_at'])
        # Create a simple lookup for outcome
        market_outcomes = df_markets.set_index('market_id')['outcome'].to_dict()
        market_questions = df_markets.set_index('market_id')['question'].to_dict()
        market_vectors = {mid: [0.1]*768 for mid in market_outcomes} # Mock embeddings

        # --- 2. Create the Profiler DataFrame (for C4) ---
        # We need all *resolved* trades to build Brier scores.
        log.info("Building profiler data...")
        df_trades['outcome'] = df_trades['market_id'].map(market_outcomes)
        # We must use *both* maker and taker wallets
        trades_maker = df_trades[['maker_address', 'price', 'outcome', 'market_id']].rename(columns={'maker_address': 'wallet_id', 'price': 'bet_price'})
        trades_taker = df_trades[['taker_address', 'price', 'outcome', 'market_id']].rename(columns={'taker_address': 'wallet_id', 'price': 'bet_price'})
        # We'll just assign a 'default' topic for now
        profiler_data = pd.concat([trades_maker, trades_taker]).dropna(subset=['outcome', 'bet_price'])
        profiler_data['entity_type'] = 'default_topic'
        
        # --- 3. Create the Event Log (for C7) ---
        log.info("Building event log...")
        events = []
        
        # a) Create NEW_CONTRACT events
        for _, row in df_markets.iterrows():
            events.append((
                row['created_at'],
                'NEW_CONTRACT',
                {
                    'id': row['market_id'],
                    'text': row['question'],
                    'vector': market_vectors[row['market_id']],
                    'liquidity': 0, # We don't have this, so we'll mock it
                    'p_market_all': row['start_price']
                }
            ))
            
        # b) Create RESOLUTION events
        for _, row in df_markets.dropna(subset=['resolution_timestamp']).iterrows():
            events.append((
                row['resolution_timestamp'],
                'RESOLUTION',
                {'id': row['market_id'], 'outcome': row['outcome']}
            ))
            
        # c) Create PRICE_UPDATE events from trades
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        for _, row in df_trades.iterrows():
            # Add Taker
            events.append((
                row['timestamp'],
                'PRICE_UPDATE',
                {
                    'id': row['market_id'],
                    'p_market_all': row['price'],
                    # We pass the trade data itself for C4 to use
                    'wallet_id': row['taker_address'],
                    'price': row['price'],
                    'volume': row['size'] * row['price'] # (size * price)
                }
            ))
            # Add Maker
            events.append((
                row['timestamp'],
                'PRICE_UPDATE',
                {
                    'id': row['market_id'],
                    'p_market_all': row['price'],
                    'wallet_id': row['maker_address'],
                    'price': row['price'],
                    'volume': row['size'] * row['price']
                }
            ))

        # --- 4. Sort and return the final event log ---
        # --- THIS LINE IS THE CRITICAL FIX FROM YOUR ORIGINAL FILE ---
        event_log = pd.DataFrame(events, columns=['timestamp', 'event_type', 'data'])
        event_log['contract_id'] = event_log['data'].apply(lambda x: x.get('id'))
        event_log['timestamp'] = pd.to_datetime(event_log['timestamp'])
        event_log = event_log.set_index('timestamp').sort_index()
        
        log.info(f"ETL complete. {len(profiler_data)} trades for profiler, {len(event_log)} total events.")
        return event_log, profiler_data

    @staticmethod
    def _run_single_backtest(config: Dict[str, Any], historical_data: pd.DataFrame, profiler_data: pd.DataFrame):
        """
        This is the "objective" function that Ray Tune will optimize.
        It runs one *REAL* C1-C6 pipeline simulation.
        (This function is unchanged from your original file)
        """
        log.debug(f"--- C7: Starting back-test run with config: {config} ---")
        try:
            # 1. Initialize all components *with this run's config*
            graph = GraphManager(is_mock=True) # Mocks the DB
            
            graph.model_brier_scores = {
                'brier_internal_model': config['brier_internal_model'],
                'brier_expert_model': 0.05, 'brier_crowd_model': 0.15,
            }
            
            # --- Instantiate REAL Pipeline ---
            linker = RelationalLinker(graph)
            ai_analyst = AIAnalyst()
            
            # --- ** NEW: Pre-train the Profiler ** ---
            # We must *first* train the profiler on all historical trades
            # so the LiveFeedHandler has Brier scores to use.
            profiler = HistoricalProfiler(graph, min_trades_threshold=config.get('min_trades_threshold', 5))
            # We pass the *real* profiler_data to the (mocked) GraphManager
            graph.mock_db['profiler_data'] = profiler_data 
            profiler.run_profiling() # This will populate the mock_db['wallets']
            
            live_feed = LiveFeedHandler(graph)
            prior_manager = PriorManager(graph, ai_analyst, live_feed)
            belief_engine = BeliefEngine(graph)
            belief_engine.k_brier_scale = config['k_brier_scale']
            
            kelly_solver = HybridKellySolver(
                analytical_edge_threshold=config['kelly_edge_thresh'],
                num_samples_k=2000 
            )
            pm = PortfolioManager(graph, kelly_solver)
            
            # 2. Initialize the simulation portfolio
            portfolio = BacktestPortfolio()
            portfolio.start_time = historical_data.index.min()
            portfolio.end_time = historical_data.index.max()
            
            current_prices = {} # {contract_id: price}
            
            # 3. --- The Replay Loop ---
            for timestamp, events in historical_data.groupby(historical_data.index):
                
                # --- A. Process all non-trade events first ---
                for _, event in events.iterrows():
                    data = event['data']
                    event_type = event['event_type']
                    contract_id = event['contract_id']
                    
                    if event_type == 'NEW_CONTRACT':
                        log.debug(f"Event: NEW_CONTRACT {contract_id}")
                        graph.add_contract(data['id'], data['text'], data['vector'], data['liquidity'], data['p_market_all'])
                        current_prices[contract_id] = data['p_market_all']
                        linker.process_pending_contracts()
                        prior_manager.process_pending_contracts()
                    
                    elif event_type == 'RESOLUTION':
                        log.debug(f"Event: RESOLUTION {contract_id}")
                        p_model = graph.mock_db['contracts'].get(contract_id, {}).get('p_model', 0.5)
                        portfolio.handle_resolution(contract_id, data['outcome'], p_model, current_prices)
                        current_prices.pop(contract_id, None)
                        graph.update_contract_status(contract_id, 'RESOLVED', {'outcome': data['outcome']})

                # --- B. Process price updates & rebalance ---
                price_updates = {e['contract_id']: e['data'] for _, e in events.iterrows() if e['event_type'] == 'PRICE_UPDATE'}
                if price_updates:
                    log.debug(f"Event: PRICE_UPDATE {list(price_updates.keys())}")
                    
                    for c_id, data in price_updates.items():
                        current_prices[c_id] = data['p_market_all']
                        if c_id in graph.mock_db['contracts']:
                            # C1/C4: Add this trade to the graph mock_db
                            # This simulates the ingestor updating the price and C4 seeing the trade
                            graph.mock_db['contracts'][c_id]['p_market_all'] = data['p_market_all']
                            # Add the trade to the mock_db for C4 to find
                            # (We create a list of all trades in this batch)
                            graph.mock_db['live_trades'] = [d['data'] for _, d in events.iterrows() if d['event_type'] == 'PRICE_UPDATE' and d['contract_id'] == c_id]
                            graph.update_contract_status(c_id, 'PENDING_ANALYSIS') # Re-trigger
                    
                    prior_manager.process_pending_contracts() 
                    belief_engine.run_fusion_process()
                    target_basket = pm.run_optimization_cycle()
                    portfolio.rebalance(target_basket, current_prices)
            
            # 4. Get final metrics
            metrics = portfolio.get_final_metrics()
            
            # 5. Report to Ray Tune
            tune.report(metrics)
            
        except Exception as e:
            log.error(f"Back-test run failed: {e}", exc_info=True)
            tune.report({'irr': -1.0, 'brier': 1.0, 'sharpe': -10.0})
            
    def run_tuning_job(self):
        """Main entry point for Component 7."""
        log.info("--- C7: Starting Hyperparameter Tuning Job ---")
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR)
        
        # --- THIS IS THE NEW ETL STEP ---
        # We load data from Dune *once* before starting the tuning job.
        try:
            # --- Define your backtest window ---
            end_date_dt = datetime.now()
            start_date_dt = end_date_dt - timedelta(days=90) # e.g., 90-day backtest
            
            end_date_str = end_date_dt.strftime('%Y-%m-%d %H:%M:%S')
            start_date_str = start_date_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            log.info(f"C7: Loading data from Dune for window: {start_date_str} to {end_date_str}")
            
            # --- Call the NEW function ---
            df_markets, df_trades = self._load_data_from_dune(start_date_str, end_date_str)
            
            if df_markets.empty or df_trades.empty:
                log.error("No data loaded from Dune. Aborting tuning job.")
                return None
                
            event_log, profiler_data = self._transform_data_to_event_log(df_markets, df_trades)
        
        except Exception as e:
            log.error(f"FATAL: Failed to load or transform Dune data: {e}", exc_info=True)
            return None
        
        # "Curry" the real data into the objective function
        trainable_with_data = tune.with_parameters(
            self._run_single_backtest,
            historical_data=event_log,
            profiler_data=profiler_data
        )
        
        search_space = {
            "brier_internal_model": tune.loguniform(0.05, 0.25),
            "k_brier_scale": tune.loguniform(0.1, 5.0),
            "kelly_edge_thresh": tune.uniform(0.05, 0.25),
            "min_trades_threshold": tune.qrandint(5, 50, 5)
        }
        
        scheduler = ASHAScheduler(metric="irr", mode="max", max_t=10, grace_period=1, reduction_factor=2)
        
        analysis = tune.run(
            trainable_with_data,
            config=search_space,
            num_samples=20, # Reduced for demo
            scheduler=scheduler,
            resources_per_trial={"cpu": 1},
            name="pm_tuning_job"
        )
        
        best_config = analysis.get_best_config(metric="irr", mode="max")
        
        log.info(f"--- C7: Tuning Job Complete ---")
        log.info(f"Best config found for max IRR:")
        log.info(best_config)
        
        ray.shutdown()
        return best_config

    def run_tuning_job_async(self):
        """Launches the tuning job in a separate process."""
        log.info("--- C7: Spawning asynchronous tuning job... ---")
        
        def run_job():
            if not ray.is_initialized():
                ray.init(logging_level=logging.ERROR)
            
            # --- This constructor is now simpler ---
            backtester = BacktestEngine(
                historical_data_path="." # (Path is now relative)
            )
            best_config = backtester.run_tuning_job()
            log.info(f"--- C7: Async Tuning Job Complete. Best config: {best_config} ---")
            ray.shutdown()

        p = multiprocessing.Process(target=run_job)
        p.start()
        log.info(f"--- C7: Job process started with PID {p.pid} ---")
        return p.pid
        
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
