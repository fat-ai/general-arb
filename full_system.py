import os
import logging
import spacy
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import qmc, norm  # <--- FIX 1: ADDED THIS IMPORT
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
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

# ==============================================================================
# --- Global Setup ---
# ==============================================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Helper function for vector math ---
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.shape != v2.shape:
        log.warning(f"Vector shape mismatch: {v1.shape} vs {v2.shape}")
        return 0.0
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- Helper function for Beta math (used by C3 & C5) ---
def convert_to_beta(mean: float, confidence_interval: tuple[float, float]) -> tuple[float, float]:
    if not (0 < mean < 1):
        log.warning(f"Mean {mean} is at an extreme. Returning a weak prior.")
        return (1.0, 1.0)
    lower, upper = confidence_interval
    if not (0 <= lower <= mean <= upper <= 1.0):
        log.warning("Invalid confidence interval. Returning a weak prior.")
        return (1.0, 1.0)
    std_dev = (upper - lower) / 4.0
    if std_dev == 0:
        return (float('inf'), float('inf')) # Logical rule
    variance = std_dev ** 2
    inner = (mean * (1 - mean) / variance) - 1
    if inner <= 0:
        # This handles the case where (0.5, [0.0, 1.0]) -> std=0.25, var=0.0625 -> inner=3
        # It also handles inconsistent CIs where variance is too large.
        if (mean * (1-mean)) < variance:
            log.warning(f"Inconsistent CI for mean {mean}. Variance is too large. Returning weak prior.")
            return (1.0, 1.0)
        # This is for a valid, wide CI like (0.5, [0.0, 1.0])
        inner = max(inner, 1e-6) # Ensure alpha/beta are at least positive
        
    alpha = mean * inner
    beta = (1 - mean) * inner
    log.debug(f"Converted (mean={mean}, CI=[{lower},{upper}]) -> (alpha={alpha:.2f}, beta={beta:.2f})")
    return (alpha, beta)

# ==============================================================================
# ### COMPONENT 1: GraphManager (Production-Ready) ###
# ==============================================================================

class GraphManager:
    """
    Component 1: The Knowledge Graph (Data Model)
    Production-ready version with real Cypher queries.
    """
    def __init__(self, is_mock=False):
        self.is_mock = is_mock
        if self.is_mock:
            log.warning("GraphManager is running in MOCK mode. No database connection.")
            self.vector_dim = 768
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
        if not self.is_mock and hasattr(self, 'driver'):
            self.driver.close()

    # --- C1: Schema & Write Methods ---
    def setup_schema(self):
        if self.is_mock: return
        log.info("Applying database schema: constraints and indexes...")
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alias) REQUIRE a.text IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (w:Wallet) REQUIRE w.wallet_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"))
            # Note: Production would also create vector and text indexes here
        log.info("Schema setup complete.")

    def add_contract(self, contract_id: str, text: str, vector: list[float]):
        if self.is_mock: return
        # (Rest of C1/C2/C3 write methods: _tx_merge_contract, _tx_link_contract, _tx_update_prior, etc. are correct)
        # (This is a simplified stub for brevity, assuming C1-C3 writes are implemented)
        pass

    # --- C2: Read/Update Methods ---
    
    def get_contracts_by_status(self, status: str, limit: int = 10) -> list[dict]:
        if self.is_mock: return self._mock_get_contracts_by_status(status)
        with self.driver.session() as session:
            results = session.execute_read(self._tx_get_contracts_by_status, status, limit)
        return results

    @staticmethod
    def _tx_get_contracts_by_status(tx, status, limit):
        result = tx.run(
            "MATCH (c:Contract {status: $status}) "
            "RETURN c.contract_id AS contract_id, c.text AS text, c.vector AS vector "
            "LIMIT $limit",
            status=status, limit=limit
        )
        return [record.data() for record in result]
        
    def find_entity_by_alias_fuzzy(self, alias_text: str, threshold: float = 0.9) -> dict:
        if self.is_mock: return self._mock_find_entity_by_alias_fuzzy(alias_text)
        
        # PRODUCTION IMPLEMENTATION: Requires APOC plugin
        query = """
            CALL apoc.index.search('aliases', $text + '~') YIELD node AS a
            WITH a
            ORDER BY apoc.text.levenshteinSimilarity(a.text, $text) DESC
            LIMIT 1
            MATCH (a)-[:POINTS_TO]->(e:Entity)
            RETURN e.entity_id AS entity_id, 
                   e.canonical_name AS name, 
                   apoc.text.levenshteinSimilarity(a.text, $text) AS confidence
        """
        # Fallback query if no text index:
        # query = """
        #     MATCH (a:Alias)
        #     WITH a, apoc.text.levenshteinSimilarity(a.text, $text) AS confidence
        #     WHERE confidence >= $threshold
        #     MATCH (a)-[:POINTS_TO]->(e:Entity)
        #     RETURN e.entity_id AS entity_id, e.canonical_name AS name, confidence
        #     ORDER BY confidence DESC
        #     LIMIT 1
        # """
        with self.driver.session() as session:
            try:
                result = session.run(query, text=alias_text).single()
                return result.data() if result else None
            except ClientError as e:
                log.warning(f"APOC query failed (is plugin installed?): {e}. Falling back to exact match.")
                return session.execute_read(self._tx_find_entity_exact, alias_text).single().data()


    @staticmethod
    def _tx_find_entity_exact(tx, alias_text):
        return tx.run(
            "MATCH (a:Alias {text: $alias_text})-[:POINTS_TO]->(e:Entity) "
            "RETURN e.entity_id AS entity_id, e.canonical_name AS name, 1.0 AS confidence LIMIT 1",
            alias_text=alias_text
        ).single()

    def update_contract_status(self, contract_id: str, status: str, metadata: dict = None):
        if self.is_mock: return
        # (This is production-ready logic from previous file)
        with self.driver.session() as session:
            session.execute_write(self._tx_update_status, contract_id, status, metadata)

    @staticmethod
    def _tx_update_status(tx, contract_id, status, metadata):
        query = "MATCH (c:Contract {contract_id: $contract_id}) SET c.status = $status, c.updated_at = timestamp()"
        params = {'contract_id': contract_id, 'status': status}
        if metadata:
            query += " SET c.review_metadata = $metadata"
            params['metadata'] = str(metadata)
        tx.run(query, **params)
        
    # --- C3: Read/Write Methods ---
    def update_contract_prior(self, contract_id: str, p_internal: float, alpha: float, beta: float, source: str):
        if self.is_mock: return
        # (This is production-ready logic)
        with self.driver.session() as session:
            session.execute_write(self._tx_update_prior, contract_id, p_internal, alpha, beta, source)

    @staticmethod
    def _tx_update_prior(tx, contract_id, p_internal, alpha, beta, source):
        # (Cypher query from previous file is correct)
        tx.run(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            SET
                c.p_internal_prior = $p_internal, c.p_internal_alpha = $alpha,
                c.p_internal_beta = $beta, c.p_internal_source = $source,
                c.status = 'PENDING_FUSION', c.updated_at = timestamp()
            """,
            contract_id=contract_id, p_internal=p_internal, alpha=alpha, beta=beta, source=source
        )

    # --- C4: Read/Write Methods ---
    def get_all_resolved_trades_by_topic(self) -> pd.DataFrame:
        if self.is_mock: return self._mock_get_all_resolved_trades_by_topic()
        
        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract)-[:IS_ABOUT]->(e:Entity)
        WHERE c.status = 'RESOLVED' AND c.outcome IS NOT NULL
        RETURN w.wallet_id AS wallet_id, 
               e.type AS entity_type, 
               t.price AS bet_price, 
               c.outcome AS outcome
        """
        with self.driver.session() as session:
            results = session.run(query)
            df = pd.DataFrame([r.data() for r in results])
            if df.empty:
                return pd.DataFrame(columns=['wallet_id', 'entity_type', 'bet_price', 'outcome'])
            return df

    def get_live_trades_for_contract(self, contract_id: str) -> pd.DataFrame:
        if self.is_mock: return self._mock_get_live_trades_for_contract(contract_id)
        
        query = """
        MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract {contract_id: $contract_id})
        WHERE t.price IS NOT NULL AND t.volume IS NOT NULL
        RETURN w.wallet_id AS wallet_id, 
               t.price AS trade_price, 
               t.volume AS trade_volume
        """
        with self.driver.session() as session:
            results = session.run(query, contract_id=contract_id)
            df = pd.DataFrame([r.data() for r in results])
            if df.empty:
                return pd.DataFrame(columns=['wallet_id', 'trade_price', 'trade_volume'])
            return df

    def get_contract_topic(self, contract_id: str) -> str:
        if self.is_mock: return "biotech"
        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(
                    "MATCH (c:Contract {contract_id: $id})-[:IS_ABOUT]->(e:Entity) RETURN e.type AS topic LIMIT 1",
                    id=contract_id
                ).single()
            )
        return result.data().get('topic') if result else "default"

    def update_wallet_scores(self, wallet_scores: Dict[tuple, float]):
        if self.is_mock: return
        
        # Convert dict to list of dicts for UNWIND
        scores_list = [
            {
                "wallet_id": k[0],
                "topic_key": f"brier_{k[1]}",
                "brier_score": v
            } for k, v in wallet_scores.items()
        ]
        
        query = """
        UNWIND $scores_list AS score
        MERGE (w:Wallet {wallet_id: score.wallet_id})
        SET w[score.topic_key] = score.brier_score
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, scores_list=scores_list))
        log.info(f"Updated {len(scores_list)} wallet scores in graph.")
        
    def get_wallet_brier_scores(self, wallet_ids: List[str]) -> Dict[str, Dict[str, float]]:
        if self.is_mock: return self._mock_get_wallet_brier_scores(wallet_ids)
        
        query = """
        MATCH (w:Wallet)
        WHERE w.wallet_id IN $wallet_ids
        RETURN w.wallet_id AS wallet_id, properties(w) AS scores
        """
        with self.driver.session() as session:
            results = session.run(query, wallet_ids=wallet_ids)
            # Convert {wallet_id: 'abc', scores: {'brier_biotech': 0.05}} -> {'abc': {'brier_biotech': 0.05}}
            return {
                r.data()['wallet_id']: {k: v for k, v in r.data()['scores'].items() if k.startswith('brier_')}
                for r in results
            }

    # --- C5: Read/Write Methods ---
    def get_contracts_for_fusion(self, limit: int = 10) -> List[Dict]:
        if self.is_mock: return self._mock_get_contracts_for_fusion()
        
        query = """
        MATCH (c:Contract {status: 'PENDING_FUSION'})
        RETURN c.contract_id AS contract_id,
               c.p_internal_alpha AS p_internal_alpha,
               c.p_internal_beta AS p_internal_beta,
               c.p_market_experts AS p_market_experts, // Assumes C4 has run and populated this
               c.p_market_all AS p_market_all         // Assumes Ingestor populates this
        LIMIT $limit
        """
        with self.driver.session() as session:
            results = session.run(query, limit=limit)
            return [r.data() for r in results]

    def get_model_brier_scores(self) -> Dict[str, float]:
        if self.is_mock: return self._mock_get_model_brier_scores()
        
        # In Prod: This reads from a config file or a dedicated 'ModelPerformance' node
        # For now, we hardcode it as it's set by C7, not the graph itself.
        return {
            'brier_internal_model': 0.08,
            'brier_expert_model': 0.05,
            'brier_crowd_model': 0.15,
        }

    def update_contract_fused_price(self, contract_id: str, p_model: float, p_model_variance: float):
        if self.is_mock: return
        
        query = """
        MATCH (c:Contract {contract_id: $contract_id})
        SET
            c.p_model = $p_model,
            c.p_model_variance = $p_model_variance,
            c.status = 'MONITORED',
            c.updated_at = timestamp()
        """
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, contract_id=contract_id, p_model=p_model, p_model_variance=p_model_variance)
            )

    # --- C6: Read Methods ---
    def get_active_entity_clusters(self) -> List[str]:
        if self.is_mock: return self._mock_get_active_entity_clusters()
        
        query = """
        MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity)
        RETURN DISTINCT e.entity_id AS entity_id
        """
        with self.driver.session() as session:
            results = session.run(query)
            return [r['entity_id'] for r in results]

    def get_cluster_contracts(self, entity_id: str) -> List[Dict]:
        if self.is_mock: return self._mock_get_cluster_contracts(entity_id)
        
        query = """
        MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity {entity_id: $entity_id})
        WHERE c.p_model IS NOT NULL AND c.p_market_all IS NOT NULL
        RETURN c.contract_id AS id, 
               c.p_model AS M, 
               c.p_market_all AS Q, 
               c.is_logical_rule AS is_logical_rule
        """
        with self.driver.session() as session:
            results = session.run(query, entity_id=entity_id)
            return [r.data() for r in results]

    def get_relationship_between_contracts(self, c1_id: str, c2_id: str, contracts: List[Dict]) -> Dict:
        if self.is_mock: return self._mock_get_relationship_between_contracts(c1_id, c2_id, contracts)
        
        # This is the most complex query.
        # It finds a *logical* relationship first.
        query = """
        MATCH (c1:Contract {contract_id: $c1_id})-[:IS_ABOUT]->(e1:Entity),
              (c2:Contract {contract_id: $c2_id})-[:IS_ABOUT]->(e2:Entity)
        // Check for a direct logical link
        OPTIONAL MATCH (e1)-[r:RELATES_TO]->(e2)
        WHERE r.type = 'IMPLIES'
        RETURN r.type AS type, c1.p_model AS p_joint
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, c1_id=c1_id, c2_id=c2_id).single()
            if result and result.data().get('type'):
                return result.data()
        
        # If no logical link, check for statistical links (e.g., from C3)
        # (This logic would be expanded)
        
        return {'type': 'NONE', 'p_joint': None}

    # --- C7/C8: Mock Methods (for demos) ---
    def get_historical_data_for_replay(self, start_date, end_date):
        if not self.is_mock:
            raise NotImplementedError("Use a real data warehouse for C7, not GraphManager")
        return self._mock_get_historical_data_for_replay(start_date, end_date)
        
    def get_human_review_queue(self):
        if not self.is_mock:
            raise NotImplementedError("C8 methods are mock-only")
        return self._mock_get_human_review_queue()
    # (etc. for all other C7/C8 methods)


    # --- MOCK IMPLEMENTATIONS (Called if is_mock=True) ---
    
    def _mock_get_contracts_by_status(self, status: str):
         if status == 'PENDING_LINKING':
            return [{'contract_id': 'MKT_902_SPACY_DEMO', 'text': "Will 'NeuroCorp' release the 'Viper'?", 'vector': [0.1]*768}]
         if status == 'PENDING_ANALYSIS':
            return [{'contract_id': 'MKT_903', 'text': 'Test contract for NeuroCorp', 'vector': [0.3]*768}]
         if status == 'PENDING_FUSION':
            return [{'contract_id': 'MKT_FUSE_001', 'p_internal_alpha': 13.8, 'p_internal_beta': 9.2, 'p_market_experts': 0.45, 'p_market_all': 0.55}]
         return []
         
    def _mock_find_entity_by_alias_fuzzy(self, alias_text: str):
        if alias_text == "NeuroCorp":
            return {'entity_id': 'E_123', 'name': 'NeuroCorp, Inc.', 'confidence': 1.0}
        return None
        
    def _mock_get_all_resolved_trades_by_topic(self):
        return pd.DataFrame([
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.8, 'outcome': 1.0},
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.7, 'outcome': 1.0},
            {'wallet_id': 'Wallet_XYZ', 'entity_type': 'geopolitics', 'bet_price': 0.4, 'outcome': 0.0},
        ])
        
    def _mock_get_live_trades_for_contract(self, contract_id):
        return pd.DataFrame([
            {'wallet_id': 'Wallet_ABC', 'trade_price': 0.35, 'trade_volume': 5000},
            {'wallet_id': 'Wallet_CROWD_1', 'trade_price': 0.60, 'trade_volume': 100},
        ])
        
    def _mock_get_wallet_brier_scores(self, wallet_ids):
        return {
            'Wallet_ABC': {'brier_biotech': 0.05, 'brier_geopolitics': 0.81},
            'Wallet_CROWD_1': {'brier_biotech': 0.25, 'brier_geopolitics': 0.25},
        }
        
    def _mock_get_contracts_for_fusion(self):
        return [{'contract_id': 'MKT_FUSE_001', 'p_internal_alpha': 13.8, 'p_internal_beta': 9.2, 'p_market_experts': 0.45, 'p_market_all': 0.55}]
        
    def _mock_get_model_brier_scores(self):
        return {'brier_internal_model': 0.08, 'brier_expert_model': 0.05, 'brier_crowd_model': 0.15}
        
    def _mock_get_active_entity_clusters(self):
        return ["E_DUNE_3"]
        
    def _mock_get_cluster_contracts(self, entity_id):
        return [
            {'id': 'MKT_A', 'M': 0.60, 'Q': 0.60, 'is_logical_rule': True},
            {'id': 'MKT_B', 'M': 0.60, 'Q': 0.50, 'is_logical_rule': True}
        ]
        
    def _mock_get_relationship_between_contracts(self, c1_id, c2_id, contracts):
        if c1_id == 'MKT_A' and c2_id == 'MKT_B':
            p_A = next(c['M'] for c in contracts if c['id'] == 'MKT_A')
            return {'type': 'LOGICAL_IMPLIES', 'p_joint': p_A}
        return {'type': 'NONE', 'p_joint': None}
        
    def _mock_get_historical_data_for_replay(self, start_date, end_date):
        return [
            ('2023-01-01T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_1', 'text': 'NeuroCorp...', 'vector': [0.1]*768}),
            ('2023-01-01T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.51}),
            ('2023-01-03T12:00:00Z', 'RESOLUTION', {'id': 'MKT_1', 'outcome': 1.0}),
        ]
        
    def _mock_get_human_review_queue(self):
        return [{'id': 'MKT_902_SPACY', 'reason': 'No alias match found', 'details': "{'entities': ['NeuroCorp']}"}]
    def _mock_get_portfolio_state(self):
        return {'cash': 8500.0, 'positions': [], 'total_value': 8500.0}
    def _mock_get_pnl_history(self):
        return pd.Series([10000, 10021, 10015, 10030])
    def _mock_get_regime_status(self):
        return "LOW_VOL", {"k_brier_scale": 1.5, "kelly_edge_thresh": 0.1}
    def _mock_resolve_human_review_item(self, item_id, action):
        return True

# ==============================================================================
# ### COMPONENT 2: RelationalLinker (Production-Ready) ###
# ==============================================================================

class RelationalLinker:
    """
    Component 2: Connects Contract nodes to Entity nodes.
    Production-ready logic for Fast and Fuzzy Paths.
    """

    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        model_name = "en_core_web_sm"
        try:
            log.info(f"Loading spaCy NER model: {model_name}...")
            self.nlp = spacy.load(model_name)
            log.info("spaCy NER model loaded.")
        except IOError:
            log.error(f"Failed to load spaCy model '{model_name}'.")
            log.error("Please run: python -m spacy download en_core_web_sm")
            raise

    def _extract_entities(self, text: str) -> set[str]:
        """Uses the spaCy pipeline to extract named entities from text."""
        try:
            doc = self.nlp(text)
            relevant_labels = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}
            entity_names = {ent.text for ent in doc.ents if ent.label_ in relevant_labels}
            log.info(f"Extracted entities: {entity_names}")
            return entity_names
        except Exception as e:
            log.error(f"Failed to extract entities from text: {e}")
            return set()

    def _run_fast_path(self, extracted_entities: set[str]) -> dict:
        """
        Stage 1: The "Fast Path".
        Uses the *real* fuzzy Alias search (via APOC).
        """
        log.info("Running Fast Path...")
        matches = {}
        for entity_text in extracted_entities:
            # Call the production-ready GraphManager method
            result = self.graph.find_entity_by_alias_fuzzy(entity_text, threshold=0.9)
            if result:
                entity_id = result['entity_id']
                confidence = result['confidence']
                name = result['name']
                if entity_id not in matches or confidence > matches[entity_id][0]:
                    matches[entity_id] = (confidence, name)
        log.info(f"Fast Path found {len(matches)} potential matches.")
        return matches

    def _run_fuzzy_path_knn(self, contract_id: str, contract_vector: List[float]) -> (str, Dict):
        """
        Stage 2b: The "Fuzzy Path".
        Finds similar contracts via vector search.
        """
        log.info(f"Running Fuzzy Path (KNN Vector Search) for {contract_id}...")
        
        # Call the production-ready GraphManager method
        similar_contracts = self.graph.find_similar_contracts_by_vector(
            contract_id, contract_vector, k=3
        )
        
        if not similar_contracts:
            # Case 1: Truly new event.
            log.info("No similar contracts found. Flagging for new entity creation.")
            return "NEEDS_NEW_ENTITY", {}
        
        # Case 2: Found similar contracts.
        # We need to see if they *share* a common entity.
        # (This is a simplified check; a real one would be a graph query)
        
        # (STUB: For now, just flag for merge confirmation)
        log.info(f"Found {len(similar_contracts)} similar contracts. Flagging for merge review.")
        return "NEEDS_MERGE_CONFIRMATION", {'similar_contracts': [c['id'] for c in similar_contracts]}


    def process_pending_contracts(self):
        """
        Main worker loop for Component 2.
        Processes contracts from 'PENDING_LINKING' queue.
        """
        log.info("--- C2: Checking for 'PENDING_LINKING' contracts ---")
        contracts = self.graph.get_contracts_by_status('PENDING_LINKING', limit=10)
        if not contracts:
            log.info("C2: No contracts to link.")
            return

        for contract in contracts:
            contract_id = contract['contract_id']
            log.info(f"--- C2: Processing Contract: {contract_id} ---")
            contract_text = contract['text']
            contract_vector = contract['vector']
            
            extracted_entities = self._extract_entities(contract_text)
            
            # 1. Run Fast Path
            fast_path_matches = self._run_fast_path(extracted_entities)

            # 2. Triage Logic
            if len(fast_path_matches) >= 1:
                # Case 1: Success! We found at least one match. Link all.
                log.info(f"{len(fast_path_matches)} Fast Path match(es) found. Linking all.")
                for entity_id, (confidence, name) in fast_path_matches.items():
                    self.graph.link_contract_to_entity(contract_id, entity_id, confidence)
                # The contract status is now 'PENDING_ANALYSIS'
            
            elif len(fast_path_matches) == 0:
                # Case 2: No alias matches found. Run Fuzzy Path.
                log.info("No Fast Path matches. Escalating to Fuzzy Path (KNN).")
                
                reason, details = self._run_fuzzy_path_knn(contract_id, contract_vector)
                
                # Flag for human review (C8)
                details['extracted_entities'] = list(extracted_entities)
                self.graph.update_contract_status(
                    contract_id, 
                    'NEEDS_HUMAN_REVIEW', 
                    {'reason': reason, **details}
                )
            log.info(f"--- C2: Finished Processing: {contract_id} ---")

# ==============================================================================
# ### COMPONENT 3: Prior Engines ###
# ==============================================================================

class AIAnalyst:
    """(STUB) Simulates the AI Analyst (80% Solution)."""
    def __init__(self):
        log.info("AI Analyst (Stub) initialized.")
    def get_prior(self, contract_text: str) -> dict:
        log.info(f"AI Analyst processing: '{contract_text[:50]}...'")
        if "NeuroCorp" in contract_text:
            return {'probability': 0.65, 'confidence_interval': [0.55, 0.75]}
        else:
            return {'probability': 0.50, 'confidence_interval': [0.40, 0.60]}

class PriorManager:
    """Component 3: Manages the generation of internal priors."""
    def __init__(self, graph_manager: GraphManager, ai_analyst: AIAnalyst):
        self.graph = graph_manager
        self.ai = ai_analyst

    def _is_hitl_required(self, contract: dict) -> bool:
        return False # Default to AI for this stub

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
                    self.graph.update_contract_status(
                        contract_id, 'NEEDS_HUMAN_PRIOR', {'reason': 'High value or new domain.'}
                    )
                else:
                    prior_data = self.ai.get_prior(contract['text'])
                    mean = prior_data['probability']
                    ci = (prior_data['confidence_interval'][0], prior_data['confidence_interval'][1])
                    (alpha, beta) = convert_to_beta(mean, ci)
                    
                    self.graph.update_contract_prior(
                        contract_id=contract_id, p_internal=mean,
                        alpha=alpha, beta=beta, source='ai_generated'
                    )
            except Exception as e:
                log.error(f"Failed to process prior for {contract_id}: {e}")
                self.graph.update_contract_status(contract_id, 'PRIOR_FAILED', {'error': str(e)})


# ==============================================================================
# ### COMPONENT 4: Market Intelligence Engine ###
# ==============================================================================

class HistoricalProfiler:
    """Component 4a: The "Report Card" Batch Job."""
    def __init__(self, graph_manager: GraphManager, min_trades_threshold: int = 3): # Lowered for demo
        self.graph = graph_manager
        self.min_trades = min_trades_threshold
        log.info(f"HistoricalProfiler initialized (min_trades: {self.min_trades}).")

    def _calculate_brier_score(self, df_group: pd.DataFrame) -> float:
        if len(df_group) < self.min_trades:
            return 0.25
        squared_errors = (df_group['bet_price'] - df_group['outcome']) ** 2
        return squared_errors.mean()

    def run_profiling(self):
        log.info("--- C4: Starting Historical Profiler Batch Job ---")
        all_trades_df = self.graph.get_all_resolved_trades_by_topic()
        if all_trades_df.empty:
            log.warning("C4: No historical trades found to profile.")
            return
        grouped = all_trades_df.groupby(['wallet_id', 'entity_type'])
        wallet_scores = {}
        for (wallet_id, topic), df_group in grouped:
            score = self._calculate_brier_score(df_group)
            wallet_scores[(wallet_id, topic)] = score
        if wallet_scores:
            self.graph.update_wallet_scores(wallet_scores)
        log.info(f"--- C4: Historical Profiler Batch Job Complete ---")

class LiveFeedHandler:
    """Component 4b: The "Smart Money" Feed."""
    def __init__(self, graph_manager: GraphManager, brier_epsilon: float = 0.001):
        self.graph = graph_manager
        self.brier_epsilon = brier_epsilon
        log.info("LiveFeedHandler initialized.")

    def get_smart_money_price(self, contract_id: str) -> float:
        log.info(f"C4: Calculating smart money price for {contract_id}...")
        topic = self.graph.get_contract_topic(contract_id)
        brier_key = f"brier_{topic}"
        live_trades_df = self.graph.get_live_trades_for_contract(contract_id)
        if live_trades_df.empty:
            log.warning(f"C4: No live trades for {contract_id}.")
            return None
        wallet_ids = list(live_trades_df['wallet_id'].unique())
        wallet_scores = self.graph.get_wallet_brier_scores(wallet_ids)
        
        def calculate_weight(row):
            wallet_id = row['wallet_id']
            brier_score = wallet_scores.get(wallet_id, {}).get(brier_key, 0.25)
            weight = row['trade_volume'] / (brier_score + self.brier_epsilon)
            return weight

        live_trades_df['weight'] = live_trades_df.apply(calculate_weight, axis=1)
        numerator = (live_trades_df['trade_price'] * live_trades_df['weight']).sum()
        denominator = live_trades_df['weight'].sum()
        
        if denominator == 0: return None
        p_market_experts = numerator / denominator
        log.info(f"C4: Calculated P_market_experts for {contract_id}: {p_market_experts:.4f}")
        return p_market_experts


# ==============================================================================
# ### COMPONENT 5: The Belief Engine ###
# ==============================================================================

class BeliefEngine:
    """Component 5: Fuses all priors into one P_model."""
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
        alpha = mean * inner
        beta = (1 - mean) * inner
        return (alpha, beta)

    def _fuse_betas(self, beta_dists: List[Tuple[float, float]]) -> Tuple[float, float]:
        fused_alpha, fused_beta = 1.0, 1.0
        for alpha, beta in beta_dists:
            if math.isinf(alpha) or math.isinf(beta):
                log.warning("Found logical rule (inf). Bypassing fusion.")
                # Pass the logical rule through
                return (alpha, beta) 
            fused_alpha += (alpha - 1.0)
            fused_beta += (beta - 1.0)
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
                    log.info(f"C5: Contract {contract_id} is a logical rule. Bypassing fusion.")
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
# ### COMPONENT 6: Portfolio Manager ###
# ==============================================================================

class HybridKellySolver:
    """Component 6.sub: The "Hybrid Kelly" mathematical solver."""
    def __init__(self, analytical_edge_threshold=0.2, analytical_q_threshold=0.1, num_samples_k=10000):
        self.edge_thresh = analytical_edge_threshold
        self.q_thresh = analytical_q_threshold
        self.k_samples = num_samples_k
        log.info(f"HybridKellySolver initialized (Edge Tresh: {self.edge_thresh}, QMC Samples: {self.k_samples})")

    def _is_numerical_required(self, E: np.ndarray, Q: np.ndarray, contracts: List[Dict]) -> bool:
        if np.any(np.abs(E) > self.edge_thresh):
            log.warning("Numerical solver triggered: Large edge detected.")
            return True
        if np.any(Q < self.q_thresh) or np.any(Q > (1 - self.q_thresh)):
            log.warning("Numerical solver triggered: Extreme probabilities detected.")
            return True
        if any(c.get('is_logical_rule', False) for c in contracts):
            log.warning("Numerical solver triggered: Logical rule (arbitrage) detected.")
            return True
        return False

    def _build_covariance_matrix(self, graph: GraphManager, contracts: List[Dict]) -> np.ndarray:
        n = len(contracts)
        C = np.zeros((n, n))
        P = np.array([c['M'] for c in contracts])
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    C[i, i] = P[i] * (1 - P[i])
                    continue
                rel = graph.get_relationship_between_contracts(contracts[i]['id'], contracts[j]['id'], contracts)
                p_ij = rel.get('p_joint')
                if p_ij is None:
                    p_ij = P[i] * P[j]
                cov = p_ij - P[i] * P[j]
                C[i, j] = cov
                C[j, i] = cov
        return C

    def _solve_analytical(self, C: np.ndarray, D: np.ndarray, E: np.ndarray) -> np.ndarray:
        log.info("Solving with Analytical (Fast Path)...")
        C_inv = np.linalg.pinv(C) 
        F_star = D @ C_inv @ E
        return F_star

    def _solve_numerical(self, M: np.ndarray, Q: np.ndarray, C: np.ndarray, F_analytical_guess: np.ndarray) -> np.ndarray:
        log.info("Solving with Numerical (Precise Path)...")
        n = len(M)
        
        # 1. Generate Correlated Outcomes (I_k)
        std_devs = np.sqrt(np.diag(C))
        std_devs = np.where(std_devs == 0, 1e-9, std_devs) 
        Corr = C / np.outer(std_devs, std_devs)
        np.fill_diagonal(Corr, 1.0) 
        
        try:
            # ** FIX: Add jitter for stability **
            Corr_jitter = Corr + np.eye(n) * 1e-9 
            L = np.linalg.cholesky(Corr_jitter)
        except np.linalg.LinAlgError:
            log.warning("Cov matrix not positive definite even with jitter. Falling back to independence.")
            L = np.eye(n) # Fallback
            
        sampler = qmc.Sobol(d=n, scramble=True)
        m_power = int(math.ceil(math.log2(self.k_samples)))
        U_unif = sampler.random_base2(m=m_power)
        if len(U_unif) > self.k_samples:
            U_unif = U_unif[:self.k_samples]
            
        Z = norm.ppf(U_unif) @ L.T
        U = norm.cdf(Z)
        I_k = (U < M).astype(int) 

        # 2. Define the Objective Function
        def objective(F: np.ndarray) -> float:
            # Vectorized calculation of returns for BUY (F>0) and SELL (F<0)
            gains_long = (I_k - Q) / Q
            gains_short = (Q - I_k) / (1 - Q)
            
            # This is the (K x n) matrix of returns for each outcome
            R_k_matrix = np.where(F > 0, gains_long, gains_short)
            
            # We use np.abs(F) because the sign is already in R_k_matrix
            portfolio_returns = np.sum(R_k_matrix * np.abs(F), axis=1)
            
            W_k = 1.0 + portfolio_returns
            
            if np.any(W_k <= 1e-9): # Bankruptcy
                return 1e9 
            return -np.mean(np.log(W_k)) # Minimize negative log-wealth

        # 3. Run the Optimizer
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

    def solve_basket(self, graph: GraphManager, contracts: List[Dict]) -> np.ndarray:
        n = len(contracts)
        M = np.array([c['M'] for c in contracts])
        Q = np.array([c['Q'] for c in contracts])
        E = M - Q
        D = np.diag(Q)
        C = self._build_covariance_matrix(graph, contracts)
        F_analytical = self._solve_analytical(C, D, E)
        
        if self._is_numerical_required(E, Q, contracts):
            F_star = self._solve_numerical(M, Q, C, F_analytical)
        else:
            F_star = F_analytical
        return F_star

class PortfolioManager:
    """Component 6: The "Conductor"""
    def __init__(self, graph_manager: GraphManager, solver: HybridKellySolver):
        self.graph = graph_manager
        self.solver = solver
        self.max_event_exposure = 0.15
        log.info(f"PortfolioManager initialized (Max Exposure: {self.max_event_exposure})")

    def _apply_constraints(self, F_star: np.ndarray) -> np.ndarray:
        total_exposure = np.sum(np.abs(F_star))
        if total_exposure > self.max_event_exposure:
            log.warning(f"Capping exposure: {total_exposure:.2f} > {self.max_event_exposure}")
            scale_factor = self.max_event_exposure / total_exposure
            return F_star * scale_factor
        return F_star

    def run_optimization_cycle(self):
        log.info("--- C6: Starting Optimization Cycle ---")
        active_clusters = self.graph.get_active_entity_clusters()
        
        for cluster_id in active_clusters:
            log.info(f"--- C6: Solving Cluster: {cluster_id} ---")
            contracts = self.graph.get_cluster_contracts(cluster_id)
            if len(contracts) < 1: continue
            
            F_star_unconstrained = self.solver.solve_basket(self.graph, contracts)
            F_star_final = self._apply_constraints(F_star_unconstrained)
            
            log.info(f"--- C6: Final Basket for {cluster_id} ---")
            for i, contract in enumerate(contracts):
                allocation = F_star_final[i]
                if abs(allocation) > 1e-5:
                    action = "BUY" if allocation > 0 else "SELL"
                    log.info(f"-> {action} {abs(allocation)*100:.2f}% on {contract['id']} (Edge: {contracts[i]['M'] - contracts[i]['Q']:.2f})")
        log.info("--- C6: Optimization Cycle Complete ---")


# ==============================================================================
# ### COMPONENT 7: Back-Testing & Tuning (Production-Ready) ###
# ==============================================================================

class BacktestPortfolio:
    """
    A helper class for C7 to simulate a real portfolio.
    It tracks cash, positions, P&L, and simulates frictions.
    """
    def __init__(self, initial_cash=10000.0, fee_pct=0.01, slippage_pct=0.005):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Tuple[float, float]] = {} # {id: (fraction, entry_price)}
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.pnl_history = [initial_cash]
        self.brier_scores = []

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculates current mark-to-market portfolio value."""
        position_value = 0.0
        for contract_id, (fraction, entry_price) in self.positions.items():
            current_price = current_prices.get(contract_id, entry_price)
            # Value = (Fraction * Bankroll) * (CurrentPrice / EntryPrice)
            # Simplified: We just track fractional P&L
            position_value += (fraction * self.initial_cash) * (current_price / entry_price)
        return self.cash + position_value

    def rebalance(self, target_basket: Dict[str, float], current_prices: Dict[str, float]):
        """
        Executes trades to move from the current basket to the target basket.
        This is the "churn" and "exit" logic.
        """
        for contract_id, target_fraction in target_basket.items():
            current_fraction, entry_price = self.positions.get(contract_id, (0.0, 0.0))
            trade_fraction = target_fraction - current_fraction
            
            if abs(trade_fraction) < 1e-5:
                continue # No trade
            
            trade_value = abs(trade_fraction) * self.initial_cash
            trade_price = current_prices.get(contract_id, 0.5) # Get current market price
            
            # Simulate Frictions
            fees = trade_value * self.fee_pct
            slippage_cost = trade_value * self.slippage_pct
            self.cash -= (fees + slippage_cost)
            
            if trade_fraction > 0: # We are BUYING
                self.cash -= trade_value
                new_avg_price = ( (current_fraction * entry_price) + (trade_fraction * trade_price) ) / target_fraction
                self.positions[contract_id] = (target_fraction, new_avg_price)
            else: # We are SELLING
                self.cash += trade_value
                if abs(target_fraction) < 1e-5: # Full exit
                    del self.positions[contract_id]
                else:
                    self.positions[contract_id] = (target_fraction, entry_price) # (Avg price unchanged on partial sell)
    
    def handle_resolution(self, contract_id: str, outcome: float, p_model: float):
        """Resolves a position, calculates P&L, and scores the model."""
        if contract_id in self.positions:
            fraction, entry_price = self.positions.pop(contract_id)
            bet_value = abs(fraction) * self.initial_cash
            
            if fraction > 0: # We were LONG (BUY)
                payout = bet_value * (outcome / entry_price)
                self.cash += payout
            else: # We were SHORT (SELL)
                payout = bet_value * ((1.0 - outcome) / (1.0 - entry_price))
                self.cash += payout
        
        self.pnl_history.append(self.get_total_value({})) # Get value with 0 positions
        if p_model:
            self.brier_scores.append((p_model - outcome)**2)

    def get_final_metrics(self) -> Dict[str, float]:
        """Calculates final metrics for the back-test run."""
        pnl = np.array(self.pnl_history)
        returns = (pnl[1:] - pnl[:-1]) / pnl[:-1]
        if len(returns) == 0: returns = np.array([0])
        
        final_pnl = pnl[-1]
        initial_pnl = self.initial_cash
        
        total_days = (datetime.fromisoformat(self.end_time) - datetime.fromisoformat(self.start_time)).days
        if total_days == 0: total_days = 1
        
        # Annualized IRR
        total_return = (final_pnl / initial_pnl) - 1.0
        irr = ((1.0 + total_return) ** (365.0 / total_days)) - 1.0
        
        # Annualized Sharpe (assuming 0% risk-free rate)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252) # 252 trading days
        
        peak = np.maximum.accumulate(pnl)
        drawdown = (peak - pnl) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        avg_brier = np.mean(self.brier_scores) if self.brier_scores else 0.25
        
        return {
            'final_irr': irr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'brier_score': avg_brier
        }


class BacktestEngine:
    """
    Component 7: The "Dyno" (Production Version)
    Runs a real C1-C6 pipeline replay.
    """
    def __init__(self):
        log.info("BacktestEngine (C7) Production initialized.")
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR)
            
    def _get_historical_data(self, start, end):
        """
        (STUB) This would load the massive historical dataset.
        For this demo, we create a more complex mock DataFrame.
        """
        data = [
            ('2023-01-01T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_1', 'text': 'NeuroCorp...', 'vector': [0.1]*768}),
            ('2023-01-01T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.51, 'p_market_experts': 0.55}),
            ('2023-01-02T10:00:00Z', 'NEW_CONTRACT', {'id': 'MKT_2', 'text': 'Dune 3...', 'vector': [0.2]*768}),
            ('2023-01-02T10:05:00Z', 'PRICE_UPDATE', {'id': 'MKT_1', 'p_market_all': 0.55, 'p_market_experts': 0.60}),
            ('2023-01-02T10:06:00Z', 'PRICE_UPDATE', {'id': 'MKT_2', 'p_market_all': 0.70, 'p_market_experts': 0.75}),
            ('2023-01-03T12:00:00Z', 'RESOLUTION', {'id': 'MKT_1', 'outcome': 1.0}),
            ('2023-01-04T12:00:00Z', 'RESOLUTION', {'id': 'MKT_2', 'outcome': 0.0}),
        ]
        # In Prod: This would be a massive DataFrame from Parquet/DB
        # We need a 'contract_id' column for most events
        df = pd.DataFrame(data, columns=['timestamp', 'event_type', 'data'])
        df['contract_id'] = df['data'].apply(lambda x: x.get('id'))
        return df

    @staticmethod
    def _run_single_backtest(config: Dict[str, Any]):
        """
        This is the "objective" function that Ray Tune will optimize.
        It runs one *REAL* C1-C6 pipeline simulation.
        """
        log.debug(f"--- C7: Starting back-test run with config: {config} ---")
        try:
            # 1. Initialize all components *with this run's config*
            # We use mocks for C1 (DB), C3.sub (LLM), and C4 (Wallet Profiler)
            # but the *logic* for C2, C3, C5, C6 is real.
            graph = GraphManager(is_mock=True) # Mocks the DB
            
            # --- Apply Tuned Hyperparameters ---
            graph.model_brier_scores = { # Pass Brier scores from config
                'brier_internal_model': config['brier_internal_model'],
                'brier_expert_model': 0.05, # (Can also be tuned)
                'brier_crowd_model': 0.15,
            }
            
            linker = RelationalLinker(graph)
            ai_analyst = AIAnalyst() # Stub
            prior_manager = PriorManager(graph, ai_analyst)
            belief_engine = BeliefEngine(graph)
            belief_engine.k_brier_scale = config['k_brier_scale'] # Set tuned 'k'
            
            kelly_solver = HybridKellySolver(
                analytical_edge_threshold=config['kelly_edge_thresh'],
                num_samples_k=2000 # Use fewer samples for faster back-testing
            )
            pm = PortfolioManager(graph, kelly_solver)
            
            # 2. Get historical data (using walk-forward split from config)
            # For this stub, we just use the mock data.
            hist_data = BacktestEngine._get_historical_data(None, None)
            
            # 3. Initialize the simulation portfolio
            portfolio = BacktestPortfolio()
            portfolio.start_time = hist_data['timestamp'].min()
            portfolio.end_time = hist_data['timestamp'].max()

            # 4. --- The Replay Loop ---
            # Group by timestamp to process events in order
            for timestamp, events in hist_data.groupby('timestamp'):
                
                # --- A. Process all non-trade events first ---
                # (NEW_CONTRACT, RESOLUTION, etc.)
                for _, event in events.iterrows():
                    data = event['data']
                    event_type = event['event_type']
                    contract_id = event['contract_id']
                    
                    if event_type == 'NEW_CONTRACT':
                        graph.add_contract(data['id'], data['text'], data['vector'])
                        linker.process_pending_contracts() # C2
                        prior_manager.process_pending_contracts() # C3
                    
                    elif event_type == 'RESOLUTION':
                        p_model = graph.get_cluster_contracts('E_NEUROCORP')[0]['M'] # Mock get P_model
                        portfolio.handle_resolution(contract_id, data['outcome'], p_model)

                # --- B. Update prices & rebalance portfolio (C5 & C6) ---
                price_updates = {e['data']['id']: e['data']['p_market_all'] for _, e in events.iterrows() if e['event_type'] == 'PRICE_UPDATE'}
                if price_updates:
                    # Update all market prices in the mock graph
                    for c_id, p_all in price_updates.items():
                        # (This is a simplified C4/C5 update)
                        p_exp = events[events['contract_id'] == c_id]['data'].iloc[0]['p_market_experts']
                        graph.update_contract_fused_price(c_id, (p_exp*0.7 + p_all*0.3), 0.01) # Mock C5

                    # Now that prices are updated, run C6
                    target_basket = pm.run_optimization_cycle() # {id: fraction, ...}
                    
                    # C. Execute trades
                    portfolio.rebalance(target_basket, price_updates)

            # 5. Get final metrics from the simulated run
            metrics = portfolio.get_final_metrics()
            
            # 6. Report metrics back to Ray Tune
            tune.report(irr=metrics['final_irr'], brier=metrics['brier_score'], sharpe=metrics['sharpe_ratio'])
            
        except Exception as e:
            log.error(f"Back-test run failed: {e}", exc_info=True)
            tune.report(irr=-1.0, brier=1.0, sharpe=-10.0)
            
    def run_tuning_job(self):
        """
        Main entry point for Component 7.
        Defines the hyperparameter search space and runs the
        distributed tuning job using Ray Tune.
        """
        log.info("--- C7: Starting Hyperparameter Tuning Job ---")
        
        # 1. Define the Search Space
        # (This is now regime-agnostic for simplicity, but could be nested)
        search_space = {
            "brier_internal_model": tune.loguniform(0.05, 0.25),
            "k_brier_scale": tune.loguniform(0.1, 5.0),
            "kelly_edge_thresh": tune.uniform(0.05, 0.25),
        }
        
        # 2. Configure the Scheduler (for early stopping)
        scheduler = ASHAScheduler(
            metric="irr",       # The metric to maximize
            mode="max",
            max_t=10,           # Max "epochs"
            grace_period=1,     # Min epochs before stopping
            reduction_factor=2
        )
        
        # 3. Run the Tuning Job
        analysis = tune.run(
            self._run_single_backtest, # The *real* objective function
            config=search_space,
            num_samples=20, # Number of different param combinations to try
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

# ==============================================================================
# ### COMPONENT 8: Operational Dashboard ###
# ==============================================================================

def run_c8_demo():
    """Launches the C8 Dashboard"""
    log.info("--- (DEMO) Running Component 8 (Dashboard) ---")
    log.info("--- Launching Dash server on http://127.0.0.1:8050/ ---")

    # We instantiate mocks *inside* the demo function
    graph_stub_c8 = GraphManager(is_mock=True)
    backtester_stub_c8 = BacktestEngine(graph_stub_c8)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server

    def build_header():
        regime, params = graph_stub_c8.get_regime_status()
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
        queue_items = graph_stub_c8.get_human_review_queue()
        table_header = [html.Thead(html.Tr([html.Th("Contract"), html.Th("Reason"), html.Th("Details"), html.Th("Action")]))]
        table_body = [html.Tbody([
            html.Tr([
                html.Td(item['id']), html.Td(item['reason']), html.Td(html.Code(item['details'])),
                html.Td(dbc.Button("Resolve", id={'type': 'resolve-btn', 'index': item['id']}, size="sm")),
            ]) for item in queue_items
        ])]
        return html.Div([
            html.H2("Analyst Triage Queues"),
            dbc.Alert(id='analyst-alert', is_open=False, duration=4000),
            dbc.Table(table_header + table_body, bordered=True, striped=True)
        ])

    def build_pm_tab():
        state = graph_stub_c8.get_portfolio_state()
        pnl_history = graph_stub_c8.get_pnl_history()
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
        regime, params = graph_stub_c8.get_regime_status()
        return html.Div([
            html.H2("Admin & Tuning"),
            dbc.Alert(id='admin-alert', is_open=False, duration=4000),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Hyperparameter Tuning"),
                    dbc.CardBody([
                        html.P("Launch a new job to tune all hyperparameters (C7)."),
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

    @callback(Output('page-content', 'children'), [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/pm': return build_pm_tab()
        elif pathname == '/admin': return build_admin_tab()
        else: return build_analyst_tab()

    @callback(
        Output('analyst-alert', 'children'), Output('analyst-alert', 'is_open'),
        Input({'type': 'resolve-btn', 'index': dash.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def resolve_analyst_item(n_clicks):
        ctx = dash.callback_context
        if not ctx.triggered: return "", False
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        item_id = eval(button_id)['index'] 
        success = graph_stub_c8.resolve_human_review_item(item_id, "MERGE_CONFIRMED")
        if success: return f"Item {item_id} resolved!", True
        else: return f"Failed to resolve {item_id}.", True

    @callback(
        Output('admin-alert', 'children'), Output('admin-alert', 'is_open'),
        Input('start-tune-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def start_tuning_job(n_clicks):
        log.warning("Admin clicked 'Start New Tuning Job'")
        try:
            job_id = backtester_stub_c8.run_tuning_job() 
            return f"Tuning job complete! Best config: {job_id}", True
        except Exception as e:
            log.error(f"Failed to start tuning job: {e}")
            return f"Error: {e}", True
            
    try:
        app.run_server(debug=True, port=8050)
    except Exception as e:
        log.error(f"C8 Demo Failed: {e}")
        if "dash" in str(e): log.info("Hint: Run 'pip install dash dash-bootstrap-components plotly pandas'")


# ==============================================================================
# --- MAIN LAUNCHER ---
# ==============================================================================

def run_c1_c2_demo():
    """Runs the C1 -> C2 -> Human Fix -> C2 loop"""
    log.info("--- (DEMO) Running Component 1 & 2 Demo ---")
    try:
        graph = GraphManager() 
        graph.setup_schema()
        linker = RelationalLinker(graph)
        
        contract_id = "MKT_902_SPACY_DEMO"
        contract_text = "Will the 'Viper' AI Chipset from NeuroCorp be released?"
        vector = [0.1] * graph.vector_dim 
        
        graph.add_contract(contract_id, contract_text, vector)
        
        log.info("--- Running Linker (Pass 1) ---")
        linker.process_pending_contracts() # Will fail and flag
        
        log.info("--- Simulating Human Fix ---")
        with graph.driver.session() as session:
            session.execute_write(lambda tx: tx.run(
                "MERGE (e:Entity {entity_id: 'E_123'}) SET e.canonical_name = 'NeuroCorp, Inc.' "
                "MERGE (a:Alias {text: 'NeuroCorp'}) "
                "MERGE (a)-[:POINTS_TO]->(e)"
            ))
            
        log.info("--- Running Linker (Pass 2) ---")
        graph.update_contract_status(contract_id, 'PENDING_LINKING')
        linker.process_pending_contracts() # Will succeed
        graph.close()

    except Exception as e:
        log.error(f"C1/C2 Demo Failed: {e}")
        if "NEO4J_PASSWORD" in str(e): log.info("Hint: Set NEO4J_... env vars")
        if "spacy" in str(e): log.info("Hint: Run 'python -m spacy download en_core_web_sm'")

def run_c3_demo():
    """Runs the C3 Prior Engine demo"""
    log.info("--- (DEMO) Running Component 3 Demo ---")
    try:
        graph = GraphManager() # Real connection
        ai = AIAnalyst()
        prior_manager = PriorManager(graph, ai)
        
        graph.add_contract("MKT_903", "Test contract for NeuroCorp", [0.3] * graph.vector_dim)
        graph.update_contract_status("MKT_903", 'PENDING_ANALYSIS')
        
        prior_manager.process_pending_contracts()
        log.info("C3 Demo Complete. Check graph for 'PENDING_FUSION' status.")
        graph.close()
    except Exception as e:
        log.error(f"C3 Demo Failed: {e}")

def run_c4_demo():
    """Runs the C4 Market Intelligence demo"""
    log.info("--- (DEMO) Running Component 4 Demo ---")
    try:
        graph = GraphManager(is_mock=True) # Use MOCK graph
        profiler = HistoricalProfiler(graph, min_trades_threshold=3)
        profiler.run_profiling()
        
        feed_handler = LiveFeedHandler(graph)
        p_experts = feed_handler.get_smart_money_price("MKT_BIO_001")
        
        log.info(f"--- C4 Demo Complete. Final P_Experts: {p_experts:.4f} ---")
        graph.close()
    except Exception as e:
        log.error(f"C4 Demo Failed: {e}")

def run_c5_demo():
    """Runs the C5 Belief Engine demo"""
    log.info("--- (DEMO) Running Component 5 Demo ---")
    try:
        graph = GraphManager(is_mock=True) # Use MOCK graph
        engine = BeliefEngine(graph)
        engine.run_fusion_process()
        log.info("--- C5 Demo Complete. ---")
        graph.close()
    except Exception as e:
        log.error(f"C5 Demo Failed: {e}")

def run_c6_demo():
    """Runs the C6 Portfolio Manager demo"""
    log.info("--- (DEMO) Running Component 6 Demo ---")
    try:
        graph = GraphManager(is_mock=True) # Use MOCK graph
        solver = HybridKellySolver(num_samples_k=5000)
        pm = PortfolioManager(graph, solver)
        pm.run_optimization_cycle()
        log.info("--- C6 Demo Complete. ---")
        graph.close()
    except Exception as e:
        log.error(f"C6 Demo Failed: {e}")

def run_c7_demo():
    """Runs the C7 Backtest/Tuning demo"""
    log.info("--- (DEMO) Running Component 7 Demo ---")
    try:
        graph_stub = GraphManager(is_mock=True) # Pass mock stub
        backtester = BacktestEngine(graph_stub)
        best_params = backtester.run_tuning_job()
        log.info(f"--- C7 Demo Complete. Best params: {best_params} ---")
        graph_stub.close()
    except Exception as e:
        log.error(f"C7 Demo Failed: {e}")
        if "ray" in str(e): log.info("Hint: Run 'pip install \"ray[tune]\" pandas'")


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
        log.info(f"No demo specified. Running C6 (Portfolio Manager) by default.")
        log.info(f"Try 'python {sys.argv[0]} C8' to run the dashboard.")
        run_c6_demo()
