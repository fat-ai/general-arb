import os
import logging
import spacy  # Switched from transformers to spaCy
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import scipy.optimize as opt
from typing import Dict, List, Tuple
# Set up basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Helper function for vector math ---
def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two numpy vectors."""
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


# ==============================================================================
# REFINED COMPONENT 1: GraphManager (with C2/C3 Read/Write Methods)
# ==============================================================================

class GraphManager:
    """
    Component 1: The Knowledge Graph (Data Model)
    Now includes all necessary read/write methods for C2 and C3.
    """
    def __init__(self):
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
        if self.driver:
            self.driver.close()

    def setup_schema(self):
        log.info("Applying database schema: constraints and indexes...")
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alias) REQUIRE a.text IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"))
            # (Vector index creation stubbed for brevity)
        log.info("Schema setup complete.")


    def add_contract(self, contract_id: str, text: str, vector: list[float]):
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {len(vector)}")
        with self.driver.session() as session:
            session.execute_write(self._tx_merge_contract, contract_id, text, vector)
        log.info(f"Merged Contract: {contract_id}")

    @staticmethod
    def _tx_merge_contract(tx, contract_id, text, vector):
        tx.run(
            """
            MERGE (c:Contract {contract_id: $contract_id})
            ON CREATE SET c.text = $text, c.vector = $vector, c.status = 'PENDING_LINKING', c.created_at = timestamp()
            ON MATCH SET c.text = $text, c.vector = $vector, c.updated_at = timestamp()
            """,
            contract_id=contract_id, text=text, vector=vector
        )

    def link_contract_to_entity(self, contract_id: str, entity_id: str, confidence: float):
        with self.driver.session() as session:
            session.execute_write(self._tx_link_contract, contract_id, entity_id, confidence)
        log.info(f"Linked Contract '{contract_id}' -> Entity '{entity_id}' with conf: {confidence}")

    @staticmethod
    def _tx_link_contract(tx, contract_id, entity_id, confidence):
        tx.run(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (c)-[r:IS_ABOUT]->(e)
            ON CREATE SET r.confidence_score = $confidence, r.created_at = timestamp()
            ON MATCH SET r.confidence_score = $confidence
            SET c.status = 'PENDING_ANALYSIS' // Update status on link
            """,
            contract_id=contract_id, entity_id=entity_id, confidence=confidence
        )


    def get_contracts_by_status(self, status: str, limit: int = 10) -> list[dict]:
        """Gets a batch of contracts with a specific status."""
        with self.driver.session() as session:
            results = session.execute_read(
                self._tx_get_contracts_by_status, status=status, limit=limit
            )
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
        """
        STUB: Simulates a fuzzy search with an exact match.
        In Prod: This would use the APOC query from the C2 review.
        """
        with self.driver.session() as session:
            result = session.execute_read(
                self._tx_find_entity_exact, alias_text=alias_text
            ).single()
        return result.data() if result else None

    @staticmethod
    def _tx_find_entity_exact(tx, alias_text):
        """Stub for fuzzy search - just does an exact match."""
        return tx.run(
            """
            MATCH (a:Alias {text: $alias_text})-[:POINTS_TO]->(e:Entity)
            RETURN e.entity_id AS entity_id, e.canonical_name AS name, 1.0 AS confidence
            LIMIT 1
            """,
            alias_text=alias_text
        ).single()

    def update_contract_status(self, contract_id: str, status: str, metadata: dict = None):
        """Updates a contract's status, e.g., to 'NEEDS_HUMAN_REVIEW'."""
        with self.driver.session() as session:
            session.execute_write(
                self._tx_update_status,
                contract_id=contract_id, status=status, metadata=metadata
            )
        log.info(f"Updated status for {contract_id} to {status}")

    @staticmethod
    def _tx_update_status(tx, contract_id, status, metadata):
        query = "MATCH (c:Contract {contract_id: $contract_id}) SET c.status = $status, c.updated_at = timestamp()"
        params = {'contract_id': contract_id, 'status': status}
        if metadata:
            query += " SET c.review_metadata = $metadata"
            params['metadata'] = str(metadata) # Store metadata as string
        tx.run(query, **params)

    def update_contract_prior(self, contract_id: str, p_internal: float, alpha: float, beta: float, source: str):
        """Writes the output of Component 3 to the Contract node."""
        with self.driver.session() as session:
            session.execute_write(
                self._tx_update_prior,
                contract_id=contract_id, p_internal=p_internal,
                alpha=alpha, beta=beta, source=source
            )
        log.info(f"Updated prior for {contract_id} from {source}.")

    @staticmethod
    def _tx_update_prior(tx, contract_id, p_internal, alpha, beta, source):
        tx.run(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            SET
                c.p_internal_prior = $p_internal,
                c.p_internal_alpha = $alpha,
                c.p_internal_beta = $beta,
                c.p_internal_source = $source,
                c.status = 'PENDING_FUSION',
                c.updated_at = timestamp()
            """,
            contract_id=contract_id, p_internal=p_internal,
            alpha=alpha, beta=beta, source=source
        )

    def get_all_resolved_trades_by_topic(self) -> pd.DataFrame:
        """
        Fetches *all* historical, resolved trades and links them to
        their Entity 'type' (topic). This is the main data source
        for the Historical Profiler.
        
        In Prod: This would be a highly optimized query, possibly
        reading from a data warehouse or pre-aggregated tables.
        
        Returns:
            A Pandas DataFrame: [wallet_id, entity_type, bet_price, outcome]
        """
        log.info("Fetching all resolved trades by topic (STUB)")
        # STUB: This would be a massive Cypher query. We will mock the output.
        # MATCH (w:Wallet)-[t:TRADED_ON]->(c:Contract)-[:IS_ABOUT]->(e:Entity)
        # WHERE c.status = 'RESOLVED'
        # RETURN w.wallet_id AS wallet_id, 
        #        e.type AS entity_type, 
        #        t.price AS bet_price, 
        #        c.outcome AS outcome
        
        # Mock data for demonstration:
        mock_data = [
            # Wallet_ABC is a 'biotech' expert
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.8, 'outcome': 1.0},
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.7, 'outcome': 1.0},
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'biotech', 'bet_price': 0.2, 'outcome': 0.0},
            # Wallet_ABC is bad at 'geopolitics'
            {'wallet_id': 'Wallet_ABC', 'entity_type': 'geopolitics', 'bet_price': 0.9, 'outcome': 0.0},
            # Wallet_XYZ is a 'geopolitics' expert
            {'wallet_id': 'Wallet_XYZ', 'entity_type': 'geopolitics', 'bet_price': 0.4, 'outcome': 0.0},
            {'wallet_id': 'Wallet_XYZ', 'entity_type': 'geopolitics', 'bet_price': 0.3, 'outcome': 0.0},
            # Wallet_XYZ is bad at 'biotech'
            {'wallet_id': 'Wallet_XYZ', 'entity_type': 'biotech', 'bet_price': 0.1, 'outcome': 1.0},
        ]
        return pd.DataFrame(mock_data)

    def get_live_trades_for_contract(self, contract_id: str) -> pd.DataFrame:
        """
        Fetches all live trades (order book) for a given contract.
        
        Returns:
            A Pandas DataFrame: [wallet_id, trade_price, trade_volume]
        """
        log.info(f"Fetching live trades for {contract_id} (STUB)")
        # STUB: This would query a live trade feed API.
        mock_data = [
            # Wallet_ABC (biotech expert) is betting NO (low price)
            {'wallet_id': 'Wallet_ABC', 'trade_price': 0.35, 'trade_volume': 5000},
            # Wallet_XYZ (geopolitics expert) is betting YES (high price)
            {'wallet_id': 'Wallet_XYZ', 'trade_price': 0.65, 'trade_volume': 1000},
            # Crowd noise
            {'wallet_id': 'Wallet_CROWD_1', 'trade_price': 0.60, 'trade_volume': 100},
            {'wallet_id': 'Wallet_CROWD_2', 'trade_price': 0.61, 'trade_volume': 150},
        ]
        return pd.DataFrame(mock_data)

    def get_contract_topic(self, contract_id: str) -> str:
        """Finds the primary 'type' of the Entity a contract is about."""
        log.info(f"Fetching topic for {contract_id} (STUB)")
        # STUB: MATCH (c:Contract {contract_id: $id})-[:IS_ABOUT]->(e:Entity) RETURN e.type LIMIT 1
        # Mocking for this example:
        if contract_id == "MKT_BIO_001":
            return "biotech"
        return "geopolitics"

    def update_wallet_scores(self, wallet_scores: Dict[tuple, float]):
        """
        Writes the calculated Brier scores back to the graph.
        
        Args:
            wallet_scores (dict): A dict mapping (wallet_id, topic) -> brier_score
        """
        log.info(f"Updating {len(wallet_scores)} wallet scores in graph (STUB)")
        # In Prod: This would be a batched Cypher UNWIND query
        # UNWIND $scores AS score_data
        # MERGE (w:Wallet {wallet_id: score_data.wallet_id})
        # SET w[score_data.topic_key] = score_data.brier
        for (wallet_id, topic), score in wallet_scores.items():
            log.debug(f"MERGE (w:Wallet {{wallet_id: '{wallet_id}'}}) SET w.brier_{topic} = {score}")
        pass
        
    def get_wallet_brier_scores(self, wallet_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches the stored Brier scores for a list of wallets.
        
        Returns:
            A dict: { wallet_id -> {topic: brier_score, ...} }
        """
        log.info(f"Fetching Brier scores for {len(wallet_ids)} wallets (STUB)")
        # STUB: MATCH (w:Wallet) WHERE w.wallet_id IN $wallet_ids RETURN w
        # Mocking for this example:
        return {
            'Wallet_ABC': {'brier_biotech': 0.05, 'brier_geopolitics': 0.81},
            'Wallet_XYZ': {'brier_biotech': 0.49, 'brier_geopolitics': 0.01},
            'Wallet_CROWD_1': {'brier_biotech': 0.25, 'brier_geopolitics': 0.25}, # Default/uninformative
            'Wallet_CROWD_2': {'brier_biotech': 0.25, 'brier_geopolitics': 0.25},
        }

    def get_contracts_for_fusion(self, limit: int = 10) -> List[Dict]:
        """
        Gets a batch of contracts that are ready for fusion.
        These have a status of 'PENDING_FUSION' (set by C3)
        or are being actively monitored.
        """
        log.info("Fetching contracts 'PENDING_FUSION' (STUB)")
        # STUB: MATCH (c:Contract {status: 'PENDING_FUSION'}) RETURN c LIMIT $limit
        # Mocking for this example:
        mock_data = [
            {
                'contract_id': 'MKT_FUSE_001',
                'p_internal_alpha': 13.8,  # From C3 (e.g., 60% mean, 0.01 var)
                'p_internal_beta': 9.2,
                'p_market_experts': 0.45,  # From C4 (smart money)
                'p_market_all': 0.55,      # From Ingestor (crowd)
            }
        ]
        return mock_data

    def get_model_brier_scores(self) -> Dict[str, float]:
        """
        Fetches the system's own Brier scores for its models.
        In Prod: This would be read from a model performance DB.
        """
        log.info("Fetching model Brier scores (STUB)")
        # STUB: These scores are generated by the Back-testing (C7) module
        # and stored in a config or model registry.
        return {
            'brier_internal_model': 0.08, # Our AI/Human model's historical score
            'brier_expert_model': 0.05,   # The "smart money" model's historical score
            'brier_crowd_model': 0.15,    # The "crowd" model's historical score
        }

    def update_contract_fused_price(self, contract_id: str, p_model: float, p_model_variance: float):
        """
        Writes the final, fused P_model and variance to the graph,
        and sets the status to 'MONITORED' (ready for C6).
        """
        log.info(f"Updating {contract_id} with fused P_model={p_model:.4f}")
        # STUB:
        # MATCH (c:Contract {contract_id: $contract_id})
        # SET
        #   c.p_model = $p_model,
        #   c.p_model_variance = $p_model_variance,
        #   c.status = 'MONITORED',
        #   c.updated_at = timestamp()
        pass

    def get_active_entity_clusters(self) -> List[str]:
        """Finds all Entity clusters that have active, monitored bets."""
        log.info("GraphManager: Finding all active clusters (STUB)")
        # In Prod: MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity) RETURN DISTINCT e.entity_id
        return ["E_DUNE_3"] # Mock: one cluster for "Dune 3"

    def get_cluster_contracts(self, entity_id: str) -> List[Dict]:
        """
        Gets all 'MONITORED' contracts for a given Entity cluster.
        This provides the M (P_model) and Q (P_market_all) vectors.
        """
        log.info(f"GraphManager: Getting contracts for cluster {entity_id} (STUB)")
        # In Prod: MATCH (c:Contract {status:'MONITORED'})-[:IS_ABOUT]->(e:Entity {entity_id:$id})
        #          RETURN c.contract_id, c.p_model, c.p_market_all, c.is_logical_rule
        # MOCK: A risk-free arbitrage. B should be >= A.
        # The market (Q) is mispriced: P(B) < P(A)
        # Our model (M) from C5 would have corrected this: P_model(B) >= P_model(A)
        mock_contracts = [
            {
                'id': 'MKT_A', # Dune 3 > $1B
                'M': 0.60,     # P_model
                'Q': 0.60,     # P_market_all
                'is_logical_rule': True # This contract is part of a logical rule
            },
            {
                'id': 'MKT_B', # Dune 3 > $500M
                'M': 0.60,     # P_model (C5 corrected it to be >= A)
                'Q': 0.50,     # P_market_all (The mispricing!)
                'is_logical_rule': True
            }
        ]
        return mock_contracts

    def get_relationship_between_contracts(self, c1_id: str, c2_id: str) -> Dict:
        """
        Finds the correlation between two contracts. This is the key to C6.
        """
        # This is a complex query that traverses the graph.
        # In Prod: MATCH (c1)-[:IS_ABOUT]->(e1), (c2)-[:IS_ABOUT]->(e2) ... etc.
        # We mock the logic:
        if c1_id == 'MKT_A' and c2_id == 'MKT_B':
            # A -> IMPLIES -> B
            return {
                'type': 'LOGICAL_IMPLIES',
                'p_joint': 0.60 # P(A, B) = P(A) = 0.60 (from MKT_A's P_model)
            }
        # STUB: Default for unlinked bets
        return {'type': 'NONE', 'p_joint': None}

    def close(self): pass
        
# ==============================================================================
# REFINED COMPONENT 2: RelationalLinker (now using spaCy)
# ==============================================================================

class RelationalLinker:
    """
    Component 2: The Relational Linker (Refined with spaCy)
    Connects Contract nodes to Entity nodes using a multi-stage pipeline.
    """

    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        
        # Load a pre-trained spaCy model
        # This is small, fast, and loads locally without network calls (if installed).
        # To install: python -m spacy download en_core_web_sm
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
            # We only care about specific entity types
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
        Tries to find a high-confidence match for each extracted entity
        using the graph's fuzzy Alias search.
        """
        log.info("Running Fast Path...")
        matches = {}
        
        for entity_text in extracted_entities:
            # We use the (stubbed) fuzzy search from GraphManager
            result = self.graph.find_entity_by_alias_fuzzy(entity_text, threshold=0.9)
            
            if result:
                entity_id = result['entity_id']
                confidence = result['confidence']
                name = result['name']
                
                if entity_id not in matches or confidence > matches[entity_id][0]:
                    matches[entity_id] = (confidence, name)
        
        log.info(f"Fast Path found {len(matches)} potential matches.")
        return matches

    def _run_fuzzy_path_disambiguation(self, contract_vector: list[float], ambiguous_entities: list[dict]) -> str:
        """
        Stage 2a: Disambiguation using vector similarity.
        (Stubbed for brevity, logic remains the same)
        """
        log.info("Running Fuzzy Path (Disambiguation)...")
        # (Logic as before, finding best match by cosine similarity)
        # ...
        if not ambiguous_entities:
            return None
        return ambiguous_entities[0]['id'] # Stub: just return the first one

    def _flag_for_human_review(self, contract_id: str, reason: str, details: dict):
        """
        Stage 3: Flags a contract for human review (Component 8).
        """
        log.warning(f"FLAGGING FOR HUMAN REVIEW: {contract_id}")
        log.warning(f"Reason: {reason}")
        
        self.graph.update_contract_status(
            contract_id,
            'NEEDS_HUMAN_REVIEW',
            {'reason': reason, **details}
        )

    def process_new_contract(self, contract_id: str):
        """
        Main entry point for Component 2.
        Processes a single new contract, finds its links, and
        either links it automatically or flags it for a human.
        """
        log.info(f"--- Processing Contract: {contract_id} ---")
        
        contract = self.graph.get_contract_details(contract_id)
        if not contract:
            log.error(f"Contract {contract_id} not found in graph.")
            return
            
        contract_text = contract['text']
        contract_vector = contract['vector']

        extracted_entities = self._extract_entities(contract_text)
        if not extracted_entities:
            self._flag_for_human_review(contract_id, "No entities found", {})
            return

        fast_path_matches = self._run_fast_path(extracted_entities)

        if len(fast_path_matches) >= 1:
            # Case 1 or 2: Simple or multiple unambiguous matches. Link all.
            log.info(f"{len(fast_path_matches)} Fast Path match(es) found. Linking all.")
            for entity_id, (confidence, name) in fast_path_matches.items():
                self.graph.link_contract_to_entity(contract_id, entity_id, confidence)

        elif len(fast_path_matches) == 0:
            # Case 3: No "Fast Path" match.
            log.info("No Fast Path matches. Flagging for human review.")
            
            # TODO: Implement "Fuzzy Path" Stage 2b (New Entity Search)
            # This would call a new GraphManager method:
            # `similar_contracts = self.graph.find_similar_contracts_by_vector(contract_vector)`
            # and pass `similar_contracts` to the human review details.
            
            self._flag_for_human_review(
                contract_id, 
                "No alias match found", 
                {'extracted_entities': list(extracted_entities)}
            )

        log.info(f"--- Finished Processing: {contract_id} ---")


# ==============================================================================
# NEW COMPONENT 3: Prior Engines (Internal Belief)
# ==============================================================================

def convert_to_beta(mean: float, confidence_interval: tuple[float, float]) -> tuple[float, float]:
    """
    Converts a human-readable probability and confidence interval
    into the Alpha and Beta parameters of a Beta distribution.
    
    Args:
        mean (float): The estimated probability (e.g., 0.65).
        confidence_interval (tuple): The (lower, upper) bounds of a ~95% CI.

    Returns:
        tuple[float, float]: The (alpha, beta) parameters.
    """
    if not (0 < mean < 1):
        log.warning(f"Mean {mean} is at an extreme. Returning a weak prior.")
        return (1.0, 1.0) # Return Beta(1,1) - a uniform (uninformative) prior

    lower, upper = confidence_interval
    if not (0 <= lower <= mean <= upper <= 1.0):
        log.warning("Invalid confidence interval. Returning a weak prior.")
        return (1.0, 1.0)

    # Approx std_dev from 95% CI (4 std deviations total)
    std_dev = (upper - lower) / 4.0
    
    if std_dev == 0:
        # Infinite confidence (logical rule)
        log.info("Zero variance detected. Returning (inf, inf) as placeholder.")
        # In practice, the system should treat this as a fixed logical constraint
        return (float('inf'), float('inf'))

    variance = std_dev ** 2
    
    # This is the core math
    # variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    # mean = alpha / (alpha + beta)
    # We simplify: alpha = mean * n, beta = (1-mean) * n
    # n = (mean * (1-mean) / variance) - 1
    
    inner = (mean * (1 - mean) / variance) - 1
    
    if inner <= 0:
        # This means the stated variance is *larger* than a uniform distribution,
        # which is mathematically inconsistent for a Beta distribution.
        log.warning(f"Inconsistent CI for mean {mean}. Variance is too large. Returning weak prior.")
        return (1.0, 1.0)
        
    alpha = mean * inner
    beta = (1 - mean) * inner
    
    log.debug(f"Converted (mean={mean}, CI=[{lower},{upper}]) -> (alpha={alpha:.2f}, beta={beta:.2f})")
    return (alpha, beta)


class AIAnalyst:
    """
    (STUB) Simulates the AI Analyst (80% Solution).
    In production, this would wrap an OpenAI/Claude/etc. API call.
    """
    def __init__(self):
        log.info("AI Analyst (Stub) initialized.")
        # In Prod: self.client = OpenAI(api_key=...)

    def get_prior(self, contract_text: str) -> dict:
        """
        Simulates the 'superforecasting' prompt and returns a
        JSON-like response.
        """
        log.info(f"AI Analyst processing: '{contract_text[:50]}...'")
        
        # --- PROMPT (Conceptual) ---
        # "You are a panel of 5 expert analysts...
        # 1. Outside View: What is the base rate for this event type?
        # 2. Inside View: What are the specific pro/con factors?
        # 3. Synthesize: Output a JSON with 'probability' and 'confidence_interval' (95% CI)."
        
        # Mocking the AI's response based on the text
        if "NeuroCorp" in contract_text:
            mock_response = {
                'probability': 0.65,
                'confidence_interval': [0.55, 0.75],
                'reasoning': 'NeuroCorp has a strong track record, but chipset releases often slip one quarter.'
            }
        else:
            mock_response = {
                'probability': 0.50,
                'confidence_interval': [0.40, 0.60],
                'reasoning': 'Defaulting to a neutral stance due to lack of specific data.'
            }
            
        return mock_response


class PriorManager:
    """
    Component 3: Manages the generation of internal priors.
    """
    def __init__(self, graph_manager: GraphManager, ai_analyst: AIAnalyst):
        self.graph = graph_manager
        self.ai = ai_analyst
        self.hitl_liquidity_threshold = 10000 # Example: Flag any $10k+ market for a human

    def _is_hitl_required(self, contract: dict) -> bool:
        """
        Checks if a contract meets the criteria for a
        mandatory human review (the 20% solution).
        """
        # (Stub: In prod, we'd fetch liquidity from the ingestor)
        # if contract.get('expected_liquidity', 0) > self.hitl_liquidity_threshold:
        #    return True
            
        # (Stub: In prod, we'd check if it's a new, unknown domain)
        # if contract.get('domain_is_new', False):
        #    return True

        return False # Default to AI

    def process_pending_contracts(self):
        """
        Main worker loop for Component 3.
        Finds contracts 'PENDING_ANALYSIS' and generates a prior.
        """
        log.info("PriorManager: Checking for contracts 'PENDING_ANALYSIS'...")
        contracts = self.graph.get_contracts_by_status('PENDING_ANALYSIS', limit=10)
        
        if not contracts:
            log.info("PriorManager: No new contracts to analyze.")
            return

        for contract in contracts:
            contract_id = contract['contract_id']
            log.info(f"PriorManager: Processing {contract_id}")
            
            try:
                if self._is_hitl_required(contract):
                    # 1. Flag for Human
                    self.graph.update_contract_status(
                        contract_id,
                        'NEEDS_HUMAN_PRIOR',
                        {'reason': 'High value or new domain.'}
                    )
                else:
                    # 2. Use AI Analyst
                    prior_data = self.ai.get_prior(contract['text'])
                    
                    mean = prior_data['probability']
                    ci = (prior_data['confidence_interval'][0], prior_data['confidence_interval'][1])
                    
                    (alpha, beta) = convert_to_beta(mean, ci)
                    
                    if alpha == float('inf'):
                        log.warning(f"{contract_id} has a logical (inf) prior. Handling as fixed rule.")
                        # (Special handling for logical rules can be added here)
                    
                    # 3. Save to Graph
                    self.graph.update_contract_prior(
                        contract_id=contract_id,
                        p_internal=mean,
                        alpha=alpha,
                        beta=beta,
                        source='ai_generated'
                    )
                    
# ==============================================================================
# NEW COMPONENT 4: Market Intelligence Engine
# ==============================================================================

class HistoricalProfiler:
    """
    Component 4a: The "Report Card" Batch Job.
    Analyzes the *entire* trade history to build Brier score "report cards"
    for every wallet, per topic.
    """
    
    def __init__(self, graph_manager: GraphManager, min_trades_threshold: int = 20):
        self.graph = graph_manager
        self.min_trades = min_trades_threshold
        log.info(f"HistoricalProfiler initialized (min_trades: {self.min_trades}).")

    def _calculate_brier_score(self, df_group: pd.DataFrame) -> float:
        """Calculates the Brier score for a single wallet/topic group."""
        if len(df_group) < self.min_trades:
            return 0.25  # Default, uninformative score
            
        # Brier Score: (1/N) * sum( (bet_price - outcome)^2 )
        squared_errors = (df_group['bet_price'] - df_group['outcome']) ** 2
        brier_score = squared_errors.mean()
        return brier_score

    def run_profiling(self):
        """
        Main method for the batch job. Fetches all trades, calculates
        scores, and updates the graph.
        
        In Prod: This would be a scheduled Ray or Spark job.
        """
        log.info("--- Starting Historical Profiler Batch Job ---")
        
        # 1. Fetch all historical, resolved trades
        all_trades_df = self.graph.get_all_resolved_trades_by_topic()
        if all_trades_df.empty:
            log.warning("No historical trades found to profile.")
            return

        # 2. Group by wallet and topic
        grouped = all_trades_df.groupby(['wallet_id', 'entity_type'])
        
        wallet_scores = {} # (wallet_id, topic) -> brier_score
        
        # 3. Calculate Brier score for each group
        for (wallet_id, topic), df_group in grouped:
            score = self._calculate_brier_score(df_group)
            wallet_scores[(wallet_id, topic)] = score
            log.debug(f"Calculated score for ({wallet_id}, {topic}): {score:.4f}")

        # 4. Write these scores back to the Graph
        if wallet_scores:
            self.graph.update_wallet_scores(wallet_scores)
            
        log.info(f"--- Historical Profiler Batch Job Complete. Updated {len(wallet_scores)} scores. ---")


class LiveFeedHandler:
    """
    Component 4b: The "Smart Money" Feed.
    Uses the pre-calculated Brier scores to generate a real-time
    "smart money price" for an active contract.
    """

    def __init__(self, graph_manager: GraphManager, brier_epsilon: float = 0.001):
        self.graph = graph_manager
        # Epsilon prevents division by zero for a "perfect" Brier score of 0
        self.brier_epsilon = brier_epsilon
        log.info("LiveFeedHandler initialized.")

    def get_smart_money_price(self, contract_id: str) -> float:
        """
        Main method. Calculates the weighted average price
        based on *expert* trades.
        
        Returns:
            The 'P_market_experts' price.
        """
        log.info(f"Calculating smart money price for {contract_id}...")
        
        # 1. Get the topic for this contract
        topic = self.graph.get_contract_topic(contract_id)
        brier_key = f"brier_{topic}" # e.g., 'brier_biotech'

        # 2. Get all live trades for this contract
        live_trades_df = self.graph.get_live_trades_for_contract(contract_id)
        if live_trades_df.empty:
            log.warning(f"No live trades found for {contract_id}.")
            return None # Or return P_market_all as a fallback

        # 3. Get the Brier scores for all wallets in the live feed
        wallet_ids = list(live_trades_df['wallet_id'].unique())
        wallet_scores = self.graph.get_wallet_brier_scores(wallet_ids)

        # 4. Calculate the weight for each trade
        def calculate_weight(row):
            wallet_id = row['wallet_id']
            # Get the wallet's score *for this specific topic*.
            # Default to 0.25 (uninformative) if no score exists.
            brier_score = wallet_scores.get(wallet_id, {}).get(brier_key, 0.25)
            
            # Weight = Volume / (Brier Score + epsilon)
            # A low Brier (good) and high volume (conviction) = high weight
            weight = row['trade_volume'] / (brier_score + self.brier_epsilon)
            return weight

        live_trades_df['weight'] = live_trades_df.apply(calculate_weight, axis=1)

        # 5. Calculate the weighted average price
        # P_experts = sum(price * weight) / sum(weight)
        numerator = (live_trades_df['trade_price'] * live_trades_df['weight']).sum()
        denominator = live_trades_df['weight'].sum()
        
        if denominator == 0:
            log.warning(f"No valid weights for {contract_id}. Returning None.")
            return None

        p_market_experts = numerator / denominator
        log.info(f"Calculated P_market_experts for {contract_id}: {p_market_experts:.4f}")
        
        return p_market_experts
                    
            except Exception as e:
                log.error(f"Failed to process prior for {contract_id}: {e}")
                self.graph.update_contract_status(contract_id, 'PRIOR_FAILED', {'error': str(e)})

# ==============================================================================
# NEW COMPONENT 5: The Belief Engine (Synthesis)
# ==============================================================================

class BeliefEngine:
    """
    Component 5: The "Synthesizer"
    Fuses all priors (Internal, Expert, Crowd) into one "ultimate"
    fair price (P_model) using Bayesian Model Averaging.
    """

    def __init__(self, graph_manager: GraphManager):
        self.graph = graph_manager
        # This 'k' factor scales Brier scores to variance.
        # It is a key hyperparameter to be tuned by Component 7.
        self.k_brier_scale = float(os.getenv('BRIER_K_SCALE', 0.5))
        
        # Load the historical performance of our models
        self.model_brier_scores = self.graph.get_model_brier_scores()
        log.info(f"BeliefEngine initialized with k={self.k_brier_scale} and model scores.")

    def _impute_beta_from_point(self, mean: float, model_name: str) -> Tuple[float, float]:
        """
        Converts a point-price estimate (e.g., P_market_experts) into a
        Beta(α, β) distribution.
        
        The variance is *imputed* from the model's historical Brier score.
        """
        if not (0 < mean < 1):
            log.warning(f"Mean {mean} for {model_name} is at an extreme. Returning weak prior.")
            return (1.0, 1.0) # Uninformative prior
            
        brier_score = self.model_brier_scores.get(f'brier_{model_name}_model', 0.25)
        
        # Core Logic: variance = k * Brier_score
        # A high Brier (bad performance) = high variance (low confidence)
        variance = self.k_brier_scale * brier_score
        
        # Prevent division by zero or math errors
        if variance == 0:
            variance = 1e-9
        if variance >= (mean * (1 - mean)):
            # This implies the model is worse than random.
            # Clamp variance to just below the max possible (for Beta(1,1)).
            variance = (mean * (1 - mean)) - 1e-9

        # Convert (mean, variance) to (alpha, beta)
        # (This is the same math from Component 3)
        inner = (mean * (1 - mean) / variance) - 1
        
        if inner <= 0:
            log.warning(f"Inconsistent variance for {model_name} (mean={mean}, var={variance}). Returning weak prior.")
            return (1.0, 1.0)
            
        alpha = mean * inner
        beta = (1 - mean) * inner
        
        log.debug(f"Imputed Beta for {model_name}: (α={alpha:.2f}, β={beta:.2f})")
        return (alpha, beta)

    def _fuse_betas(self, beta_dists: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Fuses multiple Beta distributions by multiplying their PDFs.
        This is a conjugate update.
        
        We use: Beta_fused ~ Beta(1 + sum(α_i - 1), 1 + sum(β_i - 1))
        This assumes a uniform Beta(1,1) prior.
        """
        fused_alpha = 1.0
        fused_beta = 1.0
        
        for alpha, beta in beta_dists:
            fused_alpha += (alpha - 1.0)
            fused_beta += (beta - 1.0)
            
        # Ensure alpha/beta are at least 1 (to be a valid distribution)
        fused_alpha = max(fused_alpha, 1.0)
        fused_beta = max(fused_beta, 1.0)
        
        return (fused_alpha, fused_beta)

    def _get_beta_stats(self, alpha: float, beta: float) -> Tuple[float, float]:
        """Calculates the mean and variance of a Beta distribution."""
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ( (alpha + beta)**2 * (alpha + beta + 1) )
        return (mean, variance)

    def run_fusion_process(self):
        """
        Main worker loop for Component 5.
        Finds contracts 'PENDING_FUSION' and synthesizes P_model.
        """
        log.info("BeliefEngine: Checking for contracts 'PENDING_FUSION'...")
        contracts = self.graph.get_contracts_for_fusion(limit=10)
        
        if not contracts:
            log.info("BeliefEngine: No new contracts to fuse.")
            return

        for contract in contracts:
            contract_id = contract['contract_id']
            log.info(f"BeliefEngine: Fusing price for {contract_id}")
            
            try:
                # 1. Get Model 1 (Internal Prior)
                # This was already stored as (α, β) by Component 3
                beta_internal = (contract['p_internal_alpha'], contract['p_internal_beta'])

                # 2. Get Model 2 (Expert Prior)
                # We must impute this from its point-price and Brier score
                p_experts = contract['p_market_experts']
                beta_experts = self._impute_beta_from_point(p_experts, 'expert')

                # 3. Get Model 3 (Crowd Prior)
                # We impute this as well, expecting it to have a high Brier (low confidence)
                p_crowd = contract['p_market_all']
                beta_crowd = self._impute_beta_from_point(p_crowd, 'crowd')

                # 4. Fuse all three models
                all_betas = [beta_internal, beta_experts, beta_crowd]
                (fused_alpha, fused_beta) = self._fuse_betas(all_betas)
                
                # 5. Calculate the final P_model and variance
                (p_model, p_model_variance) = self._get_beta_stats(fused_alpha, fused_beta)
                
                log.info(f"Fusion complete for {contract_id}: P_model={p_model:.4f}, Var={p_model_variance:.4f}")

                # 6. Write the final fused price back to the graph
                self.graph.update_contract_fused_price(
                    contract_id,
                    p_model,
                    p_model_variance
                )
                
            except Exception as e:
                log.error(f"Failed to fuse prior for {contract_id}: {e}")
                # (In Prod: update status to 'FUSION_FAILED')

class HybridKellySolver:
    """
    Component 6.sub: The "Hybrid Kelly" mathematical solver.
    Encapsulates the analytical and numerical solutions.
    """
    def __init__(self, analytical_edge_threshold=0.2, analytical_q_threshold=0.1):
        self.edge_thresh = analytical_edge_threshold
        self.q_thresh = analytical_q_threshold
        log.info(f"HybridKellySolver initialized (Edge Tresh: {self.edge_thresh})")

    def _is_numerical_required(self, E: np.ndarray, Q: np.ndarray) -> bool:
        """Triage logic: checks if the analytical solution is unsafe."""
        if np.any(np.abs(E) > self.edge_thresh):
            log.warning("Numerical solver triggered: Large edge detected.")
            return True
        if np.any(Q < self.q_thresh) or np.any(Q > (1 - self.q_thresh)):
            log.warning("Numerical solver triggered: Extreme probabilities detected.")
            return True
        return False

    def _build_covariance_matrix(self, graph: GraphManager, contracts: List[Dict]) -> np.ndarray:
        """Builds the n x n Covariance Matrix 'C'."""
        n = len(contracts)
        C = np.zeros((n, n))
        P = np.array([c['M'] for c in contracts]) # P_model vector
        
        for i in range(n):
            for j in range(i, n):
                c1_id = contracts[i]['id']
                c2_id = contracts[j]['id']
                
                if i == j:
                    # Variance: p * (1-p)
                    C[i, i] = P[i] * (1 - P[i])
                    continue
                
                # Get the joint probability P(i, j)
                rel = graph.get_relationship_between_contracts(c1_id, c2_id)
                p_ij = rel.get('p_joint')
                
                if p_ij is None:
                    # Case C: No link. Assume independent.
                    # P(i, j) = P(i) * P(j)
                    # Cov(i, j) = P(i, j) - P(i)P(j) = 0
                    p_ij = P[i] * P[j]
                
                # Cov(i, j) = P(i, j) - P(i)P(j)
                cov = p_ij - P[i] * P[j]
                C[i, j] = cov
                C[j, i] = cov # Matrix is symmetric
                
        return C

    def _solve_analytical(self, C: np.ndarray, D: np.ndarray, E: np.ndarray) -> np.ndarray:
        """Solves F* = D * C_inv * E"""
        log.info("Solving with Analytical (Fast Path)...")
        # Use Moore-Penrose pseudo-inverse for numerical stability
        # Handles singular matrices (e.g., perfect correlation)
        C_inv = np.linalg.pinv(C) 
        F_star = D @ C_inv @ E
        return F_star

    def _solve_numerical(self, M: np.ndarray, Q: np.ndarray, C: np.ndarray, F_analytical_guess: np.ndarray) -> np.ndarray:
        """
        Solves max(E[log(W)]) using a numerical optimizer.
        This is the "Precise Path" stub.
        """
        log.info("Solving with Numerical (Precise Path)...")
        
        def objective(F: np.ndarray) -> float:
            # This is the function to MINIMIZE, so we use -E[log(W)]
            # STUB: This would run the QMC simulation
            # E_log_W = simulate_log_wealth(F, M, Q, C)
            # For this stub, we just pretend it's a simple quadratic
            E = M - Q
            return -np.dot(F, E) # A trivial objective for stubbing
        
        n = len(M)
        # Start the optimizer from the analytical guess
        initial_guess = F_analytical_guess
        
        # Add constraints (e.g., no single bet > 50% of bankroll)
        constraints = ({'type': 'ineq', 'fun': lambda F: 0.8 - np.sum(np.abs(F))}) # sum(|F|) <= 0.8
        bounds = [(-0.5, 0.5)] * n # F_i between -50% and +50%
        
        result = opt.minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            log.warning(f"Numerical solver failed: {result.message}")
            return initial_guess # Fallback to analytical
        
        return result.x

    def solve_basket(self, graph: GraphManager, contracts: List[Dict]) -> np.ndarray:
        """Main entry point to solve for an optimal basket F*."""
        
        # 1. Build Vectors & Matrices
        n = len(contracts)
        M = np.array([c['M'] for c in contracts]) # P_model vector
        Q = np.array([c['Q'] for c in contracts]) # P_market vector
        E = M - Q                                # Edge vector
        D = np.diag(Q)                           # Scaling matrix
        
        C = self._build_covariance_matrix(graph, contracts)
        
        # 2. Solve Analytical
        F_analytical = self._solve_analytical(C, D, E)
        
        # 3. Triage
        if self._is_numerical_required(E, Q):
            # Escalate to Numerical
            F_star = self._solve_numerical(M, Q, C, F_analytical)
        else:
            F_star = F_analytical
            
        return F_star


class PortfolioManager:
    """
    Component 6: The "Conductor"
    Runs the main optimization loop.
    """
    
    def __init__(self, graph_manager: GraphManager, solver: HybridKellySolver):
        self.graph = graph_manager
        self.solver = solver
        self.max_event_exposure = 0.15 # 15% bankroll cap per event
        log.info(f"PortfolioManager initialized (Max Exposure: {self.max_event_exposure})")

    def _apply_constraints(self, F_star: np.ndarray) -> np.ndarray:
        """
        Applies final portfolio-wide constraints to the basket.
        This is a simple "scalar" in this stub.
        """
        total_exposure = np.sum(np.abs(F_star))
        if total_exposure > self.max_event_exposure:
            log.warning(f"Capping exposure: {total_exposure:.2f} > {self.max_event_exposure}")
            scale_factor = self.max_event_exposure / total_exposure
            return F_star * scale_factor
        return F_star

    def run_optimization_cycle(self):
        """
        Main worker loop for Component 6.
        Finds all active clusters, solves their optimal baskets,
        and (stubs) sending trades.
        """
        log.info("--- PM: Starting Optimization Cycle ---")
        
        active_clusters = self.graph.get_active_entity_clusters()
        
        for cluster_id in active_clusters:
            log.info(f"--- Solving Cluster: {cluster_id} ---")
            
            # 1. Get all contract data for this cluster
            contracts = self.graph.get_cluster_contracts(cluster_id)
            if len(contracts) < 2:
                log.info("Skipping cluster (needs at least 2 contracts).")
                continue
            
            # 2. Solve for the optimal basket F*
            # This is the core call to the Hybrid Solver
            F_star_unconstrained = self.solver.solve_basket(self.graph, contracts)
            
            # 3. Apply final portfolio constraints
            F_star_final = self._apply_constraints(F_star_unconstrained)
            
            # 4. Generate Trade List
            log.info(f"--- Final Basket for {cluster_id} ---")
            trade_list = []
            for i, contract in enumerate(contracts):
                allocation = F_star_final[i]
                if abs(allocation) > 1e-5: # Only log non-zero trades
                    action = "BUY" if allocation > 0 else "SELL"
                    log.info(f"{action} {abs(allocation)*100:.2f}% on {contract['id']}")
                    trade_list.append({'id': contract['id'], 'action': action, 'fraction': abs(allocation)})

            # 5. (STUB) Send to Execution Engine (Component X)
            # send_trades_to_executor(trade_list)
            
        log.info("--- PM: Optimization Cycle Complete ---")
