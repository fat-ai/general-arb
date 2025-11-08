import os
import logging
import spacy  # Switched from transformers to spaCy
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

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

    # --- Write Methods (from previous step) ---

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

    # --- NEW: Read/Update Methods for C2 & C3 ---

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
                    
            except Exception as e:
                log.error(f"Failed to process prior for {contract_id}: {e}")
                self.graph.update_contract_status(contract_id, 'PRIOR_FAILED', {'error': str(e)})
