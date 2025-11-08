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
# REFINED COMPONENT 1: GraphManager (with new read/update methods)
# ==============================================================================

class GraphManager:
    """
    Component 1: The Knowledge Graph (Data Model) - Refined Version
    Includes write, update, and read methods needed by Component 2.
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
            log.info("GraphManager connection closed.")

    # --- Schema (as before) ---
    def setup_schema(self):
        log.info("Applying database schema: constraints and indexes...")
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE")
            )
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
            )
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alias) REQUIRE a.text IS UNIQUE")
            )
            session.execute_write(
                lambda tx: tx.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            )
            # (Vector index creation stubbed for brevity)
        log.info("Schema setup complete.")

    # --- Write/Update Methods (as before) ---
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
            ON CREATE SET
                c.text = $text, c.vector = $vector, c.p_market_all = null,
                c.p_model = null, c.p_model_variance = null,
                c.status = 'PENDING_LINKING', c.created_at = timestamp(), c.updated_at = timestamp()
            ON MATCH SET
                c.text = $text, c.vector = $vector, c.updated_at = timestamp()
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
            ON CREATE SET
                r.confidence_score = $confidence, r.created_at = timestamp()
            ON MATCH SET
                r.confidence_score = $confidence
            SET c.status = 'PENDING_ANALYSIS' // Update status on link
            """,
            contract_id=contract_id, entity_id=entity_id, confidence=confidence
        )

    # --- NEW: Read/Update Methods for Component 2 ---

    def get_contract_details(self, contract_id: str) -> dict:
        """Fetches the text and vector for a contract."""
        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(
                    "MATCH (c:Contract {contract_id: $contract_id}) RETURN c.text AS text, c.vector AS vector",
                    contract_id=contract_id
                ).single()
            )
        return result.data() if result else None

    def find_entity_by_alias_fuzzy(self, alias_text: str, threshold: float = 0.9) -> dict:
        """
        Finds the single best Entity match for an alias using fuzzy matching.
        NOTE: This is a STUB. Real fuzzy search requires the APOC plugin.
        We will simulate it with a simple exact match for this stub.
        """
        # --- PRODUCTION IMPLEMENTATION (Conceptual) ---
        # query = """
        # CALL apoc.cypher.run(
        #     'MATCH (a:Alias) WHERE apoc.text.levenshteinSimilarity(a.text, $text) >= $threshold
        #     RETURN a, apoc.text.levenshteinSimilarity(a.text, $text) AS score
        #     ORDER BY score DESC LIMIT 1',
        #     {text: $alias_text, threshold: $threshold}
        # )
        # YIELD value
        # MATCH (value.a)-[:POINTS_TO]->(e:Entity)
        # RETURN e.entity_id AS entity_id, e.canonical_name AS name, value.score AS confidence
        # """
        # with self.driver.session() as session:
        #     result = session.run(query, alias_text=alias_text, threshold=threshold).single()
        # return result.data() if result else None
        
        # --- STUB IMPLEMENTATION (Simple exact match for testing) ---
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
        query = """
            MATCH (c:Contract {contract_id: $contract_id})
            SET c.status = $status, c.updated_at = timestamp()
        """
        if metadata:
            query += " SET c.review_metadata = $metadata"
        tx.run(query, contract_id=contract_id, status=status, metadata=metadata)


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
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    # Set environment variables in your shell before running
    # export NEO4J_URI="bolt://localhost:7687"
    # export NEO4J_USER="neo4j"
    # export NEO4J_PASSWORD="your_password"
    # export VECTOR_DIM="384" 
    # NOTE: Using 384, as en_core_web_sm vectors are 384-dim if we use nlp.vocab.vectors
    # For this stub, we'll just use a placeholder.
    
    # We'll use a 384-dim placeholder vector to match spaCy's "md" models
    # Or just keep 768 as our standard and use a different model.
    # For this demo, we'll assume the ingestor uses a 768-dim S-BERT.
    # The spaCy model *only* does NER, it doesn't provide the vector.
    os.environ['VECTOR_DIM'] = "768" # Re-set for clarity
    
    try:
        log.info("--- Component 1 & 2 Integration Test ---")
        
        # 1. Initialize Component 1
        graph = GraphManager()
        graph.setup_schema()
        
        # 2. Initialize Component 2
        # This will load the spaCy model.
        linker = RelationalLinker(graph)
        
        # 3. Simulate Ingestion Feed (add a contract)
        contract_id = "MKT_902_SPACY"
        contract_text = "Will the 'Viper' AI Chipset from NeuroCorp be released by Q2 2026?"
        # The ingestion feed would generate this 768-dim vector
        vector_768d = [0.1] * 768 
        
        graph.add_contract(
            contract_id=contract_id,
            text=contract_text,
            vector=vector_768d
        )
        
        # 4. Run the linker on the new contract
        # The NER will extract "NeuroCorp".
        # The Fast Path will fail (no Alias for "NeuroCorp" exists yet).
        # It will be flagged for human review.
        linker.process_new_contract(contract_id)
        
        # 5. Simulate the Human (Component 8) creating the missing links
        log.info("--- Simulating Human-in-the-Loop (Component 8) ---")
        # The human sees the review queue item:
        # "No alias match found for {'extracted_entities': ['NeuroCorp']}"
        # The human creates the new Entity and Alias:
        
        # Create the canonical Entity
        graph.driver.execute_write(
            lambda tx: tx.run(
                "MERGE (e:Entity {entity_id: 'E_123'}) "
                "SET e.canonical_name = 'NeuroCorp, Inc.', e.type = 'Organization', e.vector = $vec",
                vec=[0.2] * 768
            )
        )
        # Create the Alias
        graph.driver.execute_write(
            lambda tx: tx.run(
                "MATCH (e:Entity {entity_id: 'E_123'}) "
                "MERGE (a:Alias {text: 'NeuroCorp'}) "
                "MERGE (a)-[:POINTS_TO]->(e)"
            )
        )
        log.info("Human created Entity 'E_123' and Alias 'NeuroCorp'.")

        # 6. Re-run the linker on the *same* contract
        # (This is what would happen after the human resolves the ticket)
        log.info("--- Re-running linker post-human-fix ---")
        
        # First, we set the status back so the worker "finds" it
        graph.update_contract_status(contract_id, 'PENDING_LINKING')
        
        linker.process_new_contract(contract_id)
        
        # This time, the NER extracts "NeuroCorp".
        # The Fast Path *succeeds* finding the Alias.
        # The contract is successfully linked to Entity E_123.
        # Its status is set to 'PENDING_ANALYSIS'.
        
        graph.close()

    except Exception as e:
        log.error(f"Failed to run main example: {e}")
        if "spacy" in str(e):
            log.error("Please run: python -m spacy download en_core_web_sm")
