import os
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Set up basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class GraphManager:
    """
    Component 1: The Knowledge Graph (Data Model) - Refined Version
    Handles schema, constraints, indexes, and robust write operations
    using transactions and configuration from environment variables.
    """

    def __init__(self):
        """
        Initialize the connection to the Neo4j database using
        environment variables.
        """
        # Load configuration from environment variables
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.vector_dim = int(os.getenv('VECTOR_DIM', 768))

        if not self.password:
            raise ValueError("NEO4J_PASSWORD environment variable not set.")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            log.info(f"GraphManager connected to Neo4j at {self.uri} successfully.")
            log.info(f"Using vector dimensions: {self.vector_dim}")
        except ServiceUnavailable as e:
            log.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise
        except Exception as e:
            log.error(f"An unexpected error occurred during connection: {e}")
            raise

    def close(self):
        """Closes the database driver connection."""
        if self.driver:
            self.driver.close()
            log.info("GraphManager connection closed.")

    def setup_schema(self):
        """
        Sets up all database constraints and indexes using idempotent queries
        within a managed session.
        """
        log.info("Applying database schema: constraints and indexes...")
        
        # Use a session to batch the schema setup operations
        with self.driver.session() as session:
            # --- Node Key Constraints (Enforces Uniqueness, auto-indexes) ---
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE")
            )
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
            )
            session.execute_write(
                lambda tx: tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alias) REQUIRE a.text IS UNIQUE")
            )
            log.info("Node key constraints applied.")

            # --- Additional Indexes ---
            session.execute_write(
                lambda tx: tx.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            )
            
            # --- Vector Indexes (Configurable Dimensions) ---
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
                session.execute_write(
                    lambda tx: tx.run(
                        f"""
                        CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                        FOR (e:Entity) ON (e.vector)
                        OPTIONS {{ indexConfig: {{
                            `vector.dimensions`: {self.vector_dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """
                    )
                )
                log.info("Vector indexes applied.")
            except Exception as e:
                log.warning(f"Could not create vector indexes (requires Neo4j Enterprise/Aura with Vector plugin): {e}")

        log.info("Schema setup complete.")

    # --- FULLY IMPLEMENTED WRITE OPERATIONS ---
    # These methods are now idempotent and use transactions.

    def add_contract(self, contract_id: str, text: str, vector: list[float]):
        """
        Creates or updates a Contract node idempotently.
        Used by the Ingestion Feed.
        """
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {len(vector)}")
        
        with self.driver.session() as session:
            session.execute_write(
                self._tx_merge_contract,
                contract_id=contract_id, text=text, vector=vector
            )
        log.info(f"Merged Contract: {contract_id}")

    @staticmethod
    def _tx_merge_contract(tx, contract_id, text, vector):
        """Transaction function to create/update a Contract."""
        tx.run(
            """
            MERGE (c:Contract {contract_id: $contract_id})
            ON CREATE SET
                c.text = $text,
                c.vector = $vector,
                c.p_market_all = null,
                c.p_model = null,
                c.p_model_variance = null,
                c.status = 'PENDING_LINKING',
                c.created_at = timestamp(),
                c.updated_at = timestamp()
            ON MATCH SET
                c.text = $text, // Update text in case it changed
                c.vector = $vector,
                c.updated_at = timestamp()
            """,
            contract_id=contract_id, text=text, vector=vector
        )

    def add_entity(self, entity_id: str, name: str, e_type: str, vector: list[float]):
        """
        Creates or updates a canonical Entity node idempotently.
        Used by Component 2 (Human-in-the-Loop).
        """
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {len(vector)}")
            
        with self.driver.session() as session:
            session.execute_write(
                self._tx_merge_entity,
                entity_id=entity_id, name=name, e_type=e_type, vector=vector
            )
        log.info(f"Merged Entity: {name} ({entity_id})")

    @staticmethod
    def _tx_merge_entity(tx, entity_id, name, e_type, vector):
        """Transaction function to create/update an Entity."""
        tx.run(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            ON CREATE SET
                e.canonical_name = $name,
                e.type = $e_type,
                e.vector = $vector
            ON MATCH SET
                e.canonical_name = $name,
                e.type = $e_type,
                e.vector = $vector
            """,
            entity_id=entity_id, name=name, e_type=e_type, vector=vector
        )

    def add_alias(self, text: str, entity_id: str):
        """
        Creates an Alias node (if it doesn't exist) and links it
        to its canonical Entity idempotently.
        Used by Component 2.
        """
        with self.driver.session() as session:
            session.execute_write(
                self._tx_link_alias,
                text=text, entity_id=entity_id
            )
        log.info(f"Linked Alias '{text}' -> Entity '{entity_id}'")

    @staticmethod
    def _tx_link_alias(tx, text, entity_id):
        """Transaction function to link an Alias to an Entity."""
        tx.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (a:Alias {text: $text})
            MERGE (a)-[:POINTS_TO]->(e)
            """,
            text=text, entity_id=entity_id
        )

    def link_contract_to_entity(self, contract_id: str, entity_id: str, confidence: float):
        """
        Creates/updates the IS_ABOUT relationship idempotently.
        This is the primary output of Component 2.
        """
        with self.driver.session() as session:
            session.execute_write(
                self._tx_link_contract,
                contract_id=contract_id, entity_id=entity_id, confidence=confidence
            )
        log.info(f"Linked Contract '{contract_id}' -> Entity '{entity_id}' with conf: {confidence}")

    @staticmethod
    def _tx_link_contract(tx, contract_id, entity_id, confidence):
        """Transaction function to link a Contract to an Entity."""
        tx.run(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (c)-[r:IS_ABOUT]->(e)
            ON CREATE SET
                r.confidence_score = $confidence,
                r.created_at = timestamp(),
                c.status = 'PENDING_ANALYSIS'
            ON MATCH SET
                r.confidence_score = $confidence, // Update confidence
                c.status = 'PENDING_ANALYSIS'
            """,
            contract_id=contract_id, entity_id=entity_id, confidence=confidence
        )

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration is now loaded from environment variables ---
    # Set these in your shell before running:
    # export NEO4J_URI="bolt://localhost:7687"
    # export NEO4J_USER="neo4j"
    # export NEO4J_PASSWORD="your_password"
    # export VECTOR_DIM="768"

    try:
        manager = GraphManager()
        
        # 1. Set up the schema, constraints, and indexes
        manager.setup_schema()

        # 2. Example: The Ingestion Feed adds a new contract
        dummy_vector = [0.1] * manager.vector_dim
        manager.add_contract(
            contract_id="MKT_901_REFINED",
            text="Will the 'Viper' AI Chipset from 'NeuroCorp' be released by Q2 2026?",
            vector=dummy_vector
        )
        
        # 3. Example: Add a corresponding entity (from Component 2)
        manager.add_entity(
            entity_id="E_123",
            name="NeuroCorp, Inc.",
            e_type="Organization",
            vector=[0.2] * manager.vector_dim
        )
        
        # 4. Example: Link them (from Component 2)
        manager.link_contract_to_entity(
            contract_id="MKT_901_REFINED",
            entity_id="E_123",
            confidence=1.0
        )

        manager.close()
        
    except Exception as e:
        log.error(f"Failed to run GraphManager refined example: {e}")
