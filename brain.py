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
# REFINED COMPONENT 1: GraphManager (with C4 Read Methods)
# ==============================================================================

class GraphManager:
    # --- (Previous methods: __init__, close, setup_schema, add_contract, etc.) ---
    # (... all methods from C3 stub are assumed to be here ...)
    
    # --- NEW: Read/Update Methods for Component 4 ---

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
