# test_paper_trader.py
import pytest
import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch
from live_paper_trader_v3 import LiveTrader, PersistenceManager, AnalyticsEngine, SubscriptionManager

# --- FIXTURES ---

@pytest.fixture
def mock_persistence(tmp_path):
    """Creates a PersistenceManager backed by a temp file."""
    with patch("live_paper_trader_v3.STATE_FILE", tmp_path / "test_state.json"):
        pm = PersistenceManager()
        pm.state["cash"] = 1000.0 # Set initial cash
        return pm

@pytest.fixture
def trader(mock_persistence):
    """Instantiates a trader with mocked components."""
    t = LiveTrader()
    t.persistence = mock_persistence
    t.broker.pm = mock_persistence
    # Mock network heavy components
    t.analytics.metadata.refresh = AsyncMock()
    return t

# --- UNIT TESTS ---

@pytest.mark.asyncio
async def test_paper_broker_buy_logic(mock_persistence):
    """Test BUY logic: Cash deduction, Position creation, Avg Price calc."""
    broker = LiveTrader().broker
    broker.pm = mock_persistence
    
    # 1. Buy 100 shares @ $0.50 (Cost $50)
    await broker.execute_market_order("TOKEN_A", "BUY", 0.50, 50.0, "FPMM_1")
    
    assert mock_persistence.state["cash"] == 950.0
    assert mock_persistence.state["positions"]["TOKEN_A"]["qty"] == 100.0
    assert mock_persistence.state["positions"]["TOKEN_A"]["avg_price"] == 0.50

    # 2. Buy 100 shares @ $0.60 (Cost $60) -> Avg Price should be $0.55
    await broker.execute_market_order("TOKEN_A", "BUY", 0.60, 60.0, "FPMM_1")
    
    assert mock_persistence.state["cash"] == 890.0
    assert mock_persistence.state["positions"]["TOKEN_A"]["qty"] == 200.0
    assert pytest.approx(mock_persistence.state["positions"]["TOKEN_A"]["avg_price"]) == 0.55

@pytest.mark.asyncio
async def test_paper_broker_insufficient_funds(mock_persistence):
    """Test that broker rejects trades when broke."""
    broker = LiveTrader().broker
    broker.pm = mock_persistence
    mock_persistence.state["cash"] = 10.0
    
    # Try to buy $20 worth
    success = await broker.execute_market_order("TOKEN_B", "BUY", 0.5, 20.0, "FPMM_1")
    
    assert success is False
    assert mock_persistence.state["cash"] == 10.0 # Unchanged
    assert "TOKEN_B" not in mock_persistence.state["positions"]

@pytest.mark.asyncio
async def test_subscription_manager_sync():
    """Test that SubscriptionManager handles dirty states and batches updates."""
    sm = SubscriptionManager()
    mock_ws = AsyncMock()
    
    # Add needs
    sm.add_need(["TOKEN_1", "TOKEN_2"])
    assert sm.dirty is True
    
    # Sync
    await sm.sync(mock_ws)
    
    # Verify WS sent correct JSON
    args, _ = mock_ws.send.call_args
    sent_json = json.loads(args[0])
    assert set(sent_json["assets"]) == {"TOKEN_1", "TOKEN_2"}
    assert sm.dirty is False

    # Sync again (should do nothing)
    mock_ws.reset_mock()
    await sm.sync(mock_ws)
    mock_ws.send.assert_not_called()

# --- INTEGRATION TESTS ---

@pytest.mark.asyncio
async def test_full_signal_to_execution_flow(trader):
    """
    Simulates:
    1. Subgraph sending a Whale Trade.
    2. Trader calculating Score & Weight.
    3. Trader pre-heating subscription.
    4. Trader executing BUY order.
    """
    # 1. SETUP: Mock Metadata
    trader.analytics.metadata.fpmm_to_tokens = {"0xmarket": ["NO_TOKEN", "YES_TOKEN"]}
    
    # 2. SETUP: Mock Wallet Score
    trader.analytics.wallet_scores = {"0xwhale": 1.0} # Super high score
    
    # 3. SETUP: Mock WS Price (Pre-populate so execution works immediately)
    trader.ws_prices["YES_TOKEN"] = 0.55
    
    # 4. INJECT: Fake Subgraph Trade (Whale buys YES)
    fake_trades = [{
        "market": {"id": "0xmarket"},
        "user": {"id": "0xwhale"},
        "tradeAmount": "2000.0", # Huge volume * 1.0 score = 2000 weight (> 1000 threshold)
        "outcomeTokensAmount": "4000.0", # Positive = YES
        "timestamp": str(int(time.time()))
    }]
    
    # 5. RUN: Process the batch manually (bypassing the loop for testing)
    await trader._process_batch(fake_trades)
    
    # 6. ASSERTIONS
    
    # A. Did we execute the trade?
    assert "YES_TOKEN" in trader.persistence.state["positions"]
    pos = trader.persistence.state["positions"]["YES_TOKEN"]
    assert pos["qty"] > 0
    assert pos["market_fpmm"] == "0xmarket"
    
    # B. Did we update subscriptions?
    assert "YES_TOKEN" in trader.sub_manager.desired_assets

@pytest.mark.asyncio
async def test_stop_loss_trigger(trader):
    """Test that a price drop triggers a SELL."""
    # 1. Setup Position
    trader.persistence.state["positions"]["TOKEN_X"] = {
        "qty": 100.0, "avg_price": 0.50, "market_fpmm": "0xfpmm"
    }
    
    # 2. Simulate Price Drop to $0.30 (40% loss, > 20% SL)
    await trader._check_stop_loss("TOKEN_X", 0.30)
    
    # 3. Assert Position Closed
    assert "TOKEN_X" not in trader.persistence.state["positions"]
    # Cash should be original (1000) + proceeds (30) = 1030
    assert trader.persistence.state["cash"] == 1030.0

@pytest.mark.asyncio
async def test_graceful_shutdown(trader):
    """Test that shutdown saves state."""
    trader.persistence.save_async = AsyncMock()
    
    task = asyncio.create_task(trader.shutdown())
    await task
    
    assert trader.running is False
    trader.persistence.save_async.assert_called_once()
