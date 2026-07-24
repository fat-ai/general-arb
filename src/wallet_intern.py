#!/usr/bin/env python3
"""SINGLE SOURCE OF TRUTH for wallet address <-> integer id.

Every component that reads or writes an address MUST go through this module:
download_data_sql.py, the V2 downloader, the DB builder, sim_strat_5.py, and any
analysis script. If two components normalise differently, joins silently mismatch and
nothing downstream fails loudly -- so normalisation lives in exactly one function.

Storage model
-------------
    wallets(wallet_id INTEGER PRIMARY KEY, address TEXT UNIQUE NOT NULL)
    trades(..., user_id INTEGER, maker_id INTEGER, ...)

`address` is ALWAYS stored normalised: lowercase, 0x-prefixed, 40 hex chars.
`wallet_id` is append-only and never reused, so ids stay valid across rebuilds of
anything except the wallets table itself.

Why intern: at ~20M distinct wallets, a 42-char TEXT address costs ~43 bytes per
reference. Two address columns across ~1.3B trades is ~110GB of text; as INTEGER ids
it is ~10GB. The rebuilt DB carries twice the address information in less space.
"""
import re
import sqlite3

_HEX40 = re.compile(r'^[0-9a-f]{40}$')


class AddressError(ValueError):
    pass


def normalize_address(addr):
    """The ONE normalisation. Returns '0x' + 40 lowercase hex, or raises.

    Accepts: mixed case, missing 0x, surrounding whitespace, 32-byte left-padded
    topic words (as returned by eth_getLogs). Rejects anything else loudly --
    a silently mangled address is far worse than a crash.
    """
    if addr is None:
        return None
    s = str(addr).strip().lower()
    if s.startswith('0x'):
        s = s[2:]
    if len(s) == 64:                 # topic word: 12 zero bytes then the address
        if s[:24] != '0' * 24:
            raise AddressError(f"64-char value is not a left-padded address: {addr!r}")
        s = s[24:]
    if not _HEX40.match(s):
        raise AddressError(f"not a 20-byte hex address: {addr!r}")
    return '0x' + s


def ensure_schema(conn):
    """Idempotent. Safe to call on every process start."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallets (
            wallet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            address   TEXT UNIQUE NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wallets_address ON wallets(address)")
    conn.commit()


class WalletIntern:
    """Address -> id with an in-process cache.

    Concurrency: interning uses INSERT OR IGNORE + SELECT, so two processes racing on
    the same new address converge on one id (the UNIQUE constraint decides). Ids are
    never reassigned, so a cache entry can never go stale.
    """

    def __init__(self, conn, preload=False):
        self.conn = conn
        self._a2i = {}
        self._i2a = {}
        ensure_schema(conn)
        if preload:
            self.preload()

    def preload(self):
        for wid, addr in self.conn.execute("SELECT wallet_id, address FROM wallets"):
            self._a2i[addr] = wid
            self._i2a[wid] = addr
        return len(self._a2i)

    def get_id(self, addr, create=True):
        n = normalize_address(addr)
        if n is None:
            return None
        hit = self._a2i.get(n)
        if hit is not None:
            return hit
        row = self.conn.execute(
            "SELECT wallet_id FROM wallets WHERE address = ?", (n,)).fetchone()
        if row is None:
            if not create:
                return None
            self.conn.execute("INSERT OR IGNORE INTO wallets(address) VALUES (?)", (n,))
            row = self.conn.execute(
                "SELECT wallet_id FROM wallets WHERE address = ?", (n,)).fetchone()
        wid = row[0]
        self._a2i[n] = wid
        self._i2a[wid] = n
        return wid

    def get_ids(self, addrs, create=True):
        """Bulk intern. One round trip for all new addresses, not one per address."""
        norm = [normalize_address(a) for a in addrs]
        missing = sorted({n for n in norm if n is not None and n not in self._a2i})
        if missing:
            if create:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO wallets(address) VALUES (?)",
                    [(n,) for n in missing])
            for i in range(0, len(missing), 900):     # stay under SQLite's var limit
                batch = missing[i:i + 900]
                q = f"SELECT address, wallet_id FROM wallets WHERE address IN ({','.join('?' * len(batch))})"
                for a, w in self.conn.execute(q, batch):
                    self._a2i[a] = w
                    self._i2a[w] = a
        return [None if n is None else self._a2i.get(n) for n in norm]

    def get_address(self, wid):
        if wid is None:
            return None
        hit = self._i2a.get(wid)
        if hit is not None:
            return hit
        row = self.conn.execute(
            "SELECT address FROM wallets WHERE wallet_id = ?", (wid,)).fetchone()
        if row is None:
            return None
        self._i2a[wid] = row[0]
        self._a2i[row[0]] = wid
        return row[0]

    def commit(self):
        self.conn.commit()
