"""
secrets_gcp.py — Resolve the wallet private key without storing it on the VM.

Priority order:
  1. GCP Secret Manager  (uses the VM's attached service account — no key file)
  2. interactive getpass prompt (last-resort fallback)

The environment is deliberately NOT a source — the key is passed directly into
LiveBroker and never written to os.environ, so it can't leak via the command
line, env inheritance, or /proc/<pid>/environ.

On a Google VM, SecretManagerServiceClient() authenticates automatically via the
VM's service account (Application Default Credentials) — nothing to configure in
code. The key is fetched into memory at startup and never written to disk.

Runtime config (env):
  GCP_PROJECT             your GCP project id  (enables the Secret Manager path)
  POLYMARKET_PK_SECRET    secret name (default: "polymarket-pk")

    pip install google-cloud-secret-manager
"""
import os
import logging

log = logging.getLogger("PaperGold")

DEFAULT_SECRET_NAME = "polymarket-pk"


def _fetch_gcp_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8")

def resolve_relayer_credentials():
    """Fetch the Relayer API Key + its bound address. Returns (key, address)."""
    import os
    from google.cloud import secretmanager
    project = os.environ["GCP_PROJECT"]
    client = secretmanager.SecretManagerServiceClient()
    def _get(name):
        path = f"projects/{project}/secrets/{name}/versions/latest"
        return client.access_secret_version(name=path).payload.data.decode("utf-8").strip()
    return _get("polymarket-relayer-key"), _get("polymarket-relayer-address")


def resolve_private_key(interactive: bool = True) -> str:
    """Return the wallet private key. Resolves from GCP Secret Manager, falling
    back to an interactive prompt. Never reads it from the environment — that
    path is deliberately closed to avoid command-line / env exposure."""
    # 1. GCP Secret Manager (preferred for unattended operation).
    project = os.environ.get("GCP_PROJECT")
    secret = os.environ.get("POLYMARKET_PK_SECRET", DEFAULT_SECRET_NAME)
    if project:
        try:
            key = _fetch_gcp_secret(project, secret)
            log.info(f"🔐 Loaded private key from Secret Manager ({project}/{secret}).")
            # .strip() guards against an accidental trailing newline in the
            # secret value, which would otherwise fail key parsing.
            return key.strip()
        except Exception as e:
            log.error(f"Secret Manager fetch failed for {project}/{secret}: {e}")
            if not interactive:
                raise

    # 2. Interactive fallback.
    if interactive:
        import getpass
        return getpass.getpass("Enter wallet private key (live): ").strip()

    raise RuntimeError(
        "No private key: set GCP_PROJECT for Secret Manager, or run interactively."
    )


if __name__ == "__main__":
    # Smoke test: confirm the key is reachable WITHOUT printing it.
    logging.basicConfig(level=logging.INFO)
    try:
        k = resolve_private_key(interactive=False)
        clean = k[2:] if k.startswith("0x") else k
        is_hex = all(c in "0123456789abcdefABCDEF" for c in clean)
        print(f"✅ resolved key — length(no 0x): {len(clean)}, valid hex: {is_hex}")
        print("   (expect length 64, valid hex: True)")
    except Exception as e:
        print(f"❌ could not resolve key: {e}")
