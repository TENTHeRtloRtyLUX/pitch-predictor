import os
from typing import Optional

from supabase import create_client


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_supabase_service_client():
    """Get a Supabase client with service role key (bypasses RLS).
    
    Use this for:
    - System operations (upserting data, managing pipeline state)
    - Operations that need to bypass RLS policies
    
    Warning: This client ignores all RLS policies.
    """
    url = _require_env("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not key:
        raise ValueError(
            "Missing SUPABASE_SERVICE_ROLE_KEY (or legacy SUPABASE_KEY) in environment."
        )
    return create_client(url, key)


def get_supabase_authenticated_client(user_id: Optional[str] = None):
    """Get a Supabase client with anon key that respects RLS policies.
    
    Use this for:
    - User-facing queries
    - Operations that should respect RLS policies
    
    Args:
        user_id: Optional user ID to set in RLS context. Can be used by RLS policies
                to filter data per user. If provided, should be set in auth headers.
    
    Returns:
        Supabase client respecting RLS policies
    """
    url = _require_env("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not key:
        raise ValueError(
            "Missing SUPABASE_ANON_KEY (or legacy SUPABASE_KEY) in environment."
        )
    client = create_client(url, key)
    
    # If user_id is provided, set it in the RLS context
    # This allows RLS policies to filter data based on user identity
    if user_id:
        client.auth.set_session({"user": {"id": user_id}})
    
    return client


def get_supabase_client():
    """Convenience function - returns service client.
    
    For new code, prefer explicitly calling get_supabase_service_client()
    or get_supabase_authenticated_client() based on your use case.
    """
    return get_supabase_service_client()
