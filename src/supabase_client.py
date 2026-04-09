import os
from typing import Optional

from supabase import create_client


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_supabase_service_client():
    """Get a Supabase client with secret key (bypasses RLS).
    
    Uses modern SUPABASE_SECRET_KEY (recommended) with fallback to legacy keys.
    
    Use this for:
    - System operations (upserting data, managing pipeline state)
    - Operations that need to bypass RLS policies
    
    Warning: This client ignores all RLS policies.
    """
    url = _require_env("SUPABASE_URL")
    # Try modern key first, then fall back to legacy keys
    key = (
        os.getenv("SUPABASE_SECRET_KEY") or
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or
        os.getenv("SUPABASE_KEY")
    )
    if not key:
        raise ValueError(
            "Missing SUPABASE_SECRET_KEY in environment. "
            "(Or legacy SUPABASE_SERVICE_ROLE_KEY / SUPABASE_KEY for backward compatibility.)"
        )
    return create_client(url, key)


def get_supabase_authenticated_client(user_id: Optional[str] = None):
    """Get a Supabase client with publishable key that respects RLS policies.
    
    Uses modern SUPABASE_PUBLISHABLE_KEY (recommended) with fallback to legacy keys.
    
    Use this for:
    - User-facing queries (Streamlit app, frontend)
    - Operations that should respect RLS policies
    
    Args:
        user_id: Optional user ID to set in RLS context. Can be used by RLS policies
                to filter data per user. If provided, should be set in auth headers.
    
    Returns:
        Supabase client respecting RLS policies
    """
    url = _require_env("SUPABASE_URL")
    # Try modern key first, then fall back to legacy keys
    key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY") or
        os.getenv("SUPABASE_ANON_KEY") or
        os.getenv("SUPABASE_KEY")
    )
    if not key:
        raise ValueError(
            "Missing SUPABASE_PUBLISHABLE_KEY in environment. "
            "(Or legacy SUPABASE_ANON_KEY / SUPABASE_KEY for backward compatibility.)"
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
