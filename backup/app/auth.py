"""
API Authentication
"""
from typing import Optional
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import VALID_API_KEYS

def verify_api_key(api_key: str) -> bool:
    """
    Verify if API key is valid and active
    
    Args:
        api_key: API key from request header
        
    Returns:
        True if valid and active, False otherwise
    """
    if api_key not in VALID_API_KEYS:
        return False
    
    key_info = VALID_API_KEYS[api_key]
    return key_info.get('active', False)


def get_api_key_info(api_key: str) -> Optional[dict]:
    """
    Get information about an API key
    
    Args:
        api_key: API key
        
    Returns:
        Dictionary with key info or None
    """
    return VALID_API_KEYS.get(api_key)


def get_rate_limit(api_key: str) -> int:
    """
    Get rate limit for API key based on tier
    
    Args:
        api_key: API key
        
    Returns:
        Requests per hour allowed
    """
    key_info = get_api_key_info(api_key)
    
    if not key_info:
        return 0
    
    tier_limits = {
        'free': 100,
        'premium': 1000,
        'enterprise': 10000
    }
    
    return tier_limits.get(key_info.get('tier', 'free'), 100)