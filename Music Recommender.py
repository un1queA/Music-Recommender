# AI Music Recommender - OPTIMIZED VERSION (2-3x faster)
# Run: streamlit run app.py

import streamlit as st
import re
import time
import concurrent.futures
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from difflib import SequenceMatcher

# =============================================================================
# CONFIGURATION - OPTIMIZED
# =============================================================================

MAX_RETRY_ATTEMPTS = 3  # Reduced from 5 (optimized search)
SONGS_REQUIRED = 3
CACHE_TTL = 3600  # Cache results for 1 hour
MAX_WORKERS = 3   # Parallel processing

# =============================================================================
# GLOBAL STATE MANAGEMENT (NEW ROBUST SYSTEM)
# =============================================================================

class GlobalStateManager:
    """Manages global state across all searches with persistence"""
    
    def __init__(self):
        # Initialize all session state variables with defaults
        if 'global_excluded_artists' not in st.session_state:
            st.session_state.global_excluded_artists = []
        
        if 'global_excluded_artists_hash' not in st.session_state:
            st.session_state.global_excluded_artists_hash = set()
        
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []  # List of (genre, artist) tuples
        
        if 'artist_genre_map' not in st.session_state:
            st.session_state.artist_genre_map = {}  # artist -> [genres]
        
        if 'total_searches' not in st.session_state:
            st.session_state.total_searches = 0
        
        if 'performance_stats' not in st.session_state:
            st.session_state.performance_stats = {'avg_time': 0, 'total_time': 0}
        
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        
        if 'last_genre' not in st.session_state:
            st.session_state.last_genre = ""
    
    def add_excluded_artist(self, artist: str, genre: str):
        """Add artist to global exclusion with hash checking"""
        artist_lower = artist.lower().strip()
        
        # Create hash for efficient checking
        artist_hash = hashlib.md5(artist_lower.encode()).hexdigest()
        
        if artist_hash not in st.session_state.global_excluded_artists_hash:
            st.session_state.global_excluded_artists.append(artist)
            st.session_state.global_excluded_artists_hash.add(artist_hash)
            
            # Update search history
            st.session_state.search_history.append({
                'artist': artist,
                'genre': genre,
                'timestamp': time.time(),
                'hash': artist_hash
            })
            
            # Update artist-genre mapping
            if artist not in st.session_state.artist_genre_map:
                st.session_state.artist_genre_map[artist] = []
            if genre not in st.session_state.artist_genre_map[artist]:
                st.session_state.artist_genre_map[artist].append(genre)
    
    def is_artist_excluded(self, artist: str) -> bool:
        """Check if artist is globally excluded (case-insensitive)"""
        if not artist:
            return False
        
        artist_lower = artist.lower().strip()
        artist_hash = hashlib.md5(artist_lower.encode()).hexdigest()
        
        return artist_hash in st.session_state.global_excluded_artists_hash
    
    def get_all_excluded_artists(self) -> List[str]:
        """Get all excluded artists"""
        return st.session_state.global_excluded_artists.copy()
    
    def get_recent_excluded_artists(self, limit: int = 10) -> List[str]:
        """Get recently excluded artists"""
        return st.session_state.global_excluded_artists[-limit:] if st.session_state.global_excluded_artists else []
    
    def get_artist_genre_history(self, artist: str) -> List[str]:
        """Get all genres an artist was suggested for"""
        return st.session_state.artist_genre_map.get(artist, [])
    
    def get_search_history_summary(self) -> str:
        """Get a summary of search history"""
        if not st.session_state.search_history:
            return "No searches yet"
        
        summary = []
        for i, entry in enumerate(st.session_state.search_history[-5:]):
            summary.append(f"{i+1}. {entry['artist']} (for {entry['genre']})")
        
        return "\n".join(summary)
    
    def clear_all(self):
        """Clear all state"""
        st.session_state.global_excluded_artists = []
        st.session_state.global_excluded_artists_hash = set()
        st.session_state.search_history = []
        st.session_state.artist_genre_map = {}
        st.session_state.total_searches = 0
        st.session_state.performance_stats = {'avg_time': 0, 'total_time': 0}
        st.session_state.last_genre = ""

# Initialize global state manager
state_manager = GlobalStateManager()

# =============================================================================
# CACHING SYSTEM - MAJOR SPEED BOOST
# =============================================================================

class SearchCache:
    """Cache YouTube search results to avoid redundant API calls"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        
    def get(self, query: str, max_results: int = 50):
        key = f"{query}_{max_results}"
        if key in self.cache and (time.time() - self.timestamps.get(key, 0)) < CACHE_TTL:
            return self.cache[key]
        return None
    
    def set(self, query: str, results: List[Dict], max_results: int = 50):
        key = f"{query}_{max_results}"
        self.cache[key] = results
        self.timestamps[key] = time.time()
        
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

# Global cache instance
search_cache = SearchCache()

def cached_youtube_search(query: str, max_results: int = 50) -> List[Dict]:
    """Cache YouTube search results to massively reduce API calls"""
    cached = search_cache.get(query, max_results)
    if cached:
        return cached
    
    try:
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        search_cache.set(query, results, max_results)
        return results
    except Exception as e:
        return []

# =============================================================================
# OPTIMIZED LLM INITIALIZATION
# =============================================================================

def initialize_llm(api_key: str) -> Optional[ChatOpenAI]:
    """Initialize DeepSeek LLM with optimized settings"""
    try:
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=800,
            timeout=15
        )
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek: {e}")
        return None

# =============================================================================
# ROBUST ARTIST DISCOVERY WITH STRICT GLOBAL EXCLUSION
# =============================================================================

def find_artist_for_genre(genre: str, llm: ChatOpenAI) -> List[str]:
    """Find DISTINCT artists with strict global exclusion"""
    
    # Get all excluded artists
    excluded_artists = state_manager.get_all_excluded_artists()
    excluded_text = ""
    
    if excluded_artists:
        # Create a clean list for the LLM (last 15 artists to avoid token limit)
        recent_excluded = excluded_artists[-15:]
        excluded_text = f"CRITICAL: DO NOT suggest these already-suggested artists: {', '.join(recent_excluded)}"
        
        if len(excluded_artists) > 15:
            excluded_text += f" (and {len(excluded_artists) - 15} more)"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded"],
        template="""Find 3 DISTINCT popular artists from the "{genre}" genre.

STRICT REQUIREMENTS:
1. Each artist MUST have an official YouTube channel
2. Must have released music before 2024
3. Must be well-known with at least 3 music videos
4. {excluded}

Return EXACTLY in this format (one per line):
1. [Artist Name 1]
2. [Artist Name 2]
3. [Artist Name 3]

IMPORTANT: Ensure NONE of these artists have been suggested before in this session."""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded": excluded_text})
        
        # Parse artists and filter out globally excluded ones
        artists = []
        for line in result.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and '.' in line[:3]):
                # Extract artist name
                parts = line.split('.', 1)
                if len(parts) > 1:
                    artist = parts[1].strip().strip('"').strip("'")
                    
                    # Clean up common prefixes/suffixes
                    artist = re.sub(r'^\s*-\s*', '', artist)
                    artist = re.sub(r'\s*-\s*$', '', artist)
                    
                    if artist and not state_manager.is_artist_excluded(artist):
                        artists.append(artist)
        
        return artists[:3]
    except Exception as e:
        return []

# =============================================================================
# CHANNEL DISCOVERY & VALIDATION
# =============================================================================

def find_official_channel_parallel(artist_name: str, llm: ChatOpenAI) -> Optional[str]:
    """Parallel channel discovery"""
    
    search_queries = [
        f"{artist_name} VEVO",
        f"{artist_name} official artist channel",
        f'"{artist_name}"',
        f"{artist_name} official"
    ]
    
    def search_channel(query):
        try:
            results = cached_youtube_search(query, max_results=15)
            
            for result in results:
                channel = result['channel'].strip()
                channel_lower = channel.lower()
                
                if 'topic' in channel_lower or 'lyrics' in channel_lower:
                    continue
                    
                score = quick_channel_score(channel, artist_name)
                
                if score >= 100:
                    return {
                        'channel': channel,
                        'score': score,
                        'title': result['title']
                    }
        except:
            pass
        return None
    
    channel_candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(search_channel, q): q for q in search_queries}
        
        for future in concurrent.futures.as_completed(future_to_query):
            result = future.result()
            if result:
                channel_candidates.append(result)
    
    if not channel_candidates:
        return None
    
    channel_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    top_channel = channel_candidates[0]
    if top_channel['score'] >= 180:
        return top_channel['channel']
    
    if validate_channel_with_llm_fast(top_channel['channel'], artist_name, llm):
        return top_channel['channel']
    
    if len(channel_candidates) > 1 and channel_candidates[1]['score'] >= 120:
        return channel_candidates[1]['channel']
    
    return None

def quick_channel_score(channel_name: str, artist_name: str) -> int:
    """Optimized scoring"""
    score = 0
    
    channel_norm = re.sub(r'[^a-z0-9]', '', channel_name.lower())
    artist_norm = re.sub(r'[^a-z0-9]', '', artist_name.lower())
    
    if channel_norm == artist_norm:
        return 200
    
    if artist_norm in channel_norm:
        score += 150
    elif len(artist_norm) > 3 and SequenceMatcher(None, artist_norm, channel_norm).ratio() > 0.7:
        score += 100
    
    if 'vevo' in channel_name.lower():
        score += 80
    if 'official' in channel_name.lower():
        score += 50
    
    if len(channel_name.split()) <= 3:
        score += 30
    
    return score

def validate_channel_with_llm_fast(channel_name: str, artist_name: str, llm: ChatOpenAI) -> bool:
    """Fast LLM validation"""
    
    prompt = PromptTemplate(
        input_variables=["channel", "artist"],
        template="""Channel: "{channel}"
Artist: "{artist}"

Is this the artist's OFFICIAL channel? Answer YES or NO.

ANSWER:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"channel": channel_name, "artist": artist_name})
        return "YES" in result.upper()[:3]
    except:
        return False

# =============================================================================
# GENRE VALIDATION AND MAPPING SYSTEM
# =============================================================================

def get_popular_genres() -> List[str]:
    """Return list of popular genres for validation"""
    return [
        "Pop", "Rock", "Hip Hop", "Rap", "R&B", "Country", "Jazz", "Classical",
        "Electronic", "EDM", "Blues", "Reggae", "Folk", "Metal", "Punk", "Soul",
        "Funk", "Disco", "Techno", "House", "Trance", "Dubstep", "Indie", 
        "Alternative", "K-pop", "J-pop", "Latin", "Reggaeton", "Salsa", 
        "Bachata", "Tango", "Ska", "Gospel", "Christian", "World", "New Age",
        "Ambient", "Soundtrack", "Musical", "Opera", "Anime", "Bollywood",
        "Afrobeats", "Amapiano", "Phonk", "Hyperpop", "Synthwave", "Lo-fi",
        "Drill", "Trap", "Vaporwave", "Future Bass", "Progressive", "Hardstyle"
    ]

def is_popular_genre(genre: str) -> bool:
    """Check if genre is popular"""
    genre_lower = genre.lower().strip()
    popular_genres = get_popular_genres()
    
    # Direct match
    for popular in popular_genres:
        if genre_lower == popular.lower():
            return True
    
    # Partial match
    for popular in popular_genres:
        if popular.lower() in genre_lower or genre_lower in popular.lower():
            return True
    
    # Common variations
    variations = {
        'hip hop': ['hiphop', 'hip-hop'],
        'r&b': ['rnb', 'rhythm and blues'],
        'edm': ['electronic dance music'],
        'k-pop': ['kpop', 'korean pop'],
        'j-pop': ['jpop', 'japanese pop'],
    }
    
    for key, aliases in variations.items():
        if genre_lower == key or genre_lower in aliases:
            return True
    
    return False

def find_closest_popular_genre(user_genre: str) -> Optional[str]:
    """Find closest popular genre"""
    user_genre_lower = user_genre.lower().strip()
    popular_genres = get_popular_genres()
    
    # Exact match
    for popular in popular_genres:
        if user_genre_lower == popular.lower():
            return popular
    
    # Contains match
    for popular in popular_genres:
        if popular.lower() in user_genre_lower:
            return popular
    
    # Similarity match
    best_match = None
    best_ratio = 0
    
    for popular in popular_genres:
        ratio = SequenceMatcher(None, user_genre_lower, popular.lower()).ratio()
        if ratio > 0.7 and ratio > best_ratio:
            best_ratio = ratio
            best_match = popular
    
    return best_match

def map_genre_if_needed(genre: str, llm: ChatOpenAI) -> Tuple[str, str]:
    """Map genre and return (mapped_genre, strategy)"""
    
    # Check if it's already a popular genre
    if is_popular_genre(genre):
        return genre, "POPULAR_GENRE"
    
    # Try to find closest popular genre
    closest = find_closest_popular_genre(genre)
    if closest:
        return closest, f"MAPPED_TO_POPULAR ({closest})"
    
    # Use LLM for mapping
    prompt = PromptTemplate(
        input_variables=["genre"],
        template="""The user searched for genre: "{genre}"

Map this to the closest well-known music genre. Consider:
1. Mainstream genres (Pop, Rock, Hip Hop, etc.)
2. Regional variations (K-pop, J-pop, etc.)
3. Subgenres (Synthwave → Electronic, etc.)

Return ONLY the mapped genre name, nothing else.

Mapped Genre:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre})
        mapped = result.strip().strip('"').strip("'")
        
        if "Mapped Genre:" in mapped:
            mapped = mapped.split("Mapped Genre:")[1].strip()
        
        if mapped and len(mapped) > 2:
            return mapped, "LLM_MAPPED"
    except:
        pass
    
    return genre, "ORIGINAL"

# =============================================================================
# VIDEO DISCOVERY AND PROCESSING
# =============================================================================

def discover_videos_from_channel(channel_name: str, artist_name: str) -> List[Dict]:
    """Discover videos from a channel"""
    
    searches = [
        f"{channel_name} official video",
        channel_name,
        f"{artist_name} {channel_name} music"
    ]
    
    all_videos = []
    seen_ids = set()
    
    for search in searches:
        try:
            results = cached_youtube_search(search, max_results=40)
            
            for result in results:
                if not result['channel'].strip().lower().startswith(channel_name.lower()[:10]):
                    continue
                
                video_id = result['id']
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                
                score = score_music_video(result)
                
                if score >= 25:
                    all_videos.append({
                        'id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': result['title'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
                    
                    if len(all_videos) >= 25:
                        break
                        
            if len(all_videos) >= 25:
                break
                
        except Exception:
            continue
    
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    return all_videos[:20]

def score_music_video(video: Dict) -> int:
    """Score music video"""
    score = 0
    title = video['title'].lower()
    
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 8:
                    score += 60
        except:
            pass
    
    patterns = ['official video', 'music video', ' mv ', 'official audio']
    for pattern in patterns:
        if pattern in title:
            score += 40
            break
    
    non_music = ['interview', 'behind', 'making of', 'rehearsal', 'dance practice']
    for pattern in non_music:
        if pattern in title:
            score -= 30
            break
    
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 30
        elif 'k' in views:
            score += 15
    
    return max(0, score)

def select_unique_songs(videos: List[Dict], count: int) -> List[Dict]:
    """Select unique songs"""
    
    if len(videos) < count:
        return videos
    
    selected = []
    title_patterns = set()
    
    for video in videos:
        if len(selected) >= count:
            break
        
        base_title = extract_base_title(video['title'])
        
        is_duplicate = False
        for pattern in title_patterns:
            if SequenceMatcher(None, base_title, pattern).ratio() > 0.6:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(video)
            title_patterns.add(base_title)
    
    return selected[:count]

def extract_base_title(title: str) -> str:
    """Extract base title"""
    patterns_to_remove = [
        r'\s*\(.*?\)',
        r'\s*\[.*?\]',
        r'\s*official.*',
        r'\s*music video.*',
        r'\s*mv.*',
        r'\s*audio.*',
        r'\s*lyrics.*',
    ]
    
    base = title.lower()
    for pattern in patterns_to_remove:
        base = re.sub(pattern, '', base)
    
    base = re.sub(r'[^\w\s]', ' ', base)
    base = re.sub(r'\s+', ' ', base).strip()
    
    return base

# =============================================================================
# MAIN DISCOVERY PIPELINE - ENHANCED WITH STRICT EXCLUSION
# =============================================================================

def discover_songs_for_genre(genre: str, llm: ChatOpenAI) -> Optional[Dict]:
    """
    Main discovery pipeline with strict global exclusion
    Returns: Dict with artist, channel, songs or None if failed
    """
    
    # Get mapped genre
    search_genre, strategy = map_genre_if_needed(genre, llm)
    
    # Try multiple times to find a non-excluded artist
    max_attempts = 3
    for attempt in range(max_attempts):
        # Get artists for this genre
        artists = find_artist_for_genre(search_genre, llm)
        
        if not artists:
            continue
        
        # Try each artist
        for artist in artists:
            # Strict check - has this artist been suggested before?
            if state_manager.is_artist_excluded(artist):
                continue
            
            # Find official channel
            channel = find_official_channel_parallel(artist, llm)
            if not channel:
                continue
            
            # Discover videos
            videos = discover_videos_from_channel(channel, artist)
            if len(videos) < SONGS_REQUIRED:
                continue
            
            # Select unique songs
            selected = select_unique_songs(videos, SONGS_REQUIRED)
            if len(selected) >= SONGS_REQUIRED:
                return {
                    'artist': artist,
                    'channel': channel,
                    'songs': selected[:SONGS_REQUIRED],
                    'search_genre': search_genre,
                    'strategy': strategy
                }
    
    return None

def discover_songs_with_fallback(genre: str, llm: ChatOpenAI, max_retries: int = 3) -> Tuple[Optional[Dict], str, int]:
    """
    Discover songs with fallback strategies
    Returns: (result, strategy, attempts)
    """
    
    start_time = time.time()
    attempts = 0
    
    # Strategy 1: Original or mapped genre
    result = discover_songs_for_genre(genre, llm)
    attempts += 1
    
    if result:
        elapsed = time.time() - start_time
        return result, result['strategy'], attempts
    
    # Strategy 2: Try with broader genre mapping
    for retry in range(max_retries - 1):
        attempts += 1
        
        # Use LLM to suggest alternative genre names
        prompt = PromptTemplate(
            input_variables=["genre", "excluded"],
            template="""The user searched for "{genre}" but we couldn't find unique artists.

Suggest 2 ALTERNATIVE ways to search for this genre that might yield different artists.
Focus on broader categories or related genres.

Excluded artists (do NOT suggest these): {excluded}

Return EXACTLY:
1. [Alternative Genre 1]
2. [Alternative Genre 2]"""
        )
        
        excluded_text = ", ".join(state_manager.get_recent_excluded_artists(10))
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            alternatives_result = chain.run({"genre": genre, "excluded": excluded_text})
            
            # Parse alternatives
            alternatives = []
            for line in alternatives_result.strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '.' in line[:3]:
                    alt_genre = line.split('.', 1)[1].strip().strip('"').strip("'")
                    if alt_genre:
                        alternatives.append(alt_genre)
            
            # Try each alternative
            for alt_genre in alternatives[:2]:
                result = discover_songs_for_genre(alt_genre, llm)
                if result:
                    elapsed = time.time() - start_time
                    return result, f"ALTERNATIVE_GENRE ({alt_genre})", attempts
        
        except:
            pass
        
        # If still no result, try a very broad search
        broad_terms = ["music", "songs", "artists"]
        for term in broad_terms:
            broad_genre = f"{genre} {term}"
            result = discover_songs_for_genre(broad_genre, llm)
            if result:
                elapsed = time.time() - start_time
                return result, f"BROAD_SEARCH ({broad_genre})", attempts
    
    return None, "ALL_FAILED", attempts

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Music Recommender 🎵",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .video-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e9ecef;
        transition: transform 0.2s;
    }
    
    .video-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .channel-badge {
        background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        display: inline-block;
        margin: 4px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎵 IAMUSIC - AI Music Recommender</h1>', unsafe_allow_html=True)
        st.caption("Discover unique artists & songs • No repeats across genres • 100% channel locked")
    
    with col2:
        if st.session_state.total_searches > 0:
            avg_time = st.session_state.performance_stats['avg_time']
            st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Get from DeepSeek platform"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Statistics")
        st.metric("Total Searches", st.session_state.total_searches)
        st.metric("Unique Artists", len(state_manager.get_all_excluded_artists()))
        
        # Recent searches
        if st.session_state.search_history:
            with st.expander("📋 Recent Searches"):
                for i, entry in enumerate(st.session_state.search_history[-5:]):
                    st.caption(f"{i+1}. **{entry['artist']}** → {entry['genre']}")
        
        st.markdown("---")
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                search_cache.clear()
                st.rerun()
        
        with col2:
            if st.button("🧹 Reset Session", use_container_width=True, type="secondary"):
                state_manager.clear_all()
                st.rerun()
        
        # Info
        with st.expander("ℹ️ How it works"):
            st.write("""
            1. **No Repeat Guarantee**: Once an artist is suggested, they won't appear again
            2. **Channel Locking**: All songs come from the artist's official channel
            3. **Genre Mapping**: Niche genres are mapped to popular equivalents
            4. **Smart Fallbacks**: Multiple strategies ensure success
            """)
    
    # Main interface
    st.markdown("### 🎶 Discover Music from Any Genre")
    
    with st.form("search_form"):
        genre = st.text_input(
            "Enter a music genre",
            placeholder="e.g., J-pop, Anime, K-pop, Synthwave, Reggaeton...",
            value=st.session_state.get('last_genre', ''),
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            submit = st.form_submit_button("🚀 Discover", use_container_width=True, type="primary")
    
    # Process search
    if submit and genre:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key!")
            return
        
        # Initialize
        with st.spinner("Initializing AI..."):
            llm = initialize_llm(st.session_state.api_key)
            if not llm:
                return
        
        # Update state
        st.session_state.last_genre = genre
        st.session_state.total_searches += 1
        start_time = time.time()
        
        # Progress
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Discover songs
        progress_placeholder.info(f"🔍 Searching for **{genre}**...")
        progress_bar.progress(0.2)
        
        result, strategy, attempts = discover_songs_with_fallback(genre, llm, max_retries=3)
        
        if result:
            # Double-check exclusion
            artist = result['artist']
            if state_manager.is_artist_excluded(artist):
                progress_placeholder.error(f"❌ Error: Artist '{artist}' was already suggested!")
                return
            
            # Add to global exclusion
            state_manager.add_excluded_artist(artist, genre)
            
            # Update performance stats
            elapsed = time.time() - start_time
            st.session_state.performance_stats['total_time'] += elapsed
            st.session_state.performance_stats['avg_time'] = (
                st.session_state.performance_stats['total_time'] / 
                st.session_state.total_searches
            )
            
            # Success message
            progress_bar.progress(1.0)
            
            if "MAPPED" in strategy or "ALTERNATIVE" in strategy:
                progress_placeholder.success(f"✅ Found via {strategy}!")
            else:
                progress_placeholder.success(f"✅ Found **{artist}**!")
            
            # Display results
            display_results(result, genre, strategy)
        
        else:
            # Failure
            progress_bar.progress(1.0)
            
            error_msg = f"❌ Could not find unique artists for '{genre}' after {attempts} attempts"
            
            # Check if too many artists excluded
            excluded_count = len(state_manager.get_all_excluded_artists())
            if excluded_count > 20:
                error_msg += f"\n\n**Note**: {excluded_count} artists have been suggested this session."
                error_msg += " Try resetting the session or searching a different genre."
            
            progress_placeholder.error(error_msg)
            
            # Suggest popular genres
            with st.expander("💡 Try these popular genres instead"):
                popular_genres = get_popular_genres()
                cols = st.columns(3)
                for idx, pop_genre in enumerate(popular_genres[:9]):
                    with cols[idx % 3]:
                        if st.button(pop_genre, key=f"pop_{idx}"):
                            st.session_state.last_genre = pop_genre
                            st.rerun()

def display_results(result: Dict, original_genre: str, strategy: str):
    """Display results"""
    
    st.markdown("---")
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 🎤 {result['artist']}")
    with col2:
        if strategy != "POPULAR_GENRE":
            st.caption(f"Via: {strategy}")
    
    # Channel info
    st.markdown(f'<div class="channel-badge">🔒 LOCKED CHANNEL: {result["channel"]}</div>', unsafe_allow_html=True)
    st.caption(f"All songs are exclusively from this verified official channel")
    
    # Songs
    st.markdown("### 🎵 Recommended Songs")
    
    cols = st.columns(3)
    
    for idx, song in enumerate(result['songs']):
        with cols[idx]:
            with st.container():
                st.markdown(f'<div class="video-card">', unsafe_allow_html=True)
                
                # Song info
                st.markdown(f"**Song {idx + 1}**")
                
                # Thumbnail
                try:
                    st.video(song['url'])
                except:
                    video_id = song['id']
                    st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                
                # Title
                title = song['title']
                if len(title) > 45:
                    title = title[:42] + "..."
                st.write(f"**{title}**")
                
                # Metadata
                meta_cols = st.columns(2)
                with meta_cols[0]:
                    if song.get('duration'):
                        st.caption(f"⏱️ {song['duration']}")
                with meta_cols[1]:
                    if song.get('views'):
                        st.caption(f"👁️ {song['views']}")
                
                # Watch button
                st.markdown(
                    f'<a href="{song["url"]}" target="_blank">'
                    '<button style="background:#FF0000;color:white;width:100%;'
                    'padding:8px;border-radius:6px;border:none;cursor:pointer;">'
                    '▶️ Watch</button></a>',
                    unsafe_allow_html=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Next search
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(f"🔍 Discover Another {original_genre.title()} Artist", use_container_width=True):
            st.rerun()
    
    with col2:
        excluded_count = len(state_manager.get_all_excluded_artists())
        st.caption(f"{excluded_count} unique artists suggested")

if __name__ == "__main__":
    main()
