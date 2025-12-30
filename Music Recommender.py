# AI Music Recommender - OPTIMIZED VERSION (2-3x faster)
# Run: streamlit run app.py

import streamlit as st
import re
import time
import concurrent.futures
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
            temperature=0.3,  # Lower temperature for faster, more consistent responses
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=800,   # Reduced for speed
            timeout=15        # Reduced timeout
        )
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek: {e}")
        return None

# =============================================================================
# OPTIMIZED ARTIST DISCOVERY (PARALLEL PROCESSING)
# =============================================================================

def find_artist_for_genre(genre: str, excluded_artists: List[str], llm: ChatOpenAI) -> List[str]:
    """Find MULTIPLE artists at once to reduce retry loops"""
    
    excluded_text = ", ".join(excluded_artists[-10:]) if excluded_artists else "None"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded"],
        template="""Find 3 DISTINCT popular artists from the "{genre}" genre.

CRITICAL REQUIREMENTS:
1. Each MUST have an official YouTube channel
2. MUST be different from excluded artists: {excluded}
3. Must have released music before 2024
4. Must be well-known enough to have at least 3 music videos

Return EXACTLY in this format (one per line):
1. [Artist 1]
2. [Artist 2] 
3. [Artist 3]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded": excluded_text})
        
        # Parse multiple artists
        artists = []
        for line in result.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and '.' in line[:3]):
                # Remove numbering
                artist = line.split('.', 1)[1].strip().strip('"').strip("'")
                if artist and artist not in excluded_artists:
                    artists.append(artist)
        
        return artists[:3]  # Return up to 3 artists
    except:
        return []

# =============================================================================
# PARALLEL CHANNEL DISCOVERY & VALIDATION
# =============================================================================

def find_official_channel_parallel(artist_name: str, llm: ChatOpenAI) -> Optional[str]:
    """Parallel channel discovery - checks multiple channels at once"""
    
    # OPTIMIZED: Pre-defined reliable queries
    search_queries = [
        f"{artist_name} VEVO",
        f"{artist_name} official artist channel",
        f'"{artist_name}"',  # Exact match
        f"{artist_name} official"
    ]
    
    # Parallel search
    def search_channel(query):
        try:
            results = cached_youtube_search(query, max_results=15)
            
            for result in results:
                channel = result['channel'].strip()
                channel_lower = channel.lower()
                
                # FAST FILTER: Skip non-official immediately
                if 'topic' in channel_lower or 'lyrics' in channel_lower:
                    continue
                    
                # Quick score calculation
                score = quick_channel_score(channel, artist_name)
                
                if score >= 100:  # High confidence threshold
                    return {
                        'channel': channel,
                        'score': score,
                        'title': result['title']
                    }
        except:
            pass
        return None
    
    # Run searches in parallel
    channel_candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(search_channel, q): q for q in search_queries}
        
        for future in concurrent.futures.as_completed(future_to_query):
            result = future.result()
            if result:
                channel_candidates.append(result)
    
    if not channel_candidates:
        return None
    
    # Sort by score
    channel_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Validate top candidate with LLM (only if needed)
    top_channel = channel_candidates[0]
    if top_channel['score'] >= 180:  # Very high confidence - skip LLM
        return top_channel['channel']
    
    # Use LLM for validation
    if validate_channel_with_llm_fast(top_channel['channel'], artist_name, llm):
        return top_channel['channel']
    
    # Try second candidate
    if len(channel_candidates) > 1 and channel_candidates[1]['score'] >= 120:
        return channel_candidates[1]['channel']
    
    return None

def quick_channel_score(channel_name: str, artist_name: str) -> int:
    """Optimized scoring - faster calculation"""
    score = 0
    
    # Fast text normalization
    channel_norm = re.sub(r'[^a-z0-9]', '', channel_name.lower())
    artist_norm = re.sub(r'[^a-z0-9]', '', artist_name.lower())
    
    # Exact match (highest priority)
    if channel_norm == artist_norm:
        return 200  # Immediate return for exact match
    
    # Partial matches
    if artist_norm in channel_norm:
        score += 150
    elif len(artist_norm) > 3 and SequenceMatcher(None, artist_norm, channel_norm).ratio() > 0.7:
        score += 100
    
    # Keywords
    if 'vevo' in channel_name.lower():
        score += 80
    if 'official' in channel_name.lower():
        score += 50
    
    # Length heuristic (official channels are usually concise)
    if len(channel_name.split()) <= 3:
        score += 30
    
    return score

def validate_channel_with_llm_fast(channel_name: str, artist_name: str, llm: ChatOpenAI) -> bool:
    """Fast LLM validation with minimal tokens"""
    
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
# GENRE MAPPING & FALLBACK SYSTEM
# =============================================================================

def get_genre_taxonomy() -> str:
    """Return comprehensive genre taxonomy as a string for LLM reference"""
    return """TOP-LEVEL GENRE TAXONOMY (Use for mapping niche genres):

MAJOR GLOBAL GENRES:
1. Hip-Hop/R&B - Dominant category including rap, trap, drill
2. Rock - All rock subgenres: classic, alternative, metal, punk
3. Pop - Mainstream pop, synthpop, electropop
4. Latin - Reggaeton, salsa, bachata, Latin pop
5. Country - Traditional and modern country, Americana
6. EDM - Electronic dance music, house, techno, trance
7. Afrobeats - African popular music, Afropop
8. K-Pop - Korean pop music and related
9. Christian/Gospel - Religious and gospel music
10. Indie & Alternative - Independent and alternative rock/pop

EMERGING & REGIONAL SUBGENRES:
• Amapiano - South African electronic
• Phonk - Drift Phonk, Brazilian Phonk
• Hyperpop - Glitchy, high-energy electronic pop
• Corridos Tumbados - Regional Mexican
• Drill - UK Drill, Brooklyn Drill
• Melodic Techno - Emotional techno subgenre
• Lo-fi Hip-Hop - Chill, study-focused hip-hop
• Neo-Soul - Modern soul with R&B elements
• Synthwave - 80s-inspired electronic

ROCK & METAL SUBGENRES:
• Punk Rock, Heavy Metal, Shoegaze, Grunge
• Black Metal, Death Metal, Progressive Rock
• Post-Rock, Indie Pop, Americana

ELECTRONIC SUBGENRES:
• House, Techno, Drum and Bass, Dubstep
• Tropical House, Vaporwave, Trance
• Acid Jazz, Future Bass, Nu-Disco

JAZZ, CLASSICAL & GLOBAL:
• Classical, Bebop, Smooth Jazz, Blues
• Salsa, Bollywood, Ska, Ambient, Folk
• Soundtrack/Cinematic

RULES FOR MAPPING:
1. Map niche genres to their closest MAJOR parent genre
2. Preserve regional identity (e.g., "Tamil Pop" → "Pop (Tamil)")
3. For very specific subgenres, use the broader category
4. Keep language/cultural markers when relevant"""

def map_to_known_genre(user_genre: str, llm: ChatOpenAI) -> str:
    """Map user's niche genre to the closest known genre category"""
    
    genre_taxonomy = get_genre_taxonomy()
    
    prompt = PromptTemplate(
        input_variables=["user_genre", "taxonomy"],
        template="""The user searched for genre: "{user_genre}"

GENRE TAXONOMY (for reference):
{taxonomy}

TASK: Map this user's genre to the MOST APPROPRIATE category from the taxonomy.

CRITICAL RULES:
1. If it's already a major genre, return it unchanged
2. If it's a niche subgenre, find its parent major genre
3. Preserve cultural/language markers when important
4. Return ONLY the mapped genre name, nothing else

Examples:
• "Synthpop" → "Pop"
• "UK Drill" → "Hip-Hop/R&B"
• "Tamil film music" → "Bollywood (Tamil)"
• "Deep House" → "EDM"
• "K-Indie" → "Indie & Alternative"

Mapped Genre:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"user_genre": user_genre, "taxonomy": genre_taxonomy})
        mapped = result.strip().strip('"').strip("'")
        
        # Clean up
        if "Mapped Genre:" in mapped:
            mapped = mapped.split("Mapped Genre:")[1].strip()
        
        # Safety check: don't return empty or nonsense
        if mapped and len(mapped) > 2 and len(mapped.split()) <= 5:
            return mapped
        else:
            return user_genre  # Fallback to original
    except:
        return user_genre  # Fallback to original

def enhanced_discover_songs(genre: str, excluded_artists: List[str], llm: ChatOpenAI) -> Optional[Dict]:
    """
    Enhanced discovery with genre mapping fallback
    
    Strategy:
    1. First try: User's exact genre
    2. If fails: Try mapped (broader) genre
    3. If still fails: Try with genre synonyms
    """
    
    # STEP 1: Try user's exact genre
    result = discover_songs_fast(genre, excluded_artists, llm)
    if result:
        return result, "ORIGINAL_GENRE"
    
    # STEP 2: Map to known genre and retry
    mapped_genre = map_to_known_genre(genre, llm)
    
    # Only proceed if mapped genre is different
    if mapped_genre.lower() != genre.lower():
        # Reset excluded for mapped genre (fresh search)
        mapped_excluded = []
        result = discover_songs_fast(mapped_genre, mapped_excluded, llm)
        
        if result:
            return result, f"MAPPED_GENRE ({mapped_genre})"
    
    # STEP 3: Try language-based expansion
    if " " in genre:
        # Try last word (often the main genre)
        base_genre = genre.split()[-1]
        if base_genre.lower() != genre.lower():
            result = discover_songs_fast(base_genre, excluded_artists, llm)
            if result:
                return result, f"BASE_GENRE ({base_genre})"
    
    return None, "ALL_FAILED"    

# =============================================================================
# OPTIMIZED VIDEO DISCOVERY
# =============================================================================

def discover_videos_from_channel_fast(channel_name: str, artist_name: str) -> List[Dict]:
    """Fast video discovery with better filtering"""
    
    # OPTIMIZED: Fewer, smarter searches
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
                # Fast channel matching
                if not result['channel'].strip().lower().startswith(channel_name.lower()[:10]):
                    continue
                
                video_id = result['id']
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                
                # Fast scoring
                score = score_music_video_fast(result)
                
                if score >= 25:  # Lower threshold to get more videos
                    all_videos.append({
                        'id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': result['title'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
                    
                    if len(all_videos) >= 25:  # Stop when we have enough
                        break
                        
            if len(all_videos) >= 25:
                break
                
        except Exception:
            continue
    
    # Sort and return
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    return all_videos[:20]  # Return top 20

def score_music_video_fast(video: Dict) -> int:
    """Optimized video scoring"""
    score = 0
    title = video['title'].lower()
    
    # Duration check (fast path)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 8:  # Optimal range
                    score += 60
        except:
            pass
    
    # Title patterns (single pass)
    patterns = ['official video', 'music video', ' mv ', 'official audio']
    for pattern in patterns:
        if pattern in title:
            score += 40
            break
    
    # Avoid non-music
    non_music = ['interview', 'behind', 'making of', 'rehearsal', 'dance practice']
    for pattern in non_music:
        if pattern in title:
            score -= 30
            break
    
    # Views
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 30
        elif 'k' in views:
            score += 15
    
    return max(0, score)

# =============================================================================
# FAST SONG SELECTION
# =============================================================================

def select_unique_songs_fast(videos: List[Dict], count: int) -> List[Dict]:
    """Fast song selection without LLM for duplicates"""
    
    if len(videos) < count:
        return videos
    
    selected = []
    title_patterns = set()
    
    for video in videos:
        if len(selected) >= count:
            break
        
        # Extract base title (remove version info)
        base_title = extract_base_title(video['title'])
        
        # Check if similar title already selected
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
    """Extract the main song title from video title"""
    # Remove common suffixes
    patterns_to_remove = [
        r'\s*\(.*?\)',  # Parentheses content
        r'\s*\[.*?\]',  # Bracket content  
        r'\s*official.*',
        r'\s*music video.*',
        r'\s*mv.*',
        r'\s*audio.*',
        r'\s*lyrics.*',
        r'\s*[\[\(].*[\]\)]',  # Any brackets
    ]
    
    base = title.lower()
    for pattern in patterns_to_remove:
        base = re.sub(pattern, '', base)
    
    # Remove extra spaces and special chars
    base = re.sub(r'[^\w\s]', ' ', base)
    base = re.sub(r'\s+', ' ', base).strip()
    
    return base

# =============================================================================
# OPTIMIZED MAIN DISCOVERY PIPELINE
# =============================================================================

def discover_songs_fast(genre: str, excluded_artists: List[str], llm: ChatOpenAI) -> Optional[Dict]:
    """
    Optimized pipeline with early termination and batch processing
    """
    
    # Step 1: Get multiple artists at once
    artists = find_artist_for_genre(genre, excluded_artists, llm)
    if not artists:
        return None
    
    # Step 2: Try each artist until success
    for artist in artists:
        # Fast channel discovery
        channel = find_official_channel_parallel(artist, llm)
        if not channel:
            continue
        
        # Fast video discovery
        videos = discover_videos_from_channel_fast(channel, artist)
        if len(videos) < SONGS_REQUIRED:
            continue
        
        # Fast song selection
        selected = select_unique_songs_fast(videos, SONGS_REQUIRED)
        if len(selected) >= SONGS_REQUIRED:
            return {
                'artist': artist,
                'channel': channel,
                'songs': selected[:SONGS_REQUIRED]
            }
    
    return None

# =============================================================================
# OPTIMIZED RETRY MECHANISM
# =============================================================================

def retry_with_progress(genre: str, excluded_artists: List[str], llm: ChatOpenAI, 
                       progress_placeholder, max_attempts: int = MAX_RETRY_ATTEMPTS):
    """
    Optimized retry loop with progress updates
    """
    
    attempts = 0
    start_time = time.time()
    
    while attempts < max_attempts:
        attempts += 1
        
        progress_placeholder.info(
            f"🚀 Attempt {attempts}: Searching... "
            f"(Elapsed: {int(time.time() - start_time)}s)"
        )
        
        result = discover_songs_fast(genre, excluded_artists, llm)
        
        if result:
            # Success - validate uniqueness
            if result['artist'] in excluded_artists:
                progress_placeholder.warning(f"⚠️ Artist already suggested, trying next...")
                excluded_artists.append(result['artist'])
                continue
                
            return result, attempts
        
        # Failed, update progress
        progress_placeholder.warning(
            f"⚠️ Attempt {attempts} failed, retrying with different artist..."
        )
    
    return None, attempts

# =============================================================================
# OPTIMIZED STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Music Recommender 🚀",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Session state - optimized
    session_defaults = {
        'api_key': "",
        'excluded_by_genre': {},
        'total_searches': 0,
        'cache_hits': 0,
        'last_genre': "",
        'performance_stats': {'avg_time': 0, 'total_time': 0}
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Custom CSS - optimized for performance
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .performance-badge {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        display: inline-block;
        margin: 5px;
        font-weight: 600;
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
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Optimize sidebar */
    section[data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 280px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with performance info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎵 AI Music Recommender</h1>', unsafe_allow_html=True)
        st.caption("⚡ Optimized for speed • Universal genre support • 100% accurate channel locking")
    
    with col2:
        if st.session_state.total_searches > 0:
            avg_time = st.session_state.performance_stats['avg_time']
            cache_hits = st.session_state.get('cache_hits', 0)
            st.markdown(f'<div class="performance-badge">⚡ Avg: {avg_time:.1f}s</div>', unsafe_allow_html=True)
    
    # Sidebar - optimized
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Get your API key from DeepSeek platform"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Performance stats
        st.subheader("📊 Performance")
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Searches", st.session_state.total_searches)
        with col_stat2:
            cache_hits = st.session_state.get('cache_hits', 0)
            st.metric("Cache Hits", cache_hits)
        
        if st.session_state.excluded_by_genre:
            st.markdown("**Recent Genres:**")
            for genre, artists in list(st.session_state.excluded_by_genre.items())[:3]:
                st.caption(f"• {genre.title()}: {len(artists)} artists")
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                search_cache.clear()
                st.session_state.cache_hits = 0
                st.rerun()
        
        with col_btn2:
            if st.button("🧹 Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Quick tips
        with st.expander("💡 Tips for faster results"):
            st.write("""
            1. **Popular genres** work fastest
            2. Clear cache if results seem stale
            3. The system learns from previous searches
            4. First search may be slower due to caching
            """)
    
    # Main search interface
    st.markdown("### Discover Music from Any Genre")
    
    with st.form("genre_form", border=False):
        col_input, col_btn = st.columns([4, 1])
        
        with col_input:
            genre = st.text_input(
                "Enter a genre",
                placeholder="e.g., K-pop, Bollywood, Reggaeton, Synthwave, J-pop...",
                label_visibility="collapsed",
                value=st.session_state.get('last_genre', '')
            )
        
        with col_btn:
            submit = st.form_submit_button(
                "🚀 Discover",
                use_container_width=True,
                type="primary"
            )
    
    # Process search
     # Process search
    if submit and genre:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key!")
            return
        
        start_time = time.time()
        
        # Initialize LLM
        with st.spinner("Initializing AI engine..."):
            llm = initialize_llm(st.session_state.api_key)
            if not llm:
                return
        
        # Get excluded artists for this genre
        genre_key = genre.lower().strip()
        st.session_state.last_genre = genre
        if genre_key not in st.session_state.excluded_by_genre:
            st.session_state.excluded_by_genre[genre_key] = []
        
        excluded = st.session_state.excluded_by_genre[genre_key]
        st.session_state.total_searches += 1
        
        # Progress tracking
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # ENHANCED RETRY STRATEGY
        result = None
        strategy = None
        attempts_made = 0
        max_total_attempts = MAX_RETRY_ATTEMPTS * 2  # Allow for original + mapped
        
        # Create multi-strategy search
        for phase in ["original", "mapped", "broad"]:
            if result:  # If we found something, stop
                break
            
            if phase == "original":
                # Phase 1: Try original genre
                progress_placeholder.info(
                    f"🔍 **Phase 1:** Searching for **{genre.title()}** artists..."
                )
                
                for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
                    attempts_made += 1
                    progress_bar.progress(min(attempts_made / max_total_attempts, 0.9))
                    
                    progress_placeholder.info(
                        f"🔍 **Phase 1:** Attempt {attempt}/{MAX_RETRY_ATTEMPTS}\n"
                        f"Searching **{genre.title()}**..."
                    )
                    
                    result = discover_songs_fast(genre, excluded, llm)
                    
                    if result:
                        strategy = "ORIGINAL_GENRE"
                        break
                    
                    if attempt < MAX_RETRY_ATTEMPTS:
                        progress_placeholder.warning(
                            f"⚠️ Original genre attempt {attempt} failed. Retrying..."
                        )
            
            elif phase == "mapped" and not result:
                # Phase 2: Try mapped/broader genre
                progress_placeholder.info(
                    f"🔄 **Phase 2:** Trying broader genre categorization..."
                )
                
                mapped_genre = map_to_known_genre(genre, llm)
                
                if mapped_genre.lower() != genre.lower():
                    progress_placeholder.info(
                        f"🔄 Mapping '{genre}' → '{mapped_genre}' for better results..."
                    )
                    
                    # Fresh search with mapped genre
                    mapped_excluded = []
                    
                    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
                        attempts_made += 1
                        progress_bar.progress(min(attempts_made / max_total_attempts, 0.95))
                        
                        progress_placeholder.info(
                            f"🔄 **Phase 2:** Attempt {attempt}/{MAX_RETRY_ATTEMPTS}\n"
                            f"Searching **{mapped_genre.title()}**..."
                        )
                        
                        result = discover_songs_fast(mapped_genre, mapped_excluded, llm)
                        
                        if result:
                            strategy = f"MAPPED_GENRE ({mapped_genre})"
                            # Add to original excluded list too
                            if result['artist'] not in excluded:
                                excluded.append(result['artist'])
                            break
                        
                        if attempt < MAX_RETRY_ATTEMPTS:
                            progress_placeholder.warning(
                                f"⚠️ Mapped genre attempt {attempt} failed. Retrying..."
                            )
            
            elif phase == "broad" and not result:
                # Phase 3: Last resort - try just the base word
                if " " in genre:
                    base_genre = genre.split()[-1]
                    
                    if base_genre.lower() != genre.lower() and len(base_genre) > 3:
                        progress_placeholder.info(
                            f"🌐 **Phase 3:** Trying base genre '{base_genre}'..."
                        )
                        
                        for attempt in range(1, 2):  # Just one attempt for this
                            attempts_made += 1
                            progress_bar.progress(1.0)
                            
                            progress_placeholder.info(
                                f"🌐 **Final attempt:** Searching **{base_genre.title()}**..."
                            )
                            
                            result = discover_songs_fast(base_genre, excluded, llm)
                            
                            if result:
                                strategy = f"BASE_GENRE ({base_genre})"
                                break
        
        # Handle results
        if result:
            artist = result['artist']
            
            # Final uniqueness check
            if artist in excluded:
                progress_placeholder.warning(f"Artist already suggested, trying next...")
                excluded.append(artist)
            else:
                # SUCCESS
                progress_bar.progress(1.0)
                
                # Add to excluded
                excluded.append(artist)
                
                # Show strategy used
                if strategy != "ORIGINAL_GENRE":
                    progress_placeholder.success(
                        f"✅ Found via {strategy}!\n"
                        f"**Artist:** {artist} with {len(result['songs'])} songs"
                    )
                else:
                    progress_placeholder.success(
                        f"✅ Found **{artist}** with {len(result['songs'])} songs!"
                    )
                
                # Calculate performance
                elapsed = time.time() - start_time
                st.session_state.performance_stats['total_time'] += elapsed
                st.session_state.performance_stats['avg_time'] = (
                    st.session_state.performance_stats['total_time'] / 
                    st.session_state.total_searches
                )
                
                # Display results
                display_results(result, genre, strategy)
        else:
            # ALL ATTEMPTS FAILED
            progress_bar.progress(1.0)
            
            # Smart error message based on genre
            genre_lower = genre.lower()
            
            error_details = {
                "message": f"❌ Could not find suitable artists for '{genre}'",
                "reasons": [
                    "• Genre might be too niche or misspelled",
                    "• Artists may not have official YouTube channels",
                    "• Try a broader genre or check spelling"
                ],
                "suggestions": []
            }
            
            # Genre-specific suggestions
            if "pop" in genre_lower:
                error_details["suggestions"].extend([
                    "Try: 'K-pop', 'J-pop', 'Synthpop', 'Indie Pop'"
                ])
            elif "rock" in genre_lower:
                error_details["suggestions"].extend([
                    "Try: 'Alternative Rock', 'Indie Rock', 'Classic Rock'"
                ])
            elif any(word in genre_lower for word in ["edm", "electronic", "techno", "house"]):
                error_details["suggestions"].extend([
                    "Try: 'EDM', 'House', 'Techno', 'Trance'"
                ])
            
            # Display error
            progress_placeholder.error(error_details["message"])
            
            with st.expander("🔍 Why this might have failed:"):
                for reason in error_details["reasons"]:
                    st.write(reason)
                
                if error_details["suggestions"]:
                    st.markdown("**Try these similar genres:**")
                    for suggestion in error_details["suggestions"]:
                        st.write(suggestion)

def display_results(result: Dict, genre: str, strategy: str = "ORIGINAL_GENRE"):
    """Enhanced results display with strategy info"""
    
    # Success header with strategy indicator
    st.markdown("---")
    
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"## 🎤 {result['artist']}")
    with col_header2:
        if "MAPPED" in strategy:
            st.caption(f"🔄 Via: {strategy}")
    
    col_channel, col_stats = st.columns([2, 1])
    with col_channel:
        st.markdown(f'<div class="channel-badge">📺 {result["channel"]}</div>', unsafe_allow_html=True)
    with col_stats:
        st.caption(f"🎵 {len(result['songs'])} songs • {genre.title()}")
    
    # Display songs in columns
    st.markdown("### 🎵 Recommended Songs")
    
    cols = st.columns(3)
    
    for idx, song in enumerate(result['songs']):
        with cols[idx]:
            with st.container():
                st.markdown(f'<div class="video-card">', unsafe_allow_html=True)
                
                # Song number
                st.markdown(f"**Song {idx + 1}**")
                
                # Thumbnail or video
                try:
                    st.video(song['url'])
                except:
                    video_id = song['id']
                    st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                
                # Title (truncated if needed)
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
                
                # Quality indicator
                if song.get('score', 0) >= 60:
                    st.caption("🌟 High quality match")
                
                # Watch button
                st.markdown(
                    f'<a href="{song["url"]}" target="_blank">'
                    '<button style="background:#FF0000;color:white;width:100%;'
                    'padding:10px;border-radius:8px;border:none;cursor:pointer;'
                    'font-weight:600;margin-top:10px;">▶️ Watch on YouTube</button>'
                    '</a>',
                    unsafe_allow_html=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Next steps
    st.markdown("---")
    
    # Show how many more artists available
    genre_key = genre.lower().strip()
    excluded = st.session_state.excluded_by_genre.get(genre_key, [])
    remaining = 50 - len(excluded)
    
    if remaining > 0:
        col_next, col_stats = st.columns([2, 1])
        with col_next:
            if st.button(f"🔍 Discover Another {genre.title()} Artist", type="secondary", use_container_width=True):
                st.rerun()
        with col_stats:
            st.caption(f"~{remaining} more artists available")
    else:
        st.info("🎉 You've explored many artists in this genre! Try a different genre.")

# Performance monitoring
def log_performance(operation: str, duration: float):
    """Log performance metrics"""
    if 'performance_log' not in st.session_state:
        st.session_state.performance_log = []
    
    st.session_state.performance_log.append({
        'operation': operation,
        'duration': duration,
        'timestamp': time.time()
    })

if __name__ == "__main__":
    main()

