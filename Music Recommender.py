# AI Music Recommender - FOOLPROOF VERSION
# Run: streamlit run app.py

import streamlit as st
import re
import time
import concurrent.futures
import json
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from difflib import SequenceMatcher

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_RETRY_ATTEMPTS = 3
SONGS_REQUIRED = 3
CACHE_TTL = 3600
MAX_WORKERS = 3

# =============================================================================
# FOOLPROOF GLOBAL STATE MANAGEMENT
# =============================================================================

class AbsoluteExclusionManager:
    """Absolute exclusion system - no artist repeats EVER in same session"""
    
    def __init__(self):
        # Initialize ALL session state in one place
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'absolute_excluded_artists': [],  # Main exclusion list
            'artist_exclusion_count': 0,  # Total excluded count
            'search_session_history': [],  # Complete history of searches
            'genre_artist_map': {},  # Which artist was suggested for which genre
            'artist_genre_map': {},  # Reverse mapping
            'total_searches': 0,
            'performance_stats': {'avg_time': 0, 'total_time': 0, 'successful_searches': 0},
            'api_key': "",
            'last_genre': "",
            'search_cache': {},
            'last_search_time': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def register_artist(self, artist: str, genre: str):
        """Register an artist as absolutely excluded across ALL genres"""
        artist_normalized = self.normalize_artist_name(artist)
        
        # Add to absolute exclusion
        if artist_normalized not in st.session_state.absolute_excluded_artists:
            st.session_state.absolute_excluded_artists.append(artist_normalized)
            st.session_state.artist_exclusion_count += 1
        
        # Update search history
        history_entry = {
            'artist': artist,
            'normalized_artist': artist_normalized,
            'genre': genre,
            'timestamp': time.time(),
            'search_number': st.session_state.total_searches
        }
        st.session_state.search_session_history.append(history_entry)
        
        # Update mappings
        if genre not in st.session_state.genre_artist_map:
            st.session_state.genre_artist_map[genre] = []
        st.session_state.genre_artist_map[genre].append(artist_normalized)
        
        st.session_state.artist_genre_map[artist_normalized] = genre
    
    def is_artist_excluded(self, artist: str) -> bool:
        """Check if artist is absolutely excluded (case-insensitive, fuzzy matching)"""
        if not artist:
            return False
        
        artist_normalized = self.normalize_artist_name(artist)
        
        # Exact match
        if artist_normalized in st.session_state.absolute_excluded_artists:
            return True
        
        # Fuzzy match for similar names
        for excluded_artist in st.session_state.absolute_excluded_artists:
            if self.are_artists_similar(artist_normalized, excluded_artist):
                return True
        
        return False
    
    def normalize_artist_name(self, artist: str) -> str:
        """Normalize artist name for consistent comparison"""
        # Convert to lowercase
        normalized = artist.lower().strip()
        
        # Remove common suffixes/prefixes
        normalized = re.sub(r'\s+', ' ', normalized)  # Remove extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        
        # Remove common words that don't affect uniqueness
        common_words = ['the', 'and', '&', 'feat', 'featuring', 'ft', 'vs', 'x']
        words = normalized.split()
        filtered_words = [w for w in words if w not in common_words]
        
        return ' '.join(filtered_words).strip()
    
    def are_artists_similar(self, artist1: str, artist2: str, threshold: float = 0.8) -> bool:
        """Check if two artist names are similar using fuzzy matching"""
        return SequenceMatcher(None, artist1, artist2).ratio() >= threshold
    
    def get_exclusion_list_for_prompt(self, max_artists: int = 50) -> str:
        """Get formatted exclusion list for LLM prompt"""
        if not st.session_state.absolute_excluded_artists:
            return "No artists have been recommended yet."
        
        # Get all excluded artists
        excluded_list = st.session_state.absolute_excluded_artists.copy()
        
        # Format for LLM prompt
        if len(excluded_list) <= max_artists:
            formatted_list = ", ".join([f'"{artist}"' for artist in excluded_list])
            return f"NEVER suggest these already-recommended artists: {formatted_list}"
        else:
            # Show first 20 and last 20 if list is too long
            first_20 = excluded_list[:20]
            last_20 = excluded_list[-20:]
            formatted = ", ".join([f'"{artist}"' for artist in first_20 + ["..."] + last_20])
            return f"NEVER suggest these {len(excluded_list)} already-recommended artists (showing some): {formatted}"
    
    def get_artist_history(self, artist: str) -> List[Dict]:
        """Get search history for an artist"""
        artist_normalized = self.normalize_artist_name(artist)
        return [entry for entry in st.session_state.search_session_history 
                if entry['normalized_artist'] == artist_normalized]
    
    def clear_all_exclusions(self):
        """Clear all exclusions"""
        st.session_state.absolute_excluded_artists = []
        st.session_state.artist_exclusion_count = 0
        st.session_state.search_session_history = []
        st.session_state.genre_artist_map = {}
        st.session_state.artist_genre_map = {}
        st.session_state.total_searches = 0
        st.session_state.performance_stats = {'avg_time': 0, 'total_time': 0, 'successful_searches': 0}
        st.session_state.last_genre = ""
    
    def get_statistics(self) -> Dict:
        """Get exclusion statistics"""
        return {
            'total_excluded': len(st.session_state.absolute_excluded_artists),
            'total_searches': st.session_state.total_searches,
            'successful_searches': st.session_state.performance_stats['successful_searches'],
            'unique_genres': len(st.session_state.genre_artist_map)
        }

# Initialize absolute exclusion manager
exclusion_manager = AbsoluteExclusionManager()

# =============================================================================
# ENHANCED CACHING SYSTEM
# =============================================================================

class SmartSearchCache:
    """Smart caching with genre awareness"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str):
        if key in self.cache and (time.time() - self.timestamps.get(key, 0)) < CACHE_TTL:
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

search_cache = SmartSearchCache()

def smart_youtube_search(query: str, max_results: int = 50) -> List[Dict]:
    """Smart YouTube search with caching"""
    cache_key = f"youtube_{query}_{max_results}"
    
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        search_cache.set(cache_key, results)
        return results
    except Exception:
        return []

# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def initialize_llm(api_key: str) -> Optional[ChatOpenAI]:
    """Initialize DeepSeek LLM"""
    try:
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.8,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=20
        )
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek: {e}")
        return None

# =============================================================================
# FOOLPROOF ARTIST DISCOVERY - NO REPEATS GUARANTEED
# =============================================================================

def find_artists_with_absolute_exclusion(genre: str, llm: ChatOpenAI, num_artists: int = 5) -> List[str]:
    """
    Find artists with ABSOLUTE guarantee of no repeats
    Returns list of artist names that are NOT in global exclusion
    """
    
    # Get comprehensive exclusion list for prompt
    exclusion_text = exclusion_manager.get_exclusion_list_for_prompt()
    
    # Create a STRICT prompt that absolutely prevents repeats
    prompt_template = """You are a music expert helping users discover NEW artists they haven't heard before.

The user wants to discover artists in the "{genre}" genre.

CRITICAL REQUIREMENT - ABSOLUTELY DO NOT SUGGEST THESE ARTISTS:
{exclusion_list}

These artists have ALREADY been recommended in this session and MUST NOT be suggested again.

Requirements for suggested artists:
1. MUST be popular in the "{genre}" genre
2. MUST have an official YouTube channel with at least 3 music videos
3. MUST have released music before 2024
4. MUST be completely different from the excluded artists above

Please suggest {num_artists} artists that fit the "{genre}" genre and are NOT in the excluded list.

Format your response EXACTLY like this (one artist per line, no numbering):
Artist Name 1
Artist Name 2
Artist Name 3
Artist Name 4
Artist Name 5

Do not include any other text, explanations, or formatting."""
    
    prompt = PromptTemplate(
        input_variables=["genre", "exclusion_list", "num_artists"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            result = chain.run({
                "genre": genre,
                "exclusion_list": exclusion_text,
                "num_artists": num_artists
            })
            
            # Parse artists
            artists = []
            for line in result.strip().split('\n'):
                line = line.strip()
                if line and not line[0].isdigit() and ':' not in line and '•' not in line:
                    # Clean the artist name
                    artist = line.strip()
                    artist = re.sub(r'^\d+\.\s*', '', artist)  # Remove numbering
                    artist = re.sub(r'^-\s*', '', artist)  # Remove bullet points
                    artist = re.sub(r'^[*•]\s*', '', artist)  # Remove other bullets
                    artist = artist.strip().strip('"').strip("'")
                    
                    if artist and len(artist) > 1:
                        # Check if this artist is excluded (double-check)
                        if not exclusion_manager.is_artist_excluded(artist):
                            artists.append(artist)
            
            # Filter out any that might have slipped through
            filtered_artists = []
            for artist in artists:
                if not exclusion_manager.is_artist_excluded(artist):
                    filtered_artists.append(artist)
            
            if filtered_artists:
                return filtered_artists[:num_artists]
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
    
    return []

# =============================================================================
# CHANNEL DISCOVERY WITH VALIDATION
# =============================================================================

def find_official_channel(artist: str, llm: ChatOpenAI) -> Optional[str]:
    """Find official YouTube channel for an artist"""
    
    search_queries = [
        f"{artist} VEVO official channel",
        f"{artist} official artist channel",
        f'"{artist}" official YouTube',
        f"{artist} - Topic"  # YouTube music topic channels
    ]
    
    for query in search_queries:
        try:
            results = smart_youtube_search(query, max_results=10)
            
            for result in results:
                channel_name = result['channel'].strip()
                
                # Skip topic and lyrics channels
                if any(keyword in channel_name.lower() for keyword in ['topic', 'lyrics', 'mix', 'compilation']):
                    continue
                
                # Check if channel name contains artist name
                artist_lower = artist.lower()
                channel_lower = channel_name.lower()
                
                # Simple similarity check
                if (artist_lower in channel_lower or 
                    SequenceMatcher(None, artist_lower, channel_lower).ratio() > 0.6):
                    
                    # Additional validation
                    if 'vevo' in channel_lower or 'official' in channel_lower:
                        return channel_name
                    
                    # Use LLM to validate if unsure
                    if validate_channel_with_llm(channel_name, artist, llm):
                        return channel_name
        
        except Exception:
            continue
    
    return None

def validate_channel_with_llm(channel: str, artist: str, llm: ChatOpenAI) -> bool:
    """Validate channel using LLM"""
    
    prompt = PromptTemplate(
        input_variables=["channel", "artist"],
        template="""Channel Name: "{channel}"
Artist Name: "{artist}"

Is this channel the OFFICIAL YouTube channel for this artist?
Answer only YES or NO.

Answer:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"channel": channel, "artist": artist})
        return result.strip().upper().startswith('YES')
    except Exception:
        return False

# =============================================================================
# VIDEO DISCOVERY FROM LOCKED CHANNEL
# =============================================================================

def discover_videos_from_channel(channel_name: str, artist_name: str) -> List[Dict]:
    """Discover videos only from the locked channel"""
    
    # Search specifically for this channel
    queries = [
        f"{channel_name}",
        f"{artist_name} {channel_name} official video",
        f"site:youtube.com {channel_name} music video"
    ]
    
    all_videos = []
    seen_video_ids = set()
    
    for query in queries:
        try:
            results = smart_youtube_search(query, max_results=30)
            
            for result in results:
                # Verify it's from the correct channel
                if result['channel'].strip().lower() != channel_name.lower():
                    continue
                
                video_id = result['id']
                if video_id in seen_video_ids:
                    continue
                
                seen_video_ids.add(video_id)
                
                # Score the video
                score = score_video_quality(result, artist_name)
                
                if score > 20:  # Minimum quality threshold
                    all_videos.append({
                        'id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': result['title'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
        
        except Exception:
            continue
    
    # Sort by score and remove duplicates
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove similar videos
    unique_videos = []
    seen_titles = set()
    
    for video in all_videos:
        base_title = get_base_title(video['title'])
        if base_title not in seen_titles:
            seen_titles.add(base_title)
            unique_videos.append(video)
    
    return unique_videos[:20]  # Return top 20

def score_video_quality(video: Dict, artist: str) -> int:
    """Score video quality"""
    score = 0
    title = video['title'].lower()
    artist_lower = artist.lower()
    
    # Artist name in title (good sign)
    if artist_lower in title:
        score += 30
    
    # Official indicators
    official_indicators = ['official video', 'music video', 'official audio', 'official']
    for indicator in official_indicators:
        if indicator in title:
            score += 25
            break
    
    # Avoid non-music content
    non_music = ['interview', 'behind the scenes', 'making of', 'rehearsal', 'dance practice']
    for indicator in non_music:
        if indicator in title:
            score -= 40
            break
    
    # Duration check (2-8 minutes is typical for music videos)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                if 2 <= minutes <= 8:
                    score += 20
        except:
            pass
    
    # Views (popularity indicator)
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 30
        elif 'k' in views_text:
            score += 15
    
    return max(0, score)

def get_base_title(title: str) -> str:
    """Extract base title without extra info"""
    # Remove common suffixes
    patterns = [
        r'\s*\(.*?\)',
        r'\s*\[.*?\]',
        r'\s*official.*',
        r'\s*music video.*',
        r'\s*mv.*',
        r'\s*audio.*',
        r'\s*lyrics.*',
        r'\s*ft\..*',
        r'\s*feat\..*',
        r'\s*hd.*',
        r'\s*4k.*',
        r'\s*[\d]{4}.*'  # Year suffixes
    ]
    
    base = title.lower()
    for pattern in patterns:
        base = re.sub(pattern, '', base)
    
    # Clean up
    base = re.sub(r'[^\w\s]', ' ', base)
    base = re.sub(r'\s+', ' ', base).strip()
    
    return base

def select_unique_songs(videos: List[Dict], required: int) -> List[Dict]:
    """Select unique songs from videos"""
    if len(videos) <= required:
        return videos
    
    selected = []
    seen_title_patterns = set()
    
    for video in videos:
        if len(selected) >= required:
            break
        
        base_title = get_base_title(video['title'])
        
        # Check if similar title already selected
        is_duplicate = False
        for pattern in seen_title_patterns:
            if SequenceMatcher(None, base_title, pattern).ratio() > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(video)
            seen_title_patterns.add(base_title)
    
    return selected

# =============================================================================
# MAIN DISCOVERY ENGINE - ABSOLUTELY NO REPEATS
# =============================================================================

def discover_music_for_genre(genre: str, llm: ChatOpenAI, max_attempts: int = 5) -> Tuple[Optional[Dict], str]:
    """
    Main discovery function with ABSOLUTE no-repeat guarantee
    Returns: (result_dict, status_message)
    """
    
    # Update search count
    st.session_state.total_searches += 1
    
    # Try multiple times with different strategies
    for attempt in range(1, max_attempts + 1):
        # Get fresh batch of artists with ABSOLUTE exclusion
        artists = find_artists_with_absolute_exclusion(genre, llm, num_artists=5)
        
        if not artists:
            continue
        
        # Try each artist
        for artist in artists:
            # Final verification (should already be filtered, but double-check)
            if exclusion_manager.is_artist_excluded(artist):
                continue
            
            # Find official channel
            channel = find_official_channel(artist, llm)
            if not channel:
                continue
            
            # Discover videos from locked channel
            videos = discover_videos_from_channel(channel, artist)
            if len(videos) < SONGS_REQUIRED:
                continue
            
            # Select unique songs
            selected_songs = select_unique_songs(videos, SONGS_REQUIRED)
            if len(selected_songs) >= SONGS_REQUIRED:
                # SUCCESS - Register artist immediately
                exclusion_manager.register_artist(artist, genre)
                st.session_state.performance_stats['successful_searches'] += 1
                
                return {
                    'artist': artist,
                    'channel': channel,
                    'songs': selected_songs[:SONGS_REQUIRED],
                    'attempt': attempt
                }, "SUCCESS"
        
        # If we get here, none of the artists worked
        if attempt < max_attempts:
            time.sleep(0.5)  # Brief pause before retry
    
    return None, "NO_UNIQUE_ARTISTS_FOUND"

# =============================================================================
# POPULAR GENRE VALIDATION & FALLBACK
# =============================================================================

def get_popular_genres() -> List[str]:
    """Return comprehensive list of popular genres"""
    return [
        "Pop", "Rock", "Hip Hop", "Rap", "R&B", "Country", "Jazz", "Classical",
        "Electronic", "EDM", "Blues", "Reggae", "Folk", "Metal", "Punk", "Soul",
        "Funk", "Disco", "Techno", "House", "Trance", "Dubstep", "Indie", 
        "Alternative", "K-pop", "J-pop", "C-pop", "Latin", "Reggaeton", "Salsa", 
        "Bachata", "Tango", "Ska", "Gospel", "Christian", "World", "New Age",
        "Ambient", "Soundtrack", "Musical", "Opera", "Anime", "Bollywood",
        "Afrobeats", "Amapiano", "Phonk", "Hyperpop", "Synthwave", "Lo-fi",
        "Drill", "Trap", "Vaporwave", "Future Bass", "Progressive", "Hardstyle",
        "Grunge", "Emo", "Shoegaze", "Post-rock", "Experimental", "Acoustic"
    ]

def is_genre_popular(genre: str) -> bool:
    """Check if genre is popular"""
    genre_lower = genre.lower().strip()
    popular_genres = [g.lower() for g in get_popular_genres()]
    
    # Exact match
    if genre_lower in popular_genres:
        return True
    
    # Contains match
    for popular in popular_genres:
        if popular in genre_lower or genre_lower in popular:
            return True
    
    # Common variations
    variations = {
        'hip hop': ['hiphop', 'hip-hop'],
        'r&b': ['rnb', 'rhythm and blues'],
        'edm': ['electronic dance music'],
        'k-pop': ['kpop', 'korean pop'],
        'j-pop': ['jpop', 'japanese pop'],
        'c-pop': ['cpop', 'chinese pop'],
        'lo-fi': ['lofi', 'low fi'],
        'synthwave': ['synth wave'],
        'vaporwave': ['vapor wave'],
        'future bass': ['futurebass'],
        'hardstyle': ['hard style']
    }
    
    for base, aliases in variations.items():
        if genre_lower == base or genre_lower in aliases:
            return True
    
    return False

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC",
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
    
    .success-box {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .channel-badge {
        background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px 0;
        font-weight: 600;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎵 IAMUSIC </h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            placeholder="sk-...",
            help="Get your API key from platform.deepseek.com"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Statistics
        st.header("📊 Session Statistics")
        
        stats = exclusion_manager.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Artists", stats['total_excluded'])
        with col2:
            st.metric("Successful Searches", stats['successful_searches'])
        
        st.metric("Unique Genres", stats['unique_genres'])
        st.metric("Total Attempts", stats['total_searches'])
        
        # Recent artists
        if st.session_state.search_session_history:
            with st.expander("🎤 Recently Suggested Artists"):
                for entry in st.session_state.search_session_history[-10:]:
                    st.caption(f"**{entry['artist']}** → {entry['genre']}")
        
        st.markdown("---")
        
      # Controls
        if st.button("🧹 Reset Session", use_container_width=True, type="secondary"):
            exclusion_manager.clear_all_exclusions()
            search_cache.clear()
            st.rerun()
    
    # Main search interface
    st.markdown("### 🎶 Discover  Music")
    
    with st.form("search_form", border=False):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            genre = st.text_input(
                "Enter any music genre:",
                placeholder="e.g., J-pop, Anime, K-pop, Synthwave, Reggaeton, Bollywood...",
                value=st.session_state.get('last_genre', ''),
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button(
                "Search",
                use_container_width=True,
                type="primary"
            )
    
    # Process search
    if submit and genre:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key in the sidebar!")
            return
        
        # Update last genre
        st.session_state.last_genre = genre
        
        # Initialize LLM
        with st.spinner("🤖 Initializing AI engine..."):
            llm = initialize_llm(st.session_state.api_key)
            if not llm:
                return
        
        # Show progress
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        start_time = time.time()
        
        # Check if genre is popular
        is_popular = is_genre_popular(genre)
        
        progress_placeholder.info(f"🔍 **Phase 1:** Searching for **{genre}** artists...")
        progress_bar.progress(0.2)
        
        # Main discovery
        result, status = discover_music_for_genre(genre, llm, max_attempts=3)
        
        elapsed_time = time.time() - start_time
        
        # Update performance stats
        st.session_state.performance_stats['total_time'] += elapsed_time
        if st.session_state.performance_stats['successful_searches'] > 0:
            st.session_state.performance_stats['avg_time'] = (
                st.session_state.performance_stats['total_time'] / 
                st.session_state.performance_stats['successful_searches']
            )
        
        # Handle results
        if result:
            progress_bar.progress(1.0)
            
            # Success message
            artist = result['artist']
            stats = exclusion_manager.get_statistics()
            
            # Display results
            display_results(result, genre)
        
        else:
            progress_bar.progress(1.0)
            
            # Failure message
            stats = exclusion_manager.get_statistics()
            
            st.markdown(f"""
            <div class='warning-box'>
            <h3>⚠️ Could Not Find New Artist</h3>
            <p><b>Genre:</b> {genre}</p>
            <p><b>Status:</b> {status}</p>
            <p><b>Possible Reasons:</b></p>
            <ul>
            <li>All suitable artists for this genre may have been suggested already ({stats['total_excluded']} artists suggested so far)</li>
            <li>The genre might be too niche or misspelled</li>
            <li>Artists in this genre may not have official YouTube channels</li>
            </ul>
            <p><b>Suggestions:</b></p>
            <ul>
            <li>Try a different genre</li>
            <li>Clear session to start fresh</li>
            <li>Check spelling or try broader genre name</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show popular genres as suggestions
            if not is_popular:
                st.info("💡 **Tip:** Try one of these popular genres for better results:")
                popular_genres = get_popular_genres()
                cols = st.columns(4)
                for idx, pop_genre in enumerate(popular_genres[:8]):
                    with cols[idx % 4]:
                        if st.button(pop_genre, key=f"sugg_{pop_genre}"):
                            st.session_state.last_genre = pop_genre
                            st.rerun()

def display_results(result: Dict, genre: str):
    """Display results beautifully"""
    
    st.markdown("---")
    
    # Artist header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 🎤 **{result['artist']}**")
        st.caption(f"Recommended for **{genre}** genre")
    
    with col2:
        st.caption(f"Attempt #{result['attempt']}")
    
    # Channel info with LOCKED badge
    st.markdown(f"""
    <div style='background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 15px 0;'>
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
    <div style='font-size: 18px; color: #4A00E0;'>🔒</div>
    <div style='font-weight: 600; color: #4A00E0;'>LOCKED CHANNEL</div>
    </div>
    <div style='margin-left: 0;'>
    <div style='font-size: 0.9em; color: #666; margin-bottom: 4px;'>{result['channel']}</div>
    <div style='font-size: 0.9em; color: #666;'>All songs below are exclusively from this verified official channel</div>
    </div>
</div>
    """, unsafe_allow_html=True)
    
    # Songs
    st.markdown("### 🎵 Recommended Songs")
    
    # Display songs in columns
    cols = st.columns(3)
    
    for idx, song in enumerate(result['songs']):
        with cols[idx]:
            # Video/thumbnail
            video_id = song['id']
            try:
                st.video(song['url'])
            except:
                st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)
            
            # Title
            title = song['title']
            if len(title) > 50:
                title = title[:47] + "..."
            st.markdown(f"**{title}**")
            
            # Metadata
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                if song.get('duration'):
                    st.caption(f"⏱️ {song['duration']}")
            with col_meta2:
                if song.get('views'):
                    st.caption(f"👁️ {song['views']}")
            
            # Watch button
            st.markdown(
                f'<a href="{song["url"]}" target="_blank" style="text-decoration: none;">'
                '<button style="background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); '
                'color: white; width: 100%; padding: 10px; border-radius: 6px; '
                'border: none; cursor: pointer; font-weight: 600; margin-top: 10px;">'
                '▶️ Watch on YouTube</button></a>',
                unsafe_allow_html=True
            )
    
    # Next search option
    st.markdown("---")

if __name__ == "__main__":
    main()
