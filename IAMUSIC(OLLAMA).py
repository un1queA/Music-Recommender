# IAMUSIC - OLLAMA VERSION (Fixed Model Detection)
# Run: streamlit run IAMUSIC_OLLAMA_FIXED.py

import streamlit as st
import re
import time
import json
import requests
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
from langchain_community.llms import Ollama
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
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'absolute_excluded_artists': [],
            'artist_exclusion_count': 0,
            'search_session_history': [],
            'genre_artist_map': {},
            'artist_genre_map': {},
            'total_searches': 0,
            'performance_stats': {'avg_time': 0, 'total_time': 0, 'successful_searches': 0},
            'ollama_base_url': "http://localhost:11434",
            'ollama_model': "llama3.2:3b",  # Changed to your available model
            'last_genre': "",
            'search_cache': {},
            'last_search_time': 0,
            'available_models': [],
            'ollama_connected': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def register_artist(self, artist: str, genre: str):
        """Register an artist as absolutely excluded across ALL genres"""
        artist_normalized = self.normalize_artist_name(artist)
        
        if artist_normalized not in st.session_state.absolute_excluded_artists:
            st.session_state.absolute_excluded_artists.append(artist_normalized)
            st.session_state.artist_exclusion_count += 1
        
        history_entry = {
            'artist': artist,
            'normalized_artist': artist_normalized,
            'genre': genre,
            'timestamp': time.time(),
            'search_number': st.session_state.total_searches
        }
        st.session_state.search_session_history.append(history_entry)
        
        if genre not in st.session_state.genre_artist_map:
            st.session_state.genre_artist_map[genre] = []
        st.session_state.genre_artist_map[genre].append(artist_normalized)
        
        st.session_state.artist_genre_map[artist_normalized] = genre
    
    def is_artist_excluded(self, artist: str) -> bool:
        """Check if artist is absolutely excluded"""
        if not artist:
            return False
        
        artist_normalized = self.normalize_artist_name(artist)
        
        if artist_normalized in st.session_state.absolute_excluded_artists:
            return True
        
        for excluded_artist in st.session_state.absolute_excluded_artists:
            if self.are_artists_similar(artist_normalized, excluded_artist):
                return True
        
        return False
    
    def normalize_artist_name(self, artist: str) -> str:
        """Normalize artist name for consistent comparison"""
        normalized = artist.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
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
        
        excluded_list = st.session_state.absolute_excluded_artists.copy()
        
        if len(excluded_list) <= max_artists:
            formatted_list = ", ".join([f'"{artist}"' for artist in excluded_list])
            return f"NEVER suggest these already-recommended artists: {formatted_list}"
        else:
            first_20 = excluded_list[:20]
            last_20 = excluded_list[-20:]
            formatted = ", ".join([f'"{artist}"' for artist in first_20 + ["..."] + last_20])
            return f"NEVER suggest these {len(excluded_list)} already-recommended artists (showing some): {formatted}"
    
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
        st.session_state.search_cache = {}
    
    def get_statistics(self) -> Dict:
        """Get exclusion statistics"""
        return {
            'total_excluded': len(st.session_state.absolute_excluded_artists),
            'total_searches': st.session_state.total_searches,
            'successful_searches': st.session_state.performance_stats['successful_searches'],
            'unique_genres': len(st.session_state.genre_artist_map)
        }

exclusion_manager = AbsoluteExclusionManager()

# =============================================================================
# OLLAMA CONNECTION MANAGEMENT - FIXED VERSION
# =============================================================================

def get_available_models(base_url: str) -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
    except Exception as e:
        st.error(f"Error fetching models: {e}")
    return []

def test_ollama_connection(base_url: str, model: str = None) -> Tuple[bool, str, List[str]]:
    """Test connection to Ollama and model"""
    try:
        # Test if Ollama server is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding", []
        
        # Get available models
        data = response.json()
        available_models = [model['name'] for model in data.get('models', [])]
        
        if not available_models:
            return False, "No models found. Pull a model first.", []
        
        # If model is specified, check if it's available
        if model:
            if model not in available_models:
                # Find similar model
                model_lower = model.lower()
                for available in available_models:
                    if model_lower in available.lower() or available.lower() in model_lower:
                        return True, f"Using similar model: {available}", available_models
                
                return False, f"Model '{model}' not found. Available: {', '.join(available_models)}", available_models
            
            # Test the specific model
            test_prompt = {"model": model, "prompt": "Hello", "stream": False}
            response = requests.post(f"{base_url}/api/generate", json=test_prompt, timeout=10)
            if response.status_code == 200:
                return True, "Connection successful", available_models
            else:
                return False, f"Model test failed: {response.status_code}", available_models
        
        # If no model specified, just return success
        return True, "Ollama server is running", available_models
            
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama at {base_url}. Make sure Ollama is running.", []
    except Exception as e:
        return False, f"Error: {str(e)}", []

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
    except Exception as e:
        return []

# =============================================================================
# OLLAMA LLM INITIALIZATION - SIMPLIFIED
# =============================================================================

def initialize_llm() -> Optional[Ollama]:
    """Initialize Ollama LLM with error handling"""
    base_url = st.session_state.get('ollama_base_url', "http://localhost:11434")
    model = st.session_state.get('ollama_model', "llama3.2:3b")
    
    # Test connection
    with st.spinner("🔗 Testing Ollama connection..."):
        connected, message, available_models = test_ollama_connection(base_url, model)
        
        if not connected:
            st.error(f"❌ {message}")
            
            if available_models:
                st.info(f"📋 Available models: {', '.join(available_models)}")
                # Auto-select first available model
                if available_models:
                    st.session_state.ollama_model = available_models[0]
                    st.info(f"🔄 Auto-selected model: {available_models[0]}")
                    st.rerun()
            
            with st.expander("🔧 Setup Instructions"):
                st.write("""
                1. **Install Ollama**: [ollama.ai](https://ollama.ai)
                2. **Pull a model**:
                ```bash
                # Based on your available models:
                ollama pull llama3.2:3b
                # or try a more capable model:
                ollama pull llama3.1:8b
                ```
                3. **Start Ollama** (if not running):
                ```bash
                ollama serve
                ```
                4. **Refresh this page**
                """)
            return None
        
        st.session_state.ollama_connected = True
        st.session_state.available_models = available_models
    
    try:
        # Initialize Ollama with optimized settings
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=0.3,
            num_predict=500,  # Reduced for speed
            timeout=30,
            num_ctx=2048  # Context window
        )
    except Exception as e:
        st.error(f"Failed to initialize Ollama: {e}")
        return None

# =============================================================================
# ARTIST DISCOVERY - SIMPLIFIED FOR OLLAMA
# =============================================================================

def find_artists_with_exclusion(genre: str, llm: Ollama, num_artists: int = 5) -> List[str]:
    """Find artists with exclusion guarantee"""
    
    exclusion_text = exclusion_manager.get_exclusion_list_for_prompt()
    
    # Simplified prompt for Ollama
    prompt_template = """Find {num_artists} popular music artists in the "{genre}" genre.
    
IMPORTANT: Do NOT suggest these already-suggested artists: {exclusion_list}

Requirements:
1. Must be popular in {genre}
2. Must have an official YouTube channel
3. Must have music released before 2024
4. Must NOT be in the excluded list

Return ONLY artist names, one per line, with no numbers or bullets:

Artist Name 1
Artist Name 2
Artist Name 3
Artist Name 4
Artist Name 5"""

    prompt = PromptTemplate(
        input_variables=["genre", "exclusion_list", "num_artists"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
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
            if line and len(line) > 2:
                # Clean up
                artist = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering
                artist = re.sub(r'^[-*•]\s*', '', artist)    # Remove bullets
                artist = artist.strip('"\' ')
                
                if artist and not exclusion_manager.is_artist_excluded(artist):
                    artists.append(artist)
        
        return artists[:num_artists]
        
    except Exception as e:
        return []

# =============================================================================
# CHANNEL DISCOVERY
# =============================================================================

def find_official_channel(artist: str) -> Optional[str]:
    """Find official YouTube channel for an artist"""
    
    search_queries = [
        f"{artist} official",
        f"{artist} VEVO",
        f"{artist} official music",
        f"{artist} channel",
        f"{artist}"
    ]
    
    for query in search_queries:
        try:
            results = smart_youtube_search(query, max_results=10)
            
            for result in results:
                channel_name = result['channel'].strip()
                channel_lower = channel_name.lower()
                
                # Skip unwanted channels
                if any(kw in channel_lower for kw in ['topic', 'lyrics', 'mix', 'compilation']):
                    continue
                
                # Check if channel seems official
                artist_lower = artist.lower()
                similarity = SequenceMatcher(None, artist_lower, channel_lower).ratio()
                
                if (similarity > 0.6 or 
                    artist_lower in channel_lower or
                    any(kw in channel_lower for kw in ['vevo', 'official', 'music'])):
                    return channel_name
        
        except Exception:
            continue
    
    return None

# =============================================================================
# VIDEO DISCOVERY
# =============================================================================

def discover_videos_from_channel(channel_name: str, artist_name: str) -> List[Dict]:
    """Discover videos from the locked channel"""
    
    queries = [
        channel_name,
        f"{artist_name} {channel_name}",
        f"{channel_name} music"
    ]
    
    all_videos = []
    seen_ids = set()
    
    for query in queries:
        try:
            results = smart_youtube_search(query, max_results=20)
            
            for result in results:
                # Must be from the locked channel
                if result['channel'].strip().lower() != channel_name.lower():
                    continue
                
                video_id = result['id']
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                
                # Score video
                score = 0
                title = result['title'].lower()
                
                # Check for official indicators
                if any(kw in title for kw in ['official video', 'music video', 'mv']):
                    score += 50
                elif 'official' in title:
                    score += 30
                
                # Check duration (2-8 minutes ideal)
                if result.get('duration'):
                    try:
                        parts = result['duration'].split(':')
                        if len(parts) == 2:
                            mins = int(parts[0])
                            if 2 <= mins <= 10:
                                score += 20
                    except:
                        pass
                
                # Avoid non-music content
                if any(kw in title for kw in ['interview', 'behind', 'rehearsal', 'practice']):
                    score -= 30
                
                if score > 20:  # Minimum threshold
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
    
    # Sort by score
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove similar titles
    unique_videos = []
    seen_titles = set()
    
    for video in all_videos:
        # Simple deduplication
        title_norm = video['title'].lower()
        title_norm = re.sub(r'\s*\(.*?\)', '', title_norm)
        title_norm = re.sub(r'\s*\[.*?\]', '', title_norm)
        title_norm = re.sub(r'\s*official.*', '', title_norm)
        title_norm = re.sub(r'\s*music video.*', '', title_norm)
        title_norm = re.sub(r'[^\w\s]', ' ', title_norm)
        title_norm = re.sub(r'\s+', ' ', title_norm).strip()
        
        if title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_videos.append(video)
    
    return unique_videos[:15]

# =============================================================================
# MAIN DISCOVERY ENGINE
# =============================================================================

def discover_music_for_genre(genre: str, llm: Ollama, max_attempts: int = 3) -> Tuple[Optional[Dict], str]:
    """Main discovery function"""
    
    st.session_state.total_searches += 1
    
    for attempt in range(1, max_attempts + 1):
        artists = find_artists_with_exclusion(genre, llm, num_artists=5)
        
        if not artists:
            continue
        
        for artist in artists:
            # Final check
            if exclusion_manager.is_artist_excluded(artist):
                continue
            
            # Find channel
            channel = find_official_channel(artist)
            if not channel:
                continue
            
            # Find videos
            videos = discover_videos_from_channel(channel, artist)
            if len(videos) < SONGS_REQUIRED:
                continue
            
            # Get top songs
            selected_songs = videos[:SONGS_REQUIRED]
            
            # Register artist
            exclusion_manager.register_artist(artist, genre)
            st.session_state.performance_stats['successful_searches'] += 1
            
            return {
                'artist': artist,
                'channel': channel,
                'songs': selected_songs,
                'attempt': attempt
            }, "SUCCESS"
    
    return None, "NO_UNIQUE_ARTISTS_FOUND"

# =============================================================================
# POPULAR GENRE VALIDATION
# =============================================================================

def get_popular_genres() -> List[str]:
    """Return list of popular genres"""
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
    
    if genre_lower in popular_genres:
        return True
    
    for popular in popular_genres:
        if popular in genre_lower or genre_lower in popular:
            return True
    
    return False

# =============================================================================
# STREAMLIT UI - SIMPLIFIED
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC - Offline",
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
    }
    
    .offline-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        display: inline-block;
        font-weight: 600;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎵 IAMUSIC - Offline</h1>', unsafe_allow_html=True)
        st.caption("⚡ Local AI • No API Keys • Privacy First")
    
    with col2:
        st.markdown('<div class="offline-badge">🔌 OFFLINE MODE</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ollama Settings")
        
        # Ollama URL
        base_url = st.text_input(
            "Ollama URL",
            value=st.session_state.get('ollama_base_url', "http://localhost:11434"),
            help="Default: http://localhost:11434"
        )
        st.session_state.ollama_base_url = base_url
        
        # Get available models
        available_models = get_available_models(base_url)
        
        # Model selection
        if available_models:
            current_model = st.session_state.get('ollama_model', available_models[0])
            
            # Ensure current model is in available list
            if current_model not in available_models:
                current_model = available_models[0]
                st.session_state.ollama_model = current_model
            
            model = st.selectbox(
                "Select Model",
                options=available_models,
                index=available_models.index(current_model),
                help="Choose an Ollama model"
            )
            st.session_state.ollama_model = model
        else:
            st.warning("No models detected. Pull a model first:")
            st.code("ollama pull llama3.2:3b")
            
            model = st.selectbox(
                "Select Model",
                options=["llama3.2:3b", "llama3.1:8b", "gemma3n:e4b", "custom"],
                index=0
            )
            
            if model == "custom":
                custom_model = st.text_input("Enter model name", value="llama3.2:3b")
                st.session_state.ollama_model = custom_model
            else:
                st.session_state.ollama_model = model
        
        # Test Connection
        if st.button("🔗 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                connected, message, models = test_ollama_connection(base_url, st.session_state.ollama_model)
                if connected:
                    st.success(f"✅ {message}")
                    if models:
                        st.session_state.available_models = models
                    st.session_state.ollama_connected = True
                else:
                    st.error(f"❌ {message}")
                    if models:
                        st.info(f"Available models: {', '.join(models)}")
        
        st.markdown("---")
        
        # Stats
        st.header("📊 Stats")
        stats = exclusion_manager.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Artists", stats['total_excluded'])
        with col2:
            st.metric("Searches", stats['total_searches'])
        
        # Recent artists
        if st.session_state.search_session_history:
            with st.expander("🎤 Recent"):
                for entry in st.session_state.search_session_history[-5:]:
                    st.caption(f"**{entry['artist']}** → {entry['genre']}")
        
        st.markdown("---")
        
        # Controls
        if st.button("🧹 Reset Session", use_container_width=True, type="secondary"):
            exclusion_manager.clear_all_exclusions()
            search_cache.clear()
            st.rerun()
        
        # Quick pull commands
        with st.expander("📥 Pull Models"):
            st.write("Run in terminal:")
            st.code("ollama pull llama3.2:3b")
            st.code("ollama pull llama3.1:8b")
            st.code("ollama pull gemma3n:e4b")
    
    # Main interface
    st.markdown("### 🎶 Discover Music")
    
    with st.form("search_form"):
        genre = st.text_input(
            "Enter music genre",
            placeholder="pop, rock, k-pop, anime, hip hop...",
            value=st.session_state.get('last_genre', ''),
            label_visibility="collapsed"
        )
        
        submit = st.form_submit_button(
            "🔍 Search",
            use_container_width=True,
            type="primary"
        )
    
    # Process search
    if submit and genre:
        if not st.session_state.get('ollama_connected', False):
            st.warning("⚠️ Please test Ollama connection first!")
            return
        
        st.session_state.last_genre = genre
        
        # Initialize LLM
        with st.spinner("🤖 Initializing AI..."):
            llm = initialize_llm()
            if not llm:
                return
        
        # Search
        progress = st.progress(0)
        status = st.empty()
        
        status.info(f"🔍 Searching for {genre} artists...")
        progress.progress(30)
        
        result, message = discover_music_for_genre(genre, llm)
        
        if result:
            progress.progress(100)
            status.success(f"✅ Found {result['artist']}!")
            
            # Display results
            display_results(result, genre)
        else:
            progress.progress(100)
            status.error(f"❌ {message}")
            
            # Suggestions
            st.info("💡 **Try these tips:**")
            st.write("""
            1. Make sure Ollama is running
            2. Try a more specific genre
            3. Clear session and try again
            4. Try a different model
            """)

def display_results(result: Dict, genre: str):
    """Display results"""
    
    st.markdown("---")
    
    # Artist info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 🎤 {result['artist']}")
        st.caption(f"**Genre:** {genre}")
    
    with col2:
        st.caption(f"Attempt #{result['attempt']}")
    
    # Channel info
    st.markdown(f"""
    <div style='background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 15px 0;'>
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
    <div style='font-size: 18px; color: #4A00E0;'>🔒</div>
    <div style='font-weight: 600; color: #4A00E0;'>LOCKED CHANNEL</div>
    </div>
    <div style='margin-left: 0;'>
    <div style='font-size: 0.9em; color: #666; margin-bottom: 4px;'>{result['channel']}</div>
    <div style='font-size: 0.9em; color: #666;'>All songs are from this official channel</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Songs
    st.markdown("### 🎵 Songs")
    
    cols = st.columns(3)
    
    for idx, song in enumerate(result['songs']):
        with cols[idx]:
            # Thumbnail
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
            if song.get('duration'):
                st.caption(f"⏱️ {song['duration']}")
            if song.get('views'):
                st.caption(f"👁️ {song['views']}")
            
            # Watch button
            st.markdown(
                f'<a href="{song["url"]}" target="_blank">'
                '<button style="background: #FF0000; color: white; width: 100%; '
                'padding: 8px; border-radius: 6px; border: none; cursor: pointer; '
                'font-weight: 600; margin-top: 10px;">▶️ Watch on YouTube</button></a>',
                unsafe_allow_html=True
            )
    
    # Next search
    st.markdown("---")
    if st.button(f"🔍 Find Another {genre} Artist", use_container_width=True):
        st.rerun()

if __name__ == "__main__":
    main()
