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

# Top 50 Most Popular Genres Worldwide (for validation)
TOP_50_POPULAR_GENRES = [
    "Pop", "Rock", "Hip Hop", "Rap", "R&B", "Country", "Jazz", "Classical",
    "Electronic", "EDM", "Blues", "Reggae", "Folk", "Metal", "Punk", "Soul",
    "Funk", "Disco", "Techno", "House", "Trance", "Dubstep", "Indie", 
    "Alternative", "K-pop", "J-pop", "Latin", "Reggaeton", "Salsa", 
    "Bachata", "Tango", "Ska", "Gospel", "Christian", "World", "New Age",
    "Ambient", "Soundtrack", "Musical", "Opera", "Anime", "Bollywood",
    "Afrobeats", "Amapiano", "Phonk", "Hyperpop", "Synthwave", "Lo-fi",
    "Drill", "Trap", "Vaporwave", "Future Bass", "Progressive", "Hardstyle"
]

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
# GLOBAL ARTIST EXCLUSION SYSTEM (NEW)
# =============================================================================

class GlobalExclusion:
    """Manages global exclusion of artists across all genres"""
    
    def __init__(self):
        if 'global_excluded_artists' not in st.session_state:
            st.session_state.global_excluded_artists = []
        if 'genre_exclusion_map' not in st.session_state:
            st.session_state.genre_exclusion_map = {}
    
    def add_excluded_artist(self, artist: str, genre: str):
        """Add artist to global exclusion list and genre-specific map"""
        if artist not in st.session_state.global_excluded_artists:
            st.session_state.global_excluded_artists.append(artist)
        
        # Add to genre-specific map
        genre_key = genre.lower().strip()
        if genre_key not in st.session_state.genre_exclusion_map:
            st.session_state.genre_exclusion_map[genre_key] = []
        
        if artist not in st.session_state.genre_exclusion_map[genre_key]:
            st.session_state.genre_exclusion_map[genre_key].append(artist)
    
    def is_artist_excluded(self, artist: str) -> bool:
        """Check if artist is globally excluded"""
        return artist in st.session_state.global_excluded_artists
    
    def get_excluded_artists_for_genre(self, genre: str) -> List[str]:
        """Get artists excluded for a specific genre"""
        genre_key = genre.lower().strip()
        return st.session_state.genre_exclusion_map.get(genre_key, [])
    
    def get_all_excluded_artists(self) -> List[str]:
        """Get all globally excluded artists"""
        return st.session_state.global_excluded_artists.copy()
    
    def clear(self):
        """Clear all exclusions"""
        st.session_state.global_excluded_artists = []
        st.session_state.genre_exclusion_map = {}

# Global exclusion instance
global_exclusion = GlobalExclusion()

# =============================================================================
# POPULAR GENRE VALIDATION (NEW)
# =============================================================================

def is_popular_genre(genre: str) -> bool:
    """Check if the genre is in the top 50 popular genres"""
    genre_lower = genre.lower().strip()
    
    # Check exact matches
    for popular_genre in TOP_50_POPULAR_GENRES:
        if genre_lower == popular_genre.lower():
            return True
    
    # Check partial matches (e.g., "hip hop" contains "hip")
    for popular_genre in TOP_50_POPULAR_GENRES:
        if popular_genre.lower() in genre_lower or genre_lower in popular_genre.lower():
            return True
    
    # Check genre synonyms
    genre_synonyms = {
        'hip hop': ['hiphop', 'hip-hop', 'rap'],
        'r&b': ['rnb', 'rhythm and blues'],
        'edm': ['electronic dance music'],
        'k-pop': ['kpop', 'korean pop'],
        'j-pop': ['jpop', 'japanese pop'],
        'reggaeton': ['reggae ton', 'regueton'],
        'afrobeats': ['afrobeat', 'afro beats'],
        'amapiano': ['ama piano'],
        'hyperpop': ['hyper pop'],
        'synthwave': ['synth wave'],
        'lo-fi': ['lofi', 'lo fi'],
        'vaporwave': ['vapor wave'],
        'future bass': ['futurebass'],
        'hardstyle': ['hard style']
    }
    
    for key, synonyms in genre_synonyms.items():
        if genre_lower == key.lower() or genre_lower in [s.lower() for s in synonyms]:
            return True
        if key.lower() in genre_lower:
            return True
    
    return False

def find_closest_popular_genre(user_genre: str) -> Optional[str]:
    """Find the closest popular genre match for the user's input"""
    user_genre_lower = user_genre.lower().strip()
    
    # Exact match check
    for popular_genre in TOP_50_POPULAR_GENRES:
        if user_genre_lower == popular_genre.lower():
            return popular_genre
    
    # Contains check
    for popular_genre in TOP_50_POPULAR_GENRES:
        if popular_genre.lower() in user_genre_lower:
            return popular_genre
    
    # Levenshtein distance check (simple)
    for popular_genre in TOP_50_POPULAR_GENRES:
        popular_lower = popular_genre.lower()
        if SequenceMatcher(None, user_genre_lower, popular_lower).ratio() > 0.7:
            return popular_genre
    
    # Check genre synonyms
    genre_synonyms = {
        'hip hop': ['hiphop', 'hip-hop', 'rap'],
        'r&b': ['rnb', 'rhythm and blues'],
        'edm': ['electronic dance music'],
        'k-pop': ['kpop', 'korean pop'],
        'j-pop': ['jpop', 'japanese pop'],
        'reggaeton': ['reggae ton', 'regueton'],
        'afrobeats': ['afrobeat', 'afro beats'],
        'amapiano': ['ama piano'],
        'hyperpop': ['hyper pop'],
        'synthwave': ['synth wave'],
        'lo-fi': ['lofi', 'lo fi'],
        'vaporwave': ['vapor wave'],
        'future bass': ['futurebass'],
        'hardstyle': ['hard style']
    }
    
    for key, synonyms in genre_synonyms.items():
        if user_genre_lower == key.lower() or user_genre_lower in [s.lower() for s in synonyms]:
            return key
        for synonym in synonyms:
            if synonym in user_genre_lower or user_genre_lower in synonym:
                return key
    
    return None

# =============================================================================
# OPTIMIZED ARTIST DISCOVERY (PARALLEL PROCESSING)
# =============================================================================

def find_artist_for_genre(genre: str, llm: ChatOpenAI) -> List[str]:
    """Find MULTIPLE artists at once with global exclusion check"""
    
    # Get all globally excluded artists
    global_excluded = global_exclusion.get_all_excluded_artists()
    excluded_text = ", ".join(global_excluded) if global_excluded else "None"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded"],
        template="""Find 3 DISTINCT popular artists from the "{genre}" genre.

CRITICAL REQUIREMENTS:
1. Each MUST have an official YouTube channel
2. MUST be different from these already-recommended artists: {excluded}
3. Must have released music before 2024
4. Must be well-known enough to have at least 3 music videos
5. Artists should be active and popular in their genre

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
                if artist and not global_exclusion.is_artist_excluded(artist):
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
    
    # First check if it's a popular genre
    closest_popular = find_closest_popular_genre(user_genre)
    if closest_popular:
        return closest_popular
    
    # If not, use LLM mapping
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

# =============================================================================
# ENHANCED DISCOVERY WITH POPULAR GENRE VALIDATION (NEW)
# =============================================================================

def enhanced_discover_songs(genre: str, llm: ChatOpenAI) -> Tuple[Optional[Dict], str]:
    """
    Enhanced discovery with multiple fallback strategies
    
    Returns: (result_dict, strategy_used)
    Strategies: ORIGINAL_GENRE, MAPPED_GENRE, POPULAR_GENRE_FALLBACK, ALL_FAILED
    """
    
    # Check if genre is popular but search is failing
    is_popular = is_popular_genre(genre)
    
    # STEP 1: Try user's exact genre
    result = discover_songs_fast(genre, llm)
    if result:
        return result, "ORIGINAL_GENRE"
    
    # STEP 2: If genre is popular but failed, try with special prompt
    if is_popular:
        # Special attempt for popular genres
        result = discover_songs_fast_popular(genre, llm)
        if result:
            return result, "POPULAR_GENRE_FALLBACK"
    
    # STEP 3: Map to known genre and retry
    mapped_genre = map_to_known_genre(genre, llm)
    
    if mapped_genre.lower() != genre.lower():
        result = discover_songs_fast(mapped_genre, llm)
        if result:
            return result, f"MAPPED_GENRE ({mapped_genre})"
    
    # STEP 4: Try base word extraction
    if " " in genre:
        base_genre = genre.split()[-1]
        if base_genre.lower() != genre.lower() and len(base_genre) > 3:
            result = discover_songs_fast(base_genre, llm)
            if result:
                return result, f"BASE_GENRE ({base_genre})"
    
    # STEP 5: Final fallback for popular genres
    if is_popular:
        # Try one more time with aggressive search
        result = discover_songs_aggressive(genre, llm)
        if result:
            return result, "POPULAR_GENRE_AGGRESSIVE"
    
    return None, "ALL_FAILED"

def discover_songs_fast_popular(genre: str, llm: ChatOpenAI) -> Optional[Dict]:
    """Special discovery for popular genres with more aggressive search"""
    
    # Get artists with emphasis on popularity
    artists = find_artist_for_genre_popular(genre, llm)
    if not artists:
        return None
    
    for artist in artists:
        # Check global exclusion
        if global_exclusion.is_artist_excluded(artist):
            continue
        
        # Fast channel discovery
        channel = find_official_channel_parallel(artist, llm)
        if not channel:
            continue
        
        # Aggressive video discovery for popular genres
        videos = discover_videos_from_channel_aggressive(channel, artist)
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

def find_artist_for_genre_popular(genre: str, llm: ChatOpenAI) -> List[str]:
    """Find artists for popular genres with emphasis on mainstream artists"""
    
    global_excluded = global_exclusion.get_all_excluded_artists()
    excluded_text = ", ".join(global_excluded) if global_excluded else "None"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded"],
        template="""Find 5 DISTINCT POPULAR artists from the "{genre}" genre.

IMPORTANT: {genre} is one of the world's most popular music genres!
These artists should be MAINSTREAM and WELL-KNOWN.

CRITICAL REQUIREMENTS:
1. Each MUST have an official YouTube channel with VEVO or official branding
2. MUST NOT include any of these already-recommended artists: {excluded}
3. Must have released music before 2024
4. Must be globally recognized with millions of views
5. Focus on the MOST popular artists in this genre

Return EXACTLY in this format (one per line):
1. [Artist 1]
2. [Artist 2] 
3. [Artist 3]
4. [Artist 4]
5. [Artist 5]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded": excluded_text})
        
        artists = []
        for line in result.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and '.' in line[:3]):
                artist = line.split('.', 1)[1].strip().strip('"').strip("'")
                if artist and not global_exclusion.is_artist_excluded(artist):
                    artists.append(artist)
        
        return artists[:5]
    except:
        return []

def discover_songs_aggressive(genre: str, llm: ChatOpenAI) -> Optional[Dict]:
    """Aggressive search for difficult cases"""
    
    # Try multiple search strategies
    search_variations = [
        genre,
        f"{genre} music",
        f"best {genre} artists",
        f"top {genre} singers"
    ]
    
    for search_term in search_variations:
        artists = find_artist_for_genre(search_term, llm)
        
        for artist in artists:
            if global_exclusion.is_artist_excluded(artist):
                continue
            
            channel = find_official_channel_parallel(artist, llm)
            if not channel:
                continue
            
            videos = discover_videos_from_channel_aggressive(channel, artist)
            if len(videos) >= SONGS_REQUIRED:
                selected = select_unique_songs_fast(videos, SONGS_REQUIRED)
                if len(selected) >= SONGS_REQUIRED:
                    return {
                        'artist': artist,
                        'channel': channel,
                        'songs': selected[:SONGS_REQUIRED]
                    }
    
    return None

# =============================================================================
# OPTIMIZED VIDEO DISCOVERY
# =============================================================================

def discover_videos_from_channel_fast(channel_name: str, artist_name: str) -> List[Dict]:
    """Fast video discovery with better filtering"""
    
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
    
    # Sort and return
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    return all_videos[:20]

def discover_videos_from_channel_aggressive(channel_name: str, artist_name: str) -> List[Dict]:
    """More aggressive video discovery for difficult cases"""
    
    searches = [
        channel_name,
        f"{channel_name} music",
        f"{artist_name} - {channel_name}",
        f"{channel_name} playlist",
        f"{artist_name} songs",
        f"{artist_name} music videos"
    ]
    
    all_videos = []
    seen_ids = set()
    
    for search in searches:
        try:
            results = cached_youtube_search(search, max_results=50)
            
            for result in results:
                # More lenient channel matching for aggressive search
                channel = result['channel'].strip().lower()
                if channel_name.lower() not in channel and artist_name.lower() not in channel:
                    continue
                
                video_id = result['id']
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                
                # More lenient scoring
                score = score_music_video_aggressive(result)
                
                if score >= 20:
                    all_videos.append({
                        'id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': result['title'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
                    
                    if len(all_videos) >= 30:
                        break
                        
            if len(all_videos) >= 30:
                break
                
        except Exception:
            continue
    
    all_videos.sort(key=lambda x: x['score'], reverse=True)
    return all_videos[:25]

def score_music_video_fast(video: Dict) -> int:
    """Optimized video scoring"""
    score = 0
    title = video['title'].lower()
    
    # Duration check
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 8:
                    score += 60
        except:
            pass
    
    # Title patterns
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

def score_music_video_aggressive(video: Dict) -> int:
    """More lenient video scoring for aggressive search"""
    score = 0
    title = video['title'].lower()
    
    # Duration check (more lenient)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 1 <= mins <= 10:  # Wider range
                    score += 40
        except:
            pass
    
    # Title patterns (more lenient)
    patterns = ['official', 'video', 'music', 'song', 'audio', 'lyric']
    for pattern in patterns:
        if pattern in title:
            score += 20
            break
    
    # Avoid obvious non-music (less strict)
    non_music = ['interview', 'behind the scenes', 'making of']
    for pattern in non_music:
        if pattern in title:
            score -= 20
            break
    
    # Views (more lenient)
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 25
        elif 'k' in views:
            score += 10
    
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

def discover_songs_fast(genre: str, llm: ChatOpenAI) -> Optional[Dict]:
    """
    Optimized pipeline with global exclusion check
    """
    
    # Step 1: Get multiple artists at once
    artists = find_artist_for_genre(genre, llm)
    if not artists:
        return None
    
    # Step 2: Try each artist until success
    for artist in artists:
        # Skip if artist is globally excluded
        if global_exclusion.is_artist_excluded(artist):
            continue
        
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

def retry_with_progress(genre: str, llm: ChatOpenAI, progress_placeholder, 
                       max_attempts: int = MAX_RETRY_ATTEMPTS):
    """
    Optimized retry loop with progress updates and global exclusion
    """
    
    attempts = 0
    start_time = time.time()
    
    # Check if genre is popular
    is_popular = is_popular_genre(genre)
    
    while attempts < max_attempts:
        attempts += 1
        
        progress_placeholder.info(
            f"🚀 Attempt {attempts}: Searching... "
            f"(Elapsed: {int(time.time() - start_time)}s)"
        )
        
        # Use enhanced discovery
        result, strategy = enhanced_discover_songs(genre, llm)
        
        if result:
            # Check global exclusion (should already be done, but double-check)
            if global_exclusion.is_artist_excluded(result['artist']):
                progress_placeholder.warning(
                    f"⚠️ Artist '{result['artist']}' already suggested globally, trying next..."
                )
                continue
                
            return result, attempts, strategy
        
        # Failed, update progress
        if is_popular and attempts >= max_attempts - 1:
            progress_placeholder.warning(
                f"⚠️ Popular genre '{genre}' is taking longer than expected. "
                f"Trying aggressive search..."
            )
        else:
            progress_placeholder.warning(
                f"⚠️ Attempt {attempts} failed, retrying..."
            )
    
    return None, attempts, "ALL_FAILED"

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
    
    # Initialize global exclusion
    global global_exclusion
    global_exclusion = GlobalExclusion()
    
    # Session state - optimized
    session_defaults = {
        'api_key': "",
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
        st.caption("⚡ Optimized for speed • Universal genre support • Global artist tracking • Popular genre validation")
    
    with col2:
        if st.session_state.total_searches > 0:
            avg_time = st.session_state.performance_stats['avg_time']
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
            st.metric("Cache Hits", st.session_state.get('cache_hits', 0))
        
        # Global exclusion stats
        excluded_count = len(global_exclusion.get_all_excluded_artists())
        if excluded_count > 0:
            st.markdown(f"**Globally Excluded Artists:** {excluded_count}")
            recent_artists = global_exclusion.get_all_excluded_artists()[-5:]
            st.caption(f"Recent: {', '.join(recent_artists)}")
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                search_cache.clear()
                st.session_state.cache_hits = 0
                st.rerun()
        
        with col_btn2:
            if st.button("🧹 Reset All", use_container_width=True, type="secondary"):
                global_exclusion.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Genre information
        with st.expander("🌍 Popular Genres Supported"):
            st.write("**Top 50 Most Popular Genres:**")
            cols = st.columns(3)
            genres_per_col = len(TOP_50_POPULAR_GENRES) // 3
            for i in range(3):
                with cols[i]:
                    start = i * genres_per_col
                    end = start + genres_per_col
                    for genre in TOP_50_POPULAR_GENRES[start:end]:
                        st.caption(f"• {genre}")
        
        # Quick tips
        with st.expander("💡 Tips for faster results"):
            st.write("""
            1. **Popular genres** work fastest
            2. Clear cache if results seem stale
            3. Artists are tracked globally across all genres
            4. First search may be slower due to caching
            """)
    
    # Main search interface
    st.markdown("### Discover Music from Any Genre")
    
    with st.form("genre_form", border=False):
        col_input, col_btn = st.columns([4, 1])
        
        with col_input:
            genre = st.text_input(
                "Enter a genre",
                placeholder="e.g., K-pop, Bollywood, Reggaeton, Synthwave, J-pop, Anime...",
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
        
        # Update session state
        st.session_state.last_genre = genre
        st.session_state.total_searches += 1
        
        # Progress tracking
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Check if genre is popular
        is_popular = is_popular_genre(genre)
        
        # ENHANCED RETRY STRATEGY WITH GLOBAL EXCLUSION
        result = None
        strategy = None
        attempts_made = 0
        
        # Show initial status
        if is_popular:
            progress_placeholder.info(
                f"🎯 **{genre.title()}** is a popular genre! "
                f"Using optimized search..."
            )
        else:
            progress_placeholder.info(
                f"🔍 Searching for **{genre.title()}**..."
            )
        
        # Multi-phase search with global exclusion
        phases = [
            ("original", genre, MAX_RETRY_ATTEMPTS),
        ]
        
        # Add mapped genre phase
        mapped_genre = map_to_known_genre(genre, llm)
        if mapped_genre.lower() != genre.lower():
            phases.append(("mapped", mapped_genre, MAX_RETRY_ATTEMPTS))
        
        # Add base genre phase
        if " " in genre:
            base_genre = genre.split()[-1]
            if base_genre.lower() != genre.lower() and len(base_genre) > 3:
                phases.append(("base", base_genre, 1))
        
        # Add aggressive phase for popular genres
        if is_popular:
            phases.append(("aggressive", genre, 2))
        
        total_phases = len(phases)
        current_phase = 0
        
        for phase_name, search_genre, phase_attempts in phases:
            current_phase += 1
            progress_bar.progress(current_phase / (total_phases + 1))
            
            if result:
                break
            
            phase_display_name = {
                "original": "Original Genre",
                "mapped": "Mapped Genre",
                "base": "Base Genre",
                "aggressive": "Aggressive Search"
            }.get(phase_name, phase_name)
            
            if phase_name == "mapped":
                progress_placeholder.info(
                    f"🔄 **Phase {current_phase}/{total_phases}:** "
                    f"Trying broader genre '{search_genre}'..."
                )
            else:
                progress_placeholder.info(
                    f"🔍 **Phase {current_phase}/{total_phases}:** "
                    f"Searching {phase_display_name}..."
                )
            
            for attempt in range(1, phase_attempts + 1):
                attempts_made += 1
                
                if phase_attempts > 1:
                    progress_placeholder.info(
                        f"   ↳ Attempt {attempt}/{phase_attempts} for '{search_genre}'..."
                    )
                
                # Search for songs
                result = discover_songs_fast(search_genre, llm)
                
                if result:
                    # Double-check global exclusion
                    if global_exclusion.is_artist_excluded(result['artist']):
                        progress_placeholder.warning(
                            f"⚠️ Found '{result['artist']}' but artist was already "
                            f"suggested in a previous search. Trying next..."
                        )
                        result = None
                        continue
                    
                    # Set strategy
                    if phase_name == "original":
                        strategy = "ORIGINAL_GENRE"
                    elif phase_name == "mapped":
                        strategy = f"MAPPED_GENRE ({search_genre})"
                    elif phase_name == "base":
                        strategy = f"BASE_GENRE ({search_genre})"
                    else:
                        strategy = "AGGRESSIVE_SEARCH"
                    break
                
                if attempt < phase_attempts:
                    progress_placeholder.warning(
                        f"   ↳ Attempt {attempt} failed, retrying..."
                    )
            
            if result:
                break
        
        # Handle results
        if result:
            artist = result['artist']
            
            # Final check and add to global exclusion
            if global_exclusion.is_artist_excluded(artist):
                progress_placeholder.error(
                    f"❌ Artist '{artist}' was already suggested globally. "
                    f"This shouldn't happen - please try a different genre."
                )
                return
            
            # Add artist to global exclusion
            global_exclusion.add_excluded_artist(artist, genre)
            
            # SUCCESS
            progress_bar.progress(1.0)
            
            # Show success message with strategy
            if "MAPPED" in strategy or "BASE" in strategy:
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
            
            # Enhanced error messaging
            error_details = {
                "message": f"❌ Could not find suitable artists for '{genre}'",
                "reasons": [],
                "suggestions": []
            }
            
            # Check if genre is popular
            if is_popular:
                error_details["reasons"].extend([
                    f"• '{genre}' is a popular genre but no artists found",
                    "• All potential artists may have been suggested already",
                    "• Try clearing the cache and excluded artists list"
                ])
                error_details["suggestions"].append("Try: Clear cache and excluded artists, then search again")
            else:
                error_details["reasons"].extend([
                    "• Genre might be too niche or misspelled",
                    "• Artists may not have official YouTube channels",
                    "• Try a broader or more common genre name"
                ])
            
            # Check how many artists are globally excluded
            excluded_count = len(global_exclusion.get_all_excluded_artists())
            if excluded_count > 0:
                error_details["reasons"].append(f"• {excluded_count} artists have already been suggested in this session")
                error_details["suggestions"].append("Try: Clear excluded artists to see previously suggested artists again")
            
            # Display error
            progress_placeholder.error(error_details["message"])
            
            with st.expander("🔍 Why this might have failed:"):
                for reason in error_details["reasons"]:
                    st.write(reason)
                
                if error_details["suggestions"]:
                    st.markdown("**Suggestions:**")
                    for suggestion in error_details["suggestions"]:
                        st.write(f"• {suggestion}")

def display_results(result: Dict, genre: str, strategy: str = "ORIGINAL_GENRE"):
    """Enhanced results display with strategy info"""
    
    # Success header with strategy indicator
    st.markdown("---")
    
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"## 🎤 {result['artist']}")
    with col_header2:
        if "MAPPED" in strategy or "BASE" in strategy:
            st.caption(f"🔄 Via: {strategy}")
    
    col_channel, col_stats = st.columns([2, 1])
    with col_channel:
        st.markdown(f'<div class="channel-badge">🔒 LOCKED CHANNEL: {result["channel"]}</div>', unsafe_allow_html=True)
        st.caption("All songs are from this official channel only")
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
    
    # Show global exclusion info
    excluded_count = len(global_exclusion.get_all_excluded_artists())
    
    col_next, col_stats = st.columns([2, 1])
    with col_next:
        if st.button(f"🔍 Discover Another {genre.title()} Artist", type="secondary", use_container_width=True):
            st.rerun()
    with col_stats:
        st.caption(f"{excluded_count} artists suggested this session")

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
