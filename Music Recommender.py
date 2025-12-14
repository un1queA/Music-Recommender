# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
import time

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# LLM-POWERED SONG DEDUPLICATION (SMARTER THAN REGEX)
# =============================================================================

def are_songs_different_llm(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """
    Use DeepSeek to intelligently determine if two video titles represent DIFFERENT SONGS.
    Returns True if they're DIFFERENT songs, False if they're SAME song (different versions).
    """
    
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Analyze these two YouTube video titles from a music channel:
        
        Title 1: "{title1}"
        Title 2: "{title2}"
        
        Determine if these represent DIFFERENT SONGS or just DIFFERENT VERSIONS of the SAME SONG.
        
        CONSIDER:
        - Same song can have: (Official Video), (Live), (Acoustic), (Remix), (Lyrics), [4K], etc.
        - Different songs will have different core titles/names
        - Featured artists (ft./feat.) don't necessarily mean different songs
        - Language doesn't matter - focus on the core song identity
        - Live performances, acoustic versions, remixes, lyric videos of the SAME song = SAME SONG
        
        Return EXACTLY:
        DIFFERENT: [YES/NO]
        REASON: [Brief explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2})
        
        # Parse the result
        different = "NO"
        for line in result.strip().split('\n'):
            if line.startswith("DIFFERENT:"):
                different = line.replace("DIFFERENT:", "").strip().upper()
                break
        
        return different == "YES"
        
    except Exception as e:
        return title1 != title2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """
    Select the best music videos using LLM to ensure DIFFERENT SONGS.
    """
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) <= count:
        return videos[:count]
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    
    selected = []
    
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        is_different_song = True
        
        if selected:
            for selected_video in selected:
                if not are_songs_different_llm(video['title'], selected_video['title'], llm):
                    is_different_song = False
                    break
        
        if is_different_song:
            selected.append(video)
    
    if len(selected) < count:
        selected = sorted_videos[:count]
    
    return selected[:count]

# =============================================================================
# IMPROVED: BETTER OFFICIAL CHANNEL DETECTION
# =============================================================================

# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
import time

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# LLM-POWERED SONG DEDUPLICATION (SMARTER THAN REGEX)
# =============================================================================

def are_songs_different_llm(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """
    Use DeepSeek to intelligently determine if two video titles represent DIFFERENT SONGS.
    Returns True if they're DIFFERENT songs, False if they're SAME song (different versions).
    """
    
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Analyze these two YouTube video titles from a music channel:
        
        Title 1: "{title1}"
        Title 2: "{title2}"
        
        Determine if these represent DIFFERENT SONGS or just DIFFERENT VERSIONS of the SAME SONG.
        
        CONSIDER:
        - Same song can have: (Official Video), (Live), (Acoustic), (Remix), (Lyrics), [4K], etc.
        - Different songs will have different core titles/names
        - Featured artists (ft./feat.) don't necessarily mean different songs
        - Language doesn't matter - focus on the core song identity
        - Live performances, acoustic versions, remixes, lyric videos of the SAME song = SAME SONG
        
        Return EXACTLY:
        DIFFERENT: [YES/NO]
        REASON: [Brief explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2})
        
        # Parse the result
        different = "NO"
        for line in result.strip().split('\n'):
            if line.startswith("DIFFERENT:"):
                different = line.replace("DIFFERENT:", "").strip().upper()
                break
        
        return different == "YES"
        
    except Exception as e:
        return title1 != title2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """
    Select the best music videos using LLM to ensure DIFFERENT SONGS.
    """
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) <= count:
        return videos[:count]
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    
    selected = []
    
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        is_different_song = True
        
        if selected:
            for selected_video in selected:
                if not are_songs_different_llm(video['title'], selected_video['title'], llm):
                    is_different_song = False
                    break
        
        if is_different_song:
            selected.append(video)
    
    if len(selected) < count:
        selected = sorted_videos[:count]
    
    return selected[:count]

# =============================================================================
# IMPROVED: BETTER OFFICIAL CHANNEL DETECTION
# =============================================================================
def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def get_artist_search_queries(artist_name: str, genre: str, llm: ChatOpenAI) -> List[str]:
    """Generate queries optimized for finding official channels with verification"""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Generate 3 specific YouTube search queries to find the OFFICIAL YouTube channel for the artist "{artist_name}" (genre: {genre}).

        PRIORITIZE queries that will return:
        1. The artist's personal "Official Artist Channel" (with a music note 🎵 badge).
        2. The official "Topic" channel (e.g., "ArtistName - Topic"), which aggregates official music.
        3. Channels with verified checkmarks (✓).
        4. Queries like "[Artist] official YouTube channel" or the channel's exact local language name.

        AVOID queries that return fan channels, lyrics channels, or compilation channels.

        Return EXACTLY 3 queries separated by | (pipe symbol):
        QUERY_1 | QUERY_2 | QUERY_3"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3]
    except:
        # Fallback to smart queries
        return [
            f"{artist_name} official artist channel",
            f"{artist_name} topic",
            f"{artist_name} official YouTube"
        ]

def check_channel_verification_status(channel_name: str, artist_name: str) -> Dict[str, any]:
    """
    Attempts to fetch the channel page to check for verification and Official Artist Channel status.
    Returns a dict with verification indicators.
    """
    import requests
    from bs4 import BeautifulSoup
    
    result = {
        'is_verified': False,
        'is_official_artist': False,
        'official_description_match': 0,
        'error': None
    }
    
    # Try to find the channel URL from the channel name
    # This is simplified - in practice you'd need to handle different URL formats
    try:
        # First try to search for the channel to get its URL
        search_results = YoutubeSearch(channel_name, max_results=3).to_dict()
        if not search_results:
            result['error'] = "No search results found"
            return result
        
        # Find the exact channel match
        channel_url = None
        for res in search_results:
            if res['channel'].strip().lower() == channel_name.lower():
                # Construct channel URL from channel ID
                channel_id = res.get('channel_id') or res['id']
                if channel_id:
                    channel_url = f"https://www.youtube.com/channel/{channel_id}"
                    break
        
        if not channel_url:
            # Fallback to the first result's channel
            first_result = search_results[0]
            channel_id = first_result.get('channel_id') or first_result['id']
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
        
        # Fetch the channel page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(channel_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for verification badge (multiple possible patterns)
            # Note: YouTube's HTML structure changes frequently
            verification_indicators = [
                'verified', 'badge-style-type-verified', 'has-badge',
                'qualified-channel-title-badge'
            ]
            
            page_text = str(soup).lower()
            for indicator in verification_indicators:
                if indicator in page_text:
                    result['is_verified'] = True
                    break
            
            # Check for Official Artist Channel (music note 🎵)
            # Look for music note emoji or related text
            music_note_indicators = ['🎵', 'official artist', 'music artist', 'artist channel']
            for indicator in music_note_indicators:
                if indicator.lower() in page_text:
                    result['is_official_artist'] = True
                    break
            
            # Check description for official language
            description = soup.find('meta', attrs={'name': 'description'})
            if description and description.get('content'):
                desc_text = description['content'].lower()
                artist_lower = artist_name.lower()
                
                # Score description for officialness
                official_phrases = [
                    f'official channel of {artist_lower}',
                    f'official youtube channel of {artist_lower}',
                    f'welcome to the official {artist_lower}',
                    f'this is the official {artist_lower}',
                    f'{artist_lower} official channel'
                ]
                
                for phrase in official_phrases:
                    if phrase in desc_text:
                        result['official_description_match'] += 30
                
                # Additional points for having artist name in description
                if artist_lower in desc_text:
                    result['official_description_match'] += 20
                    
        else:
            result['error'] = f"HTTP {response.status_code}"
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def score_channel_for_artist(channel_name: str, artist_name: str, verification_status: Dict) -> Dict[str, int]:
    """
    Score a channel based on how likely it is to be the official artist channel.
    Now includes verification and official artist channel status.
    """
    scores = {
        'name_match': 0,
        'official_indicators': 0,
        'description_score': 0,
        'penalties': 0
    }
    
    artist_name_lower = artist_name.lower()
    channel_lower = channel_name.lower()
    
    # 1. NAME MATCH SCORING
    # Exact name match is very important
    clean_channel = re.sub(r'[\s\-_]+', '', channel_lower)
    clean_artist = re.sub(r'[\s\-_]+', '', artist_name_lower)
    
    if clean_artist in clean_channel or clean_channel in clean_artist:
        scores['name_match'] += 100  # Very high bonus for exact match
    
    # Contains artist name words
    artist_words = set(re.findall(r'[\w]+', artist_name_lower))
    channel_words = set(re.findall(r'[\w]+', channel_lower))
    
    common_words = artist_words.intersection(channel_words)
    if common_words:
        scores['name_match'] += len(common_words) * 20
    
    # 2. OFFICIAL INDICATORS (from verification check)
    if verification_status.get('is_verified'):
        scores['official_indicators'] += 150  # Massive bonus for verification
    
    if verification_status.get('is_official_artist'):
        scores['official_indicators'] += 200  # Even bigger bonus for Official Artist Channel
    
    # Topic channel indicator
    if 'topic' in channel_lower:
        scores['official_indicators'] += 80  # Topic channels are official for music
    
    # Official keywords in channel name
    official_keywords = ['official', 'vevo', 'music', 'artist']
    for keyword in official_keywords:
        if keyword in channel_lower:
            scores['official_indicators'] += 30
    
    # 3. DESCRIPTION SCORE (from verification check)
    scores['description_score'] = verification_status.get('official_description_match', 0)
    
    # 4. PENALTIES
    # Penalize common wrong channel patterns
    wrong_patterns = [
        'on beat', 'lyrics', 'lyric', 'compilation', 'mixes',
        'fan', 'cover', 'covers', 'best of', 'top 10', 'playlist',
        'radio', 'daily', 'mix'
    ]
    
    for pattern in wrong_patterns:
        if pattern in channel_lower:
            scores['penalties'] += 60  # Heavy penalty for wrong channels
    
    # Penalize if channel name is MUCH longer than artist name
    if len(channel_name) > len(artist_name) * 3:
        scores['penalties'] += 40
    
    # Penalize record label names without artist name
    label_terms = ['records', 'recordings', 'label', 'network', 'entertainment']
    if any(term in channel_lower for term in label_terms) and not any(word in channel_lower for word in artist_words):
        scores['penalties'] += 50
    
    return scores

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    COMPLETELY REWORKED: Finds official artist channel using smarter search,
    verification checks, and multiple verification methods.
    """
    
    # Get smart search queries from LLM
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    all_candidates = []
    artist_name_lower = artist_name.lower()
    
    st.info(f"🔍 Searching with queries: {', '.join(search_queries)}")
    
    for search_query in search_queries:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                
                # Skip if we've already evaluated this channel
                if any(c['channel'] == channel_name for c in all_candidates):
                    continue
                
                # Check verification and official status
                with st.spinner(f"Checking verification for {channel_name}..."):
                    verification_status = check_channel_verification_status(channel_name, artist_name)
                
                # Score this channel
                scores = score_channel_for_artist(channel_name, artist_name, verification_status)
                
                # Calculate total score
                total_score = (
                    scores['name_match'] +
                    scores['official_indicators'] +
                    scores['description_score'] -
                    scores['penalties']
                )
                
                # Add verification bonus separately for display
                verification_bonus = 0
                if verification_status.get('is_verified'):
                    verification_bonus += 150
                if verification_status.get('is_official_artist'):
                    verification_bonus += 200
                
                # Only consider promising candidates (threshold lowered since verification adds lots of points)
                if total_score >= 50 or verification_status.get('is_verified') or verification_status.get('is_official_artist'):
                    
                    # Additional content verification for promising candidates
                    content_score = 0
                    try:
                        # Get videos from this channel to verify artist content
                        channel_videos = YoutubeSearch(channel_name, max_results=10).to_dict()
                        channel_videos = [r for r in channel_videos if r['channel'].strip() == channel_name]
                        
                        if channel_videos:
                            # Check artist concentration
                            artist_video_count = 0
                            for video in channel_videos:
                                video_title = video['title'].lower()
                                if any(word in video_title for word in artist_name_lower.split()):
                                    artist_video_count += 1
                            
                            content_score = min(100, artist_video_count * 20)
                            total_score += content_score
                    except:
                        pass
                    
                    all_candidates.append({
                        'channel': channel_name,
                        'score': total_score,
                        'scores_breakdown': scores,
                        'verification_status': verification_status,
                        'content_score': content_score,
                        'query_used': search_query,
                        'video_title': result['title'],
                        'subscribers': result.get('channel_subscribers', 'N/A')
                    })
            
            time.sleep(0.2)  # Be nice to YouTube
        except Exception as e:
            continue
    
    # Select and verify the best channel
    if all_candidates:
        # Sort by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top candidates for debugging (optional)
        # st.write("Top channel candidates:", all_candidates[:3])
        
        # Try the top 3 candidates with enhanced verification
        for candidate in all_candidates[:3]:
            channel_name = candidate['channel']
            
            # Final verification: Deep content check
            try:
                channel_videos = YoutubeSearch(channel_name, max_results=20).to_dict()
                channel_videos = [r for r in channel_videos if r['channel'].strip() == channel_name]
                
                if channel_videos:
                    # Check for consistent artist content
                    artist_videos = 0
                    music_videos = 0
                    
                    for video in channel_videos[:15]:
                        video_title = video['title'].lower()
                        
                        # Check for artist name
                        if any(word in video_title for word in artist_name_lower.split()):
                            artist_videos += 1
                        
                        # Check for music video indicators
                        music_indicators = ['official video', 'music video', 'mv', 'audio', 'lyric']
                        if any(indicator in video_title for indicator in music_indicators):
                            music_videos += 1
                    
                    # Strong verification: At least 60% artist content OR official verification
                    artist_content_ratio = artist_videos / len(channel_videos[:15]) if channel_videos[:15] else 0
                    
                    if (artist_content_ratio >= 0.6) or candidate['verification_status'].get('is_verified'):
                        st.success(f"✅ Found channel with strong verification signals: {channel_name}")
                        return channel_name, "FOUND", search_queries
                    
                    # Moderate verification: Good content with some official indicators
                    if (artist_content_ratio >= 0.4 and music_videos >= 3) or candidate['verification_status'].get('is_official_artist'):
                        st.info(f"ℹ️ Found channel with moderate verification: {channel_name}")
                        return channel_name, "FOUND", search_queries
            
            except Exception as e:
                continue
        
        # If no channel passed enhanced verification, return the top candidate
        best_channel = all_candidates[0]['channel']
        st.warning(f"⚠️ Using best available channel: {best_channel}")
        return best_channel, "FOUND", search_queries
    
    return None, "NOT_FOUND", search_queries

def discover_channel_videos(locked_channel: str, artist_name: str, min_videos: int = 20) -> List[Dict]:
    """
    Discover videos from a channel and identify music videos.
    """
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    # Search for channel content
    search_attempts = [
        locked_channel,
        f"{locked_channel} music",
        f"{artist_name} {locked_channel}",
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=30).to_dict()
            
            for result in results:
                # Strict filter: Only videos from our exact channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                # Score video for likelihood of being music video
                score = score_video_music_likelihood(result, artist_name)
                
                if score >= 40:
                    all_videos.append({
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'channel': result['channel'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score,
                        'found_via': search_query
                    })
            
            if len(all_videos) >= min_videos:
                break
                
        except Exception as e:
            continue
    
    # Remove duplicates
    unique_videos = {}
    for video in all_videos:
        video_id = video['url'].split('v=')[1].split('&')[0]
        if video_id not in unique_videos:
            unique_videos[video_id] = video
    
    return sorted(unique_videos.values(), key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """
    Score video using structural patterns only.
    """
    score = 0
    title = video['title']
    
    # 1. DURATION
    if video.get('duration'):
        duration = video['duration']
        try:
            parts = duration.split(':')
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                total_seconds = minutes * 60 + seconds
                
                if 120 <= total_seconds <= 600:
                    score += 60
                elif 90 <= total_seconds <= 720:
                    score += 40
                elif 30 <= total_seconds <= 1200:
                    score += 20
                    
            elif len(parts) == 1:
                if int(parts[0]) > 180:
                    score += 30
        except:
            pass
    
    # 2. TITLE PATTERNS
    title_lower = title.lower()
    
    structural_patterns = [
        r'[\[\(][^\]\)]+[\]\)]',
        r'\d{4}',
        r'[A-Z]{2,}',
        r'[|•\-–—]',
    ]
    
    for pattern in structural_patterns:
        if re.search(pattern, title):
            score += 10
            break
    
    # 3. LENGTH
    title_length = len(title)
    if 20 <= title_length <= 80:
        score += 15
    elif 10 <= title_length <= 120:
        score += 10
    
    # 4. VIEWS
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 25
        elif 'k' in views_text:
            score += 15
        elif 'views' in views_text:
            score += 10
    
    # 5. AVOID NON-MUSIC
    non_music_indicators = [
        'live at', 'concert', 'interview', 'behind the scenes',
        'making of', 'documentary', 'backstage', 'rehearsal',
        'acoustic session', 'unplugged', 'q&a', 'talk'
    ]
    
    for indicator in non_music_indicators:
        if indicator in title_lower:
            score -= 30
            break
    
    # 6. FORMAT HINTS
    format_indicators = [
        '4k', 'hd', '1080p', 'official audio', 'visualizer',
        'lyric video', 'lyrics', 'audio only'
    ]
    
    for indicator in format_indicators:
        if indicator in title_lower:
            score += 5
    
    return max(0, score)

# =============================================================================
# INTELLIGENT ARTIST ROTATION & GENRE HANDLING
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """Analyze genre for popularity/availability"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Classify its popularity level:
        POPULAR: Mainstream, many artists with YouTube presence (e.g., Pop, Hip-Hop)
        MODERATE: Established niche, several professional artists (e.g., Synthwave, Math Rock)  
        NICHE: Limited artists, may lack official channels (e.g., Dungeon Synth, Microtonal)
        
        Return EXACTLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        EXPLANATION: [Brief reasoning]
        ARTIST_POOL: [Estimated available artists: "50+", "10-30", or "1-10"]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": search_attempts})
        
        level = "MODERATE"
        explanation = ""
        artist_pool = "10-30"
        
        for line in result.strip().split('\n'):
            if line.startswith("LEVEL:"):
                level = line.replace("LEVEL:", "").strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("ARTIST_POOL:"):
                artist_pool = line.replace("ARTIST_POOL:", "").strip()
        
        return {
            "level": level,
            "explanation": explanation,
            "artist_pool": artist_pool,
            "attempts": search_attempts
        }
    except:
        return {
            "level": "MODERATE",
            "explanation": "Standard music genre",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict, max_attempts: int = 3) -> Dict:
    """Find artist with intelligent rotation"""
    
    current_attempt = genre_popularity["attempts"]
    artist_pool = genre_popularity["artist_pool"]
    
    # Format excluded artists
    if excluded_artists:
        if artist_pool == "50+":
            show_last = min(10, len(excluded_artists))
        elif artist_pool == "10-30":
            show_last = min(5, len(excluded_artists))
        else:
            show_last = min(3, len(excluded_artists))
        
        shown_excluded = excluded_artists[-show_last:] if excluded_artists else []
        excluded_text = "\n".join([f"- {artist}" for artist in shown_excluded])
    else:
        excluded_text = "None"
    
    # Adjust prompt based on genre popularity
    if genre_popularity["level"] == "POPULAR":
        artist_requirement = f"Choose a DIFFERENT mainstream artist from {genre}. Many artists available."
    elif genre_popularity["level"] == "MODERATE":
        artist_requirement = f"Choose an established artist from {genre}. Be creative but realistic."
    else:
        if current_attempt >= 2:
            return {
                "status": "GENRE_TOO_NICHE",
                "message": f"Genre '{genre}' appears too niche.",
                "attempts": current_attempt
            }
        artist_requirement = f"This is a niche genre. Choose carefully."
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text", "artist_requirement", "attempt"],
        template="""Find ONE artist for music genre: {genre}
        
        {artist_requirement}
        
        REQUIREMENTS:
        1. Artist should have YouTube presence
        2. Not in excluded list: {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        CONFIDENCE: [High/Medium/Low - confidence in YouTube presence]
        NOTE: [Brief note about the artist]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text,
            "artist_requirement": artist_requirement,
            "attempt": current_attempt
        })
        
        artist = ""
        confidence = "Medium"
        note = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("NOTE:"):
                note = line.replace("NOTE:", "").strip()
        
        if not artist:
            return {
                "status": "NO_ARTIST_FOUND",
                "explanation": "Could not identify artist.",
                "attempts": current_attempt
            }
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "confidence": confidence,
            "note": note,
            "attempts": current_attempt
        }
    
    except Exception as e:
        return {
            "status": "ERROR",
            "explanation": f"Error: {str(e)}",
            "attempts": current_attempt
        }

def handle_niche_genre_fallback(genre: str, llm: ChatOpenAI, attempts: int) -> Dict:
    """Provide fallback for niche genres"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears too niche for official YouTube channels.
        
        After {attempts} attempts, please provide:
        1. Explanation of why this genre lacks official channels
        2. 3 representative song titles
        3. Suggestions for exploring this genre
        
        Return EXACTLY:
        EXPLANATION: [Explanation]
        SONG_1: [Song 1]
        SONG_2: [Song 2]
        SONG_3: [Song 3]
        SUGGESTIONS: [Suggestions]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": attempts})
        
        explanation = ""
        songs = []
        suggestions = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("SONG_1:"):
                songs.append(line.replace("SONG_1:", "").strip())
            elif line.startswith("SONG_2:"):
                songs.append(line.replace("SONG_2:", "").strip())
            elif line.startswith("SONG_3:"):
                songs.append(line.replace("SONG_3:", "").strip())
            elif line.startswith("SUGGESTIONS:"):
                suggestions = line.replace("SUGGESTIONS:", "").strip()
        
        return {
            "explanation": explanation,
            "songs": songs[:3],
            "suggestions": suggestions,
            "genre": genre,
            "attempts": attempts
        }
    
    except:
        return {
            "explanation": f"'{genre}' is too niche for official YouTube content.",
            "songs": [f"{genre} song 1", f"{genre} song 2", f"{genre} song 3"],
            "suggestions": f"Try searching '{genre}' on Bandcamp or SoundCloud.",
            "genre": genre,
            "attempts": attempts
        }

# =============================================================================
# STREAMLIT APP (UNCHANGED UI)
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC ",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'genre_attempts' not in st.session_state:
        st.session_state.genre_attempts = {}
    if 'genre_popularity' not in st.session_state:
        st.session_state.genre_popularity = {}
    
    # Custom CSS
    st.markdown("""
    <style>
    .universal-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
            
    }
    .artist-match-high {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .artist-match-medium {
        background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .ai-badge {
        background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔊 I AM MUSIC")

    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-..."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        st.write(f"Total sessions: {len(st.session_state.sessions)}")
        st.write(f"Excluded artists: {len(st.session_state.excluded_artists)}")
        
        if st.session_state.genre_attempts:
            st.markdown("#### Genre Attempts")
            for genre, attempts in list(st.session_state.genre_attempts.items())[-5:]:
                st.write(f"**{genre[:15]}...**: {attempts} searches")
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.sessions = []
            st.session_state.excluded_artists = []
            st.session_state.genre_attempts = {}
            st.session_state.genre_popularity = {}
            st.rerun()
    
    # Main input with Enter key support
    st.markdown("### Enter Any Music Genre")

    # Create a centered layout using columns
    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        with st.form(key="search_form"):
            genre_input = st.text_input(
                "Genre name:",
                placeholder="e.g., Tamil Pop, K-pop, Reggaeton, Synthwave...",
                label_visibility="collapsed",
                key="genre_input"
            )
            
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")
    
    # Process search only when form is submitted
    if submitted and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Track genre attempts
        if genre_input not in st.session_state.genre_attempts:
            st.session_state.genre_attempts[genre_input] = 0
        st.session_state.genre_attempts[genre_input] += 1
        current_attempt = st.session_state.genre_attempts[genre_input]
        
        # Analyze genre popularity
        if genre_input not in st.session_state.genre_popularity:
            with st.spinner("Analyzing genre..."):
                genre_analysis = analyze_genre_popularity(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = genre_analysis
        else:
            genre_analysis = st.session_state.genre_popularity[genre_input]
            genre_analysis["attempts"] = current_attempt
        
        # Step 1: Find artist with rotation
        with st.spinner(f"Finding {genre_input} artist..."):
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
        
        # Handle niche genre case
        if artist_result["status"] == "GENRE_TOO_NICHE":
            st.markdown("---")
            st.error("🎵 **Niche Genre Detected**")
            
            with st.spinner("Preparing genre explanation..."):
                niche_fallback = handle_niche_genre_fallback(genre_input, llm, current_attempt)
            
            st.write(niche_fallback["explanation"])
            
            st.markdown("### 🎶 Representative Songs")
            cols = st.columns(3)
            for idx, song in enumerate(niche_fallback["songs"][:3]):
                with cols[idx]:
                    st.info(f"**Song {idx+1}**")
                    st.code(song)
                    st.caption("⚠️ No official video available")
            
            session_data = {
                "genre": genre_input,
                "status": "NICHE_GENRE",
                "explanation": niche_fallback["explanation"],
                "songs": niche_fallback["songs"],
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] != "ARTIST_FOUND":
            st.error(f"Error: {artist_result.get('explanation', 'Unknown error')}")
            return
        
        # Artist found
        artist_name = artist_result["artist"]
        
        # Check duplicate
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} already suggested! Searching for alternative...")
            return
        
        st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Confidence: {artist_result.get('confidence', 'Medium')} | Note: {artist_result.get('note', '')}")
        
        # Step 2: Find and lock official channel
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Finding official YouTube channel"):
            locked_channel, channel_status, search_queries = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "FOUND" and locked_channel:
            st.success(f"✅ **Artist Channel Locked:** {locked_channel}")
            
            # Step 3: Discover videos
            with st.spinner(f"Exploring {locked_channel} for {artist_name} music videos..."):
                available_videos = discover_channel_videos(locked_channel, artist_name)
            
            if not available_videos:
                st.error(f"❌ No music videos found in {locked_channel}")
                st.info(f"Channel might not have public music videos. Try searching '{artist_name}' manually.")
                
                session_data = {
                    "genre": genre_input,
                    "artist": artist_name,
                    "locked_channel": locked_channel,
                    "status": "NO_VIDEOS_IN_CHANNEL",
                    "attempt": current_attempt,
                    "timestamp": time.time()
                }
                st.session_state.sessions.append(session_data)
                return
            
            # Select best 3 music videos
            with st.spinner(f"🤖 Using AI to select different songs..."):
                selected_videos = select_best_music_videos(available_videos, 3, llm)
            
            if len(selected_videos) < 3:
                st.warning(f"⚠️ Only found {len(selected_videos)} suitable music videos")
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎵 {artist_name} Music Videos from {locked_channel}")
            
            # Show AI-powered selection info
            st.success(f"✅ **AI-Selected {len(selected_videos)} Distinct Songs**")
            
            videos_found = 0
            cols = st.columns(3)
            
            for idx, video in enumerate(selected_videos):
                with cols[idx % 3]:
                    st.markdown(f'<div class="video-success">', unsafe_allow_html=True)
                    st.markdown(f"**Song {idx+1}**")
                    
                    # Display video
                    try:
                        st.video(video['url'])
                    except:
                        video_id = video['url'].split('v=')[1].split('&')[0]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Title and info
                    display_title = video['title']
                    if len(display_title) > 40:
                        display_title = display_title[:37] + "..."
                    
                    st.write(f"**{display_title}**")
                    
                    if video.get('duration'):
                        st.caption(f"Duration: {video['duration']}")
                    
                    if video.get('score'):
                        score = video['score']
                        if score >= 70:
                            st.markdown(f'<span class="artist-match-high">Match: {score}/100</span>', unsafe_allow_html=True)
                        elif score >= 40:
                            st.markdown(f'<span class="artist-match-medium">Match: {score}/100</span>', unsafe_allow_html=True)
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    videos_found += 1
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "locked_channel": locked_channel,
                "status": "COMPLETE",
                "videos_found": videos_found,
                "total_videos": len(selected_videos),
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            
            st.session_state.sessions.append(session_data)
            
            # Summary
            st.markdown("---")
            if videos_found == 3:
                st.success(f"🎉 **Perfect! Found 3 AI-selected songs from {locked_channel}**")
            else:
                st.warning(f"⚠️ Found {videos_found}/3 videos in the channel")
        
        else:
            # No channel found
            st.error(f"❌ Could not find official channel for {artist_name}")
            st.info(f"Try searching '{artist_name}' directly on YouTube.")
            
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "status": "NO_CHANNEL",
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)

if __name__ == "__main__":
    main()

def discover_channel_videos(locked_channel: str, artist_name: str, min_videos: int = 20) -> List[Dict]:
    """
    Discover videos from a channel and identify music videos.
    """
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    # Search for channel content
    search_attempts = [
        locked_channel,
        f"{locked_channel} music",
        f"{artist_name} {locked_channel}",
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=30).to_dict()
            
            for result in results:
                # Strict filter: Only videos from our exact channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                # Score video for likelihood of being music video
                score = score_video_music_likelihood(result, artist_name)
                
                if score >= 40:
                    all_videos.append({
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'channel': result['channel'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score,
                        'found_via': search_query
                    })
            
            if len(all_videos) >= min_videos:
                break
                
        except Exception as e:
            continue
    
    # Remove duplicates
    unique_videos = {}
    for video in all_videos:
        video_id = video['url'].split('v=')[1].split('&')[0]
        if video_id not in unique_videos:
            unique_videos[video_id] = video
    
    return sorted(unique_videos.values(), key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """
    Score video using structural patterns only.
    """
    score = 0
    title = video['title']
    
    # 1. DURATION
    if video.get('duration'):
        duration = video['duration']
        try:
            parts = duration.split(':')
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                total_seconds = minutes * 60 + seconds
                
                if 120 <= total_seconds <= 600:
                    score += 60
                elif 90 <= total_seconds <= 720:
                    score += 40
                elif 30 <= total_seconds <= 1200:
                    score += 20
                    
            elif len(parts) == 1:
                if int(parts[0]) > 180:
                    score += 30
        except:
            pass
    
    # 2. TITLE PATTERNS
    title_lower = title.lower()
    
    structural_patterns = [
        r'[\[\(][^\]\)]+[\]\)]',
        r'\d{4}',
        r'[A-Z]{2,}',
        r'[|•\-–—]',
    ]
    
    for pattern in structural_patterns:
        if re.search(pattern, title):
            score += 10
            break
    
    # 3. LENGTH
    title_length = len(title)
    if 20 <= title_length <= 80:
        score += 15
    elif 10 <= title_length <= 120:
        score += 10
    
    # 4. VIEWS
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 25
        elif 'k' in views_text:
            score += 15
        elif 'views' in views_text:
            score += 10
    
    # 5. AVOID NON-MUSIC
    non_music_indicators = [
        'live at', 'concert', 'interview', 'behind the scenes',
        'making of', 'documentary', 'backstage', 'rehearsal',
        'acoustic session', 'unplugged', 'q&a', 'talk'
    ]
    
    for indicator in non_music_indicators:
        if indicator in title_lower:
            score -= 30
            break
    
    # 6. FORMAT HINTS
    format_indicators = [
        '4k', 'hd', '1080p', 'official audio', 'visualizer',
        'lyric video', 'lyrics', 'audio only'
    ]
    
    for indicator in format_indicators:
        if indicator in title_lower:
            score += 5
    
    return max(0, score)

# =============================================================================
# INTELLIGENT ARTIST ROTATION & GENRE HANDLING
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """Analyze genre for popularity/availability"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Classify its popularity level:
        POPULAR: Mainstream, many artists with YouTube presence (e.g., Pop, Hip-Hop)
        MODERATE: Established niche, several professional artists (e.g., Synthwave, Math Rock)  
        NICHE: Limited artists, may lack official channels (e.g., Dungeon Synth, Microtonal)
        
        Return EXACTLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        EXPLANATION: [Brief reasoning]
        ARTIST_POOL: [Estimated available artists: "50+", "10-30", or "1-10"]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": search_attempts})
        
        level = "MODERATE"
        explanation = ""
        artist_pool = "10-30"
        
        for line in result.strip().split('\n'):
            if line.startswith("LEVEL:"):
                level = line.replace("LEVEL:", "").strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("ARTIST_POOL:"):
                artist_pool = line.replace("ARTIST_POOL:", "").strip()
        
        return {
            "level": level,
            "explanation": explanation,
            "artist_pool": artist_pool,
            "attempts": search_attempts
        }
    except:
        return {
            "level": "MODERATE",
            "explanation": "Standard music genre",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict, max_attempts: int = 3) -> Dict:
    """Find artist with intelligent rotation"""
    
    current_attempt = genre_popularity["attempts"]
    artist_pool = genre_popularity["artist_pool"]
    
    # Format excluded artists
    if excluded_artists:
        if artist_pool == "50+":
            show_last = min(10, len(excluded_artists))
        elif artist_pool == "10-30":
            show_last = min(5, len(excluded_artists))
        else:
            show_last = min(3, len(excluded_artists))
        
        shown_excluded = excluded_artists[-show_last:] if excluded_artists else []
        excluded_text = "\n".join([f"- {artist}" for artist in shown_excluded])
    else:
        excluded_text = "None"
    
    # Adjust prompt based on genre popularity
    if genre_popularity["level"] == "POPULAR":
        artist_requirement = f"Choose a DIFFERENT mainstream artist from {genre}. Many artists available."
    elif genre_popularity["level"] == "MODERATE":
        artist_requirement = f"Choose an established artist from {genre}. Be creative but realistic."
    else:
        if current_attempt >= 2:
            return {
                "status": "GENRE_TOO_NICHE",
                "message": f"Genre '{genre}' appears too niche.",
                "attempts": current_attempt
            }
        artist_requirement = f"This is a niche genre. Choose carefully."
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text", "artist_requirement", "attempt"],
        template="""Find ONE artist for music genre: {genre}
        
        {artist_requirement}
        
        REQUIREMENTS:
        1. Artist should have YouTube presence
        2. Not in excluded list: {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        CONFIDENCE: [High/Medium/Low - confidence in YouTube presence]
        NOTE: [Brief note about the artist]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text,
            "artist_requirement": artist_requirement,
            "attempt": current_attempt
        })
        
        artist = ""
        confidence = "Medium"
        note = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("NOTE:"):
                note = line.replace("NOTE:", "").strip()
        
        if not artist:
            return {
                "status": "NO_ARTIST_FOUND",
                "explanation": "Could not identify artist.",
                "attempts": current_attempt
            }
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "confidence": confidence,
            "note": note,
            "attempts": current_attempt
        }
    
    except Exception as e:
        return {
            "status": "ERROR",
            "explanation": f"Error: {str(e)}",
            "attempts": current_attempt
        }

def handle_niche_genre_fallback(genre: str, llm: ChatOpenAI, attempts: int) -> Dict:
    """Provide fallback for niche genres"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears too niche for official YouTube channels.
        
        After {attempts} attempts, please provide:
        1. Explanation of why this genre lacks official channels
        2. 3 representative song titles
        3. Suggestions for exploring this genre
        
        Return EXACTLY:
        EXPLANATION: [Explanation]
        SONG_1: [Song 1]
        SONG_2: [Song 2]
        SONG_3: [Song 3]
        SUGGESTIONS: [Suggestions]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": attempts})
        
        explanation = ""
        songs = []
        suggestions = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("SONG_1:"):
                songs.append(line.replace("SONG_1:", "").strip())
            elif line.startswith("SONG_2:"):
                songs.append(line.replace("SONG_2:", "").strip())
            elif line.startswith("SONG_3:"):
                songs.append(line.replace("SONG_3:", "").strip())
            elif line.startswith("SUGGESTIONS:"):
                suggestions = line.replace("SUGGESTIONS:", "").strip()
        
        return {
            "explanation": explanation,
            "songs": songs[:3],
            "suggestions": suggestions,
            "genre": genre,
            "attempts": attempts
        }
    
    except:
        return {
            "explanation": f"'{genre}' is too niche for official YouTube content.",
            "songs": [f"{genre} song 1", f"{genre} song 2", f"{genre} song 3"],
            "suggestions": f"Try searching '{genre}' on Bandcamp or SoundCloud.",
            "genre": genre,
            "attempts": attempts
        }

# =============================================================================
# STREAMLIT APP (UNCHANGED UI)
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC ",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'genre_attempts' not in st.session_state:
        st.session_state.genre_attempts = {}
    if 'genre_popularity' not in st.session_state:
        st.session_state.genre_popularity = {}
    
    # Custom CSS
    st.markdown("""
    <style>
    .universal-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
            
    }
    .artist-match-high {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .artist-match-medium {
        background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .ai-badge {
        background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔊 I AM MUSIC")

    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-..."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        st.write(f"Total sessions: {len(st.session_state.sessions)}")
        st.write(f"Excluded artists: {len(st.session_state.excluded_artists)}")
        
        if st.session_state.genre_attempts:
            st.markdown("#### Genre Attempts")
            for genre, attempts in list(st.session_state.genre_attempts.items())[-5:]:
                st.write(f"**{genre[:15]}...**: {attempts} searches")
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.sessions = []
            st.session_state.excluded_artists = []
            st.session_state.genre_attempts = {}
            st.session_state.genre_popularity = {}
            st.rerun()
    
    # Main input with Enter key support
    st.markdown("### Enter Any Music Genre")

    # Create a centered layout using columns
    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        with st.form(key="search_form"):
            genre_input = st.text_input(
                "Genre name:",
                placeholder="e.g., Tamil Pop, K-pop, Reggaeton, Synthwave...",
                label_visibility="collapsed",
                key="genre_input"
            )
            
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")
    
    # Process search only when form is submitted
    if submitted and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Track genre attempts
        if genre_input not in st.session_state.genre_attempts:
            st.session_state.genre_attempts[genre_input] = 0
        st.session_state.genre_attempts[genre_input] += 1
        current_attempt = st.session_state.genre_attempts[genre_input]
        
        # Analyze genre popularity
        if genre_input not in st.session_state.genre_popularity:
            with st.spinner("Analyzing genre..."):
                genre_analysis = analyze_genre_popularity(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = genre_analysis
        else:
            genre_analysis = st.session_state.genre_popularity[genre_input]
            genre_analysis["attempts"] = current_attempt
        
        # Step 1: Find artist with rotation
        with st.spinner(f"Finding {genre_input} artist..."):
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
        
        # Handle niche genre case
        if artist_result["status"] == "GENRE_TOO_NICHE":
            st.markdown("---")
            st.error("🎵 **Niche Genre Detected**")
            
            with st.spinner("Preparing genre explanation..."):
                niche_fallback = handle_niche_genre_fallback(genre_input, llm, current_attempt)
            
            st.write(niche_fallback["explanation"])
            
            st.markdown("### 🎶 Representative Songs")
            cols = st.columns(3)
            for idx, song in enumerate(niche_fallback["songs"][:3]):
                with cols[idx]:
                    st.info(f"**Song {idx+1}**")
                    st.code(song)
                    st.caption("⚠️ No official video available")
            
            session_data = {
                "genre": genre_input,
                "status": "NICHE_GENRE",
                "explanation": niche_fallback["explanation"],
                "songs": niche_fallback["songs"],
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] != "ARTIST_FOUND":
            st.error(f"Error: {artist_result.get('explanation', 'Unknown error')}")
            return
        
        # Artist found
        artist_name = artist_result["artist"]
        
        # Check duplicate
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} already suggested! Searching for alternative...")
            return
        
        st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Confidence: {artist_result.get('confidence', 'Medium')} | Note: {artist_result.get('note', '')}")
        
        # Step 2: Find and lock official channel
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Finding official YouTube channel"):
            locked_channel, channel_status, search_queries = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "FOUND" and locked_channel:
            st.success(f"✅ **Artist Channel Locked:** {locked_channel}")
            
            # Step 3: Discover videos
            with st.spinner(f"Exploring {locked_channel} for {artist_name} music videos..."):
                available_videos = discover_channel_videos(locked_channel, artist_name)
            
            if not available_videos:
                st.error(f"❌ No music videos found in {locked_channel}")
                st.info(f"Channel might not have public music videos. Try searching '{artist_name}' manually.")
                
                session_data = {
                    "genre": genre_input,
                    "artist": artist_name,
                    "locked_channel": locked_channel,
                    "status": "NO_VIDEOS_IN_CHANNEL",
                    "attempt": current_attempt,
                    "timestamp": time.time()
                }
                st.session_state.sessions.append(session_data)
                return
            
            # Select best 3 music videos
            with st.spinner(f"🤖 Using AI to select different songs..."):
                selected_videos = select_best_music_videos(available_videos, 3, llm)
            
            if len(selected_videos) < 3:
                st.warning(f"⚠️ Only found {len(selected_videos)} suitable music videos")
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎵 {artist_name} Music Videos from {locked_channel}")
            
            # Show AI-powered selection info
            st.success(f"✅ **AI-Selected {len(selected_videos)} Distinct Songs**")
            
            videos_found = 0
            cols = st.columns(3)
            
            for idx, video in enumerate(selected_videos):
                with cols[idx % 3]:
                    st.markdown(f'<div class="video-success">', unsafe_allow_html=True)
                    st.markdown(f"**Song {idx+1}**")
                    
                    # Display video
                    try:
                        st.video(video['url'])
                    except:
                        video_id = video['url'].split('v=')[1].split('&')[0]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Title and info
                    display_title = video['title']
                    if len(display_title) > 40:
                        display_title = display_title[:37] + "..."
                    
                    st.write(f"**{display_title}**")
                    
                    if video.get('duration'):
                        st.caption(f"Duration: {video['duration']}")
                    
                    if video.get('score'):
                        score = video['score']
                        if score >= 70:
                            st.markdown(f'<span class="artist-match-high">Match: {score}/100</span>', unsafe_allow_html=True)
                        elif score >= 40:
                            st.markdown(f'<span class="artist-match-medium">Match: {score}/100</span>', unsafe_allow_html=True)
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    videos_found += 1
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "locked_channel": locked_channel,
                "status": "COMPLETE",
                "videos_found": videos_found,
                "total_videos": len(selected_videos),
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            
            st.session_state.sessions.append(session_data)
            
            # Summary
            st.markdown("---")
            if videos_found == 3:
                st.success(f"🎉 **Perfect! Found 3 AI-selected songs from {locked_channel}**")
            else:
                st.warning(f"⚠️ Found {videos_found}/3 videos in the channel")
        
        else:
            # No channel found
            st.error(f"❌ Could not find official channel for {artist_name}")
            st.info(f"Try searching '{artist_name}' directly on YouTube.")
            
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "status": "NO_CHANNEL",
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)

if __name__ == "__main__":
    main()


