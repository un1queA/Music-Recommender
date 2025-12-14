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
    """Generate queries optimized for finding official channels"""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""For the artist "{artist_name}" (genre: {genre}), generate 3 YouTube search queries
        that will best find their OFFICIAL CHANNEL.
        
        FOCUS ON:
        - Official artist channels (not fan channels)
        - Topic channels (YouTube's official music channels)
        - Channels with verified checkmarks
        - Channels with the artist's exact name
        
        AVOID:
        - Fan channels, compilation channels
        - Lyrics channels
        - "Official Music" channels (often not the actual artist)
        
        Return EXACTLY 3 queries separated by | (pipe symbol):
        QUERY_1 | QUERY_2 | QUERY_3"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3]
    except:
        # Improved fallback queries
        return [
            f"{artist_name} official",
            f"{artist_name} topic",
            f"@{artist_name.replace(' ', '')}"
        ]

def parse_subscriber_count(subscriber_text: str) -> int:
    """Parse subscriber count from text (e.g., '1.2M subscribers' -> 1200000)"""
    if not subscriber_text:
        return 0
    
    subscriber_text = subscriber_text.lower().replace('subscribers', '').replace('subscriber', '').strip()
    
    try:
        if 'm' in subscriber_text:
            return int(float(subscriber_text.replace('m', '').strip()) * 1_000_000)
        elif 'k' in subscriber_text:
            return int(float(subscriber_text.replace('k', '').strip()) * 1_000)
        else:
            # Try to parse as plain number
            return int(''.join(filter(str.isdigit, subscriber_text)))
    except:
        return 0

def score_channel_for_artist(channel_name: str, artist_name: str, video_title: str, 
                            subscriber_text: str = "") -> Dict[str, int]:
    """
    IMPROVED: Score a channel based on how likely it is to be the official artist channel.
    Returns scores for different categories.
    """
    scores = {
        'name_match': 0,
        'official_indicators': 0,
        'subscriber_score': 0,
        'content_quality': 0,
        'penalties': 0
    }
    
    artist_name_lower = artist_name.lower()
    channel_lower = channel_name.lower()
    title_lower = video_title.lower()
    
    # 1. NAME MATCH SCORING (Most important)
    
    # Exact name match (BANGTANTV for BTS) - HIGHEST SCORE
    clean_channel = re.sub(r'[\s\-_]+', '', channel_lower)
    clean_artist = re.sub(r'[\s\-_]+', '', artist_name_lower)
    
    if clean_artist in clean_channel or clean_channel in clean_artist:
        scores['name_match'] += 80
    
    # Contains artist name words
    artist_words = set(re.findall(r'[\w]+', artist_name_lower))
    channel_words = set(re.findall(r'[\w]+', channel_lower))
    
    common_words = artist_words.intersection(channel_words)
    if common_words:
        scores['name_match'] += len(common_words) * 15
    
    # 2. OFFICIAL INDICATORS
    
    # Topic channel (YouTube's official music channel)
    if 'topic' in channel_lower:
        scores['official_indicators'] += 60
    
    # Official keywords in channel name
    official_keywords = ['official', 'vevo', 'music', 'entertainment']
    for keyword in official_keywords:
        if keyword in channel_lower:
            scores['official_indicators'] += 20
    
    # Handle format (@channelname)
    if channel_name.startswith('@'):
        scores['official_indicators'] += 15
    
    # 3. SUBSCRIBER SCORE (Official channels have more subscribers)
    if subscriber_text:
        subscribers = parse_subscriber_count(subscriber_text)
        if subscribers > 10_000_000:
            scores['subscriber_score'] += 50
        elif subscribers > 1_000_000:
            scores['subscriber_score'] += 40
        elif subscribers > 100_000:
            scores['subscriber_score'] += 30
        elif subscribers > 10_000:
            scores['subscriber_score'] += 20
    
    # 4. CONTENT QUALITY BASED ON VIDEO TITLE
    
    # Official video patterns
    official_patterns = [
        r'\[official\s+[a-z]+\]',
        r'\(official\s+[a-z]+\)',
        r'official\s+music\s+video',
        r'official\s+audio',
        r'official\s+mv'
    ]
    
    for pattern in official_patterns:
        if re.search(pattern, title_lower):
            scores['content_quality'] += 20
    
    # Contains artist name in video title
    if any(word in title_lower for word in artist_words):
        scores['content_quality'] += 25
    
    # 5. PENALTIES (Avoid wrong channels)
    
    # Penalize common wrong channel patterns
    wrong_patterns = [
        'on beat', 'lyrics', 'lyric', 'compilation',
        'mixes', 'mix', 'fan', 'cover', 'covers',
        'best of', 'top 10', 'playlist', 'radio'
    ]
    
    for pattern in wrong_patterns:
        if pattern in channel_lower:
            scores['penalties'] += 40
        if pattern in title_lower:
            scores['penalties'] += 20
    
    # Penalize if channel name is MUCH longer than artist name
    if len(channel_name) > len(artist_name) * 3:
        scores['penalties'] += 30
    
    # Penalize record label names without artist name
    if any(label in channel_lower for label in ['records', 'recordings', 'label', 'network']) and not any(word in channel_lower for word in artist_words):
        scores['penalties'] += 40
    
    return scores

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    IMPROVED: Better channel detection that finds official artist channels.
    Now with better scoring and verification.
    """
    
    # Get smart search queries
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    # Add channel-specific searches
    channel_specific_queries = [
        f"@{artist_name.replace(' ', '')}",
        f"{artist_name} topic channel",
        f"{artist_name} official channel"
    ]
    
    all_queries = search_queries + channel_specific_queries
    all_candidates = []
    
    artist_name_lower = artist_name.lower()
    
    for search_query in all_queries:
        try:
            results = YoutubeSearch(search_query, max_results=20).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                
                # Skip if we've already evaluated this channel
                if any(c['channel'] == channel_name for c in all_candidates):
                    continue
                
                # Score this channel
                scores = score_channel_for_artist(
                    channel_name, 
                    artist_name, 
                    result['title'],
                    result.get('channel_subscribers', '')
                )
                
                # Calculate total score
                total_score = (
                    scores['name_match'] +
                    scores['official_indicators'] +
                    scores['subscriber_score'] +
                    scores['content_quality'] -
                    scores['penalties']
                )
                
                # Only consider promising candidates
                if total_score >= 70:
                    # Verify channel content
                    try:
                        # Get more videos from this channel to verify
                        channel_videos = YoutubeSearch(channel_name, max_results=15).to_dict()
                        channel_videos = [r for r in channel_videos if r['channel'].strip() == channel_name]
                        
                        if channel_videos:
                            # Check artist concentration in channel
                            artist_video_count = 0
                            for video in channel_videos[:10]:
                                video_title = video['title'].lower()
                                artist_words = set(re.findall(r'[\w]+', artist_name_lower))
                                
                                if any(word in video_title for word in artist_words):
                                    artist_video_count += 1
                            
                            concentration_score = min(50, artist_video_count * 10)
                            total_score += concentration_score
                            
                            # Additional verification: Check for music videos
                            music_video_count = 0
                            for video in channel_videos[:10]:
                                title = video['title'].lower()
                                duration = video.get('duration', '')
                                
                                # Check for music video patterns
                                music_indicators = [
                                    'official', 'mv', 'music video', 'audio',
                                    'lyric', 'performance', 'live'
                                ]
                                
                                if any(indicator in title for indicator in music_indicators):
                                    music_video_count += 1
                                
                                # Check duration (music videos are typically 2-10 minutes)
                                if duration:
                                    try:
                                        if ':' in duration:
                                            parts = duration.split(':')
                                            if len(parts) == 2:
                                                minutes = int(parts[0])
                                                if 2 <= minutes <= 10:
                                                    music_video_count += 1
                                    except:
                                        pass
                            
                            music_score = min(30, music_video_count * 3)
                            total_score += music_score
                    except:
                        # If verification fails, use original score
                        pass
                    
                    all_candidates.append({
                        'channel': channel_name,
                        'score': total_score,
                        'scores_breakdown': scores,
                        'query_used': search_query,
                        'video_title': result['title'],
                        'subscribers': result.get('channel_subscribers', 'N/A')
                    })
            
            time.sleep(0.1)  # Be nice to YouTube
        except Exception as e:
            continue
    
    # Select and verify the best channel
    if all_candidates:
        # Sort by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # DEBUG: Show top 3 candidates in console (optional)
        # print(f"Top candidates for {artist_name}:")
        # for i, cand in enumerate(all_candidates[:3]):
        #     print(f"{i+1}. {cand['channel']} - Score: {cand['score']}")
        
        # Try the top candidate first
        best_candidate = all_candidates[0]
        best_channel = best_candidate['channel']
        
        # Final deep verification
        try:
            channel_videos = YoutubeSearch(best_channel, max_results=20).to_dict()
            channel_videos = [r for r in channel_videos if r['channel'].strip() == best_channel]
            
            if channel_videos:
                # Check artist presence
                artist_videos = 0
                for video in channel_videos[:15]:
                    video_title = video['title'].lower()
                    artist_words = set(re.findall(r'[\w]+', artist_name_lower))
                    
                    if any(word in video_title for word in artist_words):
                        artist_videos += 1
                
                # Require at least 30% artist content
                if artist_videos >= max(3, len(channel_videos[:15]) * 0.3):
                    return best_channel, "FOUND", search_queries
                
                # If top candidate fails, try next best
                if len(all_candidates) > 1:
                    for candidate in all_candidates[1:3]:  # Try next 2
                        alt_channel = candidate['channel']
                        alt_videos = YoutubeSearch(alt_channel, max_results=15).to_dict()
                        alt_videos = [r for r in alt_videos if r['channel'].strip() == alt_channel]
                        
                        if alt_videos:
                            artist_videos = 0
                            for video in alt_videos[:10]:
                                video_title = video['title'].lower()
                                if any(word in video_title for word in artist_name_lower.split()):
                                    artist_videos += 1
                            
                            if artist_videos >= max(2, len(alt_videos[:10]) * 0.3):
                                return alt_channel, "FOUND", search_queries
        except:
            # If verification fails, return best candidate anyway
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

