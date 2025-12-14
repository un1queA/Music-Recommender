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
from langchain.chains import LLMChain

# =============================================================================
# LLM-POWERED SONG DEDUPLICATION
# =============================================================================

def are_songs_different_llm(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """Use LLM to determine if two videos represent different songs."""
    
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Analyze these YouTube video titles:
        Title 1: "{title1}"
        Title 2: "{title2}"
        
        Are these DIFFERENT SONGS or just different versions of the SAME SONG?
        
        Return EXACTLY:
        DIFFERENT: [YES/NO]
        REASON: [Brief explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2})
        different = "NO"
        for line in result.strip().split('\n'):
            if line.startswith("DIFFERENT:"):
                different = line.replace("DIFFERENT:", "").strip().upper()
                break
        return different == "YES"
    except Exception:
        return title1 != title2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select the best music videos using LLM to ensure different songs."""
    
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
# IMPROVED OFFICIAL CHANNEL DETECTION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API."""
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
    """Generate queries optimized for finding official music channels."""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Generate 3 YouTube search queries to find the OFFICIAL MUSIC CHANNEL for "{artist_name}".

        CRITICAL REQUIREMENTS:
        1. MUST exclude news, facts, updates, rumors, or any non-music content
        2. Focus on OFFICIAL music content only
        3. Include terms like "official music", "music channel", "topic channel"
        4. Exclude terms like "news", "facts", "update", "daily", "interview"
        
        Return EXACTLY 3 queries separated by |:
        QUERY_1 | QUERY_2 | QUERY_3"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3]
    except Exception:
        # Fallback to guaranteed music-focused queries
        return [
            f"{artist_name} official music",
            f"{artist_name} topic",
            f"{artist_name} music channel"
        ]

def parse_subscriber_count(subscriber_text: str) -> int:
    """Parse subscriber count from text."""
    if not subscriber_text:
        return 0
    
    subscriber_text = subscriber_text.lower().replace('subscribers', '').replace('subscriber', '').strip()
    
    try:
        if 'm' in subscriber_text:
            return int(float(subscriber_text.replace('m', '').strip()) * 1_000_000)
        elif 'k' in subscriber_text:
            return int(float(subscriber_text.replace('k', '').strip()) * 1_000)
        else:
            return int(''.join(filter(str.isdigit, subscriber_text)))
    except Exception:
        return 0

def analyze_channel_content(channel_name: str, artist_name: str) -> Dict[str, any]:
    """
    Analyze channel content to determine if it's a music channel.
    Returns detailed content analysis.
    """
    result = {
        'music_video_count': 0,
        'artist_content_ratio': 0,
        'non_music_penalty': 0,
        'total_videos_checked': 0
    }
    
    try:
        # Search for videos from this channel
        search_results = YoutubeSearch(channel_name, max_results=20).to_dict()
        
        # Filter for videos from this specific channel
        channel_videos = []
        for video in search_results:
            if video['channel'].strip() == channel_name:
                channel_videos.append(video)
        
        if not channel_videos:
            return result
        
        result['total_videos_checked'] = len(channel_videos)
        artist_name_lower = artist_name.lower()
        
        music_indicators = [
            'official mv', 'music video', 'official video', 'mv',
            'audio', 'lyric video', 'performance', 'live stage',
            'comeback', 'album', 'single', 'release'
        ]
        
        non_music_penalties = [
            'news', 'facts', 'update', 'daily', 'rumor', 'gossip',
            'interview', 'talk show', 'behind', 'documentary',
            'reaction', 'react', 'compilation', 'edit'
        ]
        
        for video in channel_videos:
            title_lower = video['title'].lower()
            
            # Check for music content
            for indicator in music_indicators:
                if indicator in title_lower:
                    result['music_video_count'] += 1
                    break
            
            # Check for artist content
            if artist_name_lower in title_lower:
                # This is content about the artist
                pass
            
            # Penalize non-music content
            for penalty in non_music_penalties:
                if penalty in title_lower:
                    result['non_music_penalty'] += 3
                    break
        
        # Calculate artist content ratio
        artist_videos = sum(1 for video in channel_videos if artist_name_lower in video['title'].lower())
        if channel_videos:
            result['artist_content_ratio'] = artist_videos / len(channel_videos)
        
    except Exception:
        pass
    
    return result

def score_channel_for_artist(channel_name: str, artist_name: str, channel_data: Dict) -> Dict[str, int]:
    """
    Score a channel based on how likely it is to be the official artist channel.
    MUCH stricter to exclude non-music channels.
    """
    scores = {
        'name_match': 0,
        'official_indicators': 0,
        'content_score': 0,
        'popularity_score': 0,
        'penalties': 0
    }
    
    artist_name_lower = artist_name.lower()
    channel_lower = channel_name.lower()
    
    # === CRITICAL: HEAVY PENALTIES FOR NON-MUSIC ===
    
    # 1. CRITICAL: Penalize news/facts channels heavily
    non_music_penalty_terms = [
        'news', 'facts', 'update', 'daily', 'rumor', 'gossip',
        'magazine', 'blog', 'vlog', 'talk', 'show', 'reaction',
        'compilation', 'edit', 'mashup', 'mix', 'lyrics only'
    ]
    
    for term in non_music_penalty_terms:
        if term in channel_lower:
            scores['penalties'] += 100  # Very heavy penalty
            break
    
    # 2. Penalize if it doesn't contain music terms
    if not any(term in channel_lower for term in ['music', 'official', 'topic', 'artist']):
        scores['penalties'] += 30
    
    # === POSITIVE SCORING ===
    
    # 3. Strong official music indicators
    strong_music_indicators = [
        'official music', 'music official', 'vevo', 'topic',
        'artist channel', 'music channel', 'records', 'entertainment'
    ]
    
    for indicator in strong_music_indicators:
        if indicator in channel_lower:
            scores['official_indicators'] += 50
    
    # 4. Exact name matching is very important
    clean_channel = re.sub(r'[\s\-_]+', '', channel_lower)
    clean_artist = re.sub(r'[\s\-_]+', '', artist_name_lower)
    
    if clean_artist == clean_channel:
        scores['name_match'] += 100  # Perfect match
    elif clean_artist in clean_channel:
        scores['name_match'] += 60  # Contains artist name
    else:
        # Check for word overlap
        artist_words = set(re.findall(r'[a-z]+', artist_name_lower))
        channel_words = set(re.findall(r'[a-z]+', channel_lower))
        common_words = artist_words.intersection(channel_words)
        if common_words:
            scores['name_match'] += len(common_words) * 20
    
    # 5. Topic channel with artist name
    if 'topic' in channel_lower and artist_name_lower in channel_lower:
        scores['official_indicators'] += 80
    
    # 6. Handle language-specific channels appropriately
    # "TWICE JAPAN OFFICIAL" is still official but not primary
    language_terms = ['japan', 'japanese', 'korea', 'korean', 'english', 'global']
    for term in language_terms:
        if term in channel_lower:
            # It's official but not primary - give moderate score
            scores['official_indicators'] += 30
            # But penalize if it's not the main channel
            if 'official' not in channel_lower:
                scores['penalties'] += 20
            break
    
    # 7. Content analysis score
    if 'content_analysis' in channel_data:
        analysis = channel_data['content_analysis']
        
        # Reward high music video count
        if analysis['music_video_count'] > 5:
            scores['content_score'] += 40
        elif analysis['music_video_count'] > 2:
            scores['content_score'] += 20
        
        # Reward high artist content ratio
        if analysis['artist_content_ratio'] > 0.7:
            scores['content_score'] += 50
        elif analysis['artist_content_ratio'] > 0.4:
            scores['content_score'] += 25
        
        # Apply non-music penalty
        scores['penalties'] += analysis['non_music_penalty']
    
    # 8. Popularity score (subscribers)
    if 'subscribers' in channel_data and channel_data['subscribers']:
        subs = parse_subscriber_count(channel_data['subscribers'])
        if subs > 1_000_000:
            scores['popularity_score'] += 40
        elif subs > 100_000:
            scores['popularity_score'] += 25
        elif subs > 10_000:
            scores['popularity_score'] += 10
    
    # 9. Additional penalties
    # Penalize if channel name is too long (often indicates fan channels)
    if len(channel_name) > len(artist_name) * 2.5:
        scores['penalties'] += 30
    
    # Penalize record labels without artist name
    label_terms = ['records', 'recordings', 'label', 'network']
    if any(term in channel_lower for term in label_terms) and not any(word in channel_lower for word in artist_name_lower.split()):
        scores['penalties'] += 40
    
    return scores

def filter_and_select_best_channel(candidates: List[Dict], artist_name: str) -> Optional[Dict]:
    """
    Smart filtering to select the best channel.
    Removes candidates that are clearly wrong.
    """
    if not candidates:
        return None
    
    artist_name_lower = artist_name.lower()
    filtered_candidates = []
    
    for candidate in candidates:
        channel_name = candidate['channel'].lower()
        total_score = candidate['score']
        
        # Skip if score is too low
        if total_score < 80:
            continue
        
        # CRITICAL: Skip if it's clearly a non-music channel
        skip_terms = ['news', 'facts', 'update', 'rumor', 'gossip', 'magazine']
        if any(term in channel_name for term in skip_terms):
            continue
        
        # Check if it contains artist name or related terms
        # Allow for cases like "BANGTANTV" for "BTS"
        artist_words = set(artist_name_lower.split())
        channel_words = set(re.findall(r'[a-z]+', channel_name))
        
        # Common music-related words that might indicate the channel
        music_related = {'music', 'official', 'topic', 'vevo', 'channel'}
        
        if not artist_words.intersection(channel_words) and not music_related.intersection(channel_words):
            # No word overlap at all, probably wrong
            continue
        
        # Additional check: channel should have reasonable content
        if 'content_analysis' in candidate:
            analysis = candidate['content_analysis']
            if analysis['music_video_count'] == 0 and analysis['total_videos_checked'] > 5:
                # No music videos found in channel - likely not a music channel
                continue
        
        filtered_candidates.append(candidate)
    
    if not filtered_candidates:
        # If we filtered everything out, return the original best
        candidates.sort(key=lambda x: x['score'], reverse=True)
        if candidates and candidates[0]['score'] > 50:
            return candidates[0]
        return None
    
    # Sort by score
    filtered_candidates.sort(key=lambda x: x['score'], reverse=True)
    return filtered_candidates[0]

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    Finds official artist channel using smarter search and filtering.
    """
    
    # Get smart search queries from LLM
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    all_candidates = []
    
    for search_query in search_queries:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                
                # Skip if we've already evaluated this channel
                if any(c['channel'] == channel_name for c in all_candidates):
                    continue
                
                # Analyze channel content
                content_analysis = analyze_channel_content(channel_name, artist_name)
                
                # Prepare channel data for scoring
                channel_data = {
                    'subscribers': result.get('channel_subscribers', ''),
                    'content_analysis': content_analysis
                }
                
                # Score this channel
                scores = score_channel_for_artist(channel_name, artist_name, channel_data)
                
                # Calculate total score
                total_score = (
                    scores['name_match'] +
                    scores['official_indicators'] +
                    scores['content_score'] +
                    scores['popularity_score'] -
                    scores['penalties']
                )
                
                # Only consider promising candidates
                if total_score >= 50:
                    all_candidates.append({
                        'channel': channel_name,
                        'score': total_score,
                        'scores_breakdown': scores,
                        'content_analysis': content_analysis,
                        'query_used': search_query,
                        'video_title': result['title'],
                        'subscribers': result.get('channel_subscribers', 'N/A')
                    })
            
            time.sleep(0.1)  # Be nice to YouTube
        except Exception:
            continue
    
    # Select and verify the best channel
    if all_candidates:
        # Use smart filtering
        best_candidate = filter_and_select_best_channel(all_candidates, artist_name)
        
        if best_candidate:
            best_channel = best_candidate['channel']
            
            # Final verification: Check channel content one more time
            try:
                channel_videos = YoutubeSearch(best_channel, max_results=15).to_dict()
                channel_videos = [r for r in channel_videos if r['channel'].strip() == best_channel]
                
                if channel_videos:
                    # Check for music content
                    music_videos = 0
                    for video in channel_videos[:10]:
                        title_lower = video['title'].lower()
                        if any(term in title_lower for term in ['music video', 'mv', 'official video', 'audio']):
                            music_videos += 1
                    
                    # Require at least some music content
                    if music_videos >= 2:
                        return best_channel, "FOUND", search_queries
                    else:
                        # Try next best candidate
                        all_candidates.sort(key=lambda x: x['score'], reverse=True)
                        if len(all_candidates) > 1:
                            second_best = all_candidates[1]
                            return second_best['channel'], "FOUND", search_queries
            
            except Exception:
                # If verification fails, return best candidate anyway
                return best_channel, "FOUND", search_queries
        
        # If no candidate passed filter, try the original best
        if all_candidates:
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_channel = all_candidates[0]['channel']
            return best_channel, "FOUND", search_queries
    
    return None, "NOT_FOUND", search_queries

def discover_channel_videos(locked_channel: str, artist_name: str, min_videos: int = 20) -> List[Dict]:
    """Discover videos from a locked channel."""
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    search_attempts = [
        locked_channel,
        f"{locked_channel} music",
        f"{artist_name} {locked_channel}",
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=30).to_dict()
            
            for result in results:
                if result['channel'].strip() != locked_channel:
                    continue
                
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
                
        except Exception:
            continue
    
    # Remove duplicates
    unique_videos = {}
    for video in all_videos:
        video_id = video['url'].split('v=')[1].split('&')[0]
        if video_id not in unique_videos:
            unique_videos[video_id] = video
    
    return sorted(unique_videos.values(), key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """Score video for likelihood of being a music video."""
    score = 0
    title = video['title']
    
    # Duration check
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
        except Exception:
            pass
    
    # Title patterns
    title_lower = title.lower()
    
    # Music video indicators
    music_indicators = [
        'official mv', 'music video', 'official video', 'mv',
        'audio', 'lyric video', 'performance', 'live stage'
    ]
    
    for indicator in music_indicators:
        if indicator in title_lower:
            score += 30
            break
    
    # Artist name in title
    if artist_name.lower() in title_lower:
        score += 20
    
    # Views
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 25
        elif 'k' in views_text:
            score += 15
    
    # Avoid non-music
    non_music_indicators = [
        'interview', 'behind the scenes', 'making of',
        'documentary', 'backstage', 'rehearsal', 'talk'
    ]
    
    for indicator in non_music_indicators:
        if indicator in title_lower:
            score -= 30
            break
    
    return max(0, score)

# =============================================================================
# ARTIST & GENRE HANDLING
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """Analyze genre for popularity."""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Return EXACTLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        EXPLANATION: [Brief reasoning]
        ARTIST_POOL: [50+/10-30/1-10]"""
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
    except Exception:
        return {
            "level": "MODERATE",
            "explanation": "Standard music genre",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict, max_attempts: int = 3) -> Dict:
    """Find artist with intelligent rotation."""
    
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
    
    # Adjust prompt
    if genre_popularity["level"] == "POPULAR":
        artist_requirement = f"Choose a DIFFERENT mainstream artist from {genre}."
    elif genre_popularity["level"] == "MODERATE":
        artist_requirement = f"Choose an established artist from {genre}."
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
        Not in excluded list: {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        CONFIDENCE: [High/Medium/Low]
        NOTE: [Brief note]"""
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
    """Provide fallback for niche genres."""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears too niche.
        
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
    
    except Exception:
        return {
            "explanation": f"'{genre}' is too niche for official YouTube content.",
            "songs": [f"{genre} song 1", f"{genre} song 2", f"{genre} song 3"],
            "suggestions": f"Try searching '{genre}' on Bandcamp or SoundCloud.",
            "genre": genre,
            "attempts": attempts
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC",
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
    
    # Main input
    st.markdown("### Enter Any Music Genre")
    
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
    
    # Process search
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
        
        # Step 1: Find artist
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
        
        with st.spinner("Finding official YouTube channel..."):
            locked_channel, channel_status, search_queries = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "FOUND" and locked_channel:
            st.success(f"✅ **Artist Channel Locked:** {locked_channel}")
            
            # Step 3: Discover videos
            with st.spinner(f"Exploring {locked_channel} for music videos..."):
                available_videos = discover_channel_videos(locked_channel, artist_name)
            
            if not available_videos:
                st.error(f"❌ No music videos found in {locked_channel}")
                st.info(f"Channel might not have public music videos.")
                
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
                    except Exception:
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
