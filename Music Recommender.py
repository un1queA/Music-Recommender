# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Set
from youtube_search import YoutubeSearch
from urllib.parse import quote_plus
import time
import hashlib

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# UNIVERSAL LANGUAGE-AGNOSTIC DISCOVERY ENGINE
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.8,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1500,
            timeout=45
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def get_artist_key(artist_name: str) -> str:
    """Create a normalized key for artist deduplication"""
    # Remove special characters, lowercase, and hash for consistent key
    normalized = re.sub(r'[^\w\s]', '', artist_name.lower().strip())
    normalized = ' '.join(normalized.split())  # Remove extra spaces
    return hashlib.md5(normalized.encode()).hexdigest()[:8]

def create_universal_discovery_chains(llm: ChatOpenAI, excluded_artists: List[str]):
    """Create chains that work for ANY genre worldwide with duplicate prevention"""
    
    # Format excluded artists for the prompt
    excluded_text = ""
    if excluded_artists:
        excluded_text = f"\n\nIMPORTANT: DO NOT suggest these artists (already suggested in this session):\n"
        excluded_text += "\n".join([f"- {artist}" for artist in excluded_artists[:10]])  # Show first 10
        if len(excluded_artists) > 10:
            excluded_text += f"\n- ... and {len(excluded_artists) - 10} more artists"
    
    # Chain 1: Genre interpretation and artist discovery
    genre_interpreter_prompt = PromptTemplate(
        input_variables=["genre_input"],
        template="""Interpret this music genre input: "{genre_input}"
        
        This could be:
        - A well-known genre (pop, rock, hip-hop)
        - A regional genre (K-pop, Reggaeton, Afrobeats)
        - A niche/subgenre (witch house, dungeon synth, hexd)
        - A fusion genre (folktronica, jazz-hop)
        - A language-specific genre (Mandopop, J-pop, French hip-hop)
        - A traditional/folk genre (Flamenco, Gamelan, Fado)
        - Even a made-up or very specific genre{excluded_artists_text}
        
        Your task:
        1. UNDERSTAND what the user means (even if obscure)
        2. Find a REAL artist who genuinely represents this style
        3. The artist MUST have YouTube presence with official videos
        4. The artist must NOT be in the excluded list above
        
        Return EXACTLY:
        GENRE_INTERPRETATION: [Your understanding of the genre]
        ARTIST: [Artist Name]
        ARTIST_CONTEXT: [Why they fit this genre, region/language info]
        SEARCH_STRATEGY: [3-5 search approaches for YouTube, include native language terms if relevant]
        
        Be creative but accurate. For obscure genres, find the closest legitimate artist.""".replace("{excluded_artists_text}", excluded_text)
    )
    
    genre_chain = LLMChain(
        llm=llm, 
        prompt=genre_interpreter_prompt,
        output_key="genre_analysis"
    )
    
    # Chain 2: Song selection with cultural awareness
    songs_prompt = PromptTemplate(
        input_variables=["genre_analysis"],
        template="""Based on this analysis: {genre_analysis}
        
        Select 3 songs that:
        1. Best represent this artist's work in this genre/style
        2. Are likely to have official YouTube videos
        3. Include major hits AND genre-representative tracks
        
        Consider:
        - Local title variations (include native script if relevant)
        - International title versions
        - Release eras that define the genre
        
        Return EXACTLY:
        SONG_1_TITLE: [Title in original language/script]
        SONG_1_CONTEXT: [Why this song represents the genre]
        SONG_2_TITLE: [Title in original language/script]  
        SONG_2_CONTEXT: [Why this song represents the genre]
        SONG_3_TITLE: [Title in original language/script]
        SONG_3_CONTEXT: [Why this song represents the genre]
        
        Preserve original language titles when appropriate."""
    )
    
    songs_chain = LLMChain(
        llm=llm,
        prompt=songs_prompt,
        output_key="songs_analysis"
    )
    
    return SequentialChain(
        chains=[genre_chain, songs_chain],
        input_variables=["genre_input"],
        output_variables=["genre_analysis", "songs_analysis"],
        verbose=False
    )

def parse_universal_analysis(genre_analysis: str, songs_analysis: str) -> Dict:
    """Parse the LLM's analysis for any genre worldwide"""
    
    result = {
        "genre_interpretation": "",
        "artist": "",
        "artist_context": "",
        "search_strategies": [],
        "songs": [],
        "confidence": 0.7,
        "artist_key": ""
    }
    
    # Parse genre analysis
    for line in genre_analysis.strip().split('\n'):
        if line.startswith("GENRE_INTERPRETATION:"):
            result["genre_interpretation"] = line.replace("GENRE_INTERPRETATION:", "").strip()
        elif line.startswith("ARTIST:"):
            result["artist"] = line.replace("ARTIST:", "").strip()
            result["artist_key"] = get_artist_key(result["artist"])
        elif line.startswith("ARTIST_CONTEXT:"):
            result["artist_context"] = line.replace("ARTIST_CONTEXT:", "").strip()
        elif line.startswith("SEARCH_STRATEGY:"):
            strategies = line.replace("SEARCH_STRATEGY:", "").strip()
            result["search_strategies"] = [s.strip() for s in strategies.split('|') if s.strip()]
    
    # Parse songs analysis
    current_song = None
    for line in songs_analysis.strip().split('\n'):
        if line.startswith("SONG_1_TITLE:"):
            current_song = {"title": line.replace("SONG_1_TITLE:", "").strip()}
        elif line.startswith("SONG_1_CONTEXT:") and current_song:
            current_song["context"] = line.replace("SONG_1_CONTEXT:", "").strip()
            result["songs"].append(current_song.copy())
        elif line.startswith("SONG_2_TITLE:"):
            current_song = {"title": line.replace("SONG_2_TITLE:", "").strip()}
        elif line.startswith("SONG_2_CONTEXT:") and current_song:
            current_song["context"] = line.replace("SONG_2_CONTEXT:", "").strip()
            result["songs"].append(current_song.copy())
        elif line.startswith("SONG_3_TITLE:"):
            current_song = {"title": line.replace("SONG_3_TITLE:", "").strip()}
        elif line.startswith("SONG_3_CONTEXT:") and current_song:
            current_song["context"] = line.replace("SONG_3_CONTEXT:", "").strip()
            result["songs"].append(current_song.copy())
    
    # Adjust confidence based on analysis quality
    if result["artist"] and len(result["songs"]) == 3:
        result["confidence"] = 0.85
    if not result["search_strategies"]:
        result["confidence"] = max(0.5, result["confidence"] - 0.2)
    
    return result

def is_duplicate_artist(artist_name: str, session_state) -> bool:
    """Check if artist has already been suggested in this session"""
    if not artist_name:
        return False
    
    artist_key = get_artist_key(artist_name)
    
    # Check in exploration history
    for exploration in session_state.get('exploration_history', []):
        if get_artist_key(exploration.get('artist', '')) == artist_key:
            return True
    
    # Check in excluded artists list
    for excluded in session_state.get('excluded_artists', []):
        if get_artist_key(excluded) == artist_key:
            return True
    
    return False

def handle_duplicate_artist(artist_name: str, genre_input: str, llm: ChatOpenAI, session_state, max_attempts: int = 3) -> Optional[Dict]:
    """Handle duplicate artist by trying to find alternative artist"""
    
    for attempt in range(max_attempts):
        st.info(f"🔄 Artist '{artist_name}' was already suggested. Finding alternative (attempt {attempt + 1}/{max_attempts})...")
        
        # Add to temporary exclusion for this search
        current_excluded = session_state.get('excluded_artists', []) + [artist_name]
        
        # Create new chain with updated exclusions
        discovery_chain = create_universal_discovery_chains(llm, current_excluded)
        
        try:
            chain_result = discovery_chain({"genre_input": genre_input})
            analysis = parse_universal_analysis(
                chain_result["genre_analysis"], 
                chain_result["songs_analysis"]
            )
            
            if analysis["artist"] and not is_duplicate_artist(analysis["artist"], session_state):
                st.success(f"✅ Found alternative artist: {analysis['artist']}")
                return analysis
        
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)[:50]}")
            continue
    
    return None

def universal_youtube_search(query: str, max_results: int = 10) -> List[Dict]:
    """Robust YouTube search that handles any language/script"""
    try:
        clean_query = re.sub(r'[^\w\s\-\.\']', ' ', query, flags=re.UNICODE)
        clean_query = ' '.join(clean_query.split())
        
        results = YoutubeSearch(clean_query, max_results=max_results).to_dict()
        return results[:max_results]
    
    except Exception as e:
        st.warning(f"Search issue for '{query[:30]}...': {str(e)[:50]}")
        return []

def find_best_video_for_song(song_title: str, artist: str, search_strategies: List[str]) -> Dict:
    """Find the best YouTube video for a song using multiple search strategies"""
    
    all_search_attempts = []
    
    # Generate search variations
    search_variations = []
    search_variations.append(f"{artist} {song_title} official")
    search_variations.append(f"{song_title} {artist}")
    
    for strategy in search_strategies[:3]:
        search_variations.append(f"{song_title} {strategy}")
    
    search_variations.append(f'"{song_title}" "{artist}"')
    search_variations.append(f"{song_title} music video")
    
    best_result = None
    best_score = 0
    
    for search_query in search_variations:
        try:
            results = universal_youtube_search(search_query, max_results=5)
            
            for result in results:
                score = calculate_video_relevance(result, song_title, artist, search_query)
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "url": f"https://www.youtube.com/watch?v={result['id']}",
                        "title": result['title'],
                        "channel": result['channel'],
                        "duration": result.get('duration', 'N/A'),
                        "views": result.get('views', 'N/A'),
                        "score": score,
                        "found_via": search_query,
                        "found": True
                    }
                    
                    all_search_attempts.append({
                        "query": search_query,
                        "score": score,
                        "result_count": len(results)
                    })
        
        except Exception:
            continue
    
    if best_result and best_score >= 25:
        best_result["search_attempts"] = all_search_attempts
        return best_result
    
    # Fallback
    fallback_results = universal_youtube_search(f"{artist} music", max_results=15)
    if fallback_results:
        return {
            "url": f"https://www.youtube.com/watch?v={fallback_results[0]['id']}",
            "title": fallback_results[0]['title'],
            "channel": fallback_results[0]['channel'],
            "score": 20,
            "found_via": f"Fallback: {artist} music",
            "found": True,
            "note": "Exact song not found, showing artist content"
        }
    
    return {
        "found": False,
        "error": "No video found",
        "search_suggestions": [
            f"Try YouTube search: '{artist} {song_title}'",
            f"Try YouTube search: '{song_title}' in {artist}'s language"
        ]
    }

def calculate_video_relevance(result: Dict, song_title: str, artist: str, search_query: str) -> int:
    """Calculate how relevant a video is to the search (language-agnostic)"""
    
    score = 0
    title = result['title'].lower()
    channel = result['channel'].lower()
    
    song_lower = song_title.lower()
    artist_lower = artist.lower()
    
    if song_lower in title:
        score += 40
    elif any(word in title for word in song_lower.split()[:3]):
        score += 25
    
    if artist_lower in channel:
        score += 30
    
    official_indicators = [
        'official', 'mv', 'music video', 
        '공식', '公式', 'oficjalny', 'официальный',
        '【mv】', '【official】', '(official)', '[official]'
    ]
    
    for indicator in official_indicators:
        if indicator in title.lower():
            score += 25
            break
    
    if any(term in title for term in ['live', 'concert', 'performance', 'stage']):
        score -= 10
    elif 'audio' in title:
        score += 15
    elif 'video' in title:
        score += 10
    
    if result.get('views'):
        try:
            views_text = result['views'].replace(',', '').replace(' views', '')
            if 'K' in views_text:
                views = float(views_text.replace('K', '')) * 1000
                score += min(20, views / 50000)
            elif 'M' in views_text:
                score += 25
        except:
            pass
    
    if result.get('duration'):
        try:
            time_parts = result['duration'].split(':')
            if len(time_parts) == 2:
                minutes = int(time_parts[0])
                if 2 <= minutes <= 10:
                    score += 10
        except:
            pass
    
    return score

# =============================================================================
# STREAMLIT APP WITH DUPLICATE PREVENTION
# =============================================================================

def main():
    st.set_page_config(
        page_title="🌐 Universal Music Explorer",
        page_icon="🎶",
        layout="wide"
    )
    
    # Initialize session state with duplicate tracking
    if 'exploration_history' not in st.session_state:
        st.session_state.exploration_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'artist_keys' not in st.session_state:
        st.session_state.artist_keys = set()
    
    # Custom CSS
    st.markdown("""
    <style>
    .duplicate-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .artist-history {
        font-size: 0.85em;
        color: #666;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # UI Header
    st.markdown('<h1>🌐 Universal Music Explorer</h1>', unsafe_allow_html=True)
    st.markdown("""
    Discover music from **ANY genre worldwide** with automatic duplicate prevention.
    """)
    
    # Show duplicate prevention status
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Required for understanding diverse genres"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.markdown("### 🚫 Duplicate Prevention")
        st.write(f"Artists excluded this session: {len(st.session_state.excluded_artists)}")
        
        if st.session_state.excluded_artists:
            with st.expander("View excluded artists"):
                for i, artist in enumerate(st.session_state.excluded_artists[-10:]):
                    st.write(f"{i+1}. {artist}")
        
        st.markdown("---")
        st.markdown("### 🌍 Exploration Stats")
        st.write(f"Genres explored: {len(st.session_state.exploration_history)}")
        
        if st.session_state.exploration_history:
            st.markdown("#### Recent Explorations")
            for item in reversed(st.session_state.exploration_history[-3:]):
                genre_display = item.get('genre', 'Unknown')[:15]
                artist_display = item.get('artist', 'Unknown')[:15]
                with st.expander(f"{genre_display} → {artist_display}"):
                    st.write(f"**Found:** {item.get('videos_found', 0)}/3 videos")
                    st.write(f"**Confidence:** {item.get('confidence', 0)*100:.0f}%")
        
        # Clear history button
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.exploration_history = []
            st.session_state.excluded_artists = []
            st.session_state.artist_keys = set()
            st.success("✅ History cleared! All artists can be suggested again.")
            st.rerun()
    
    # Main input
    st.markdown("### 🎯 Enter Any Music Genre")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        genre_input = st.text_input(
            "Genre name or description:",
            placeholder="Examples: 'vaporwave', 'Balkan brass', 'math rock', 'Kizomba', 'phonk', 'Swedish death metal', 'city pop', 'qawwali', '...or ANY style you can imagine'",
            label_visibility="collapsed",
            key="genre_input"
        )
    
    with col2:
        st.write("")
        explore_button = st.button(
            "🚀 Explore Genre",
            type="primary",
            use_container_width=True
        )
    
    # Genre examples
    if not genre_input:
        st.markdown("---")
        st.markdown("### 💡 Inspiration (Try These or Anything Else!)")
        
        example_categories = [
            ("Regional Styles", ["Bhangra", "Tango", "Highlife", "Samba", "Fado"]),
            ("Global Pop", ["K-pop", "J-pop", "Mandopop", "Latin pop", "Arabic pop"]),
            ("Niche/Subgenres", ["synthwave", "lo-fi hip hop", "dubstep", "emo rap", "hyperpop"]),
            ("Traditional", ["gamelan", "flamenco", "bluegrass", "fado", "qawwali"]),
            ("Fusion/Experimental", ["jazz fusion", "folktronica", "metalcore", "trip hop", "avant-garde"])
        ]
        
        for category_name, examples in example_categories:
            st.markdown(f"**{category_name}**")
            cols = st.columns(len(examples))
            for idx, example in enumerate(examples):
                with cols[idx]:
                    if st.button(example, use_container_width=True):
                        st.session_state.genre_input = example
                        st.rerun()
            st.write("")
    
    # Process exploration with duplicate prevention
    if explore_button and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key in the sidebar!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        with st.spinner(f"🌍 Exploring '{genre_input}' worldwide..."):
            try:
                # Get current excluded artists
                excluded_artists = st.session_state.excluded_artists
                
                # Create discovery chain with current exclusions
                discovery_chain = create_universal_discovery_chains(llm, excluded_artists)
                chain_result = discovery_chain({"genre_input": genre_input})
                
                # Parse results
                analysis = parse_universal_analysis(
                    chain_result["genre_analysis"], 
                    chain_result["songs_analysis"]
                )
                
                if not analysis["artist"]:
                    st.error("Could not interpret this genre. Try being more specific or descriptive.")
                    return
                
                # Check for duplicates
                if is_duplicate_artist(analysis["artist"], st.session_state):
                    st.markdown('<div class="duplicate-warning">⚠️ This artist was already suggested in this session!</div>', unsafe_allow_html=True)
                    
                    # Try to find alternative
                    alternative = handle_duplicate_artist(
                        analysis["artist"], 
                        genre_input, 
                        llm, 
                        st.session_state,
                        max_attempts=2
                    )
                    
                    if alternative:
                        analysis = alternative
                    else:
                        st.error("❌ Could not find a new artist for this genre. All suitable artists have been suggested.")
                        st.info("Try a different genre or clear history to start fresh.")
                        return
                
                # Add to excluded artists
                if analysis["artist"] not in st.session_state.excluded_artists:
                    st.session_state.excluded_artists.append(analysis["artist"])
                    st.session_state.artist_keys.add(analysis["artist_key"])
                
                # Display interpretation
                st.success(f"**Interpretation:** {analysis['genre_interpretation']}")
                
                with st.expander("📋 Analysis Details", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**🎤 Artist Found**")
                        st.markdown(f"### {analysis['artist']}")
                        st.write(analysis['artist_context'])
                        st.metric("Confidence", f"{analysis['confidence']*100:.0f}%")
                        
                        # Show duplicate prevention status
                        st.markdown('<div class="artist-history">✅ This artist is now excluded from future suggestions</div>', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown("**🔍 Search Strategy**")
                        for strategy in analysis['search_strategies'][:3]:
                            st.code(strategy)
                
                st.markdown("---")
                st.markdown(f"### 🎵 {analysis['artist']}'s Genre-Defining Songs")
                
                # Find and display videos
                videos_found = 0
                video_results = []
                
                for idx, song in enumerate(analysis['songs'][:3]):
                    with st.spinner(f"Searching for '{song['title']}'..."):
                        video_result = find_best_video_for_song(
                            song['title'], 
                            analysis['artist'], 
                            analysis['search_strategies']
                        )
                    
                    video_results.append(video_result)
                    
                    if video_result.get("found"):
                        videos_found += 1
                
                # Display in grid
                cols = st.columns(3)
                
                for idx, (song, video_result) in enumerate(zip(analysis['songs'][:3], video_results)):
                    with cols[idx]:
                        st.markdown(f'<div><strong>Song {idx+1}</strong><br>{song["title"]}</div>', unsafe_allow_html=True)
                        
                        if song.get("context"):
                            with st.expander("Why this song?", expanded=False):
                                st.write(song["context"])
                        
                        if video_result.get("found"):
                            try:
                                st.video(video_result["url"])
                            except:
                                st.image(f"https://img.youtube.com/vi/{video_result['url'].split('v=')[1].split('&')[0]}/hqdefault.jpg", 
                                         use_column_width=True)
                            
                            with st.expander("Video Details", expanded=False):
                                st.write(f"**Title:** {video_result['title'][:60]}...")
                                st.write(f"**Channel:** {video_result['channel']}")
                                if video_result.get('duration') != 'N/A':
                                    st.write(f"**Duration:** {video_result['duration']}")
                                if video_result.get('views') != 'N/A':
                                    st.write(f"**Views:** {video_result['views']}")
                                st.write(f"**Relevance:** {video_result['score']}/100")
                                st.write(f"**Found via:** `{video_result['found_via'][:40]}...`")
                            
                            st.markdown(
                                f'<a href="{video_result["url"]}" target="_blank">'
                                '<button style="background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); '
                                'color: white; border: none; padding: 10px 20px; border-radius: 6px; '
                                'cursor: pointer; width: 100%; margin-top: 10px; font-weight: bold;">'
                                '▶ Watch on YouTube</button></a>',
                                unsafe_allow_html=True
                            )
                            
                            if video_result.get("note"):
                                st.info(video_result["note"])
                        
                        else:
                            st.warning("🎵 Video not found")
                            if video_result.get("search_suggestions"):
                                st.caption("Try searching manually:")
                                for suggestion in video_result["search_suggestions"][:2]:
                                    st.code(suggestion)
                
                # Store in history
                st.session_state.exploration_history.append({
                    "genre": genre_input,
                    "artist": analysis["artist"],
                    "artist_key": analysis["artist_key"],
                    "interpretation": analysis["genre_interpretation"],
                    "videos_found": videos_found,
                    "confidence": analysis["confidence"],
                    "timestamp": time.time()
                })
                
                # Summary
                st.markdown("---")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    if videos_found == 3:
                        st.success(f"✅ Perfect match! Found all 3 songs.")
                    elif videos_found > 0:
                        st.warning(f"⚠️ Found {videos_found}/3 songs. Some content may be region-restricted.")
                    else:
                        st.error(f"❌ No videos found. This might be an extremely niche genre.")
                
                with summary_col2:
                    st.info(f"**Total unique artists explored:** {len(st.session_state.excluded_artists)}")
                
                # Next exploration
                st.markdown("---")
                st.markdown("### 🔄 Explore Another Genre")
                
                # Suggest genres that haven't been explored
                suggested_genres = [
                    "dream pop", "grime", "bossa nova", "reggae", "ambient",
                    "shoegaze", "trip hop", "hardstyle", "soca", "rai"
                ]
                
                # Filter out genres similar to recent ones
                recent_genres = [exp.get('genre', '').lower() for exp in st.session_state.exploration_history[-5:]]
                available_genres = [g for g in suggested_genres if g not in recent_genres][:5]
                
                if available_genres:
                    quick_cols = st.columns(len(available_genres))
                    for col_idx, idea in enumerate(available_genres):
                        with quick_cols[col_idx]:
                            if st.button(idea, use_container_width=True):
                                st.session_state.genre_input = idea
                                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Exploration failed: {str(e)}")
                st.info("""
                **Troubleshooting tips:**
                1. Try a slightly different genre description
                2. Be more specific (e.g., 'modern jazz' instead of just 'jazz')
                3. Check your API key is valid
                4. Some extremely niche genres might have no YouTube presence
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <strong>🌐 Universal Music Explorer with Duplicate Prevention</strong><br>
    No artist suggestions will repeat in this session. Clear history to reset.<br>
    <em>Exploring the world's musical diversity without repetition.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
