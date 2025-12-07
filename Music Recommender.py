# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional
from youtube_search import YoutubeSearch
from urllib.parse import quote_plus
import time

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
            temperature=0.8,  # Higher for creative genre interpretation
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1500,
            timeout=45
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def create_universal_discovery_chains(llm: ChatOpenAI):
    """Create chains that work for ANY genre worldwide"""
    
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
        - Even a made-up or very specific genre
        
        Your task:
        1. UNDERSTAND what the user means (even if obscure)
        2. Find a REAL artist who genuinely represents this style
        3. The artist MUST have YouTube presence with official videos
        
        Return EXACTLY:
        GENRE_INTERPRETATION: [Your understanding of the genre]
        ARTIST: [Artist Name]
        ARTIST_CONTEXT: [Why they fit this genre, region/language info]
        SEARCH_STRATEGY: [3-5 search approaches for YouTube, include native language terms if relevant]
        
        Be creative but accurate. For obscure genres, find the closest legitimate artist."""
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
    
    # Create sequential chain
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
        "confidence": 0.7  # Default confidence
    }
    
    # Parse genre analysis
    for line in genre_analysis.strip().split('\n'):
        if line.startswith("GENRE_INTERPRETATION:"):
            result["genre_interpretation"] = line.replace("GENRE_INTERPRETATION:", "").strip()
        elif line.startswith("ARTIST:"):
            result["artist"] = line.replace("ARTIST:", "").strip()
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

def universal_youtube_search(query: str, max_results: int = 10) -> List[Dict]:
    """Robust YouTube search that handles any language/script"""
    try:
        # Clean query for better search results
        clean_query = re.sub(r'[^\w\s\-\.\']', ' ', query, flags=re.UNICODE)
        clean_query = ' '.join(clean_query.split())  # Remove extra spaces
        
        results = YoutubeSearch(clean_query, max_results=max_results).to_dict()
        
        # Filter out obviously irrelevant results
        filtered = []
        for r in results:
            # Keep all results for now - let scoring handle relevance
            filtered.append(r)
        
        return filtered[:max_results]
    
    except Exception as e:
        st.warning(f"Search issue for '{query[:30]}...': {str(e)[:50]}")
        return []

def find_best_video_for_song(song_title: str, artist: str, search_strategies: List[str]) -> Dict:
    """Find the best YouTube video for a song using multiple search strategies"""
    
    all_search_attempts = []
    
    # Generate search variations
    search_variations = []
    
    # 1. Direct searches
    search_variations.append(f"{artist} {song_title} official")
    search_variations.append(f"{song_title} {artist}")
    
    # 2. Add strategies from LLM
    for strategy in search_strategies[:3]:  # Use top 3 strategies
        search_variations.append(f"{song_title} {strategy}")
    
    # 3. Language-agnostic variations
    search_variations.append(f'"{song_title}" "{artist}"')
    search_variations.append(f"{song_title} music video")
    
    # Try all variations
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
    
    if best_result and best_score >= 25:  # Lower threshold for niche genres
        best_result["search_attempts"] = all_search_attempts
        return best_result
    
    # Fallback: Try broader search
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
    
    # Convert to searchable strings (preserve non-Latin scripts)
    song_lower = song_title.lower()
    artist_lower = artist.lower()
    
    # 1. Title contains song title (highest weight)
    if song_lower in title:
        score += 40
    elif any(word in title for word in song_lower.split()[:3]):
        score += 25
    
    # 2. Artist in channel name
    if artist_lower in channel:
        score += 30
    
    # 3. Official indicators (multi-language)
    official_indicators = [
        'official', 'mv', 'music video', 
        '공식', '公式', 'oficjalny', 'официальный',
        '【mv】', '【official】', '(official)', '[official]'
    ]
    
    for indicator in official_indicators:
        if indicator in title.lower():
            score += 25
            break
    
    # 4. Clean audio/video (not live)
    if any(term in title for term in ['live', 'concert', 'performance', 'stage']):
        score -= 10
    elif 'audio' in title:
        score += 15
    elif 'video' in title:
        score += 10
    
    # 5. Popularity signals
    if result.get('views'):
        try:
            views_text = result['views'].replace(',', '').replace(' views', '')
            if 'K' in views_text:
                views = float(views_text.replace('K', '')) * 1000
                score += min(20, views / 50000)  # Cap at 20 points
            elif 'M' in views_text:
                score += 25  # High popularity bonus
        except:
            pass
    
    # 6. Duration check (filter out very long/short)
    if result.get('duration'):
        try:
            time_parts = result['duration'].split(':')
            if len(time_parts) == 2:
                minutes = int(time_parts[0])
                if 2 <= minutes <= 10:  # Typical song length
                    score += 10
        except:
            pass
    
    return score

# =============================================================================
# STREAMLIT APP - TRULY UNIVERSAL
# =============================================================================

def main():
    st.set_page_config(
        page_title="🌐 Universal Music Explorer",
        page_icon="🎶",
        layout="wide"
    )
    
    # Initialize session state
    if 'exploration_history' not in st.session_state:
        st.session_state.exploration_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Custom CSS for better multi-language display
    st.markdown("""
    <style>
    .universal-title {
        font-family: "Segoe UI", "Noto Sans", sans-serif;
        font-weight: 300;
    }
    .song-title {
        font-size: 1.1em;
        margin-bottom: 5px;
        line-height: 1.4;
    }
    .genre-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # UI Header
    st.markdown('<h1 class="universal-title">🌐 Universal Music Explorer</h1>', unsafe_allow_html=True)
    st.markdown("""
    Discover music from **ANY genre worldwide** - from mainstream to ultra-niche, 
    any language, any region, any style. The system adapts to your input.
    """)
    
    # API Configuration
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
        st.markdown("### 🌍 Exploration Stats")
        st.write(f"Genres explored: {len(st.session_state.exploration_history)}")
        
        if st.session_state.exploration_history:
            st.markdown("#### Recent Explorations")
            for item in reversed(st.session_state.exploration_history[-5:]):
                genre_display = item.get('genre', 'Unknown')[:20]
                artist_display = item.get('artist', 'Unknown')[:15]
                with st.expander(f"{genre_display} → {artist_display}..."):
                    st.write(f"**Found:** {item.get('videos_found', 0)}/3 videos")
                    st.write(f"**Confidence:** {item.get('confidence', 0)*100:.0f}%")
    
    # Main input - completely open
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
        st.write("")  # Spacer
        explore_button = st.button(
            "🚀 Explore Genre",
            type="primary",
            use_container_width=True,
            use_container_width=True
        )
    
    # Genre examples without hardcoding
    if not genre_input:
        st.markdown("---")
        st.markdown("### 💡 Inspiration (Try These or Anything Else!)")
        
        # Generate example categories dynamically
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
    
    # Process exploration
    if explore_button and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key in the sidebar!")
            return
        
        # Initialize
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        with st.spinner(f"🌍 Exploring '{genre_input}' worldwide..."):
            try:
                # Create and run universal discovery
                discovery_chain = create_universal_discovery_chains(llm)
                chain_result = discovery_chain({"genre_input": genre_input})
                
                # Parse results
                analysis = parse_universal_analysis(
                    chain_result["genre_analysis"], 
                    chain_result["songs_analysis"]
                )
                
                if not analysis["artist"]:
                    st.error("Could not interpret this genre. Try being more specific or descriptive.")
                    return
                
                # Display interpretation
                st.success(f"**Interpretation:** {analysis['genre_interpretation']}")
                
                with st.expander("📋 Analysis Details", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**🎤 Artist Found**")
                        st.markdown(f"### {analysis['artist']}")
                        st.write(analysis['artist_context'])
                        st.metric("Confidence", f"{analysis['confidence']*100:.0f}%")
                    
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
                
                # Display in a clean grid
                cols = st.columns(3)
                
                for idx, (song, video_result) in enumerate(zip(analysis['songs'][:3], video_results)):
                    with cols[idx]:
                        # Song header
                        st.markdown(f'<div class="song-title"><strong>Song {idx+1}</strong><br>{song["title"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        if song.get("context"):
                            with st.expander("Why this song?", expanded=False):
                                st.write(song["context"])
                        
                        # Video display
                        if video_result.get("found"):
                            # Embed video
                            try:
                                st.video(video_result["url"])
                            except:
                                st.image("https://img.youtube.com/vi/{}/hqdefault.jpg".format(
                                    video_result["url"].split('v=')[1].split('&')[0]
                                ), use_column_width=True)
                            
                            # Video info
                            with st.expander("Video Details", expanded=False):
                                st.write(f"**Title:** {video_result['title'][:60]}...")
                                st.write(f"**Channel:** {video_result['channel']}")
                                if video_result.get('duration') != 'N/A':
                                    st.write(f"**Duration:** {video_result['duration']}")
                                if video_result.get('views') != 'N/A':
                                    st.write(f"**Views:** {video_result['views']}")
                                st.write(f"**Relevance:** {video_result['score']}/100")
                                st.write(f"**Found via:** `{video_result['found_via'][:40]}...`")
                            
                            # Watch button
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
                    "interpretation": analysis["genre_interpretation"],
                    "videos_found": videos_found,
                    "confidence": analysis["confidence"],
                    "timestamp": time.time()
                })
                
                # Summary and next steps
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
                    if videos_found < 3:
                        st.info(f"**Tips:** Try searching '{analysis['artist']}' directly on YouTube for more content.")
                
                # Next exploration suggestion
                st.markdown("---")
                st.markdown("### 🔄 Explore Another Genre")
                
                quick_cols = st.columns(5)
                quick_ideas = ["dream pop", "grime", "bossa nova", "reggae", "ambient"]
                
                for col_idx, idea in enumerate(quick_ideas):
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
    
    # Footer with universal message
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <strong>🌐 Universal Music Explorer</strong><br>
    This system adapts to ANY music genre worldwide using AI interpretation.<br>
    No genre is too niche, no language unsupported, no region excluded.<br>
    <em>Exploring the world's musical diversity, one genre at a time.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
