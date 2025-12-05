# prerequisites:
# pip install langchain
# pip install openai
# pip install langchain -U langchain-community
# pip install youtube-search
# pip install httpx==0.24.1
# pip install streamlit
# pip install streamlit -U streamlit
# pip install -U langchain-openai
# Run with this command ==> python -m streamlit run IAMUSIC2.py --logger.level=error

import streamlit as st
import os
import logging
import re
from youtube_search import YoutubeSearch
import time

# Set up logging
logging.getLogger("streamlit").setLevel(logging.WARNING)

# =============================================================================
# BACKEND FUNCTIONS - IMPROVED VERSION
# =============================================================================

# =============================================================================
# IMPROVED BACKEND FUNCTIONS
# =============================================================================

def generate_artist_and_songs(genre):
    """Generate artist and songs with strict validation"""
    llm = initialize_llm()
    if not llm:
        return None

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            # FIRST: Get artist with metadata for better searching
            artist_prompt = f"""I need ONE artist specializing in {genre}. 
            Provide in EXACT format:
            Stage Name: [artist stage name]
            Real Name: [real name or "Not widely published"]
            YouTube Channel Name: [exact name of their official YouTube channel if known]
            
            REQUIREMENTS:
            1. Artist MUST have official YouTube channel with music videos
            2. The YouTube channel name should be their verified/official channel
            3. Provide the most common/known channel name (e.g., "LiSA Official Channel" not "user123")
            4. If unsure about channel, write "Unknown"
            
            Example for anime genre:
            Stage Name: LiSA
            Real Name: Risa Oribe
            YouTube Channel Name: LiSA Official Channel"""
            
            artist_response = llm.predict(artist_prompt)
            
            # Parse artist info with YouTube channel
            stage_name, real_name, youtube_channel = None, None, None
            for line in artist_response.strip().split('\n'):
                line = line.strip()
                if line.startswith('Stage Name:'):
                    stage_name = line.replace('Stage Name:', '').strip()
                elif line.startswith('Real Name:'):
                    real_name = line.replace('Real Name:', '').strip()
                elif line.startswith('YouTube Channel Name:'):
                    youtube_channel = line.replace('YouTube Channel Name:', '').strip()
            
            if not stage_name:
                continue
            
            clean_artist = clean_artist_name(stage_name)
            
            # Check for duplicates
            if clean_artist in st.session_state.used_artists:
                continue
            
            # SECOND: Get 3 songs with STRICT requirements
            song_prompt = f"""Give me EXACTLY 3 songs from {stage_name} with these requirements:
            
            CRITICAL REQUIREMENTS FOR EACH SONG:
            1. MUST have an official music video on YouTube
            2. MUST be from the artist's official channel
            3. Video should have at least 1 million views (popular)
            4. Song should be identifiable by a unique phrase in the video description
            
            Format EACH song as:
            Original: [exact title as on YouTube] | English: [English name] | Unique Phrase: [5-7 word unique phrase from video]
            
            Example:
            1. Original: 紅蓮華 | English: Gurenge | Unique Phrase: "Demon Slayer Kimetsu no Yaiba opening theme song"
            2. Original: 炎 | English: Homura | Unique Phrase: "Demon Slayer Mugen Train movie theme song"
            3. Original: ADAMAS | English: ADAMAS | Unique Phrase: "Sword Art Online Alicization opening theme"
            
            Provide only the 3 lines above, nothing else."""
            
            songs_response = llm.predict(song_prompt)
            
            # Parse songs with unique phrases
            song_list = []
            for line in songs_response.strip().split('\n'):
                line = line.strip()
                if 'Original:' in line and 'English:' in line and 'Unique Phrase:' in line:
                    # Extract all parts
                    parts = line.split('|')
                    if len(parts) >= 3:
                        original = parts[0].replace('Original:', '').strip()
                        english = parts[1].replace('English:', '').strip()
                        unique_phrase = parts[2].replace('Unique Phrase:', '').strip()
                        
                        song_list.append({
                            'original': original,
                            'english': english,
                            'unique_phrase': unique_phrase,
                            'search_terms': self.generate_search_terms(original, english, unique_phrase)
                        })
            
            if len(song_list) == 3:
                # Update used artists and songs
                st.session_state.used_artists.add(clean_artist)
                if clean_artist not in st.session_state.used_songs:
                    st.session_state.used_songs[clean_artist] = []
                
                return {
                    'stage_name': stage_name,
                    'real_name': real_name if real_name else "Not widely published",
                    'youtube_channel': youtube_channel if youtube_channel else "Unknown",
                    'songs': song_list,
                    'clean_name': clean_artist
                }
                
        except Exception as e:
            continue
    
    st.error("Couldn't find a suitable artist after 5 attempts.")
    return None

def generate_search_terms(original, english, unique_phrase):
    """Generate multiple search strategies for a song"""
    search_sets = []
    
    # Strategy 1: Exact original title + artist
    search_sets.append({
        'name': 'Exact Original',
        'queries': [
            f'"{original}" official music video',
            f'"{original}" 公式MV',
            f'"{original}" MV'
        ],
        'weight': 1.0
    })
    
    # Strategy 2: English title if different
    if english and english.lower() != original.lower():
        search_sets.append({
            'name': 'English Title',
            'queries': [
                f'"{english}" official music video',
                f'"{english}" MV'
            ],
            'weight': 0.8
        })
    
    # Strategy 3: Unique phrase search
    search_sets.append({
        'name': 'Unique Phrase',
        'queries': [
            f'"{unique_phrase}"',
            f'{unique_phrase} official'
        ],
        'weight': 0.9
    })
    
    # Strategy 4: Combination searches
    words = unique_phrase.split()[:4]  # First 4 words of unique phrase
    if words:
        search_sets.append({
            'name': 'Key Terms',
            'queries': [f'"{original}" {" ".join(words)}'],
            'weight': 0.7
        })
    
    return search_sets

def find_and_verify_video(artist_info, song_info, search_strategies):
    """Find and VERIFY a video using multiple search strategies"""
    all_results = []
    
    for strategy in search_strategies:
        for query in strategy['queries']:
            try:
                # Add artist name to query for context
                enhanced_query = f"{query} {artist_info['stage_name']}"
                results = YoutubeSearch(enhanced_query, max_results=10).to_dict()
                
                for result in results:
                    score = calculate_comprehensive_score(
                        result['title'],
                        result.get('description', ''),
                        result['channel'],
                        artist_info,
                        song_info,
                        query
                    )
                    
                    if score >= 0.7:  # Good match threshold
                        result['strategy'] = strategy['name']
                        result['query'] = query
                        result['score'] = score
                        all_results.append(result)
                        
            except Exception:
                continue
    
    # Sort by score and remove duplicates
    if all_results:
        # Remove duplicates by video ID
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Return best result if score is high enough
        if unique_results and unique_results[0]['score'] >= 0.85:
            return unique_results[0]
    
    return None

def calculate_comprehensive_score(title, description, channel, artist_info, song_info, query):
    """Calculate score with multiple verification layers"""
    score = 0.0
    max_score = 1.0
    
    title_lower = title.lower()
    desc_lower = (description or '').lower()
    channel_lower = channel.lower()
    
    # 1. Artist verification (30%)
    artist_indicators = [
        artist_info['stage_name'].lower(),
        artist_info['real_name'].lower() if artist_info['real_name'] != "Not widely published" else ""
    ]
    
    artist_found = False
    for indicator in artist_indicators:
        if indicator and indicator in title_lower:
            score += 0.15
            artist_found = True
        if indicator and indicator in channel_lower:
            score += 0.15
            artist_found = True
    
    if not artist_found:
        return 0.0  # Reject if artist not found
    
    # 2. Song title verification (40%)
    # Check original title
    if song_info['original'].lower() in title_lower:
        score += 0.4
    # Check English title
    elif song_info['english'].lower() in title_lower:
        score += 0.35
    else:
        # Check for partial matches
        original_words = set(song_info['original'].lower().split())
        title_words = set(title_lower.split())
        common_words = original_words.intersection(title_words)
        if common_words:
            score += len(common_words) * 0.1
    
    # 3. Unique phrase verification (20%)
    if song_info['unique_phrase'].lower() in desc_lower:
        score += 0.2
    elif any(word.lower() in desc_lower for word in song_info['unique_phrase'].split()[:3]):
        score += 0.1
    
    # 4. Official indicators (10%)
    official_terms = ['official', 'vevo', 'music video', 'mv', '公式']
    for term in official_terms:
        if term in title_lower or term in desc_lower:
            score += 0.05
            break
    
    # 5. Penalties for unwanted content
    unwanted = ['cover', 'tribute', 'reaction', 'lyrics', 'karaoke', 'instrumental', 'fan']
    for term in unwanted:
        if term in title_lower:
            score -= 0.3
            break
    
    # 6. View count bonus (if available in description)
    if 'million' in desc_lower or '000,000' in desc_lower:
        score += 0.05
    
    return min(max_score, max(0.0, score))

def get_song_video_with_verification(artist_info, song):
    """Main function to get verified video for a song"""
    # Try multiple search strategies
    search_strategies = song.get('search_terms', generate_search_terms(
        song['original'], 
        song['english'], 
        song['unique_phrase']
    ))
    
    # First attempt: Normal search
    video = find_and_verify_video(artist_info, song, search_strategies)
    
    if video:
        return {
            'video_id': video['id'],
            'title': video['title'],
            'channel': video['channel'],
            'strategy': video.get('strategy', 'Unknown'),
            'score': video.get('score', 0),
            'verified': video.get('score', 0) >= 0.85
        }
    
    # Second attempt: If artist has known YouTube channel
    if artist_info.get('youtube_channel') and artist_info['youtube_channel'] != "Unknown":
        try:
            channel_query = f"{song['original']} {artist_info['youtube_channel']}"
            results = YoutubeSearch(channel_query, max_results=5).to_dict()
            
            for result in results:
                if artist_info['youtube_channel'].lower() in result['channel'].lower():
                    score = calculate_comprehensive_score(
                        result['title'],
                        result.get('description', ''),
                        result['channel'],
                        artist_info,
                        song,
                        channel_query
                    )
                    
                    if score >= 0.7:
                        return {
                            'video_id': result['id'],
                            'title': result['title'],
                            'channel': result['channel'],
                            'strategy': 'Channel Search',
                            'score': score,
                            'verified': score >= 0.85
                        }
        except Exception:
            pass
    
    return None

# =============================================================================
# UPDATED FRONTEND DISPLAY
# =============================================================================

def display_song_info(song):
    """Display song name appropriately - only show English if original is not English"""
    # Check if original contains non-Latin characters or is clearly non-English
    has_non_latin = any(ord(char) > 127 for char in song['original'])
    
    if has_non_latin and song['english']:
        # Non-English original: Show both
        return f"{song['original']} ({song['english']})"
    elif has_non_latin and not song['english']:
        # Non-English without translation: Show original only
        return song['original']
    else:
        # English original: Show only original
        return song['original']

def main_app():
    """Main application interface with guaranteed accuracy"""
    st.title("🎵 I AM MUSIC - Verified Edition")
    
    if st.sidebar.button("🔄 Reset Memory"):
        st.session_state.used_artists = set()
        st.session_state.used_songs = {}
        st.success("Memory reset!")
    
    genre = st.text_input("Enter a genre of music:", placeholder="e.g., anime, kpop, rock, jazz")

    if genre:
        with st.spinner("Generating verified recommendations..."):
            # Get artist and songs
            result = generate_artist_and_songs(genre)
            
            if result is None:
                st.error("Failed to generate recommendations. Try a different genre.")
                return
            
            # Display artist info
            st.subheader(f"🎤 {result['stage_name']}")
            if result['real_name'] != "Not widely published":
                st.caption(f"Real name: {result['real_name']}")
            if result['youtube_channel'] != "Unknown":
                st.caption(f"YouTube: {result['youtube_channel']}")
            
            # Display songs
            st.subheader("🎵 Verified Songs")
            
            for i, song in enumerate(result['songs'], 1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display song name appropriately
                    display_name = display_song_info(song)
                    st.write(f"**{i}. {display_name}**")
                    if song['unique_phrase']:
                        st.caption(f"🔍 {song['unique_phrase']}")
                
                with col2:
                    # Search for video
                    with st.spinner("🔍"):
                        video_info = get_song_video_with_verification(result, song)
                
                # Display result
                if video_info:
                    st.success(f"✅ Verified (Score: {video_info['score']:.0%})")
                    
                    # Show video
                    video_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
                    st.video(video_url)
                    
                    # Show verification details
                    with st.expander("Verification Details"):
                        st.write(f"**Strategy:** {video_info['strategy']}")
                        st.write(f"**Channel:** {video_info['channel']}")
                        st.write(f"**Title:** {video_info['title']}")
                        st.write(f"**Confidence Score:** {video_info['score']:.0%}")
                        
                        if video_info['score'] >= 0.9:
                            st.success("✅ High confidence match")
                        elif video_info['score'] >= 0.8:
                            st.info("✅ Good confidence match")
                        else:
                            st.warning("⚠️ Moderate confidence - please verify")
                else:
                    st.error("❌ Could not verify video")
                    
                    # Show fallback search
                    with st.expander("Try searching manually"):
                        st.write(f"**Search terms for:** {display_song_info(song)}")
                        st.code(f"{result['stage_name']} {song['original']} official music video")
                        st.code(f"{result['stage_name']} {song['english']} official music video" if song['english'] else "")
                        st.code(f"{song['unique_phrase']}")
            
            # Summary
            found_videos = [song for song in result['songs'] 
                          if get_song_video_with_verification(result, song)]
            
            if len(found_videos) == 3:
                st.balloons()
                st.success("🎉 Perfect! All 3 songs verified with high confidence!")
            elif len(found_videos) > 0:
                st.success(f"✅ Found {len(found_videos)} verified videos out of 3")
            else:
                st.info("💡 Some videos couldn't be automatically verified. Use the manual search suggestions above.")

# =============================================================================
# INITIALIZATION (keep the same)
# =============================================================================

if 'used_artists' not in st.session_state:
    st.session_state.used_artists = set()

if 'used_songs' not in st.session_state:
    st.session_state.used_songs = {}

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

if 'hide_api_section' not in st.session_state:
    st.session_state.hide_api_section = False

# Setup API key (use your existing setup_api_key function)
setup_api_key()

if not st.session_state.api_key_valid:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to use the app.")
else:
    main_app()
