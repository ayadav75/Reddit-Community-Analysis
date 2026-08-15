import requests
import pandas as pd
import time
import json
import csv
import os
import glob
import random

# --- CONFIGURATION ---
USER_AGENTS = [
    "DataMiningProject/1.0 (Student_Coursework_1225158272; +http://localhost)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0 Safari/537.36",
]

CUSTOM_USER_AGENT = (
    "DataMiningProject/1.0 (Student_Coursework_1225158272; +http://localhost)"
)


# --- HELPER: Request with Retry Logic ---
def make_request_with_backoff(url, params, retries=5):
    for attempt in range(retries):
        try:
            # ROTATION: Pick a random browser identity for this request
            current_headers = {'User-Agent': USER_AGENTS[retries-1]}
            
            response = requests.get(url, headers=current_headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response
            
            elif response.status_code == 429:
                # Exponential backoff
                wait_time = (attempt) * 60  
                print(f"      [!] Rate Limit (429). Sleeping {wait_time}s...")
                print(f"          (Tip: Toggle Airplane Mode to reset IP if this persists)")
                time.sleep(wait_time)
                continue 
            
            else:
                print(f"      [!] Error {response.status_code}")
                return response 
                
        except Exception as e:
            print(f"      [!] Exception: {e}")
            time.sleep(5)
    
    return None


# --- 1. CRAWLER FUNCTION ---
def crawl_subreddit_robust(subreddit, limit=1000):
    posts_dict = {} 
    
    strategies = [
        ("top", "month"),
        ("top", "year"),
        ("top", "week"),
        ("rising", None),
        ("hot", None),
        ("new", None),
        ("best", None),
    ]

    
    print(f"--- Starting Crawl for r/{subreddit} ---")

    for sort_type, time_filter in strategies:
        if len(posts_dict) >= limit:
            break
            
        print(f"--> Strategy: Sort='{sort_type}', Time='{time_filter}'")
        base_url = f"https://www.reddit.com/r/{subreddit}/{sort_type}.json"
        after_token = None
        
        for page_num in range(15): 
            if len(posts_dict) >= limit:
                break

            params = {'limit': 100, 'after': after_token}
            if time_filter:
                params['t'] = time_filter

            # Pass request to the helper that handles headers/rotation
            response = make_request_with_backoff(base_url, params)
            
            if not response or response.status_code != 200: 
                break 

            try:
                data = response.json()
                children = data.get('data', {}).get('children', [])
            except:
                print("      Error parsing JSON.")
                break
            
            if not children: 
                break

            new_posts_count = 0
            for post in children:
                p_data = post.get('data', {})
                post_id = p_data.get('id')
                upvotes = p_data.get('ups', 0)
                
                if upvotes < 10:
                    continue
                
                if post_id and post_id not in posts_dict:
                    posts_dict[post_id] = p_data
                    new_posts_count += 1

            print(f"      Page {page_num+1}: Found {new_posts_count} new. Total: {len(posts_dict)}")

            after_token = data['data'].get('after')
            if not after_token: break
            
            # RANDOM SLEEP: Mimics human reading time to avoid detection
            sleep_time = random.uniform(3.0, 7.0)
            time.sleep(sleep_time)

    return list(posts_dict.values())

# --- 2. PROCESSOR FUNCTION ---
def process_json_to_csv(json_filename, csv_filename):
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"   -> Warning: JSON {json_filename} is empty.")
            return

        df['title'] = df['title'].fillna('')
        df['selftext'] = df['selftext'].fillna('')
        df['combined_text'] = df['title'] + " " + df['selftext']
        
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.strftime('%Y-%m-%d')
        else:
            df['date'] = ''

        df['upvotes'] = df.get('ups', 0)
        df['comments'] = df.get('num_comments', 0)
        df['upvote_ratio'] = df.get('upvote_ratio', 0.0)
        df['flair'] = df.get('link_flair_text', '').fillna('None')

        cols_to_keep = ['id', 'subreddit', 'date', 'combined_text', 'title', 'flair', 'upvotes', 'upvote_ratio', 'comments', 'url']
        final_cols = [c for c in cols_to_keep if c in df.columns]
        final_df = df[final_cols]
        
        final_df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        print(f"   -> Converted {len(final_df)} records to {csv_filename}")
        
    except Exception as e:
        print(f"   -> Failed to convert to CSV: {e}")

# --- 3. MERGE FUNCTION ---
def merge_all_data():
    print("\n===============================")
    print("MERGING DATASETS")
    print("===============================")
    all_files = glob.glob("data_processed/*.csv")
    df_list = []
    for filename in all_files:
        if "reddit_ALL_data.csv" in filename: continue
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if df_list:
        master_df = pd.concat(df_list, axis=0, ignore_index=True)
        master_path = "reddit_ALL_data.csv"
        master_df.to_csv(master_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        print(f"SUCCESS: Combined {len(master_df)} rows into '{master_path}'")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    os.makedirs("data_raw", exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)

    # 15 communities
    target_communities = [
        "technology", "programming", "ChatGPT",      
        "wallstreetbets", "Bitcoin", "investing",    
        "worldnews", "politics", "europe",           
        "movies", "gaming", "nba", "Music",          
        "AmIOverreacting", "NoStupidQuestions"       
    ]
    
    for comm in target_communities:
        print(f"\n===============================")
        print(f"PROCESSING: r/{comm}")
        print(f"===============================")
        
        json_path = f"data_raw/{comm}_raw.json"
        csv_path = f"data_processed/{comm}_data.csv"

        # RESUME LOGIC
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing = json.load(f)
                if len(existing) > 100: 
                    print(f"   [SKIP] Found {len(existing)} posts. Moving on.")
                    process_json_to_csv(json_path, csv_path)
                    continue
            except:
                pass 
        
        raw_data = crawl_subreddit_robust(comm, limit=1000)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=4)
        print(f"   Saved raw JSON to {json_path}")
        
        process_json_to_csv(json_path, csv_path)
        
        # Sleep between communities
        time.sleep(5)

    merge_all_data()