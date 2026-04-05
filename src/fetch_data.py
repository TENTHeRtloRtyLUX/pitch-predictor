import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def fetch_all_pitches():
    all_pitches = []
    page = 0
    page_size = 1000

    while True:
        response = supabase.table("pitches").select("*").range(
            page * page_size, (page + 1) * page_size - 1
        ).execute()

        if not response.data:
            break

        all_pitches.extend(response.data)
        page += 1

        if page % 100 == 0:
            print(f"Fetched {len(all_pitches)} pitches so far...")

    df = pd.DataFrame(all_pitches)
    print(f"Total pitches fetched: {len(df)}")
    return df

if __name__ == "__main__":
    df = fetch_all_pitches()
    df.to_csv("data/training_data.csv", index=False)
    print("Saved to data/training_data.csv")


