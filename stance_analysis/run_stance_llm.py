import pandas as pd
import json
import re
from tqdm import tqdm
import time
from openai import OpenAI
import pandas as pd

# Load dataset
df = pd.read_csv('../moldova.csv')

pd.set_option('display.float_format', '{:.0f}'.format)

key = "gemini_key.txt"
output = "stances/labeled_data.json"
start_idx = 0
end_idx = 10000



predict_prompt_CHAIN_OF_THOUGHT = """
Tweet Classification Task

Overview:
You are an expert tweet classifier. You will *only* output a JSON array of 10 value: 8 `"yes"`/`"no"` values plus, the 100-char tweet snippet, the tweet id (full id), and the index provided—nothing else.  

Task:
Task: Given a single tweet, first determine whether it mentions Russia or Europe in the political context of the Moldovan Elections and, if so, whether it is pro, anti, or neutral toward each.  

Output Format:
For each tweet, return: [pro_Russia, anti_Russia, neutral_Russia, pro_Europe, anti_Europe, neutral_Europe, non_relevant_Russia, non_relevant_Europe, 100_char, tweet id, index]


Step-by-Step Process:

Step 1: Country Mention Detection
For each country (Europe and Russia):


Mention Detection Rules:
- Russia mentioned: Tweet contains "Stoianoglo" (case-insensitive) OR their Twitter handle OR content about Russia in a political context RELEVANT TO the Moldovan Elections
- Europe mentioned: Tweet contains "Sandu" (case-insensitive) OR their Twitter handle OR content about Europe in a political context RELEVANT TO the Moldovan Elections 


Step 2: Sentiment Classification
For each mention detected, evaluate each sentiment category independently:


- Pro-[Country] = “yes” if:
- the tweet contains any explicitly positive framing or sentiment about the country. That includes, but is not limited to, support or endorsement statements, praise of policies or leaders, celebration of achievements or milestones, positive economic or cultural reporting, expressions of solidarity or partnership, and favorable comparisons.
- Examples: "Praise Russia", "EU embraces Moldova", "Vote for Sandu!"


Anti-[Country] = “yes” if:
- the tweet contains any explicitly negative framing or sentiment about the country. That includes, but is not limited to, direct criticism of leaders or policies, negative economic or social reporting, exposure of scandals or abuses, calls for sanctions or boycotts, negative comparisons or superlatives, depictions of aggression/threat, and expressions of shame or condemnation.
- Examples: "Pro -Russian actors are found to have been buying voices in the Moldova elections", "Paying off voters just like Putin did in Moldova. Shame on you", "Its not the right moment to join EU"


Neutral-[Country] = “yes” if:
- the tweet mentions the country in a factual, descriptive, or informational way without any positive (Pro) or negative (Anti) language.By flagging strictly non-evaluative mentions—news briefs, logistics, simple name references, event timings, or purely descriptive geography—you capture every case where the country appears but no sentiment is attached.
- Examples: "Stoianoglo spoke today at the presidential conference", "The European Union will allocate a record €1.8 billion to Moldova to support the country’s plan to join the bloc, European Commission President Ursula von der Leyen said Thursday https://t.co/WmZoKBRXKK"


Non-relevant to [Country] = "yes" if:
- Country is NOT mentioned at all in the tweet

Step 3: Tweet Extraction:
Extract the first 100 characters of the tweet. This will be the "100_char" value of the corresponding array

Important Rules:
1. Mixed sentiment allowed: A tweet can be both pro-Country AND anti-Country if it praises them for one thing and criticizes them for another
2. If not mentioned: non-relevant = "yes", all others = "no"
3. If mentioned: At least one of pro/anti/neutral must be "yes", non-relevant = "no"
4. Neutral rules: neutral = "yes" only if there's mention but NO pro or anti sentiment detected
5. Case insensitive: Match names regardless of capitalization


Example of mixed sentiment:
- "XXXX" → pro-Europe = "yes", anti-Europe = "yes", neutral-Europe = "no"
- "XXXX" → pro-Russia = "yes", anti-Russia = "yes", neutral-Russia = "no"

The following are examples and their correct outputs:

--- START OF EXAMPLES ---

index: 10, id: 1845987224627929088, input tweet: "XXXX"

Output: XXXX

index: 11, id: 184598722462342488, input tweet: "XXXX"
Output: XXXX

index: 15, id: 184928301923832912, input tweet: "XXXX"

Output: XXXX

index: 44, id: 184965698956524651, input tweet: "XXXX"

Output: XXXX

index: 90, id: 132843290213289028, input tweet: "XXXX"

Output: XXXX

--- END OF EXAMPLES ---

Process Each Tweet:
1. Read the tweet carefully
2. Check if Russia is mentioned → classify sentiment
3. Check if Europe is mentioned → classify sentiment 
4. Format as JSON array with exactly 10 values. Return ONLY the JSON array per tweet

CRITICAL: Output only raw JSON arrays. 
CRITICAL: Output a single JSON itme for every group of tweets, with a raw array for each tweet.
CRITICAL: All values should be "yes" or "no" (in quotes).
CRITICAL: Add a backslash to every quote symbol (") appearing in the tweet text snippet you're returning, to avoid any parsing issue.
CRITICAL: Similarly, remove any square bracket ([ and ]) appearing in the tweet text snippet you're returning, to avoid any parsing issue.


"""

with open(key, 'r') as file:
    gemini_key = file.read().rstrip()

client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def Gemini(client, input_string, prompt="You are a precise and disciplined \
           Moldova elections tweet classification assistant. Always output \
           only the requested structured JSON format.", model="gemini-2.5-flash"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
              "role": "system",
              "content": prompt
            },
            {
              "role": "user",
              "content": input_string
            }
        ],
        n=1
    )
    return response.choices[0].message.content

def generate_prompt(initial_prompt, start, ids, tweets, length_limit):
    prompt = initial_prompt
    index = 0
    for tweet in tweets:
        if len(prompt) + len(tweet) + 50 < length_limit:
            prompt = prompt + " \nindex: {}, id: {}, input tweet: {}. ".format(start + index, ids[index], tweet)
            index = index + 1
    return prompt, index


batch_size = 20

filtered_df = df[df['is_retweet'] == False].reset_index(drop=True)

manual_label_list = filtered_df['translatedContentText'].tolist() 

filtered_df['twitterData.tweetId'] = filtered_df['twitterData.tweetId'].astype('Int64')

id_list = [str(v) if pd.notna(v) else "NA" for v in filtered_df['twitterData.tweetId']]

gemini_responde = ""

max_retries = 1
for start in tqdm(range(start_idx, end_idx, batch_size)):
    batch = manual_label_list[start : start + batch_size]
    ids = id_list[start : start + batch_size]

    retry_count = 0
    success = False

    while not success and retry_count < max_retries:
        try:
            print(f"Batch {start}:{start+batch_size}")
            time.sleep(2)
            final_prompt, index = generate_prompt(predict_prompt_CHAIN_OF_THOUGHT, start, ids, batch, 1000000)
            gemini_response = Gemini(client, final_prompt)

            start_of_JSON = gemini_response.find('[')
            end_of_JSON = gemini_response.rfind(']') + 1
            JSON_list = gemini_response[start_of_JSON:end_of_JSON]

            JSON_list = re.sub(r'(?<=[,\[\s])\s*no"(?=\s*[,\]])', '"no"', JSON_list)
            JSON_list = re.sub(r'(?<=[,\[\s])\s*yes"(?=\s*[,\]])', '"yes"', JSON_list)
            JSON_list = re.sub(r'"+([^"]+?)"+', r'"\1"', JSON_list)

            Gemini_list = json.loads(JSON_list)

            mapping = {"no": 0, "yes": 1}

            with open(output, 'a', encoding='utf-8') as f:
                f.write(gemini_response + '\n')

            # Parsing and scoring succeeded
            success = True  

        except json.JSONDecodeError as e:
            retry_count += 1
            print(f"[Batch {start}-{start+batch_size}] JSON decoding failed (attempt {retry_count}/{max_retries}). Retrying...")
            time.sleep(2)  

    if not success:
        print(f"[Batch {start}-{start+batch_size}] Failed after {max_retries} attempts. Skipping this batch.")
        with open(output, 'a', encoding='utf-8') as f:
                f.write(gemini_response + '\n')

