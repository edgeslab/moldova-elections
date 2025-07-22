import pandas as pd
import json
import re
from sklearn.metrics import f1_score
from tqdm import tqdm
import time

from importlib.metadata import version
import openai
from openai import OpenAI
import pandas as pd

# Load dataset
df = pd.read_csv('../moldova_new.csv')

client = openai.OpenAI(api_key="gemini-key3.txt")



predict_prompt_CHAIN_OF_THOUGHT = """
Tweet Classification Task

Overview:
You are an expert tweet classifier. You will *only* output a JSON array of nine `"yes"`/`"no"` values plus the 100-char tweet snippet—nothing else.  

Task:
Task: Given a single tweet, first determine whether it mentions Russia or Europe in the political context of the Moldovan Elections and, if so, whether it is pro, anti, or neutral toward each.  

Output Format:
For each tweet, return: [pro_Russia, anti_Russia, neutral_Russia, pro_Europe, anti_Europe, neutral_Europe, non_relevant_Russia, non_relevant_Europe, 100_char]


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
- "The EU has contributed a great amount of money to support Sandu, but they have dismissed her political race completely" → pro-Europe = "yes", anti-Europe = "yes", neutral-Europe = "no"
- "Putin recently announced his efforts of increasing communication between Russia and Moldova. This news came shortly after Russia being exposed for medling with the elections" → pro-Russia = "yes", anti-Russia = "yes", neutral-Russia = "no"

The following are examples and their correct outputs:

--- START OF EXAMPLES ---

Input tweet: "Don’t forget Georgia and Moldova, Russians have been slicing off pieces of Eastern European countries for a while."

Output: The tweet mentions both Russia and Europe in a political context relevant to the Moldovan elections. The tweet does not mention anything positive about Russia, so the tweet is not pro-Russian. The tweet says "Russians have been slicing off pieces of Eastern European countries for a while", which portrays Russia as oppressors, so the tweet is anti-Russian. Since the tweet expressed criticism of Russia, it is not neutral-Russian. The tweet does not give explicit support or praise of Europe, so the tweet is not pro-Europe. The tweet does not give explicit criticism or opposition towards Europe, so it is not anti-Europe. Since the tweet mentions Europe but does not contain any supportive or critical content of Europe, the tweet is neutral-Europe. Since the tweet mentions Russia in a political context relevant to the Moldovan elections, it is not non relevant to Russia. Since the tweet mentions Europe in a political context relevant to the Moldovan elections, it is not non relevant to Europe. So the answer is  ["no", "yes", "no",  "no", "no", "yes",  "no", "no", "Don’t forget Georgia and Moldova, Russians have been slicing off pieces of Eastern European countrie"]

Input tweet: "Congratulations @Sandumaiamd for victory in the first round of the presidential elections in #Moldova! With 42.45% of the votes, it will go to the second round on November 3. Sandu remains a symbol of pro-European reforms and the struggle for the rule of law. 🇲🇩👏 #elections2024"

Output: The tweet does not mention Russia in a political context relevant to the Moldovan elections. The tweet does mention Europe in a political context relevant to the Moldovan elections because it uses "@Sandumaiamd". Since the tweet does not mention Russia, the tweet is not pro-Russian. Since the tweet does not mention Russia, the tweet is not anti-Russian. Since the tweet does not mention Russia, the tweet is not neutral-Russia. The tweet congratulates Sandu for winning the first round of the elections, so the tweet is pro-Europe. The tweet does not explicitly criticize criticize or oppose Europe in any way, so the tweet is not anti-Europe. Since the tweet contains explicit praise of Europe, it is not neutral-Europe. Since the tweet does not mention Russia in a political context relevant to the Moldovan elections, it is non relevant to Russia. Since the tweet mentions Europe in a political context relevant to the Moldovan elections, it is not non relevant to Europe. So the answer is ["no", "no", "no", "yes", "no", "no", "yes", "no", "Congratulations @Sandumaiamd for victory in the first round of the presidential elections in #Moldov"]

Input tweet: "@RksNews It's not the right moment to join EU. EU & NATO promised 'stability & prosperity' to Ukraine but they succeed to guarantee only Poverty, Devastation, Identity & Independence loss,and finally, a WAR !! The same 'stability & prosperity' they are promising to Moldova & Georgia now"

Output: The tweet does not mention Russia in a political context relevant to the Moldovan elections. The tweet does mention Europe in a political context relevant to the Moldovan elections because it expresses that its not the right moment for Moldova to join EU. Since the tweet does not mention Russia, the tweet is not pro-Russian. Since the tweet does not mention Russia, the tweet is not anti-Russian. Since the tweet does not mention Russia, the tweet is not neutral-Russia.  The tweet does not give explicit support or praise of Europe, so the tweet is not pro-Europe. The tweet expresses opposition against Moldova joining EU due to EU's false promises of stability and prosperity, so it is anti-Europe. Since the tweet expressed opposition towards Europe, it is not neutral-Europe. Since the tweet does not mention Russia in a political context relevant to the Moldovan elections, it is non relevant to Russia. Since the tweet mentions Europe in a political context relevant to the Moldovan elections, it is not non relevant to Europe. So the answer is  ["no", "no", "no", "no", "yes", "no", "yes", "no", "@RksNews It's not the right moment to join EU. EU & NATO promised 'stability & prosperity' to Ukrain"]

Input tweet: "Maia Sandu Speră să câștige alegerile prezidențiale din primul vur. Nu are nicio șansă. ESTE EVIENT Că va Fi Turul Doi, Despre Asta Vorbesc Deja și Propagandiștii Pas, între maia sandu și candatulu porului alexandru stoianoglo, Pe Care îl Susțințințin Socialiștii și EU Personal. Vreau să avertizez din nou guvernarea: dacă alexandru stoianoglo, Candidatul poporului, nu va fi fi va fi exclus din cursa electorală, noi ne rezervăm Dreptul de a boicota alegerile și vom Chema oamenii în stradă. :::::::: In the PM, Maya Sandu expects to win the presidential election from the first round. Zero chances. It is obvious that there will be a second round, the propagandists of PAS, between Maya Sandu and People’s Candidate Alexander Stoyanoglo, who are supported by the socialists and I personally are already talking about this. I want to warn the authorities again: if the people's candidate Alexander Pisanoglo is not registered or excluded from the election race, we will use the right to boycott the elections and call people to the streets. https://www.facebook.com/400101714817025/videos/1482974625677350"

Output: The tweet mentions both Russia and Europe in a political context relevant to the Moldovan elections. The tweet expresses support for Alexandru Stoianoglo, so the tweet is pro-Russian. The tweet does not mention any criticism or opposition towards Russia, so the tweet is not anti-Russian. Since the tweet expressed support of Russia, it is not neutral-Russian. The tweet does not give explicit support or praise of Europe, so the tweet is not pro-Europe. The tweet expresses criticism of Maia Sandu by saying she has no chance to win the presidential elections in the first round, so the tweet is anti-Europe. Since the tweet expressed opposition towards Europe, it is not neutral-Europe. Since the tweet mentions Russia in a political context relevant to the Moldovan elections, it is not non relevant to Russia. Since the tweet mentions Europe in a political context relevant to the Moldovan elections, it is not non relevant to Europe. So the answer is  ["yes", "no", "no", "no", "yes", "no", "no", "no", "Maia Sandu Speră să câștige alegerile prezidențiale din primul vur. Nu are nicio șansă. ESTE EVIENT"]

Input tweet: "not like he's there anymore lmao"

Output: The tweet does not mention either Russia or Europe in a political context relevant to the Moldovan elections. Since the tweet does not mention Russia, the tweet is not pro-Russian. Since the tweet does not mention Russia, the tweet is not anti-Russian. Since the tweet does not mention Russia, the tweet is not neutral-Russia. Since the tweet does not mention Europe, the tweet is not pro-Europe. Since the tweet does not mention Europe, the tweet is not anti-Europe. Since the tweet does not mention Europe, the tweet is not neutral-Europe. Since the tweet does not mention Russia in a political context relevant to the Moldovan elections, it is non relevant to Russia. Since the tweet does not mention Europe in a political context relevant to the Moldovan elections, it is non relevant to Europe. So the answer is  ["no", "no", "no", "no", "no", "no", "yes", "yes", "not like he's there anymore lmao"]

--- END OF EXAMPLES ---

Process Each Tweet:
1. Read the tweet carefully
2. Check if Russia is mentioned → classify sentiment
3. Check if Europe is mentioned → classify sentiment 
4. Format as JSON array with exactly 8 values. Return ONLY the JSON array per tweet

CRITICAL: Output only raw JSON arrays. 
CRITICAL: All values should be "yes" or "no" (in quotes).


"""


with open('gemini_key3.txt', 'r') as file:
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

def generate_prompt(initial_prompt, tweets, length_limit):
    prompt = initial_prompt
    index = 0
    for tweet in tweets:
        if len(prompt) + len(tweet) + 5 < length_limit:
            index = index + 1
            prompt = prompt + " \n{}. ".format(index) + tweet
    return prompt, index


batch_size = 20

filtered_df = df[df['is_retweet'] == False]

print(len(filtered_df))

# same as list as "manual_label.txt"
manual_label_list = filtered_df['translatedContentText'].tolist() 

gemini_responde = ""

for start in tqdm(range(7380, 8101, batch_size)):
    batch = manual_label_list[start : start + batch_size]

    max_retries = 3
    retry_count = 0
    success = False

    while not success and retry_count < max_retries:
        try:
            print(f"Batch {start}-{start+batch_size}]")
            time.sleep(2)
            final_prompt, index = generate_prompt(predict_prompt_CHAIN_OF_THOUGHT, batch, 1000000)
            gemini_response = Gemini(client, final_prompt)

            start_of_JSON = gemini_response.find('[')
            end_of_JSON = gemini_response.rfind(']') + 1
            JSON_list = gemini_response[start_of_JSON:end_of_JSON]

            JSON_list = re.sub(r'(?<=[,\[\s])\s*no"(?=\s*[,\]])', '"no"', JSON_list)
            JSON_list = re.sub(r'(?<=[,\[\s])\s*yes"(?=\s*[,\]])', '"yes"', JSON_list)
            JSON_list = re.sub(r'"+([^"]+?)"+', r'"\1"', JSON_list)

            Gemini_list = json.loads(JSON_list)

            # Gemini_list_processed = []
            # for row in Gemini_list:
            #     if not isinstance(row, list):
            #         continue

            #     processed_row = row[1:-1]
            #     Gemini_list_processed.append(processed_row)

            # Process to keep only "yes" or "no" per row
            Gemini_list_processed = []
            for item in Gemini_list:
                if isinstance(item, list):
                    row = [v.strip().casefold() for v in item if isinstance(v, str) and v.strip().casefold() in ["yes", "no"]]
                    if row:
                        Gemini_list_processed.append(row)
                elif isinstance(item, str):
                    if item.strip().casefold() in ["yes", "no"]:
                        Gemini_list_processed.append([item.strip().casefold()])    

            mapping = {"no": 0, "yes": 1}

            with open("labeled_stances_gemini_CoT.txt", 'a', encoding='utf-8') as f:
                f.write(gemini_response + '\n')

            success = True  # Parsing and scoring succeeded

        except json.JSONDecodeError as e:
            retry_count += 1
            print(f"[Batch {start}-{start+batch_size}] JSON decoding failed (attempt {retry_count}/{max_retries}). Retrying...")
            time.sleep(2)  

        # except Exception as e:
        #     retry_count += 1
        #     print(f"[Batch {start}-{start+batch_size}] Error: {e} (attempt {retry_count}/{max_retries}). Retrying...")
        #     time.sleep(2)

    if not success:
        print(f"[Batch {start}-{start+batch_size}] Failed after {max_retries} attempts. Skipping this batch.")
        with open("labeled_stances_gemini_CoT.txt", 'a', encoding='utf-8') as f:
                f.write(gemini_response + '\n')

