#%%
import json
import requests
import pandas as pd

# load the data
df = pd.read_json("./chatbot_intent_classifier_data.json")
df["data"] = df["data"].apply(json.loads)

# we drop the project id because it only has one value
df = df.drop(["project_id"], axis=1)

# split the text and the intent into two columns
get_text = lambda x: x["text"]
df["text"] = df["data"].apply(lambda x: x["text"])
df["intent"] = df["data"].apply(lambda x: x["intent"])
df = df.drop(["data"], axis=1)
#%%
# for a given index get back an API compatible dictionary
def get_post_data(idx, df):
    text = df.iloc[idx]["text"]
    intent = df.iloc[idx]["intent"]
    if intent:
        return {"text": text, "intent": intent}
    return {"text": text}


# API Settings
url = "https://api.humanloop.com/projects/411/predict"
header = {
    "x-api-key": "fe20f8db14ca9cec47e9b1c9d982da9f",
    "Content-Type": "application/json",
}

# get the predictions for all the data points
i = 0
all_predictions = []
# iterate over the dataframe, reduced to be divisible by 20 (the recommended batch size)
# NOTE this is a very hacky way to do this but it is fast to write. without the time constraint it could be easily rewritten to be asyncronous and more general.
while i < len(df) - 7:
    post_list = [get_post_data(i + j, df) for j in range(20)]
    post_data = {"data": post_list, "n_best": 1}
    post_data = json.dumps(post_data)
    r = requests.post(url, headers=header, data=post_data)
    response = r.json()
    batch_prediction = [
        # get the values we need from the response
        response[i]["predictions"][0]["value"]
        for i in range(len(post_list))
    ]
    all_predictions.extend(batch_prediction)
    i = i + 20

# get the remaining datapoints
post_list = [get_post_data(j, df) for j in range(len(df) - 7, len(df))]
post_data = {"data": post_list, "n_best": 1}
post_data = json.dumps(post_data)
r = requests.post(url, headers=header, data=post_data)
response = r.json()
batch_prediction = [
    response[i]["predictions"][0]["value"] for i in range(len(post_list))
]
all_predictions.extend(batch_prediction)

# add predictions to the dataframe
df["prediction"] = all_predictions
# helper function to get prediction results from list
def get_list_item(l, item):
    try:
        return l[0][item]
    except:
        return


#%%
# add columns for prediction results
df["intent_prediction"] = df["prediction"].apply(get_list_item, args=("label",))
df["confidence"] = df["prediction"].apply(get_list_item, args=("confidence",))
# drop predition column
df = df.drop("prediction", axis=1)
# save it for future use
df.to_pickle("dataframe.pkl")
