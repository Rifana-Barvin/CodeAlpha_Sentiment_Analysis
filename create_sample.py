
import pandas as pd

input_file = "data/training.1600000.processed.noemoticon.csv"
output_file = "data/sentiment_data.csv"

column_names = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

negative_samples = []
positive_samples = []

# Read the large dataset in smaller chunks
for chunk in pd.read_csv(
    input_file,
    encoding="latin-1",
    header=None,
    names=column_names,
    chunksize=50000
):
    negative = chunk[chunk["sentiment"] == 0]
    positive = chunk[chunk["sentiment"] == 4]

    if len(negative_samples) < 10000:
        negative_samples.append(
            negative.head(10000 - sum(len(x) for x in negative_samples))
        )

    if len(positive_samples) < 10000:
        positive_samples.append(
            positive.head(10000 - sum(len(x) for x in positive_samples))
        )

    if (
        sum(len(x) for x in negative_samples) >= 10000
        and sum(len(x) for x in positive_samples) >= 10000
    ):
        break

# Combine both classes
df = pd.concat(negative_samples + positive_samples, ignore_index=True)

# Keep only sentiment and text
df = df[["sentiment", "text"]]

# Convert numeric labels to readable labels
df["sentiment"] = df["sentiment"].map({
    0: "negative",
    4: "positive"
})

# Remove missing and duplicate tweets
df.dropna(inplace=True)
df.drop_duplicates(subset="text", inplace=True)

# Save the balanced dataset
df.to_csv(output_file, index=False, encoding="utf-8")

print("Balanced sample dataset created!")
print("Dataset shape:", df.shape)
print("\nSentiment counts:")
print(df["sentiment"].value_counts())
print("\nFirst five rows:")
print(df.head())
