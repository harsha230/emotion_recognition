# Main Dataset

### Geral Info:

**Name:** RAF-DB (Real-world Affective Faces Database)

**Why was chosen:** This dataset provides a substantial amount of data (see details below), and the images focus exclusively on faces. In addition, the labeling process is rigorous and consistent, which makes the dataset more reliable.

### Number of Samples:

In total, 29,672 real-world facial images.

After an algorithm that calculated reliability estimation, they define two main subsets:

- **Single-label subset (basic emotions): 15,339 images across 7 basic classes (surprise, fear, disgust, happiness, sadness, anger, neutral).**
    - This is the one we are using.
- Two-tab subset (compound emotions): 3,954 images across 11 compound classes (they drop “fearfully disgusted” for being too rare).

The paper does not give exact numeric counts per class in text. But looking into the data I got from Kaggle, we have:

**Train:**

- 1 (surprise): 1290
- 2 (fear): 281
- 3 (disgust): 717
- 4 (happiness): 4772
- 5 (sadness): 1982
- 6 (anger): 705
- 7 (neutral): 2524

**Test:**

- 1 (surprise): 329
- 2 (fear): 74
- 3 (disgust): 160
- 4 (happiness): 1185
- 5 (sadness): 478
- 6 (anger): 162
- 7 (neutral): 680

As we can see, in the test subset we have very few examples of each class (such as fear and anger). As we are not going to perform any training, I opted to merge this 2 subsets, so we have a bigger test set.

**Bigger test (all):**

- 1 (surprise): 1619
- 2 (fear): 355
- 3 (disgust): 877
- 4 (happiness): 5957
- 5 (sadness): 2460
- 6 (anger): 867
- 7 (neutral): 3204

### Where the images were extracted from:

**Source**: 

- Images were collected from Flickr using the Flickr image search API.

**Collection process:**

- They query Flickr with emotion-related keywords such as “smile, giggle, cry, rage, scared, frightened, terrified, shocked, astonished, disgust, expressionless,” corresponding to the six basic emotions plus neutral.
- URLs returned by the API (in XML format) are fed into an automatic downloader to save the images in batches.

### How the data was labeled:

**Crowdsourcing with many annotators:**

- 315 annotators (students and staff from universities) initially participated.
- Each annotator received a one-hour tutorial on basic psychological knowledge about emotion.

**Labeling protocol:**

- Annotators saw one face at a time on a custom annotation website and had to choose one of 7 basic categories (surprise, fear, disgust, happiness, sadness, anger, neutral) that best described the *most apparent* expression.
- Each image was labeled about 40 times independently by different annotators.

**Reliability filtering (EM):**

- They use an Expectation–Maximization (EM) framework to estimate each annotator’s reliability (αᵢ) and image difficulty (βⱼ), modeling the probability that a given label matches the latent “true” label.
- After this procedure, labels from 285 annotators are retained; the overall Cronbach’s Alpha is 0.966, indicating high internal consistency.

### Other interesting characteristics:

- You need to send a email for them to get the oficial data.
- I sent the email but they didn’t answer it, so I’m using a version that is available on Kaggle: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset