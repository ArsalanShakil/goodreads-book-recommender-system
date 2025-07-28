# GoodReads Recommendation Systems

-----

## Ongoing Work

I'm currently building a **Collaborative Filtering Recommendation System** for this dataset. Stay tuned for updates in the `Collaborative_Filtering.ipynb` notebook\!

-----

## Citations

  * Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18. [bibtex]
  * Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. [bibtex]
  * SEMrush. "Goodreads.com - Overview." SEMrush, https://www.semrush.com/website/goodreads.com/overview/
  * McLean, Kristen. "Book Sales in the U.S. Are Stronger Than Ever." Publishers Weekly, 19 Apr. 2022, https://www.publishersweekly.com/pw/by-topic/industry-news/bookselling/article/89040-book-sales-in-the-u-s-are-stronger-than-ever.html

-----

## Instructions for Viewing this Project

To best understand this project, please view the files in the following order:

1.  Read Me
2.  [Data](https://mengtingwan.github.io/data/goodreads.html) (only if you wish to download the data I used)
3.  Capstone Presentation.pdf (optional)
4.  `data_prep.ipynb`
5.  `Content_Based.ipynb`

-----

## Background

**GoodReads** serves as a central hub for users to explore books, offering details like summaries, ISBNs, authors, page counts, and various editions. Crucially, it's also a user-driven platform, aggregating user ratings and enabling written reviews. GoodReads plays a vital role at the intersection of a user's interest in a book and their decision to purchase it, acting as a middleman in the potential book sale lifecycle.

As illustrated by SEMrush, a web traffic tracking site, 42% of users navigate directly to GoodReads, while another 40% arrive via a Google search. After engaging with the site, 42% return to Google, and 22% proceed to Amazon. This 22% is likely heading to Amazon to consider purchasing the book they were researching. Users who return to Google may also plan to make purchases through other channels.

GoodReads' influence is further amplified by the overall health of the book industry. Publishers Weekly reports that print book sales reached their highest levels since the trade magazine began collecting data in 2004. They noted, "led by growth in adult fiction, annual print volume in the U.S. reached 826.6 million units, rising 9% over the prior year. It’s the first time annual sales volume exceeded 800 million units" (McLean).

-----

## Research Question

GoodReads is an immensely popular site, averaging approximately 230 million views per month, and the book industry is projected to continue its impressive growth (SEMrush) (McLean). Therefore, GoodReads' ability to provide value to its users is critical for the book industry's profitability. My research question is: **Can I build an accurate recommendation system for users?** I will develop two recommendation systems: a **content-based recommender**, which takes a book title and suggests similar books, and a **collaborative filtering recommender**, which identifies similar users and recommends books that those similar users have read but the requesting user has not.

-----

## Data Understanding

My data, scraped in 2017 by researchers Wan and McAuley, can be found [here](https://mengtingwan.github.io/data/goodreads.html).

I utilize four datasets from their collection:

1.  **`meta_gr.csv`**: Contains over 2.3 million rows, each representing an individual book with metadata such as ISBN, description, links, and authors.
2.  **`goodreads_book_genres_initial.json`**: Lists the genres assigned to each book.
3.  **`goodreads_book_authors.json`**: Lists all author names assigned to each book.
4.  **`goodreads_reviews_spoiler.json.gz`**: Contains over 1.3 million individual rows, with each row representing one user's text review and rating (out of 5) for a specific book.

-----

## Data Preparation

From these four datasets, I cleaned and prepared three datasets crucial for my two recommendation systems:

**A. User-Facing Recommendations (Book Metadata)**: I used dataset (1) as the primary source. Since it lacked genre information, I incorporated data from dataset (2) by joining on `book_ids`. Similarly, I used dataset (3) to add author names. I then removed some erroneous values from the `publication_year` column.

**B. Per-User Rating Dataset**: I created this by grouping dataset (4) by user, resulting in one row per user. I concatenated all their review text and then subsetted to include only users who had written at least 5 reviews.

**C. Per-Book Metadata + Reviews Dataset**: Finally, I combined datasets (A) and the pre-subsetted (B) to create a dataset containing all metadata for each book plus all associated text data (user reviews and summaries). I removed null values in the text and genre data, and then imputed nulls in `page_numbers` using genre-specific averages. This resulted in one row per book.

Dataset (B) was used for collaborative filtering, and dataset (C) was used for content-based recommendations. Dataset (A) was utilized in both systems to return recommendations.

-----

## Content-Based Recommender

A **content-based recommendation system** leverages various features of products to calculate similarity scores, which are then used to suggest the most similar items to a given input. In this project, users will be prompted to enter a book title and the desired number of recommendations. My system will then return that many similar books.

My model utilized the following features:

1.  **Concatenated Text Data**: This consisted of the book's summary and all its text reviews. I processed this text data using Natural Language Processing (NLP) techniques:

      * **Lemmatization**: Using `WordNetLemmatizer` to reduce words to their semantic root (e.g., "charged" and "charging" become "charg").
      * **Stop Word Removal**: Eliminating non-semantic words.
      * **Length Filtering**: Keeping only words 3 or more letters long.
      * **Tokenization**: Using `RegExpTokenizer`.
        This text data is invaluable for providing detailed information about a book. For example, if many users mention "horse" in their reviews, the recommender system can identify other books related to animals.

    This text data was then **vectorized using `TfidfVectorizer`**. This vectorizer is highly effective for content-based classification because it assigns importance weights to certain tokens using a TF-IDF score. A higher TF-IDF score indicates that a word is more significant in a particular document compared to its importance across all documents.

2.  **Book's Average Rating**: The average rating out of 5. Some users may be interested only in what others consider high-quality writing.

3.  **Total Rating Count**: The number of users who had rated that book. This feature served as a proxy for book popularity. More ratings generally indicate a more widely read book. Users may prefer more mainstream or more niche books.

4.  **Number of Pages**: The total page count of the book. Presumably, some users prefer longer versus shorter books.

5.  **Book Genres**: The genres of the books were **one-hot encoded**. Ten distinct genres were identified.

-----

## Collaborative Filtering (CF)

A **collaborative filtering recommendation system (CF)** first takes explicit rating data provided by each user (in this case, a numeric score from 1-5). It then transforms each user and their ratings into a vector, creating a sparse matrix where each row represents a user and columns represent individual content items. When a user requests a recommendation, the recommender calculates a similarity metric between the requesting user and all other users. It then selects the most similar users to the requestor and recommends books that these similar users have rated but the requestor has not. In this project, I use **cosine similarity** to identify the most similar users (i.e., the closest vectors).

Similar to the content-based recommender, I built a function that accepts user input and returns recommendations. The CF function takes a user's ID, calculates similarity scores, and returns the desired number of recommendations. To conserve computing power and time, instead of pre-calculating a similarity matrix containing scores for every user against every other user, I calculated scores only for the requesting user against all others.

-----

## Next Steps

GoodReads offers numerous other metrics that could be integrated into these recommendation systems to enhance their accuracy and informational value.

For example, a popular feature on the site is the ability to add titles to custom "lists," such as a "Want to Read" list. Users can compile books they're interested in, keeping track of their upcoming reads. This action represents a valuable data point; books frequently added to "Want to Read" lists are likely more popular. Furthermore, the GoodReads site could be scraped for additional information like book format (Audio, eBook, or print). This would add another feature to the model, allowing for even more precise recommendations tailored to a user's preferred format.

Ultimately, recommendation systems like these are broadly applicable across various websites and for countless other product types.

-----

## Repo Structure

```
├── data
│   ├── book_tags.csv
│   ├── books.csv
│   ├── tags.csv
│   └── to_read.csv
├── Images
│   ├── readme_header.png
│   └── goodreads_traffic.png
├── data_prep.ipynb
├── data_cleaning_scratch.ipynb
├── Content_Based.ipynb
├── Collaborative_Filtering.ipynb
├── Capstone Presentation.pdf
├── .gitignore
├── LICENSE
└── README.md
```
