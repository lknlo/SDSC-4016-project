import json
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs

# set seaborn styling globally
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# set paths to the Yelp dataset files
BUSINESS_PATH = 'yelp_academic_dataset_business.json'
REVIEW_PATH = 'yelp_academic_dataset_review.json'
USER_PATH = 'yelp_academic_dataset_user.json'


# define function to read JSON data into polars DataFrame
def parse_json_to_df(file_path, max_records=None):
    """
    Parse JSON file line by line and return a polars DataFrame
    with optional limit on number of records to process
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(file_path)}")):
            if max_records and i >= max_records:
                break
            data.append(json.loads(line))

    # convert to polars DataFrame
    return pl.DataFrame(data)


# load the data with limits to avoid memory issues during development
# increase or remove limits as needed based on your hardware capabilities
print("Loading business data...")
business_df = parse_json_to_df(BUSINESS_PATH, max_records=None)
print("Loading review data...")
review_df = parse_json_to_df(REVIEW_PATH, max_records=100000)
print("Loading user data...")
user_df = parse_json_to_df(USER_PATH, max_records=50000)

print(f"Loaded {business_df.height} businesses, {review_df.height} reviews, and {user_df.height} users")


# DATA TYPE 1: RATINGS DATA
# extract and analyze the ratings distribution
def process_ratings_data(review_df):
    print("\n=== Processing Ratings Data ===")
    # basic rating statistics
    rating_stats = review_df.select(pl.col("stars")).describe()
    print("Rating Statistics:")
    print(rating_stats)

    # plot rating distribution using seaborn
    plt.figure(figsize=(10, 6))
    ratings_count = review_df.group_by("stars").agg(pl.len().alias("count")).sort("stars")
    ratings_df = pd.DataFrame({"stars": ratings_count["stars"].to_numpy(),
                               "count": ratings_count["count"].to_numpy()})

    ax = sns.barplot(x="stars", y="count", data=ratings_df, hue="stars", legend=False)

    # add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=10, color='black',
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

    plt.title('Distribution of Ratings', fontsize=16)
    plt.xlabel('Rating (Stars)', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.tight_layout()
    plt.savefig('rating_distribution.png', dpi=300)
    plt.close()

    # calculate user rating statistics
    user_ratings = review_df.group_by("user_id").agg(
        pl.len().alias("rating_count"),
        pl.mean("stars").alias("avg_rating"),
        pl.std("stars").alias("rating_std")
    )

    # fill NaN in std (users with only 1 rating)
    user_ratings = user_ratings.with_columns(
        pl.col("rating_std").fill_null(0).alias("rating_std")
    )

    # calculate business rating statistics
    business_ratings = review_df.group_by("business_id").agg(
        pl.len().alias("rating_count"),
        pl.mean("stars").alias("avg_rating"),
        pl.std("stars").alias("rating_std")
    )

    business_ratings = business_ratings.with_columns(
        pl.col("rating_std").fill_null(0).alias("rating_std")
    )

    # plot user rating behavior
    plt.figure(figsize=(10, 6))
    user_rating_df = user_ratings.filter(pl.col("rating_count") > 5).to_pandas()
    sns.histplot(data=user_rating_df, x="avg_rating", bins=20, kde=True)
    plt.title('Distribution of Average User Ratings', fontsize=16)
    plt.xlabel('Average Rating', fontsize=14)
    plt.ylabel('Number of Users', fontsize=14)
    plt.tight_layout()
    plt.savefig('user_avg_rating_distribution.png', dpi=300)
    plt.close()

    return user_ratings, business_ratings


# DATA TYPE 2: TEXT DATA
# process review text data
def process_text_data(review_df):
    print("\n=== Processing Text Data ===")
    # take a sample for text analysis
    text_sample = review_df

    # verify that 'text' column exists
    if "text" not in text_sample.columns:
        print("Column 'text' not found in review data. Available columns:", text_sample.columns)
        return None, None

    # basic text statistics
    text_sample = text_sample.with_columns([
        pl.col("text").map_elements(len, return_dtype=pl.Int64).alias("text_length"),
        pl.col("text").map_elements(lambda x: len(x.split()), return_dtype=pl.Int64).alias("word_count")
    ])

    text_stats = text_sample.select(["text_length", "word_count"]).describe()
    print("Text Statistics:")
    print(text_stats)

    # plot text length distribution with seaborn
    plt.figure(figsize=(10, 6))
    text_df = text_sample.select(["word_count", "stars"]).to_pandas()

    # use seaborn's histplot with KDE
    sns.histplot(data=text_df, x="word_count", bins=50, kde=True)
    plt.title('Distribution of Review Word Count', fontsize=16)
    plt.xlabel('Word Count', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.xlim(0, text_df["word_count"].quantile(0.99))  # Limit x-axis to 99th percentile to handle outliers
    plt.tight_layout()
    plt.savefig('word_count_distribution.png', dpi=300)
    plt.close()

    # plot word count by rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=text_df, x="stars", y="word_count")
    plt.title('Review Word Count by Rating', fontsize=16)
    plt.xlabel('Rating (Stars)', fontsize=14)
    plt.ylabel('Word Count', fontsize=14)
    plt.ylim(0, text_df["word_count"].quantile(0.95))  # Limit y-axis to handle outliers
    plt.tight_layout()
    plt.savefig('word_count_by_rating.png', dpi=300)
    plt.close()

    # extract basic text features for each review
    print("Extracting text features...")

    def avg_word_length(text):
        words = text.split()
        if not words:
            return 0
        return np.mean([len(word) for word in words])

    text_features = text_sample.select([
        "review_id", "user_id", "business_id", "text", "text_length", "word_count"
    ])

    text_features = text_features.with_columns([
        pl.col("text").map_elements(avg_word_length, return_dtype=pl.Float64).alias("avg_word_length"),
        pl.col("text").map_elements(lambda x: x.count('!'), return_dtype=pl.Int64).alias("exclamation_count"),
        pl.col("text").map_elements(lambda x: x.count('?'), return_dtype=pl.Int64).alias("question_count"),
        pl.col("text").map_elements(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1),
                                    return_dtype=pl.Float64).alias("uppercase_ratio")
    ])

    # save the processed texts for later embedding or tokenization
    processed_texts = {
        'review_id': text_sample["review_id"].to_list(),
        'text': text_sample["text"].to_list(),
        'stars': text_sample["stars"].to_list()
    }

    return text_features, processed_texts


# DATA TYPE 3: SPATIO-TEMPORAL DATA
# process location and time data
def process_spatiotemporal_data(business_df, review_df):
    print("\n=== Processing Spatio-Temporal Data ===")

    # extract location features
    # check if necessary columns exist
    required_cols = ["business_id", "latitude", "longitude", "city", "state"]
    missing_cols = [col for col in required_cols if col not in business_df.columns]

    if missing_cols:
        print(f"Missing columns in business data: {missing_cols}")
        print("Available columns:", business_df.columns)
        # create placeholder data if needed
        loc_data = {"business_id": business_df["business_id"]}

        for col in ["latitude", "longitude", "city", "state"]:
            if col not in business_df.columns:
                # add placeholder data
                if col in ["latitude", "longitude"]:
                    loc_data[col] = pl.Series([0.0] * business_df.height)
                else:
                    loc_data[col] = pl.Series(["unknown"] * business_df.height)
            else:
                loc_data[col] = business_df[col]

        location_df = pl.DataFrame(loc_data)
    else:
        location_df = business_df.select(required_cols)

    # plot business locations on a world map if data is available
    if not (location_df["latitude"].is_null().any() or location_df["longitude"].is_null().any()):
        # Business Density Heatmap with better world map background
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            plt.figure(figsize=(14, 10))

            # set up the map projection
            ax = plt.axes(projection=ccrs.PlateCarree())

            # add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            ax.add_feature(cfeature.RIVERS)

            # set map extent (adjust as needed based on your data)
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

            # get the data
            loc_df = location_df.to_pandas()

            # create the heatmap using hexbin
            h = ax.hexbin(loc_df["longitude"], loc_df["latitude"],
                          gridsize=50, cmap='viridis', alpha=0.7,
                          mincnt=1, transform=ccrs.PlateCarree())

            plt.colorbar(h, label='Business Density')

            plt.title('Business Density Heatmap (World)', fontsize=16)
            plt.tight_layout()
            plt.savefig('business_density_heatmap.png', dpi=300)
            plt.close()

            # create an additional map focused on the US (or primary region in your data)
            plt.figure(figsize=(14, 10))

            # set up the map projection focusing on US/North America
            ax = plt.axes(projection=ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5))

            # add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')

            # set map extent for US (adjust based on your data)
            ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

            # create the scatter plot
            scatter = ax.scatter(loc_df["longitude"], loc_df["latitude"],
                                 c=loc_df.groupby("state").ngroup(),
                                 alpha=0.6, s=10, transform=ccrs.PlateCarree())

            plt.title('Business Locations by State (US)', fontsize=16)
            plt.tight_layout()
            plt.savefig('business_locations_us_map.png', dpi=300)
            plt.close()

        except ImportError:
            # fallback method if cartopy is not available
            print("Cartopy not available. Using matplotlib for map visualization.")

            plt.figure(figsize=(14, 10))

            # create a simplified world map background
            from matplotlib.path import Path
            import matplotlib.patches as patches

            # simplified world map outline (rectangle with grid)
            ax = plt.subplot(111)

            # set plot limits
            plt.xlim(-180, 180)
            plt.ylim(-90, 90)

            # draw basic grid
            plt.grid(linestyle='--', alpha=0.5)

            # add simple continent outlines (very simplified)
            rect = patches.Rectangle((-180, -90), 360, 180,
                                     fill=False, edgecolor='gray',
                                     linewidth=1, alpha=0.5)
            ax.add_patch(rect)

            # get the data
            loc_df = location_df.to_pandas()

            # create the heatmap using hexbin
            h = plt.hexbin(loc_df["longitude"], loc_df["latitude"],
                           gridsize=50, cmap='viridis', alpha=0.8, mincnt=1)

            plt.colorbar(h, label='Business Density')

            # add continent labels for reference
            plt.text(0, 0, 'Equator', fontsize=10, ha='center', va='center', alpha=0.7)
            plt.text(0, 40, 'Europe/Asia', fontsize=10, ha='center', va='center', alpha=0.7)
            plt.text(-100, 40, 'North America', fontsize=10, ha='center', va='center', alpha=0.7)
            plt.text(-60, -20, 'South America', fontsize=10, ha='center', va='center', alpha=0.7)
            plt.text(20, -25, 'Australia', fontsize=10, ha='center', va='center', alpha=0.7)

            plt.title('Business Density Heatmap', fontsize=16)
            plt.xlabel('Longitude', fontsize=14)
            plt.ylabel('Latitude', fontsize=14)
            plt.tight_layout()
            plt.savefig('business_density_heatmap.png', dpi=300)
            plt.close()

    # process business city distribution
    city_counts = location_df.group_by("city").agg(pl.len().alias("count")).sort("count", descending=True).head(15)

    plt.figure(figsize=(14, 8))
    city_df = pd.DataFrame({"city": city_counts["city"].to_list(),
                            "count": city_counts["count"].to_list()})

    # use Seaborn barplot for better aesthetics - FIX: Assign hue instead of using palette
    sns.barplot(data=city_df, x="city", y="count", hue="city", legend=False)
    plt.title('Top 15 Cities by Number of Businesses', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Businesses', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # add value labels on top of bars
    for i, v in enumerate(city_df["count"]):
        plt.text(i, v + 0.1, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('top_cities.png', dpi=300)
    plt.close()

    # process temporal data from reviews
    # check if date column exists
    if "date" not in review_df.columns:
        print("Column 'date' not found in review data. Available columns:", review_df.columns)

        # create placeholder temporal data
        review_temp_df = review_df.select(["review_id", "business_id", "user_id"])
        review_temp_df = review_temp_df.with_columns([
            pl.lit(2020).alias("year"),
            pl.lit(1).alias("month"),
            pl.lit(1).alias("day"),
            pl.lit(0).alias("day_of_week"),
            pl.lit(12).alias("hour")
        ])
    else:
        # convert date to datetime and extract components
        try:
            review_temp_df = review_df.with_columns([
                pl.col("date").str.to_datetime().alias("date_time")
            ])

            review_temp_df = review_temp_df.with_columns([
                pl.col("date_time").dt.year().alias("year"),
                pl.col("date_time").dt.month().alias("month"),
                pl.col("date_time").dt.day().alias("day"),
                pl.col("date_time").dt.weekday().alias("day_of_week"),
                pl.col("date_time").dt.hour().alias("hour")
            ])

            # plot review temporal patterns using Seaborn
            # reviews by Year
            plt.figure(figsize=(12, 6))
            year_counts = review_temp_df.group_by("year").agg(pl.len().alias("count")).sort("year")
            year_df = pd.DataFrame({"year": year_counts["year"].to_list(),
                                    "count": year_counts["count"].to_list()})

            sns.lineplot(data=year_df, x="year", y="count", marker='o', linewidth=2.5)
            plt.title('Reviews by Year', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('reviews_by_year.png', dpi=300)
            plt.close()

            # reviews by Month
            plt.figure(figsize=(12, 6))
            month_counts = review_temp_df.group_by("month").agg(pl.len().alias("count")).sort("month")
            month_df = pd.DataFrame({"month": month_counts["month"].to_list(),
                                     "count": month_counts["count"].to_list()})

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # fIX: Check if month value is valid before mapping to month_names
            month_df['month_name'] = month_df['month'].apply(
                lambda x: month_names[x - 1] if 1 <= x <= 12 else f"Month {x}"
            )

            # fIX: Use hue instead of palette
            sns.barplot(data=month_df, x="month_name", y="count", hue="month_name", legend=False)
            plt.title('Reviews by Month', fontsize=16)
            plt.xlabel('Month', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)

            # add value labels on top of bars
            for i, v in enumerate(month_df["count"]):
                plt.text(i, v + 0.1, str(v), ha='center', fontsize=9)

            plt.tight_layout()
            plt.savefig('reviews_by_month.png', dpi=300)
            plt.close()

            # reviews by Day of Week
            plt.figure(figsize=(12, 6))
            dow_counts = review_temp_df.group_by("day_of_week").agg(pl.len().alias("count")).sort("day_of_week")
            dow_df = pd.DataFrame({"day_of_week": dow_counts["day_of_week"].to_list(),
                                   "count": dow_counts["count"].to_list()})

            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # fIX: Check if day_of_week value is valid before mapping to day_names
            dow_df['day_name'] = dow_df['day_of_week'].apply(
                lambda x: day_names[x] if 0 <= x < len(day_names) else f"Day {x}"
            )

            # fIX: Use hue instead of palette
            sns.barplot(data=dow_df, x="day_name", y="count", hue="day_name", legend=False)
            plt.title('Reviews by Day of Week', fontsize=16)
            plt.xlabel('Day of Week', fontsize=14)
            plt.ylabel('Number of Reviews', fontsize=14)

            # add value labels on top of bars
            for i, v in enumerate(dow_df["count"]):
                plt.text(i, v + 0.1, str(v), ha='center', fontsize=10)

            plt.tight_layout()
            plt.savefig('reviews_by_dow.png', dpi=300)
            plt.close()

            # create a heatmap of reviews by month and day of week
            review_heatmap = review_temp_df.select(["month", "day_of_week"]).to_pandas()

            # fIX: Filter out invalid month/day_of_week values before creating pivot table
            review_heatmap = review_heatmap[
                (review_heatmap["month"] >= 1) &
                (review_heatmap["month"] <= 12) &
                (review_heatmap["day_of_week"] >= 0) &
                (review_heatmap["day_of_week"] < 7)
                ]

            # create the crosstab
            review_heatmap_pivot = pd.crosstab(review_heatmap['month'], review_heatmap['day_of_week'])

            # get the actual day_of_week values present in the pivot table
            actual_days = sorted(review_heatmap_pivot.columns.tolist())
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # match day names to the actual day indices in the pivot table
            column_labels = []
            for day_idx in actual_days:
                if 0 <= day_idx < len(day_names):
                    column_labels.append(day_names[day_idx])
                else:
                    column_labels.append(f'Day {day_idx}')

            # replace indices with month and day names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            row_labels = []
            for month_idx in review_heatmap_pivot.index:
                if 1 <= month_idx <= 12:
                    row_labels.append(month_names[month_idx - 1])
                else:
                    row_labels.append(f'Month {month_idx}')

            # apply the labels
            review_heatmap_pivot.index = row_labels
            review_heatmap_pivot.columns = column_labels

            plt.figure(figsize=(12, 8))
            sns.heatmap(review_heatmap_pivot, annot=True, fmt="d", cmap="viridis")
            plt.title('Reviews by Month and Day of Week', fontsize=16)
            plt.xlabel('Day of Week', fontsize=14)
            plt.ylabel('Month', fontsize=14)
            plt.tight_layout()
            plt.savefig('reviews_month_dow_heatmap.png', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error processing date information: {e}")
            import traceback
            traceback.print_exc()

            # create placeholder temporal data
            review_temp_df = review_df.select(["review_id", "business_id", "user_id"])
            review_temp_df = review_temp_df.with_columns([
                pl.lit(2020).alias("year"),
                pl.lit(1).alias("month"),
                pl.lit(1).alias("day"),
                pl.lit(0).alias("day_of_week"),
                pl.lit(12).alias("hour")
            ])

    # create a dataframe with spatiotemporal features
    spatio_temporal_df = review_temp_df.select([
        "review_id", "business_id", "user_id", "year", "month", "day", "day_of_week", "hour"
    ])

    # join with location data
    spatio_temporal_df = spatio_temporal_df.join(
        location_df.select(["business_id", "latitude", "longitude", "city", "state"]),
        on="business_id"
    )

    return location_df, spatio_temporal_df


# cOMBINING ALL DATA TYPES FOR RECOMMENDATION SYSTEM
def prepare_data_for_deep_learning(user_rating_stats, business_rating_stats,
                                   text_features, processed_texts,
                                   spatio_temporal_df, review_df):
    print("\n=== Preparing Combined Dataset for Deep Learning ===")

    # handle case where text features processing failed
    if text_features is None:
        print("Text features are missing. Creating placeholder text features...")
        # create placeholder text features from a sample of reviews
        sample_reviews = review_df
        text_features = sample_reviews.select(["review_id", "user_id", "business_id"])

        # add placeholder text feature columns
        text_features = text_features.with_columns([
            pl.lit(100).alias("text_length"),
            pl.lit(20).alias("word_count"),
            pl.lit(5.0).alias("avg_word_length"),
            pl.lit(0).alias("exclamation_count"),
            pl.lit(0).alias("question_count"),
            pl.lit(0.1).alias("uppercase_ratio")
        ])

        # create placeholder processed_texts
        if processed_texts is None:
            processed_texts = {
                'review_id': sample_reviews["review_id"].to_list(),
                'text': ['placeholder text'] * len(sample_reviews),
                'stars': sample_reviews["stars"].to_list()
            }

    # convert to lists for easier set operations
    user_ids_ratings = set(user_rating_stats["user_id"].to_list())
    user_ids_text = set(text_features["user_id"].to_list())
    user_ids_spatiotemporal = set(spatio_temporal_df["user_id"].to_list())

    business_ids_ratings = set(business_rating_stats["business_id"].to_list())
    business_ids_text = set(text_features["business_id"].to_list())
    business_ids_spatiotemporal = set(spatio_temporal_df["business_id"].to_list())

    # identify common users and businesses across all datasets
    common_users = user_ids_ratings.intersection(user_ids_text).intersection(user_ids_spatiotemporal)
    common_businesses = business_ids_ratings.intersection(business_ids_text).intersection(business_ids_spatiotemporal)

    print(f"Number of common users across all datasets: {len(common_users)}")
    print(f"Number of common businesses across all datasets: {len(common_businesses)}")

    # filter to common entities
    filtered_reviews = review_df.filter(
        (pl.col("user_id").is_in(common_users)) &
        (pl.col("business_id").is_in(common_businesses))
    )

    # if we have no common entities, use a more relaxed approach
    if filtered_reviews.height == 0:
        print("No common entities found across all datasets. Using a more relaxed filter...")
        # just require users and businesses to be in ratings data
        filtered_reviews = review_df.filter(
            (pl.col("user_id").is_in(user_ids_ratings)) &
            (pl.col("business_id").is_in(business_ids_ratings))
        )

    # create user and business ID mapping (for embedding lookup)
    unique_users = filtered_reviews.select("user_id").unique()
    unique_businesses = filtered_reviews.select("business_id").unique()

    user_id_list = unique_users["user_id"].to_list()
    business_id_list = unique_businesses["business_id"].to_list()

    user_id_map = {id: idx for idx, id in enumerate(user_id_list)}
    business_id_map = {id: idx for idx, id in enumerate(business_id_list)}

    # create dataset with mapped IDs
    # apply the mapping using a different approach
    # create dictionaries for the mappings
    user_mapping = {u: i for i, u in enumerate(user_id_list)}
    business_mapping = {b: i for i, b in enumerate(business_id_list)}

    # define a function to apply the mapping
    def map_user_id(user_id):
        return user_mapping.get(user_id, -1)

    def map_business_id(business_id):
        return business_mapping.get(business_id, -1)

    # apply the mapping
    filtered_reviews = filtered_reviews.with_columns([
        pl.col("user_id").map_elements(map_user_id, return_dtype=pl.Int32).alias("user_idx"),
        pl.col("business_id").map_elements(map_business_id, return_dtype=pl.Int32).alias("business_idx")
    ])

    # ensure no unmapped values
    filtered_reviews = filtered_reviews.filter(
        (pl.col("user_idx") >= 0) & (pl.col("business_idx") >= 0)
    )

    # join with user rating stats
    final_df = filtered_reviews.join(
        user_rating_stats,
        on="user_id",
        how="left"
    )

    # join with business rating stats (with suffix to differentiate)
    business_rating_stats = business_rating_stats.rename({
        "rating_count": "rating_count_business",
        "avg_rating": "avg_rating_business",
        "rating_std": "rating_std_business"
    })

    final_df = final_df.join(
        business_rating_stats,
        on="business_id",
        how="left"
    )

    # join with text features to get word count info
    if text_features is not None:
        # select only the review_id and text features we want to include
        text_features_subset = text_features.select([
            "review_id", "text_length", "word_count", "avg_word_length",
            "exclamation_count", "question_count", "uppercase_ratio"
        ])

        # join with the main dataframe
        final_df = final_df.join(
            text_features_subset,
            on="review_id",
            how="left"
        )

    # join with spatiotemporal features
    spatio_df_subset = spatio_temporal_df.select([
        "review_id", "latitude", "longitude", "year", "month", "day_of_week"
    ])

    final_df = final_df.join(
        spatio_df_subset,
        on="review_id",
        how="left"
    )

    # fill missing values with means
    numeric_cols = [
        "rating_count", "avg_rating", "rating_std",
        "rating_count_business", "avg_rating_business", "rating_std_business",
        "latitude", "longitude", "year", "month", "day_of_week"
    ]

    # add text feature columns if they exist
    text_numeric_cols = ["text_length", "word_count", "avg_word_length",
                         "exclamation_count", "question_count", "uppercase_ratio"]

    for col in text_numeric_cols:
        if col in final_df.columns:
            numeric_cols.append(col)

    # get available numeric columns
    available_numeric_cols = [col for col in numeric_cols if col in final_df.columns]

    # fill missing values
    for col in available_numeric_cols:
        col_mean = final_df[col].mean()
        final_df = final_df.with_columns([
            pl.col(col).fill_null(col_mean)
        ])

    # convert to pandas for saving (avoids compatibility issues)
    final_pandas_df = final_df.to_pandas()

    # visualize final dataset distributions
    num_features = min(len(available_numeric_cols), 9)
    rows = (num_features + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(available_numeric_cols[:num_features]):
        sns.histplot(final_pandas_df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    # hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300)
    plt.close()

    # create a correlation heatmap of numeric features
    plt.figure(figsize=(14, 12))
    numeric_df = final_pandas_df[available_numeric_cols]
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="viridis", annot=True, fmt=".2f",
                square=True, linewidths=.5)

    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png', dpi=300)
    plt.close()

    # save the processed data
    print("Saving processed data...")
    final_pandas_df.to_csv('processed_yelp_data.csv', index=False)

    # save mappings for later use
    with open('user_id_map.json', 'w') as f:
        json.dump(user_id_map, f)
    with open('business_id_map.json', 'w') as f:
        json.dump(business_id_map, f)

    # save review texts separately for NLP processing
    filtered_text_data = {
        'review_id': [],
        'text': [],
        'stars': []
    }

    review_id_set = set(final_df["review_id"].to_list())
    processed_review_ids = set(processed_texts['review_id'])

    for idx, review_id in enumerate(processed_texts['review_id']):
        if review_id in review_id_set:
            filtered_text_data['review_id'].append(review_id)
            filtered_text_data['text'].append(processed_texts['text'][idx])
            filtered_text_data['stars'].append(processed_texts['stars'][idx])

    # ensure we have at least some text data
    if len(filtered_text_data['review_id']) == 0:
        print("No matching review texts found. Using placeholder text data.")
        sample_reviews = final_df.select("review_id", "stars").sample(min(1000, final_df.height))
        filtered_text_data = {
            'review_id': sample_reviews["review_id"].to_list(),
            'text': ['sample review text'] * sample_reviews.height,
            'stars': sample_reviews["stars"].to_list()
        }

    with open('processed_review_texts.json', 'w') as f:
        json.dump(filtered_text_data, f)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Number of unique users: {final_df.select('user_id').unique().height}")
    print(f"Number of unique businesses: {final_df.select('business_id').unique().height}")

    # generate a summary report with key dataset statistics
    summary_data = {
        'Metric': [
            'Total Reviews',
            'Unique Users',
            'Unique Businesses',
            'Average Rating',
            'Average Reviews per User',
            'Average Reviews per Business',
            'Average Review Word Count'
        ],
        'Value': [
            final_df.height,
            final_df.select('user_id').unique().height,
            final_df.select('business_id').unique().height,
            final_df['stars'].mean(),
            final_df.height / final_df.select('user_id').unique().height,
            final_df.height / final_df.select('business_id').unique().height,
            final_df['word_count'].mean() if 'word_count' in final_df.columns else 'N/A'
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # format numeric values for better display
    summary_df['Value'] = summary_df.apply(lambda row:
                                           f"{row['Value']:.2f}" if isinstance(row['Value'], (int, float)) else row[
                                               'Value'],
                                           axis=1
                                           )

    # create a visual summary table
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    plt.title('Yelp Dataset Summary Statistics', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('dataset_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    return final_df, user_id_map, business_id_map


# execute the data processing pipeline
print("Starting data processing pipeline...")
try:
    user_ratings, business_ratings = process_ratings_data(review_df)
    text_features, processed_texts = process_text_data(review_df)
    location_df, spatio_temporal_df = process_spatiotemporal_data(business_df, review_df)

    # combine all data for deep learning
    final_df, user_id_map, business_id_map = prepare_data_for_deep_learning(
        user_ratings, business_ratings, text_features, processed_texts, spatio_temporal_df, review_df)

    # create a summary of the pipeline execution
    print("\n=== Data Processing Complete ===")
    print("Generated visualizations:")
    print("- Rating distribution (rating_distribution.png)")
    print("- User average rating distribution (user_avg_rating_distribution.png)")
    print("- Word count distribution (word_count_distribution.png)")
    print("- Word count by rating (word_count_by_rating.png)")
    print("- Business density heatmap (business_density_heatmap.png)")
    print("- Business locations US map (business_locations_us_map.png)")
    print("- Top cities by business count (top_cities.png)")
    print("- Reviews by year (reviews_by_year.png)")
    print("- Reviews by month (reviews_by_month.png)")
    print("- Reviews by day of week (reviews_by_dow.png)")
    print("- Reviews month/day heatmap (reviews_month_dow_heatmap.png)")
    print("- Feature distributions (feature_distributions.png)")
    print("- Feature correlation heatmap (feature_correlation_heatmap.png)")
    print("- Dataset summary (dataset_summary.png)")

    print("\nGenerated data files:")
    print("- processed_yelp_data.csv (Main dataset for modeling)")
    print("- user_id_map.json (User ID to index mapping)")
    print("- business_id_map.json (Business ID to index mapping)")
    print("- processed_review_texts.json (Review text data)")

except Exception as e:
    import traceback

    print(f"Error in data processing: {e}")
    traceback.print_exc()