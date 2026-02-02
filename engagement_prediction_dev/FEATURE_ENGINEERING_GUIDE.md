# Feature Engineering Implementation Guide

This document provides detailed specifications for implementing all features for the engagement prediction models.

## 1. BASE FORMULA FEATURES

These features mirror the algorithmic scoring in `base_score_calculation.txt`.

### 1.1 Content Length Features

```python
def extract_content_length_features(content):
    """
    Extract word count and assign length category/score.
    
    Returns:
        dict with keys:
        - word_count: int
        - word_count_category: str ['too_short', 'short', 'optimal', 'acceptable', 'long', 'too_long']
        - length_score: int (based on base formula)
    """
    word_count = len(str(content).split())
    
    # Categorization
    if word_count < 50:
        category = 'too_short'
        score = -12
    elif 50 <= word_count < 80:
        category = 'short'
        score = -3
    elif 80 <= word_count < 100:
        category = 'good'
        score = 5
    elif 100 <= word_count <= 200:
        category = 'optimal'
        score = 8
    elif 200 < word_count <= 300:
        category = 'acceptable'
        score = 3
    elif 300 < word_count <= 400:
        category = 'long'
        score = -15
    else:
        category = 'too_long'
        score = -15
    
    return {
        'word_count': word_count,
        'word_count_category': category,
        'length_score': score
    }
```

### 1.2 Hook Pattern Detection

Analyze the first sentence for viral hook patterns.

```python
import re

def extract_hook_features(content):
    """
    Detect hook patterns in the first sentence.
    Only the first matching hook counts.
    
    Returns:
        dict with keys:
        - hook_type: str or None
        - hook_score: int
        - has_never_narrative: bool
        - has_specific_time_hook: bool
        - has_quote_hook: bool
        - has_contrarian_hook: bool
        - has_belief_transformation: bool
        - has_everyone_hook: bool
        - has_recency_signal: bool
    """
    sentences = content.split('.')
    first_sentence = sentences[0].lower() if sentences else ""
    
    hooks = []
    features = {
        'has_never_narrative': False,
        'has_specific_time_hook': False,
        'has_quote_hook': False,
        'has_contrarian_hook': False,
        'has_belief_transformation': False,
        'has_everyone_hook': False,
        'has_just_realized': False,
        'has_its_official': False,
        'has_recency_signal': False
    }
    
    # Never narrative (e.g., "Never thought I'd...")
    if re.search(r'\bnever\b', first_sentence):
        hooks.append(('never_narrative', 15))
        features['has_never_narrative'] = True
    
    # Specific time (e.g., "It's 2:47am", "At 3pm yesterday")
    if re.search(r'\b\d{1,2}:\d{2}\s*(am|pm)\b', first_sentence):
        hooks.append(('specific_time', 12))
        features['has_specific_time_hook'] = True
    
    # Quote hook (starts with quotation marks)
    if re.match(r'^["\']', first_sentence):
        hooks.append(('quote_hook', 10))
        features['has_quote_hook'] = True
    
    # Contrarian (Stop/Start)
    if re.search(r'\b(stop|start)\b.*\b(start|stop)\b', first_sentence):
        hooks.append(('contrarian', 7))
        features['has_contrarian_hook'] = True
    
    # Belief transformation
    if re.search(r'\b(used to think|used to believe)\b', first_sentence):
        hooks.append(('belief_transformation', 6))
        features['has_belief_transformation'] = True
    
    # Everyone's/Everyone is
    if re.search(r"\beveryone('s| is)\b", first_sentence):
        hooks.append(('everyone_hook', 5))
        features['has_everyone_hook'] = True
    
    # Just realized
    if re.search(r'\bjust realized\b', first_sentence):
        hooks.append(('just_realized', 5))
        features['has_just_realized'] = True
    
    # It's official / Today
    if re.search(r"\b(it's official|today)\b", first_sentence):
        hooks.append(('its_official', 6))
        features['has_its_official'] = True
    
    # Recency signals
    if re.search(r'\b(hours ago|last week|yesterday|this morning)\b', first_sentence):
        hooks.append(('recency_signal', 4))
        features['has_recency_signal'] = True
    
    # Select first matching hook
    if hooks:
        hook_type, hook_score = hooks[0]
    else:
        hook_type, hook_score = None, 0
    
    features['hook_type'] = hook_type
    features['hook_score'] = hook_score
    
    return features
```

### 1.3 Power Pattern Detection

Multiple power patterns can match (not mutually exclusive).

```python
def extract_power_patterns(content):
    """
    Detect multiple power patterns throughout the content.
    Each pattern adds to the score.
    
    Returns:
        dict with:
        - power_pattern_count: int
        - power_pattern_score: int (sum of all matches)
        - has_underdog_story: bool
        - has_transformation_narrative: bool
        - has_powerful_cta: bool
        - has_hidden_truth: bool
        - has_vulnerability: bool
        - has_family_story: bool
        - has_specific_numbers: bool
        - has_learning_adversity: bool
        - has_list_format: bool
        - has_contrast: bool
        - has_aspirational: bool
        - has_direct_address: bool
        - has_personal_story: bool
    """
    content_lower = content.lower()
    
    patterns = []
    features = {}
    
    # Underdog/immigrant story
    if re.search(r'\b(immigrant|came from nothing|grew up poor)\b', content_lower):
        patterns.append(('underdog', 9))
        features['has_underdog_story'] = True
    else:
        features['has_underdog_story'] = False
    
    # Transformation narrative
    if re.search(r'\b(used to be|transformed|went from.*to)\b', content_lower):
        patterns.append(('transformation', 8))
        features['has_transformation_narrative'] = True
    else:
        features['has_transformation_narrative'] = False
    
    # Powerful CTA question
    if re.search(r'\?$', content) and len(content.split('?')) > 1:
        patterns.append(('cta_question', 8))
        features['has_powerful_cta'] = True
    else:
        features['has_powerful_cta'] = False
    
    # "Nobody posts about" (hidden truth)
    if re.search(r'\bnobody (posts|talks) about\b', content_lower):
        patterns.append(('hidden_truth', 10))
        features['has_hidden_truth'] = True
    else:
        features['has_hidden_truth'] = False
    
    # Vulnerability/authenticity
    if re.search(r'\b(struggled|failed|vulnerable|honest|truth is)\b', content_lower):
        patterns.append(('vulnerability', 7))
        features['has_vulnerability'] = True
    else:
        features['has_vulnerability'] = False
    
    # Family/parenting
    if re.search(r'\b(my (son|daughter|child|kids|family)|parenting)\b', content_lower):
        patterns.append(('family', 8))
        features['has_family_story'] = True
    else:
        features['has_family_story'] = False
    
    # Specific numbers/data
    if re.search(r'\b\d+%|\b\d+x\b|\b\$\d+', content_lower):
        patterns.append(('specific_numbers', 4))
        features['has_specific_numbers'] = True
    else:
        features['has_specific_numbers'] = False
    
    # Learning from adversity
    if re.search(r'\b(learned|lesson|mistake taught me)\b', content_lower):
        patterns.append(('learning', 5))
        features['has_learning_adversity'] = True
    else:
        features['has_learning_adversity'] = False
    
    # List/bullet format
    if re.search(r'(^|\n)\s*[\d\-•]', content):
        patterns.append(('list_format', 5))
        features['has_list_format'] = True
    else:
        features['has_list_format'] = False
    
    # Contrast/comparison
    if re.search(r'\b(vs|versus|but|however|instead)\b', content_lower):
        patterns.append(('contrast', 5))
        features['has_contrast'] = True
    else:
        features['has_contrast'] = False
    
    # Aspirational payoff
    if re.search(r'\b(achieve|success|dream|goal)\b', content_lower):
        patterns.append(('aspirational', 6))
        features['has_aspirational'] = True
    else:
        features['has_aspirational'] = False
    
    # Direct address
    if re.search(r'\byou (become|will|can|are)\b', content_lower):
        patterns.append(('direct_address', 3))
        features['has_direct_address'] = True
    else:
        features['has_direct_address'] = False
    
    # Personal story markers
    if re.search(r'\b(my story|my journey|i remember)\b', content_lower):
        patterns.append(('personal_story', 5))
        features['has_personal_story'] = True
    else:
        features['has_personal_story'] = False
    
    features['power_pattern_count'] = len(patterns)
    features['power_pattern_score'] = sum(score for _, score in patterns)
    
    return features
```

### 1.4 Media Type Features

```python
def extract_media_features(media_type):
    """
    Encode media type and assign score.
    
    Returns:
        dict with:
        - has_video: bool
        - has_carousel: bool
        - has_image: bool
        - has_article: bool
        - media_score: int
        - media_type_encoded: str
    """
    media_lower = str(media_type).lower() if pd.notna(media_type) else 'none'
    
    features = {
        'has_video': 'video' in media_lower,
        'has_carousel': 'carousel' in media_lower,
        'has_image': 'image' in media_lower,
        'has_article': 'article' in media_lower,
        'media_type_encoded': media_lower
    }
    
    # Assign score
    if features['has_video']:
        features['media_score'] = 10
    elif features['has_carousel']:
        features['media_score'] = 8
    elif features['has_image']:
        features['media_score'] = 5
    else:
        features['media_score'] = 0
    
    return features
```

### 1.5 Link Penalty Features

```python
def extract_link_features(content, content_links):
    """
    Detect external links and calculate penalties.
    
    Returns:
        dict with:
        - has_external_link: bool
        - link_count: int
        - link_penalty_score: int
    """
    has_link = False
    link_count = 0
    
    # Check content_links column
    if pd.notna(content_links) and len(str(content_links)) > 0:
        has_link = True
        link_count = len(str(content_links).split(','))
    
    # Also check for URLs in content
    url_pattern = r'https?://\S+|www\.\S+'
    urls_in_content = re.findall(url_pattern, str(content))
    if urls_in_content:
        has_link = True
        link_count = max(link_count, len(urls_in_content))
    
    link_penalty = -18 if has_link else 0
    
    return {
        'has_external_link': has_link,
        'link_count': link_count,
        'link_penalty_score': link_penalty
    }
```

### 1.6 Promotional Content Detection

```python
def extract_promotional_features(content):
    """
    Detect promotional language intensity.
    
    Returns:
        dict with:
        - promotional_intensity: int (0-10)
        - is_heavy_promotion: bool
        - is_product_focused: bool
        - has_promotional_content: bool
        - promotional_penalty: int
    """
    content_lower = str(content).lower()
    
    promo_keywords = [
        'product', 'demo', 'launch', 'our product', 'we built',
        'sign up', 'register', 'buy now', 'purchase', 'sale',
        'discount', 'offer', 'limited time', 'check out', 'visit'
    ]
    
    promo_score = sum(1 for kw in promo_keywords if kw in content_lower)
    
    is_heavy = promo_score >= 6
    is_product = promo_score >= 4
    has_promo = promo_score >= 2
    
    # Calculate penalty
    if is_heavy:
        penalty = -12
    elif is_product:
        penalty = -8
    elif has_promo:
        penalty = -4
    else:
        penalty = 0
    
    return {
        'promotional_intensity': min(promo_score, 10),
        'is_heavy_promotion': is_heavy,
        'is_product_focused': is_product,
        'has_promotional_content': has_promo,
        'promotional_penalty': penalty
    }
```

### 1.7 Base Score Calculation

```python
def calculate_base_score(row):
    """
    Calculate the complete base score using all features.
    This mirrors the algorithm in base_score_calculation.txt.
    """
    score = 50  # Starting baseline
    
    # Add all feature scores
    score += row.get('length_score', 0)
    score += row.get('hook_score', 0)
    score += row.get('power_pattern_score', 0)
    score += row.get('media_score', 0)
    score += row.get('link_penalty_score', 0)
    score += row.get('promotional_penalty', 0)
    
    # Pattern density bonus
    total_patterns = row.get('power_pattern_count', 0) + (1 if row.get('hook_type') else 0)
    if total_patterns >= 6:
        score += 12
    elif total_patterns >= 4:
        score += 7
    elif total_patterns >= 3:
        score += 4
    elif total_patterns < 2:
        score -= 7
    
    # Low-effort link post penalty
    if (row.get('word_count', 0) < 80 and 
        row.get('has_external_link', False) and 
        total_patterns < 2):
        score -= 15
    
    # Link preview spam
    if (row.get('word_count', 0) < 60 and 
        row.get('has_external_link', False) and 
        row.get('media_score', 0) == 0):
        score -= 10
    
    # Cap between 0-100
    score = max(0, min(100, score))
    
    return score
```

## 2. ADVANCED NLP FEATURES

### 2.1 Text Statistics

```python
import emoji
import re

def extract_text_statistics(content):
    """
    Extract basic text statistics.
    """
    content_str = str(content)
    
    sentences = re.split(r'[.!?]+', content_str)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    words = content_str.split()
    word_count = len(words)
    unique_words = len(set(words))
    
    char_count = len(content_str)
    
    # Emoji analysis
    emoji_list = emoji.emoji_list(content_str)
    emoji_count = len(emoji_list)
    unique_emojis = len(set([e['emoji'] for e in emoji_list]))
    
    # Punctuation
    question_count = content_str.count('?')
    exclamation_count = content_str.count('!')
    punctuation = sum(1 for c in content_str if c in '.,;:!?')
    
    return {
        'sentence_count': sentence_count,
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'char_count': char_count,
        'unique_word_ratio': unique_words / word_count if word_count > 0 else 0,
        'emoji_count': emoji_count,
        'unique_emoji_count': unique_emojis,
        'question_count': question_count,
        'exclamation_count': exclamation_count,
        'punctuation_density': punctuation / char_count if char_count > 0 else 0
    }
```

### 2.2 Sentiment Analysis

```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def extract_sentiment_features(content):
    """
    Extract sentiment using VADER and TextBlob.
    """
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(str(content))
    
    blob = TextBlob(str(content))
    
    # Categorize sentiment
    compound = vader_scores['compound']
    if compound >= 0.05:
        category = 'positive'
    elif compound <= -0.05:
        category = 'negative'
    else:
        category = 'neutral'
    
    return {
        'sentiment_vader_pos': vader_scores['pos'],
        'sentiment_vader_neg': vader_scores['neg'],
        'sentiment_vader_neu': vader_scores['neu'],
        'sentiment_vader_compound': vader_scores['compound'],
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'sentiment_category': category
    }
```

### 2.3 Named Entity Recognition

```python
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_ner_features(content):
    """
    Extract named entities using spaCy.
    """
    doc = nlp(str(content)[:1000000])  # Limit length for performance
    
    entities = {
        'PERSON': 0,
        'ORG': 0,
        'GPE': 0,  # Geo-political entity
        'DATE': 0,
        'MONEY': 0,
        'PERCENT': 0
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] += 1
    
    return {
        'person_mentions': entities['PERSON'],
        'org_mentions': entities['ORG'],
        'location_mentions': entities['GPE'],
        'date_mentions': entities['DATE'],
        'money_mentions': entities['MONEY'],
        'percent_mentions': entities['PERCENT'],
        'total_entities': len(doc.ents),
        'entity_diversity': len(set([ent.label_ for ent in doc.ents]))
    }
```

### 2.4 Readability Metrics

```python
import textstat

def extract_readability_features(content):
    """
    Calculate various readability metrics.
    """
    content_str = str(content)
    
    flesch_ease = textstat.flesch_reading_ease(content_str)
    flesch_grade = textstat.flesch_kincaid_grade(content_str)
    gunning_fog = textstat.gunning_fog(content_str)
    
    # Categorize readability
    if flesch_ease >= 80:
        category = 'very_easy'
    elif flesch_ease >= 60:
        category = 'easy'
    elif flesch_ease >= 40:
        category = 'moderate'
    else:
        category = 'difficult'
    
    return {
        'flesch_reading_ease': flesch_ease,
        'flesch_kincaid_grade': flesch_grade,
        'gunning_fog_index': gunning_fog,
        'readability_category': category
    }
```

### 2.5 Hashtag Features

```python
def extract_hashtag_features(num_hashtags, hashtags, hashtag_followers):
    """
    Extract hashtag-related features.
    """
    num_tags = int(num_hashtags) if pd.notna(num_hashtags) else 0
    
    # Parse hashtag text
    hashtag_list = []
    if pd.notna(hashtags):
        # Extract hashtags from text
        hashtag_list = re.findall(r'#\w+', str(hashtags))
    
    avg_hashtag_length = 0
    if hashtag_list:
        avg_hashtag_length = sum(len(tag) for tag in hashtag_list) / len(hashtag_list)
    
    followers = int(hashtag_followers) if pd.notna(hashtag_followers) else 0
    
    return {
        'num_hashtags': num_tags,
        'avg_hashtag_length': avg_hashtag_length,
        'hashtag_followers': followers,
        'has_hashtags': num_tags > 0
    }
```

## 3. DERIVED FEATURES

### 3.1 Influencer-Level Features

```python
def add_influencer_features(df):
    """
    Add influencer-level aggregated features.
    """
    # Calculate per-influencer statistics
    influencer_stats = df.groupby('name').agg({
        'reactions': ['mean', 'median', 'std'],
        'comments': ['mean', 'median', 'std'],
        'followers': 'first'
    }).reset_index()
    
    influencer_stats.columns = [
        'name', 
        'influencer_avg_reactions', 'influencer_median_reactions', 'influencer_std_reactions',
        'influencer_avg_comments', 'influencer_median_comments', 'influencer_std_comments',
        'influencer_followers'
    ]
    
    # Merge back
    df = df.merge(influencer_stats, on='name', how='left')
    
    # Calculate deviation from average
    df['reactions_deviation'] = df['reactions'] - df['influencer_avg_reactions']
    df['comments_deviation'] = df['comments'] - df['influencer_avg_comments']
    
    # Engagement rate
    df['engagement_rate'] = (df['reactions'] / df['followers']) * 1000
    
    return df
```

### 3.2 Combined Quality Scores

```python
def calculate_quality_scores(row):
    """
    Calculate composite quality and virality scores.
    """
    # Content quality (length + readability + structure)
    quality_score = 0
    
    if row.get('word_count_category') in ['optimal', 'good', 'acceptable']:
        quality_score += 30
    
    if row.get('readability_category') in ['easy', 'moderate']:
        quality_score += 20
    
    if row.get('has_list_format', False):
        quality_score += 15
    
    if row.get('sentence_count', 0) >= 3:
        quality_score += 10
    
    quality_score += row.get('unique_word_ratio', 0) * 25
    
    # Virality potential (hooks + patterns + media)
    virality_score = 0
    virality_score += row.get('hook_score', 0)
    virality_score += row.get('power_pattern_score', 0)
    virality_score += row.get('media_score', 0)
    
    if row.get('has_vulnerability', False):
        virality_score += 10
    
    if row.get('sentiment_vader_compound', 0) > 0.5:  # Very positive
        virality_score += 5
    
    # Authenticity vs promotion ratio
    authenticity_score = 100 - (row.get('promotional_intensity', 0) * 10)
    if row.get('has_personal_story', False):
        authenticity_score += 10
    if row.get('has_vulnerability', False):
        authenticity_score += 10
    
    return {
        'content_quality_score': min(100, quality_score),
        'virality_potential_score': min(100, virality_score),
        'authenticity_score': min(100, max(0, authenticity_score))
    }
```

## 4. FEATURE EXTRACTION MASTER FUNCTION

```python
def extract_all_features(df):
    """
    Master function to extract all features from the dataset.
    """
    print("Extracting features...")
    
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row.get('content', '')
        
        feature_dict = {}
        
        # Base formula features
        feature_dict.update(extract_content_length_features(content))
        feature_dict.update(extract_hook_features(content))
        feature_dict.update(extract_power_patterns(content))
        feature_dict.update(extract_media_features(row.get('media_type')))
        feature_dict.update(extract_link_features(content, row.get('content_links')))
        feature_dict.update(extract_promotional_features(content))
        
        # NLP features
        feature_dict.update(extract_text_statistics(content))
        feature_dict.update(extract_sentiment_features(content))
        feature_dict.update(extract_ner_features(content))
        feature_dict.update(extract_readability_features(content))
        feature_dict.update(extract_hashtag_features(
            row.get('num_hashtags'),
            row.get('hashtags'),
            row.get('hashtag_followers')
        ))
        
        # Calculate base score
        feature_dict['base_score'] = calculate_base_score(feature_dict)
        
        features_list.append(feature_dict)
    
    # Create feature dataframe
    features_df = pd.DataFrame(features_list)
    
    # Concatenate with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # Add influencer-level features
    result_df = add_influencer_features(result_df)
    
    # Add quality scores
    quality_scores = result_df.apply(calculate_quality_scores, axis=1, result_type='expand')
    result_df = pd.concat([result_df, quality_scores], axis=1)
    
    print(f"Feature extraction complete. Total features: {len(result_df.columns)}")
    
    return result_df
```

## 5. USAGE EXAMPLE

```python
# Load data
df = pd.read_csv('data/cleaned_data.csv')

# Extract all features
df_with_features = extract_all_features(df)

# Save
df_with_features.to_csv('data/feature_engineered_data.csv', index=False)

print(f"Shape: {df_with_features.shape}")
print(f"New features added: {len(df_with_features.columns) - len(df.columns)}")
```

## 6. FEATURE SUMMARY

### Total Feature Count: ~80-100 features

**Categories:**
- Base Formula: 25-30 features
- Text Statistics: 10 features
- Sentiment: 7 features
- NER: 8 features
- Readability: 4 features
- Hashtags: 4 features
- Influencer-level: 8 features
- Quality Scores: 3 features
- Original columns: ~15-20 features

This comprehensive feature set provides multiple angles for predicting engagement and can be used for feature selection in the modeling phase.
