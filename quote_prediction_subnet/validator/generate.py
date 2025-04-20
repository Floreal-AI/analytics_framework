import numpy as np

def generate_interaction():
    """
    Generate synthetic customer interaction data with features and targets.
    
    Returns:
        dict: Contains 'features' (dict of feature values) and 'targets' (dict with quote_delivery_time and quote_acceptance).
    """
    # Generate features based on dataset statistics
    features = {
        'interaction_duration': np.random.normal(117.13, 12.36),
        'question_count': round(np.random.normal(5.53, 1.08)),
        'repetition_count': round(np.random.normal(1.11, 1.06)),
        'entity_completeness': np.random.normal(94.21, 5.62) / 100,
        'avg_confidence_score': np.random.normal(0.87, 0.03),
        'low_confidence_flag': np.random.binomial(1, 0.41),
        'sentiment_score': np.random.normal(0.71, 0.09),
        'emotion_intensity': np.random.normal(0.83, 0.12),
        'surface_area': np.random.normal(60.06, 10.53),
        'detection_canine_flag': np.random.binomial(1, 0.70),
        'quote_value': np.random.normal(669.18, 80.51),
        'user_query_count': round(np.random.normal(1.82, 0.79)),
        'session_length': np.random.normal(25.85, 2.59),
    }
    
    # Compute quote_delivery_time based on key features with noise
    quote_delivery_time = (
        0.8 * features['interaction_duration'] +
        2 * features['question_count'] +
        3 * features['repetition_count'] +
        np.random.normal(0, 5)
    )
    
    # Compute quote_acceptance based on sentiment and emotion
    score = 0.5 * features['sentiment_score'] + 0.3 * features['emotion_intensity']
    quote_acceptance = 1 if score > 0.5 else 0
    
    return {
        'features': features,
        'targets': {
            'quote_delivery_time': quote_delivery_time,
            'quote_acceptance': quote_acceptance
        }
    }