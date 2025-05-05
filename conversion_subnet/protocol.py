import bittensor as bt
from typing import Optional, Dict, Union, List
from typing_extensions import TypedDict

class ConversationFeatures(TypedDict, total=False):
    """
    TypedDict for conversation features used in the analytics framework.
    
    This type definition makes it easier to catch typos and ensures all code
    using these features has consistent expectations about what's available.
    
    Features marked as required=True are essential for basic functioning.
    """
    # Required features (must be present)
    session_id: str                           # Unique identifier for the conversation
    conversation_duration_seconds: float      # Total duration of conversation in seconds
    has_target_entity: int                    # Binary indicator if target entity was found (0 or 1)
    entities_collected_count: int             # Number of entities collected during conversation
    message_ratio: float                      # Ratio of agent messages to user messages
    
    # Optional features
    conversation_duration_minutes: float      # Total duration in minutes
    hour_of_day: int                          # Hour when conversation started (0-23)
    day_of_week: int                          # Day of week when conversation started (0-6)
    is_business_hours: int                    # Binary indicator if during business hours (0 or 1)
    is_weekend: int                           # Binary indicator if weekend (0 or 1)
    time_to_first_response_seconds: float     # Time between first user message and first agent response
    avg_response_time_seconds: float          # Average time between messages
    max_response_time_seconds: float          # Maximum time between messages
    min_response_time_seconds: float          # Minimum time between messages
    avg_agent_response_time_seconds: float    # Average time for agent to respond
    avg_user_response_time_seconds: float     # Average time for user to respond
    response_time_stddev: float               # Standard deviation of response times
    response_gap_max: float                   # Maximum gap between responses
    messages_per_minute: float                # Average message rate
    total_messages: int                       # Total count of messages
    user_messages_count: int                  # Count of user messages
    agent_messages_count: int                 # Count of agent messages
    avg_message_length_user: float            # Average length of user messages
    max_message_length_user: int              # Maximum length of user messages
    min_message_length_user: int              # Minimum length of user messages
    total_chars_from_user: int                # Total characters from user
    avg_message_length_agent: float           # Average length of agent messages
    max_message_length_agent: int             # Maximum length of agent messages
    min_message_length_agent: int             # Minimum length of agent messages
    total_chars_from_agent: int               # Total characters from agent
    question_count_agent: int                 # Count of questions asked by agent
    questions_per_agent_message: float        # Average questions per agent message
    question_count_user: int                  # Count of questions asked by user
    questions_per_user_message: float         # Average questions per user message
    sequential_user_messages: int             # Count of sequential user messages
    sequential_agent_messages: int            # Count of sequential agent messages
    avg_entity_confidence: float              # Average confidence of entity recognition
    min_entity_confidence: float              # Minimum confidence of entity recognition
    entity_collection_rate: float             # Rate of entity collection
    repeated_questions: int                   # Count of repeated questions
    message_alternation_rate: float           # Rate of message alternation between user and agent

class PredictionOutput(TypedDict):
    """
    TypedDict for miner prediction outputs.
    
    This ensures consistent prediction format and makes it easier to validate
    outputs from miners.
    """
    conversion_happened: int                  # Binary indicator if conversion happened (0 or 1)
    time_to_conversion_seconds: float         # Time to conversion in seconds (-1.0 if no conversion)

# Create a default prediction that always passes validation
DEFAULT_PREDICTION = {
    'conversion_happened': 0,
    'time_to_conversion_seconds': -1.0
}

class ConversionSynapse(bt.Synapse):
    """
    A synapse protocol for the AI Agent Task Performance Subnet, inheriting from bt.Synapse.
    Facilitates communication between validators and miners by defining input features and output predictions.

    Attributes:
        features: Dictionary of 40 conversation features (e.g., conversation_duration_seconds, has_target_entity).
        prediction: Dictionary with miner predictions (conversion_happened, time_to_conversion_seconds).
        confidence: Predicted probability for conversion_happened (from model's predict_proba).
        response_time: Time taken by miner to respond (seconds).
        miner_uid: Unique identifier for the miner.
    """
    # Input: 40 conversation features sent by validator
    features: ConversationFeatures

    # Output: Predictions and confidence returned by miner
    # Initialize with a valid default to avoid validation errors
    prediction: PredictionOutput = DEFAULT_PREDICTION  
    confidence: Optional[float] = None

    # Metadata for scoring
    response_time: float = 0.0
    miner_uid: int = 0

    def __post_init__(self):
        """
        Post-initialization hook to ensure data types are correct.
        Converts integer fields to integers if they were provided as floats.
        """
        if self.features:
            integer_fields = [
                'hour_of_day', 'day_of_week', 'is_business_hours', 'is_weekend',
                'total_messages', 'user_messages_count', 'agent_messages_count',
                'max_message_length_user', 'min_message_length_user', 'total_chars_from_user',
                'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent',
                'question_count_agent', 'question_count_user', 'sequential_user_messages',
                'sequential_agent_messages', 'entities_collected_count', 'has_target_entity',
                'repeated_questions'
            ]
            
            for field in integer_fields:
                if field in self.features and self.features[field] is not None:
                    try:
                        self.features[field] = int(self.features[field])
                    except (ValueError, TypeError):
                        # If conversion fails, log warning but continue
                        bt.logging.warning(f"Failed to convert {field} to integer: {self.features[field]}")
        
        # Ensure prediction is valid
        self.set_prediction(self.prediction)

    def set_prediction(self, prediction: Optional[Union[Dict, PredictionOutput]]) -> None:
        """
        Set the prediction with validation and type conversion.
        
        Args:
            prediction: The prediction to set, or None to use default
        """
        if prediction is None or not prediction:
            # Use default prediction
            self.prediction = DEFAULT_PREDICTION.copy()
            return
            
        # Make a copy to avoid modifying the original
        result = prediction.copy()
        
        # Ensure required fields exist
        if 'conversion_happened' not in result:
            result['conversion_happened'] = 0
            
        if 'time_to_conversion_seconds' not in result:
            result['time_to_conversion_seconds'] = -1.0
            
        # Fix data types
        try:
            result['conversion_happened'] = int(result['conversion_happened'])
        except (ValueError, TypeError):
            result['conversion_happened'] = 0
            
        try:
            result['time_to_conversion_seconds'] = float(result['time_to_conversion_seconds'])
        except (ValueError, TypeError):
            result['time_to_conversion_seconds'] = -1.0
            
        # Set the validated prediction
        self.prediction = result

    def deserialize(self) -> Union[PredictionOutput, Dict]:
        """
        Deserialize the miner's response.

        Returns:
            PredictionOutput: The miner's prediction (conversion_happened, time_to_conversion_seconds).
        """
        if self.prediction is None:
            return DEFAULT_PREDICTION.copy()
        return self.prediction

