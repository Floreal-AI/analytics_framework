"""
Challenge generation and sending system for the validator.

This module replaces the simple generate.py with a more sophisticated challenge
system that creates realistic conversation features and sends them to miners
for binary conversion prediction.
"""

import uuid
import random
import time
import asyncio
import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

from conversion_subnet.protocol import ConversionSynapse, ConversationFeatures
from conversion_subnet.utils.log import logger
from conversion_subnet.validator.utils import validate_features
from conversion_subnet.validator.reward import Validator
from conversion_subnet.validator.forward import get_external_validation
from conversion_subnet.validator.validation_client import ValidationError

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
Faker.seed(seed)
fake = Faker()

class ConversationChallengeGenerator:
    """
    Generates realistic conversation challenges for testing miners.
    
    This class creates diverse conversation scenarios with varying
    complexity and conversion likelihood to thoroughly test miner performance.
    """
    
    def __init__(self, historical_data_path: Optional[str] = None):
        """
        Initialize the challenge generator.
        
        Args:
            historical_data_path: Path to historical data for realistic distributions
        """
        self.historical_stats = None
        self.challenge_templates = self._create_challenge_templates()
        
        # Load historical data if available for more realistic distributions
        if historical_data_path and Path(historical_data_path).exists():
            try:
                self._load_historical_stats(historical_data_path)
                logger.info(f"Loaded historical statistics from {historical_data_path}")
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
    
    def _load_historical_stats(self, data_path: str) -> None:
        """Load historical conversation statistics for realistic generation."""
        df = pd.read_csv(data_path)
        
        # Extract statistics for realistic value generation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.historical_stats = {}
        
        for col in numeric_columns:
            if col in ['id', 'target']:
                continue
            self.historical_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    
    def _create_challenge_templates(self) -> List[Dict]:
        """
        Create different challenge templates representing various conversation scenarios.
        
        Returns:
            List of challenge template dictionaries
        """
        templates = [
            {
                'name': 'high_engagement_conversion',
                'description': 'High engagement conversation likely to convert',
                'base_params': {
                    'total_messages_range': (15, 50),
                    'conversation_duration_range': (300, 1800),  # 5-30 minutes
                    'message_ratio_range': (1.2, 2.0),
                    'entities_collected_range': (3, 8),
                    'conversion_probability': 0.8
                }
            },
            {
                'name': 'quick_exit',
                'description': 'Short conversation with early exit',
                'base_params': {
                    'total_messages_range': (1, 5),
                    'conversation_duration_range': (30, 180),  # 0.5-3 minutes
                    'message_ratio_range': (0.5, 1.5),
                    'entities_collected_range': (0, 2),
                    'conversion_probability': 0.1
                }
            },
            {
                'name': 'medium_engagement',
                'description': 'Medium length conversation with moderate engagement',
                'base_params': {
                    'total_messages_range': (6, 20),
                    'conversation_duration_range': (180, 600),  # 3-10 minutes
                    'message_ratio_range': (0.8, 1.5),
                    'entities_collected_range': (1, 5),
                    'conversion_probability': 0.4
                }
            },
            {
                'name': 'extended_exploration',
                'description': 'Long conversation with extensive exploration but no conversion',
                'base_params': {
                    'total_messages_range': (20, 60),
                    'conversation_duration_range': (600, 2400),  # 10-40 minutes
                    'message_ratio_range': (1.0, 1.8),
                    'entities_collected_range': (2, 6),
                    'conversion_probability': 0.2
                }
            },
            {
                'name': 'business_hours_professional',
                'description': 'Professional conversation during business hours',
                'base_params': {
                    'total_messages_range': (8, 25),
                    'conversation_duration_range': (240, 900),  # 4-15 minutes
                    'message_ratio_range': (1.1, 1.6),
                    'entities_collected_range': (2, 6),
                    'conversion_probability': 0.6,
                    'force_business_hours': True
                }
            }
        ]
        return templates
    
    def generate_challenge_conversation(self, template_name: Optional[str] = None) -> ConversationFeatures:
        """
        Generate a realistic conversation challenge.
        
        Args:
            template_name: Specific template to use, or None for random selection
            
        Returns:
            ConversationFeatures: Generated conversation features
        """
        # Select template
        if template_name:
            template = next((t for t in self.challenge_templates if t['name'] == template_name), None)
            if not template:
                logger.warning(f"Template '{template_name}' not found, using random template")
                template = random.choice(self.challenge_templates)
        else:
            template = random.choice(self.challenge_templates)
        
        logger.debug(f"Generating challenge using template: {template['name']}")
        
        # Generate base conversation features
        features = self._generate_base_features(template['base_params'])
        
        # Add derived and calculated features
        features = self._add_derived_features(features)
        
        # Validate and adjust features for logical consistency
        features = validate_features(features)
        
        return features
    
    def _generate_base_features(self, params: Dict) -> Dict:
        """Generate base conversation features based on template parameters."""
        session_id = str(uuid.uuid4())
        
        # Time-based features
        if params.get('force_business_hours', False):
            hour_of_day = random.randint(9, 17)  # 9 AM to 5 PM
            day_of_week = random.randint(0, 4)   # Monday to Friday
            is_business_hours = 1
            is_weekend = 0
        else:
            hour_of_day = random.randint(0, 23)
            day_of_week = random.randint(0, 6)
            is_business_hours = 1 if (9 <= hour_of_day <= 17 and day_of_week < 5) else 0
            is_weekend = 1 if day_of_week >= 5 else 0
        
        # Message and conversation features
        total_messages = random.randint(*params['total_messages_range'])
        conversation_duration_seconds = random.randint(*params['conversation_duration_range'])
        conversation_duration_minutes = conversation_duration_seconds / 60.0
        
        # Message distribution
        message_ratio = random.uniform(*params['message_ratio_range'])
        user_messages = max(1, int(total_messages / (1 + message_ratio)))
        agent_messages = total_messages - user_messages
        
        # Entity features
        entities_collected_count = random.randint(*params['entities_collected_range'])
        has_target_entity = 1 if entities_collected_count > 0 and random.random() > 0.3 else 0
        
        # Message timing features
        messages_per_minute = total_messages / max(1, conversation_duration_minutes)
        
        # Response time features (in seconds)
        avg_response_time = random.uniform(5.0, 30.0)
        response_time_variation = random.uniform(0.5, 2.0)
        
        min_response_time = max(1.0, avg_response_time - response_time_variation * 10)
        max_response_time = avg_response_time + response_time_variation * 20
        response_time_stddev = response_time_variation * 5
        
        time_to_first_response = random.uniform(2.0, 20.0)
        avg_agent_response_time = random.uniform(8.0, 25.0)
        avg_user_response_time = random.uniform(15.0, 60.0)
        response_gap_max = max_response_time * random.uniform(1.2, 2.0)
        
        # Message length features
        avg_user_message_length = random.uniform(20.0, 80.0)
        avg_agent_message_length = random.uniform(50.0, 150.0)
        
        max_user_message_length = int(avg_user_message_length * random.uniform(1.5, 3.0))
        min_user_message_length = max(1, int(avg_user_message_length * random.uniform(0.3, 0.8)))
        
        max_agent_message_length = int(avg_agent_message_length * random.uniform(1.5, 2.5))
        min_agent_message_length = max(1, int(avg_agent_message_length * random.uniform(0.4, 0.8)))
        
        total_chars_from_user = int(user_messages * avg_user_message_length)
        total_chars_from_agent = int(agent_messages * avg_agent_message_length)
        
        # Question features
        question_count_user = random.randint(0, max(1, user_messages // 3))
        question_count_agent = random.randint(1, max(1, agent_messages // 2))
        
        questions_per_user_message = question_count_user / max(1, user_messages)
        questions_per_agent_message = question_count_agent / max(1, agent_messages)
        
        # Sequential message features
        sequential_user_messages = random.randint(1, max(1, min(5, user_messages)))
        sequential_agent_messages = random.randint(1, max(1, min(3, agent_messages)))
        
        # Entity confidence features
        if entities_collected_count > 0:
            avg_entity_confidence = random.uniform(0.7, 0.95)
            min_entity_confidence = avg_entity_confidence - random.uniform(0.05, 0.15)
            min_entity_confidence = max(0.5, min_entity_confidence)
            entity_collection_rate = entities_collected_count / max(1, total_messages) * random.uniform(0.8, 1.2)
        else:
            avg_entity_confidence = -1.0
            min_entity_confidence = -1.0
            entity_collection_rate = 0.0
        
        # Behavioral features
        repeated_questions = 1 if random.random() < 0.3 else 0
        message_alternation_rate = random.uniform(0.6, 1.0)
        
        # Conversion outcome (based on template probability)
        conversion_happened = 1 if random.random() < params['conversion_probability'] else 0
        time_to_conversion_minutes = -1.0
        time_to_conversion_seconds = -1.0
        
        if conversion_happened:
            # Conversion typically happens in last 20-80% of conversation
            conversion_point = random.uniform(0.2, 0.8)
            time_to_conversion_minutes = conversation_duration_minutes * conversion_point
            time_to_conversion_seconds = time_to_conversion_minutes * 60
        
        features = {
            'session_id': session_id,
            'isWeekend': float(is_weekend),
            'dayOfWeek': float(day_of_week),
            'hourOfDay': float(hour_of_day),
            'messageRatio': message_ratio,
            'totalMessages': float(total_messages),
            'responseGapMax': response_gap_max,
            'hasTargetEntity': float(has_target_entity),
            'isBusinessHours': float(is_business_hours),
            'repeatedQuestions': float(repeated_questions),
            'messagesPerMinute': messages_per_minute,
            'questionCountUser': float(question_count_user),
            'userMessagesCount': float(user_messages),
            'agentMessagesCount': float(agent_messages),
            'questionCountAgent': float(question_count_agent),
            'responseTimeStddev': response_time_stddev,
            'avgEntityConfidence': avg_entity_confidence,
            'minEntityConfidence': min_entity_confidence,
            'totalCharsFromUser': float(total_chars_from_user),
            'entityCollectionRate': entity_collection_rate,
            'totalCharsFromAgent': float(total_chars_from_agent),
            'avgMessageLengthUser': avg_user_message_length,
            'maxMessageLengthUser': float(max_user_message_length),
            'minMessageLengthUser': float(min_user_message_length),
            'avgMessageLengthAgent': avg_agent_message_length,
            'entitiesCollectedCount': float(entities_collected_count),
            'maxMessageLengthAgent': float(max_agent_message_length),
            'messageAlternationRate': message_alternation_rate,
            'minMessageLengthAgent': float(min_agent_message_length),
            'sequentialUserMessages': float(sequential_user_messages),
            'avgResponseTimeSeconds': avg_response_time,
            'maxResponseTimeSeconds': max_response_time,
            'minResponseTimeSeconds': min_response_time,
            'sequentialAgentMessages': float(sequential_agent_messages),
            'questionsPerUserMessage': questions_per_user_message,
            'timeToConversionMinutes': time_to_conversion_minutes,
            'timeToConversionSeconds': time_to_conversion_seconds,
            'questionsPerAgentMessage': questions_per_agent_message,
            'conversationDurationMinutes': conversation_duration_minutes,
            'conversationDurationSeconds': float(conversation_duration_seconds),
            'avgUserResponseTimeSeconds': avg_user_response_time,
            'timeToFirstResponseSeconds': time_to_first_response,
            'avgAgentResponseTimeSeconds': avg_agent_response_time,
            # Ground truth for validation
            'target': float(conversion_happened)
        }
        
        return features
    
    def _add_derived_features(self, features: Dict) -> Dict:
        """Add derived features and ensure logical consistency."""
        # Ensure values that should be -1 when no entities are present
        if features['entitiesCollectedCount'] == 0:
            features['avgEntityConfidence'] = -1.0
            features['minEntityConfidence'] = -1.0
            features['entityCollectionRate'] = 0.0
        
        # Ensure conversion timing features are -1 when no conversion
        if features['target'] == 0:
            features['timeToConversionMinutes'] = -1.0
            features['timeToConversionSeconds'] = -1.0
        
        # Ensure response time features are consistent
        if features['totalMessages'] <= 1:
            features['avgResponseTimeSeconds'] = -1.0
            features['maxResponseTimeSeconds'] = -1.0
            features['minResponseTimeSeconds'] = -1.0
            features['responseTimeStddev'] = -1.0
            features['avgUserResponseTimeSeconds'] = -1.0
            features['avgAgentResponseTimeSeconds'] = -1.0
            features['timeToFirstResponseSeconds'] = -1.0
        
        return features
    
    def generate_batch_challenges(self, batch_size: int, template_distribution: Optional[Dict[str, float]] = None) -> List[ConversationFeatures]:
        """
        Generate a batch of conversation challenges.
        
        Args:
            batch_size: Number of challenges to generate
            template_distribution: Distribution of templates to use (template_name -> probability)
            
        Returns:
            List of generated conversation features
        """
        challenges = []
        
        # Default template distribution if none provided
        if template_distribution is None:
            template_distribution = {
                'high_engagement_conversion': 0.2,
                'quick_exit': 0.3,
                'medium_engagement': 0.3,
                'extended_exploration': 0.1,
                'business_hours_professional': 0.1
            }
        
        # Generate challenges according to distribution
        for _ in range(batch_size):
            template_name = np.random.choice(
                list(template_distribution.keys()),
                p=list(template_distribution.values())
            )
            
            challenge = self.generate_challenge_conversation(template_name)
            challenges.append(challenge)
        
        return challenges
    
    def create_synapse_from_features(self, features: ConversationFeatures, miner_uid: int = 0) -> ConversionSynapse:
        """
        Create a ConversionSynapse from conversation features.
        
        Args:
            features: Conversation features dictionary
            miner_uid: UID of the target miner
            
        Returns:
            ConversionSynapse ready to send to miner
        """
        # Remove target and other non-feature fields
        feature_dict = {k: v for k, v in features.items() 
                       if k not in ['target', 'time_to_conversion_seconds', 'time_to_conversion_minutes']}
        
        synapse = ConversionSynapse(
            features=feature_dict,
            miner_uid=miner_uid
        )
        
        return synapse
    
    async def process_challenges_one_by_one(
        self, 
        miner_forward_function: callable,
        num_challenges: int = 5,
        miner_uid: int = 0,
        template_distribution: Optional[Dict[str, float]] = None,
        include_validation: bool = True,
        delay_between_challenges: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process challenges one by one with detailed step-by-step execution.
        
        This method implements sequential challenge processing:
        1. Generate a challenge
        2. Send to miner
        3. Wait for response
        4. Validate response (if validation enabled)
        5. Calculate reward
        6. Log detailed results
        7. Repeat for next challenge
        
        Args:
            miner_forward_function: Function to call miner (e.g., miner.forward)
            num_challenges: Number of challenges to process
            miner_uid: UID of the target miner
            template_distribution: Distribution of templates to use
            include_validation: Whether to fetch external validation
            delay_between_challenges: Delay in seconds between challenges
            
        Returns:
            Dict containing detailed results of all challenges
        """
        logger.info(f"Starting one-by-one challenge processing: {num_challenges} challenges for miner {miner_uid}")
        
        # Initialize validator for reward calculation
        validator = Validator()
        
        # Results tracking
        results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'miner_uid': miner_uid,
                'num_challenges': num_challenges,
                'include_validation': include_validation,
                'template_distribution': template_distribution or 'default'
            },
            'challenges': [],
            'summary': {
                'total_challenges': 0,
                'successful_predictions': 0,
                'successful_validations': 0,
                'total_reward': 0.0,
                'average_reward': 0.0,
                'total_processing_time': 0.0,
                'prediction_accuracy': 0.0
            }
        }
        
        start_time = time.time()
        
        # Process challenges one by one
        for challenge_idx in range(num_challenges):
            challenge_start_time = time.time()
            
            logger.info(f"Processing challenge {challenge_idx + 1}/{num_challenges}")
            
            challenge_result = {
                'challenge_id': challenge_idx + 1,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': None,
                'processing_steps': []
            }
            
            try:
                # Step 1: Generate challenge
                step_start = time.time()
                
                features = self.generate_challenge_conversation(
                    template_name=self._select_template_for_challenge(template_distribution)
                )
                
                ground_truth_target = features['target']
                synapse = self.create_synapse_from_features(features, miner_uid)
                
                step_time = time.time() - step_start
                challenge_result['processing_steps'].append({
                    'step': 'generate_challenge',
                    'duration': round(step_time, 4),
                    'success': True,
                    'details': {
                        'session_id': features['session_id'],
                        'template_used': self._identify_template(features),
                        'ground_truth': ground_truth_target,
                        'feature_count': len(synapse.features),
                        'total_messages': features.get('totalMessages', 0),
                        'conversation_duration': features.get('conversationDurationMinutes', 0)
                    }
                })
                
                logger.debug(f"  Generated challenge: session_id={features['session_id']}, ground_truth={ground_truth_target}")
                
                # Step 2: Send to miner and get prediction
                step_start = time.time()
                
                try:
                    miner_response = await asyncio.to_thread(miner_forward_function, synapse)
                    
                    # Handle different response formats
                    if isinstance(miner_response, dict):
                        prediction = miner_response
                    elif hasattr(miner_response, 'prediction'):
                        prediction = miner_response.prediction
                    else:
                        # Fallback for simple numeric responses
                        prediction = {
                            'conversion_happened': int(miner_response) if isinstance(miner_response, (int, float)) else 0,
                            'time_to_conversion_seconds': 60.0 if miner_response == 1 else -1.0
                        }
                    
                    step_time = time.time() - step_start
                    challenge_result['processing_steps'].append({
                        'step': 'miner_prediction',
                        'duration': round(step_time, 4),
                        'success': True,
                        'details': {
                            'prediction': prediction,
                            'confidence': getattr(miner_response, 'confidence', 0.8) if hasattr(miner_response, 'confidence') else 0.8,
                            'response_time': step_time
                        }
                    })
                    
                    logger.debug(f"  Miner prediction: {prediction}")
                    
                except Exception as pred_error:
                    step_time = time.time() - step_start
                    challenge_result['processing_steps'].append({
                        'step': 'miner_prediction',
                        'duration': round(step_time, 4),
                        'success': False,
                        'error': str(pred_error)
                    })
                    logger.error(f"  Miner prediction failed: {pred_error}")
                    continue
                
                # Step 3: Get validation (ground truth) if enabled
                validation_result = None
                if include_validation:
                    step_start = time.time()
                    
                    try:
                        # Try to get external validation
                        test_pk = features.get('session_id', f"challenge_{challenge_idx}")
                        validation_result = await get_external_validation(test_pk)
                        
                        ground_truth = {
                            'conversion_happened': 1 if validation_result.labels else 0,
                            'time_to_conversion_seconds': 60.0 if validation_result.labels else -1.0
                        }
                        
                        step_time = time.time() - step_start
                        challenge_result['processing_steps'].append({
                            'step': 'external_validation',
                            'duration': round(step_time, 4),
                            'success': True,
                            'details': {
                                'validation_source': 'external_api',
                                'ground_truth': ground_truth,
                                'test_pk': test_pk
                            }
                        })
                        
                        logger.debug(f"  External validation: {ground_truth}")
                        
                    except (ValidationError, Exception) as val_error:
                        # NO FALLBACKS - let validation errors bubble up
                        step_time = time.time() - step_start
                        challenge_result['processing_steps'].append({
                            'step': 'external_validation',
                            'duration': round(step_time, 4),
                            'success': False,
                            'error': str(val_error),
                        })
                        
                        logger.error(f"  External validation failed: {val_error}")
                        # Re-raise the error instead of using fallback
                        raise ValidationError(f"External validation failed for test_pk {test_pk}: {val_error}") from val_error
                else:
                    # Use generated ground truth
                    ground_truth = {
                        'conversion_happened': int(ground_truth_target),
                        'time_to_conversion_seconds': features.get('timeToConversionSeconds', -1.0)
                    }
                    
                    challenge_result['processing_steps'].append({
                        'step': 'use_generated_ground_truth',
                        'duration': 0.0,
                        'success': True,
                        'details': {'ground_truth': ground_truth}
                    })
                
                # Step 4: Calculate reward
                step_start = time.time()
                
                try:
                    # Update synapse with prediction for reward calculation
                    synapse.prediction = prediction
                    synapse.confidence = challenge_result['processing_steps'][-2]['details'].get('confidence', 0.8)
                    synapse.response_time = challenge_result['processing_steps'][-2]['details']['response_time']
                    
                    reward = validator.reward(ground_truth, synapse)
                    
                    step_time = time.time() - step_start
                    challenge_result['processing_steps'].append({
                        'step': 'reward_calculation',
                        'duration': round(step_time, 4),
                        'success': True,
                        'details': {
                            'reward': round(reward, 6),
                            'prediction_correct': prediction['conversion_happened'] == ground_truth['conversion_happened'],
                            'ground_truth': ground_truth,
                            'prediction': prediction
                        }
                    })
                    
                    # Update results
                    results['summary']['total_reward'] += reward
                    results['summary']['successful_predictions'] += 1
                    if include_validation and validation_result:
                        results['summary']['successful_validations'] += 1
                    
                    logger.info(f"  Reward calculated: {reward:.6f}")
                    challenge_result['success'] = True
                    
                except Exception as reward_error:
                    step_time = time.time() - step_start
                    challenge_result['processing_steps'].append({
                        'step': 'reward_calculation',
                        'duration': round(step_time, 4),
                        'success': False,
                        'error': str(reward_error)
                    })
                    logger.error(f"  Reward calculation failed: {reward_error}")
                
                # Calculate challenge processing time
                challenge_processing_time = time.time() - challenge_start_time
                challenge_result['total_processing_time'] = round(challenge_processing_time, 4)
                
                logger.info(f"  Challenge {challenge_idx + 1} completed in {challenge_processing_time:.3f}s")
                
            except Exception as e:
                challenge_result['error'] = str(e)
                logger.error(f"  Challenge {challenge_idx + 1} failed: {e}")
            
            # Store challenge result
            results['challenges'].append(challenge_result)
            results['summary']['total_challenges'] += 1
            
            # Add delay between challenges
            if challenge_idx < num_challenges - 1 and delay_between_challenges > 0:
                logger.debug(f"  Waiting {delay_between_challenges}s before next challenge...")
                await asyncio.sleep(delay_between_challenges)
        
        # Calculate final summary statistics
        total_time = time.time() - start_time
        results['summary']['total_processing_time'] = round(total_time, 4)
        
        if results['summary']['successful_predictions'] > 0:
            results['summary']['average_reward'] = round(
                results['summary']['total_reward'] / results['summary']['successful_predictions'], 6
            )
            
            # Calculate prediction accuracy
            correct_predictions = sum(
                1 for challenge in results['challenges']
                if challenge['success'] and any(
                    step['step'] == 'reward_calculation' and step['success'] and step['details']['prediction_correct']
                    for step in challenge['processing_steps']
                )
            )
            results['summary']['prediction_accuracy'] = round(
                correct_predictions / results['summary']['successful_predictions'], 3
            )
        
        results['metadata']['end_time'] = datetime.now().isoformat()
        results['metadata']['total_duration'] = round(total_time, 4)
        
        logger.info(f"One-by-one processing completed: {results['summary']['successful_predictions']}/{num_challenges} successful")
        logger.info(f"Total reward: {results['summary']['total_reward']:.6f}, Average: {results['summary']['average_reward']:.6f}")
        logger.info(f"Prediction accuracy: {results['summary']['prediction_accuracy']:.1%}")
        
        return results
    
    def _select_template_for_challenge(self, template_distribution: Optional[Dict[str, float]]) -> Optional[str]:
        """Select a template based on distribution or randomly."""
        if template_distribution is None:
            return None  # Will use random selection in generate_challenge_conversation
        
        return np.random.choice(
            list(template_distribution.keys()),
            p=list(template_distribution.values())
        )
    
    def _identify_template(self, features: Dict) -> str:
        """Identify which template was likely used based on features."""
        total_messages = features.get('totalMessages', 0)
        duration_minutes = features.get('conversationDurationMinutes', 0)
        target = features.get('target', 0)
        
        if total_messages <= 5 and duration_minutes <= 3:
            return 'quick_exit'
        elif total_messages >= 20 and target == 0:
            return 'extended_exploration'
        elif target == 1 and total_messages >= 15:
            return 'high_engagement_conversion'
        elif features.get('isBusinessHours', 0) == 1:
            return 'business_hours_professional'
        else:
            return 'medium_engagement'


# Convenience functions for easy usage
def generate_single_challenge(template_name: Optional[str] = None) -> ConversationFeatures:
    """Generate a single conversation challenge."""
    generator = ConversationChallengeGenerator()
    return generator.generate_challenge_conversation(template_name)

def generate_challenge_batch(batch_size: int, template_distribution: Optional[Dict[str, float]] = None) -> List[ConversationFeatures]:
    """Generate a batch of conversation challenges."""
    generator = ConversationChallengeGenerator()
    return generator.generate_batch_challenges(batch_size, template_distribution)

def create_test_synapse(miner_uid: int = 0, template_name: Optional[str] = None) -> Tuple[ConversionSynapse, float]:
    """
    Create a test synapse with ground truth for validation.
    
    Args:
        miner_uid: UID of target miner
        template_name: Template to use for generation
        
    Returns:
        Tuple of (synapse, ground_truth_target)
    """
    generator = ConversationChallengeGenerator()
    features = generator.generate_challenge_conversation(template_name)
    ground_truth = features['target']
    
    synapse = generator.create_synapse_from_features(features, miner_uid)
    
    return synapse, ground_truth

async def process_miner_challenges_one_by_one(
    miner_forward_function: callable,
    num_challenges: int = 5,
    miner_uid: int = 0,
    template_distribution: Optional[Dict[str, float]] = None,
    include_validation: bool = True,
    delay_between_challenges: float = 0.5
) -> Dict[str, Any]:
    """
    Convenience function for one-by-one challenge processing.
    
    This function provides easy access to sequential challenge processing
    without needing to instantiate the ConversationChallengeGenerator class.
    
    Args:
        miner_forward_function: Function to call miner (e.g., miner.forward)
        num_challenges: Number of challenges to process sequentially
        miner_uid: UID of the target miner
        template_distribution: Distribution of templates to use
        include_validation: Whether to fetch external validation
        delay_between_challenges: Delay in seconds between challenges
        
    Returns:
        Dict containing detailed results of all challenges
        
    Example:
        ```python
        from conversion_subnet.miner.miner import BinaryClassificationMiner
        from conversion_subnet.validator.send_challenge import process_miner_challenges_one_by_one
        
        miner = BinaryClassificationMiner()
        
        results = await process_miner_challenges_one_by_one(
            miner_forward_function=miner.forward,
            num_challenges=3,
            miner_uid=123,
            include_validation=False,
            delay_between_challenges=1.0
        )
        
        print(f"Total reward: {results['summary']['total_reward']}")
        print(f"Average reward: {results['summary']['average_reward']}")
        print(f"Accuracy: {results['summary']['prediction_accuracy']}")
        ```
    """
    generator = ConversationChallengeGenerator()
    return await generator.process_challenges_one_by_one(
        miner_forward_function=miner_forward_function,
        num_challenges=num_challenges,
        miner_uid=miner_uid,
        template_distribution=template_distribution,
        include_validation=include_validation,
        delay_between_challenges=delay_between_challenges
    )


if __name__ == "__main__":
    # Example usage and testing
    generator = ConversationChallengeGenerator()
    
    print("Testing Challenge Generator")
    print("=" * 50)
    
    # Test each template
    for template in generator.challenge_templates:
        print(f"\nTesting template: {template['name']}")
        challenge = generator.generate_challenge_conversation(template['name'])
        print(f"  Session ID: {challenge['session_id']}")
        print(f"  Total Messages: {challenge['totalMessages']}")
        print(f"  Duration: {challenge['conversationDurationMinutes']:.1f} minutes")
        print(f"  Conversion: {challenge['target']}")
        
    # Test batch generation
    print("\nTesting batch generation...")
    batch = generator.generate_batch_challenges(5)
    print(f"Generated {len(batch)} challenges")
    
    # Test synapse creation
    print("\nTesting synapse creation...")
    synapse, ground_truth = create_test_synapse(miner_uid=123)
    print(f"Synapse created for miner {synapse.miner_uid}")
    print(f"Ground truth: {ground_truth}")
    print(f"Features count: {len(synapse.features)}") 