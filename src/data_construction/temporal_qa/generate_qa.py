import os
import json
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
import random
import traceback


# =========================
# Config
# =========================

os.environ["OPENAI_API_KEY"] = ""
client = AsyncOpenAI( api_key=os.environ.get("OPENAI_API_KEY"), )


# =========================
# LLM Call
# =========================

async def get_gpt_response(system_prompt, user_prompt):
    """Get response from GPT"""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        response = completion.choices[0].message.content.strip()
        return response
        
    except Exception as e:
        print(f"Error in GPT response: {e}")
    
    return None



# =========================
# Data Loading and Processing
# =========================

async def load_sampled_events(filename, year):
    """Load sampled events from a JSON file created by the sampling script"""
    try:
        # Get the current directory and construct the full path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(current_dir)
        file_path = os.path.join(workspace_root, f"output_dataset/{year}", filename)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        print(f"Loading sampled events file: {file_path}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Load the file
        all_events = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and 'topics' in data:
                    for topic in data['topics']:
                        topic_title = topic.get('topic_title', '')
                        topic_category = topic.get('topic_category', '')
                        start_date = topic.get('start_date', '')
                        last_date = topic.get('last_date', '')
                        
                        for event in topic.get('events', []):
                            event['date'] = event.get('event_date', '')
                            event['topic_title'] = topic_title
                            event['topic_category'] = topic_category
                            event['topic_timespan'] = {'start': start_date, 'end': last_date}
                            all_events.append(event)
                    
                    print(f"Loaded {len(all_events)} events from {len(data['topics'])} topics")
                else:
                    print(f"Error: Unexpected data format in {filename}")
                    return None
                        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except MemoryError as e:
            print(f"Memory error while loading file: {e}")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
        
        print(f"\nSuccessfully loaded {len(all_events)} events from {filename}")
        return all_events
    
    except Exception as e:
        print(f"Error loading sampled events: {e}")
        traceback.print_exc()  # Add stack trace for better debugging
        return None



# =========================
# QA Generation
# =========================

async def select_random_text(event):
    """
    Select a random text from the event
    """
    if 'event_text' in event and event['event_text']:
        return random.choice(event['event_text'])
    return event.get('event_sum', '')

def clean_result_phrases(results):
    """
    Removes known prefixes like 'Impact: ', 'Status: ', etc. from result strings.
    Returns List[str]
    """
    prefixes = ["Impact: ", "Status: ", "Agenda: ", "Policy: ", "Action: "]
    cleaned = []
    for r in results:
        for prefix in prefixes:
            if r.startswith(prefix):
                r = r[len(prefix):] 
                break
        cleaned.append(r.strip())
    return cleaned


async def generate_qa(selected_event, question_type):
    if question_type is None:
        question_type = random.choice(["Result Recognition", "Entity Tracking"])
    
    # Date
    event_date = selected_event.get("event_date", "an unspecified date")

    # Entities
    people = selected_event.get("People", [])
    locations = selected_event.get("Location", [])

    # Results
    results = selected_event.get("Result", [])
    results = clean_result_phrases(results)

    # Select a random event_text from the event, then truncate the text at length of 60000 (2/3 of GPT-4 128K token limit)
    selected_text = await select_random_text(selected_event)
    if selected_text:
        words = selected_text.split()
        if len(words) > 60000:
            selected_text = ' '.join(words[:60000]) + '...'
        else:
            selected_text = ' '.join(words[:len(words)]) + '...'

    # Design constraints applied to the system prompt
    system_prompt = """
        You are a QA generator for a benchmark dataset that evaluates temporal reasoning over structured event data and news context.

        Your task is to generate a simple, clear question based on the given event metadata, following strict quality constraints.

        # **QUESTION QUALITY CONSTRAINTS**
        - Answer Concealment: Do NOT include the answer target or lexically identical phrases from the Event Context.
        - Temporal Validity: The question must target time-varying attributes (AVOID static attributes like birthplace, parentage) and be explicitly tied to a temporal reference (e.g., currently, after).

        # **ANSWER QUALITY CONSTRAINTS**
        - Contextual Groundedness: The answer must be supported by the Event Context.
        - Specificity Alignment: The specificity of the answer must align with the specificity of the reported information.
    """

    ## Objective ##
    # Result Recognition -> Focus on Results
    if question_type == "Result Recognition":
        if not results:
            return None
        answer_target = random.choice(results)

        user_prompt = f"""
            # Task Description
            As a structural QA generator, you are generating a simple question for evaluating whether an LLM can recognize the result of an event.
            Write a fluent and concise question that asks about the result, based on the event context.

            Event Date: {event_date}
            Event Context: {selected_text}
            Answer Target (Result): {answer_target}

            # Question Format
            - "What was the result of [event or sub-event]?"

            # Desired Example
            - Answer Target (Result): "Robert Gordon University revoked Trump's honorary degree"
            - Question: "What was the result of Donald Trump's comments about Muslims?"
            
            Now write the question.
        """

    # Entity Tracking -> Focus on Entity
    else:  
        entities = people + locations
        # original entities selection
        '''
        if not entities or not results:
            return None
        answer_target = random.choice(entities)
        '''

        # (modified) check if answer_target is in selected_text
        mentioned_entities = [e for e in entities if e in selected_text]


        # (modified) if not mentioned_entities -> return None (07.24 ver)
        if not mentioned_entities:
            return None
        else:
            answer_target = random.choice(mentioned_entities)
        
            
        user_prompt = f"""
            # Task Description
            As a structural QA generator, you are generating a simple question for evaluating whether an LLM can track the status of an entity after an event.
            For the entity and its associated event context (e.g., describing a change in condition or status), write a simple, clear question that asks about the key detail.

            Event Date: {event_date}  
            Event Context: {selected_text}
            Answer Target (Entity): {answer_target}

            # Question Format
            - "Who/Where is currently [role/status]?"

            # Desired Example
            - Answer Target (Entity): "Tim Kaine"  
            - Question: "Who is currently the vice-presidential running mate of Hillary Clinton?"

            Now write the question.
        """

    #question_text = await get_llama_response(system_prompt, user_prompt)
    question_text = await get_gpt_response(system_prompt, user_prompt)

    if not question_text:
        return None

    qa = {
        "question_type": question_type,
        "question_text": question_text,
        "answer_target": answer_target
    }
    print(qa)

    return qa, selected_text


# =========================
# Quality Validation
# =========================

async def validate_qa_quality(qa, selected_text, question_type):
    if question_type == "Entity Tracking":
        system_prompt = """
            You are a QA quality verification expert. Your task is to evaluate a question-answer pair against strict quality criteria.

            The evaluation dimensions are:

            [Question Quality]
            1. Answer Concealment:
            - The question must NOT reveal the answer target.
            - The question must NOT contain lexically identical or near-identical phrases from the event context that trivially expose the answer.

            2. Temporal Validity:
            - The question must target time-varying attributes (e.g., role, status, position, location, outcome).
            - The question must NOT ask about static attributes (e.g., birthplace, parentage, nationality, permanent identity).
            - The question must be explicitly or implicitly tied to a time reference (e.g., currently, after the event, recently).

            [Answer Quality]
            3. Contextual Groundedness:
            - The answer must be grounded in the provided event context.
            - The answer must not rely on external world knowledge beyond the given context.

            4. Specificity Alignment:
            - The specificity level of the answer must match what is supported in the event context.
            - The answer must not be overly coarse (e.g., "USA" when "Boston" is supported) or overly fine-grained beyond the context.

            For each criterion, decide PASS or FAIL and provide a short justification.

            Return a JSON object in the following format:
            {
            "is_valid": true/false,
            "verdicts": {
                "answer_concealment": {"pass": true/false, "reason": "..."},
                "temporal_validity": {"pass": true/false, "reason": "..."},
                "contextual_groundedness": {"pass": true/false, "reason": "..."},
                "specificity_alignment": {"pass": true/false, "reason": "..."}
            },
            "overall_reason": "short summary"
            }
        """
    else: # Result Recognition
        system_prompt = """
            You are a QA quality verification expert. Your task is to evaluate a question-answer pair against strict quality criteria.

            The evaluation dimensions are:

            [Question Quality]
            1. Temporal Validity:
            - The question must be explicitly or implicitly tied to a time reference (e.g., currently, after the event, recently).

            [Answer Quality]
            2. Contextual Groundedness:
            - The answer must be grounded in the provided event context.
            - The answer must not rely on external world knowledge beyond the given context.

            3. Specificity Alignment:
            - The specificity level of the answer must match what is supported in the event context.
            - The answer must not be overly coarse (e.g., "USA" when "Boston" is supported) or overly fine-grained beyond the context.

            For each criterion, decide PASS or FAIL and provide a short justification.

            Return a JSON object in the following format:
            {
            "is_valid": true/false,
            "verdicts": {
                "answer_concealment": {"pass": true/false, "reason": "..."},
                "temporal_validity": {"pass": true/false, "reason": "..."},
                "contextual_groundedness": {"pass": true/false, "reason": "..."},
                "specificity_alignment": {"pass": true/false, "reason": "..."}
            },
            "overall_reason": "short summary"
            }
        """
    
    user_prompt = f"""
    Event Context: {selected_text}
    Question: {qa['question_text']}
    Answer Target: {qa['answer_target']}
    Question Type: {qa['question_type']}
    
    Determine if this QA pair is valid according to the quality criteria.
    """
    
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0
        )

        response = completion.choices[0].message.content.strip()
        result = json.loads(response)

        return result

    except Exception as e:
        print(f"Error in QA quality validation: {e}")
        return {
            "is_valid": False,
            "verdicts": {
                "answer_concealment": {"pass": False, "reason": "Validation failed, defaulting to fail"},
                "temporal_validity": {"pass": False, "reason": "Validation failed, defaulting to fail"},
                "contextual_groundedness": {"pass": False, "reason": "Validation failed, defaulting to fail"},
                "specificity_alignment": {"pass": False, "reason": "Validation failed, defaulting to fail"}
            },
            "overall_reason": "Validation error, defaulting to fail"
        }
    

# =========================
# Question Rephrasing
# =========================

async def rephrase_question(question_text, question_type, answer_target):
    """
    Rephrase questions to make them more fluent and natural.
    Returns the rephrased question.
    """
    system_prompt = """
    You are a question rephrasing expert. Your task is to make questions more fluent, natural, and clear while preserving their meaning.
    
    Guidelines:
    - Make questions sound more conversational and natural
    - Improve clarity and readability
    - Maintain the same question type and intent
    - Use active voice when possible
    - Avoid awkward phrasing
    
    Examples:
    - "What was the result of the Fijian government's discussions about changing the national flag on August 18, 2016?" 
      → "What decision did the Fijian government reach regarding the change of the national flag on August 18, 2016?"
    
    Return only the rephrased question, no additional text.
    """
    
    user_prompt = f"""
    Question Type: {question_type}
    Original Question Text: {question_text}
    Answer Target: {answer_target}
    
    Rephrase only the Original Question Text to make it more fluent and natural.
    """
    
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        rephrased_question = completion.choices[0].message.content.strip()
        return rephrased_question
        
    except Exception as e:
        print(f"Error in question rephrasing: {e}")
        # Return original question if rephrasing fails
        return question_text
    


# =========================
# QA Generation with Quality Validation
# =========================

async def generate_qa_with_quality_validation(selected_event, question_type, max_attempts=3):
    """
    Generate QA pair with quality validation and question rephrasing.
    Iterates until a valid QA pair is found or max attempts reached.
    """
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} for {question_type}")
        
        # === Question Generation ===
        qa, selected_text = await generate_qa(selected_event, question_type)
        if not qa:
            print(f"QA generation failed on attempt {attempt + 1}")
            continue
        
        # === QA Quality Validation Stage ===
        result = await validate_qa_quality(qa, selected_text, question_type)
        if not result.get("is_valid"):
            print(f"QA quality validation failed: {result.get('overall_reason')}")
            if attempt < max_attempts - 1:
                print("Retrying with new question generation...")
                continue
            else:
                print("Max attempts reached, skipping this question type")
                return None
        else:
            print(f"Validation passed: {result.get('overall_reason')}")
        
        # === Question Rephrasing Stage ===
        rephrased_question = await rephrase_question(qa['question_text'], question_type, qa['answer_target'])
        qa['question_text'] = rephrased_question
        
        print(f"Question rephrased: {rephrased_question}")
        
        # If we reach here, we have a valid temporal QA pair
        return qa
    
    return None


# =========================
# Choice Generation
# =========================

async def generate_choices_candidates(qa, selected_event, client):
    answer_target = qa["answer_target"]
    question_text = qa["question_text"]

    # System prompt for choice generation (9 distractors + 1 answer)
    system_prompt = """
        You are a QA choice generator. Your task is to generate **10** plausible answer choices for a multiple-choice question.
        The first choice will be the correct answer. The other 9 choices should be incorrect but plausible.
        Ensure that all choices are at the same granularity level as the correct answer.
        For example:
        - If the correct answer is a country (e.g., 'United States'), generate other countries.
        - If the correct answer is a person (e.g., 'Barack Obama'), generate other persons' names.
        - If the correct answer is a location (e.g., 'Paris'), generate other cities or locations.
        
        The choices should be relevant and similar in scope to the correct answer.

        Return the result as a JSON object with the following format:
        {
            "choices": [list of 10 choices, first is correct answer],
            "question": "...",
            "answer": "..."
        }
    """

    user_prompt = f"""
        Question: {question_text}
        Correct Answer: {answer_target}
        
        Generate 10 candidate answer choices, where:
        - The first is the correct answer.
        - The next 9 should be plausible, but incorrect.
    """
    
    try:
        completion = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
    
        # Parse the response and extract the 10 choices
        raw_response = completion.choices[0].message.content.strip()

        # JSON으로 파싱
        model_data = json.loads(raw_response)
        choices = model_data.get("choices", [])
        answer = model_data.get("answer", "")

        
        # 정답이 첫 번째 선택지로 들어가도록 설정
        if answer != choices[0]:
            choices = [answer] + choices[:9]  # 정답을 첫 번째로 배치
        
        return {
            "question": question_text,
            "choices": choices,
            "answer": answer_target
        }

    except Exception as e:
        print(f"Error generating choices candidates: {e}")
        return None



async def select_choices(candidate_choices, client):
    question_text = candidate_choices["question"]
    choices = candidate_choices["choices"]
    answer_target = choices[0]  # answer is the first element of choices

    system_prompt = """
    You are a QA validation assistant. 
    Given a question and a list of 10 candidate answer choices, your task is to select the 4 most relevant choices.
    The first choice will be the correct answer.
    
    Validation Criteria:
    - Contextual Plausibility: The choices must be plausible and relevant to the question context.
    - Specificity Consistency: The choices must match the specificity level of the correct answer.
    - Mutual Exclusivity: The selected choices must be mutually exclusive, with no redundancy or semantic overlap.

    Your task is to select the best 4 choices based on these criteria and return them, ensuring that the correct answer is always included.
    """
    
    user_prompt = f"""
    Question: {question_text}
    Choices: {choices}
    Answer target : {answer_target}
    
    Selct 4 choices from choices list including answer_target.
    Ensure that the correct answer is always included in the selection.
    Return a JSON object with the final selected 4 choices and the answer target.
    """

    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            # model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            #response_format="json_object",
            max_tokens=300,
            temperature=0.7
        )
        
        # Parse the response and extract the selected choices
        raw_response = completion.choices[0].message.content.strip()
        # print(f"Model response(select_choices): {raw_response}")  # Just for check
        data = json.loads(raw_response)
        final_choices = data.get("choices") or data.get("selected_choices", [])
        
        if len(final_choices) != 4:
            return None
        
        # Ensure the correct answer is included in the selection
        if answer_target not in final_choices:
            final_choices[0] = answer_target  # Ensure correct answer is included if it's missing

        # Shuffle the choices to randomize order
        random.shuffle(final_choices)

        # Determine the letter for the answer (a-d)
        final_answer_letter = ['a', 'b', 'c', 'd'][final_choices.index(answer_target)]
        
        return {
            "question": question_text,
            "choices": { 
                "a": final_choices[0], 
                "b": final_choices[1], 
                "c": final_choices[2], 
                "d": final_choices[3] 
            },
            "answer": final_answer_letter,
            "candidate_choices": choices  # Return the original candidate choices list as well
        }
    
    except Exception as e:
        print(f"Error selecting choices: {e}")
        return None



# =========================
# Sanity Checks
# =========================

def check_null_redundancy_for_choices(qa_entry):
    """
    Rule-based validation for QA pairs. Returns (is_valid, reasons)
    """
    reasons = []
    is_valid = True

    # Check for Choice Redundancy or Empty Values
    choices = qa_entry.get('choices', {})
    choice_values = list(choices.values())
    if len(set(choice_values)) != 4 or any(not c.strip() for c in choice_values):
        is_valid = False
        reasons.append("Choices are not unique or contain empty values.")

    return is_valid, reasons

async def check_redundancy_for_qa_pairs(qa_pairs, client):
    """
    Check for redundancy in the list of QA pairs within an event entry.
    Checks redundancy separately for each question type (Result Recognition and Entity Tracking).
    Returns a dict with redundancy status for each question type.
    """
    # Initialize result structure
    redundancy_result = {
        'result_recognition': {'is_redundant': False, 'reason': 'No questions of this type'},
        'entity_tracking': {'is_redundant': False, 'reason': 'No questions of this type'}
    }
    
    # Separate questions by type
    result_recognition_questions = []
    entity_tracking_questions = []
    
    for i, qa in enumerate(qa_pairs):
        question_type = qa.get("question_type", "")
        if question_type == "Result Recognition":
            result_recognition_questions.append((i, qa))
        elif question_type == "Entity Tracking":
            entity_tracking_questions.append((i, qa))
    
    # Check redundancy for Result Recognition questions
    if len(result_recognition_questions) == 0:
        redundancy_result['result_recognition'] = {
            'is_redundant': False, 
            'reason': 'No Result Recognition questions found'
        }
    elif len(result_recognition_questions) == 1:
        redundancy_result['result_recognition'] = {
            'is_redundant': False, 
            'reason': 'Only one Result Recognition question - no redundancy possible'
        }
    else:
        # Check redundancy for Result Recognition questions
        rr_redundancy = await _check_qa_type_redundancy(
            result_recognition_questions, client, "Result Recognition"
        )
        redundancy_result['result_recognition'] = rr_redundancy
    
    # Check redundancy for Entity Tracking questions
    if len(entity_tracking_questions) == 0:
        redundancy_result['entity_tracking'] = {
            'is_redundant': False, 
            'reason': 'No Entity Tracking questions found'
        }
    elif len(entity_tracking_questions) == 1:
        redundancy_result['entity_tracking'] = {
            'is_redundant': False, 
            'reason': 'Only one Entity Tracking question - no redundancy possible'
        }
    else:
        # Check redundancy for Entity Tracking questions
        et_redundancy = await _check_qa_type_redundancy(
            entity_tracking_questions, client, "Entity Tracking"
        )
        redundancy_result['entity_tracking'] = et_redundancy
    
    return redundancy_result

async def _check_qa_type_redundancy(questions_of_type, client, question_type):
    """
    Helper function to check redundancy for questions of a specific type.
    """
    system_prompt = f"""
    You are a QA redundancy detection expert. Your task is to identify redundant question-answer pairs within a set of {question_type} questions from the same event.
    
    A QA pair is considered redundant if:
    1. The questions are semantically similar (asking essentially the same thing)
    2. The answer targets are the same or very similar
    3. The questions ask for the same information in different ways
    
    Analyze whether the questions are asking for the same information in different ways.
    
    Return a JSON object with:
    {{
        "is_redundant": true/false,
        "reasons": ["explanation of findings"]
    }}
    """
    
    # Prepare the questions data for analysis
    qa_data_for_analysis = []
    for i, qa in questions_of_type:
        qa_data_for_analysis.append({
            "index": i,
            "question": qa.get("question", ""),
            "answer_target": qa.get("answer_target", ""),
            "question_type": qa.get("question_type", ""),
            "referenced_article": qa.get("referenced_article", "")[:200] + "..." if len(qa.get("referenced_article", "")) > 200 else qa.get("referenced_article", "")
        })
    
    user_prompt = f"""
    Analyze the following {question_type} questions for redundancy. Each question belongs to the same event entry.
    
    Questions:
    {json.dumps(qa_data_for_analysis, indent=2, ensure_ascii=False)}
    
    Return your analysis in the specified JSON format.
    """
    
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0
        )
        
        response = completion.choices[0].message.content.strip()
        result = json.loads(response)
        
        is_redundant = result.get("is_redundant", False)
        reasons = result.get("reasons", [])
        
        if not reasons:
            if is_redundant:
                reasons.append(f"Found redundant {question_type} questions")
            else:
                reasons.append(f"No redundancy detected in {question_type} questions")
        
        return {
            'is_redundant': is_redundant,
            'reason': '; '.join(reasons)
        }
        
    except Exception as e:
        return {
            'is_redundant': False,
            'reason': f'Error during redundancy check: {str(e)}'
        }


async def sanity_check(event_entry, client):
    """
    Integrated sanity check for an event entry.
    Performs:
      1. Per-QA rule-based checks (null / duplicate choices)
      2. Per-QA structural checks (missing fields)
      3. Event-level redundancy check (per question type)

    Returns:
    {
        "is_valid": bool,
        "qa_checks": [
            {
                "index": int,
                "is_valid": bool,
                "reasons": [str, ...]
            }, ...
        ],
        "redundancy_check": {...}
    }
    """
    qa_pairs = event_entry.get("qa_pairs", [])
    qa_checks = []
    overall_valid = True

    # === Per-QA checks ===
    for idx, qa in enumerate(qa_pairs):
        reasons = []
        is_valid = True

        # ---- Structural checks ----
        if not qa.get("question"):
            is_valid = False
            reasons.append("FAILED: Missing question text.")

        if not qa.get("choices") or not isinstance(qa["choices"], dict):
            is_valid = False
            reasons.append("FAILED: Missing or invalid choices dict.")

        if not qa.get("answer"):
            is_valid = False
            reasons.append("FAILED: Missing answer label.")

        # ---- Choice null / redundancy check ----
        null_valid, null_reasons = check_null_redundancy_for_choices(qa)
        if not null_valid:
            is_valid = False
            for r in null_reasons:
                reasons.append(f"FAILED: {r}")

        # ---- Finalize QA result ----
        if not is_valid:
            overall_valid = False

        qa_checks.append({
            "index": idx,
            "is_valid": is_valid,
            "reasons": reasons if reasons else ["PASSED: QA entry passed sanity checks."]
        })

    # === Event-level redundancy check ===
    try:
        redundancy_result = await check_redundancy_for_qa_pairs(qa_pairs, client)
    except Exception as e:
        redundancy_result = {
            "result_recognition": {
                "is_redundant": False,
                "reason": f"Error during redundancy check: {str(e)}"
            },
            "entity_tracking": {
                "is_redundant": False,
                "reason": f"Error during redundancy check: {str(e)}"
            }
        }

    # If any redundancy detected, mark overall invalid
    if redundancy_result.get("result_recognition", {}).get("is_redundant") or \
       redundancy_result.get("entity_tracking", {}).get("is_redundant"):
        overall_valid = False

    return {
        "is_valid": overall_valid,
        "qa_checks": qa_checks,
        "redundancy_check": redundancy_result
    }
        


async def generate_multichoice_qa(events):
    """Generate QA pairs for each article in each event with structured data"""
    event_qa_data = []
    
    for event in events:
        try:
            # === Structured Field Sanity Check ===
            has_people = bool(event.get("People"))
            has_location = bool(event.get("Location"))
            has_result = bool(event.get("Result"))
            articles = event.get('event_text', [])

            if not (has_people or has_location) or not has_result or not articles:
                print("Skipping event due to insufficient structured fields or no articles.")
                continue

            # === Prepare Structured Data ===
            structured_data = {
                "People": event.get("People", []),
                "Location": event.get("Location", []),
                "Result": event.get("Result", []),
                "Relations": event.get("Relations", []),
                "Event Attributes": event.get("Event Attributes", {})
            }

            # === Generate QA Pairs for Each Article ===
            qa_pairs = []
            
            max_articles_per_event = 2 
            articles = random.sample(articles, max_articles_per_event) if len(articles) > max_articles_per_event else articles

            for article in articles:
                try:
                    # Create a modified event with the specific article as context
                    article_event = event.copy()
                    article_event['event_text'] = [article]

                    qa_pairs_for_article = []
                    for question_type in ["Result Recognition", "Entity Tracking"]:
                        # === QA Generation with Validation ===
                        qa = await generate_qa_with_quality_validation(article_event, question_type, max_attempts=3)
                        if not qa:
                            print(f"Skipping {question_type}: No valid QA pair found after max attempts.")
                            continue

                        # === Generate Candidate Choices ===
                        candidate_choices = await generate_choices_candidates(qa, event, client)
                        if not candidate_choices:
                            print(f"Skipping article: Candidate choices generation failed.")
                            continue

                        # === Select 4 Choices ===
                        final_choices = await select_choices(candidate_choices, client)
                        if not final_choices:
                            print(f"Skipping article: Final choice selection failed.")
                            continue

                        print(f'Selected Choices:\n a: {final_choices["choices"]["a"]},\n b: {final_choices["choices"]["b"]},\n c: {final_choices["choices"]["c"]},\n d: {final_choices["choices"]["d"]}')

                        # === Final QA Format with Context ===
                        qa_entry = {
                            "referenced_article": article,
                            "question_type": qa["question_type"],
                            "question": final_choices["question"],
                            "candidate_choices": final_choices['candidate_choices'],
                            "choices": final_choices["choices"],
                            "answer": final_choices["answer"],
                        }

                        qa_pairs_for_article.append(qa_entry)

                    qa_pairs.extend(qa_pairs_for_article)
                    
                except Exception as e:
                    print(f"Error generating QA for article: {e}")
                    continue

            # === Create Event Entry ===
            if qa_pairs:
                event_entry = {
                    "event_date": event.get("event_date"),
                    "event_sum": event.get("event_sum", ""),
                    "event_text": articles,
                    "structured_data": structured_data,
                    "qa_pairs": qa_pairs
                }
                
                # === Sanity Checks ===
                sanity_check_result = await sanity_check(event_entry, client)
                event_entry["sanity_check"] = sanity_check_result
                
                event_qa_data.append(event_entry)
                print(f"Generated {len(qa_pairs)} QA pairs for event on {event.get('event_date')}\n\n")
    
        except Exception as e:
            print(f"Error processing event: {e}")
            continue
        
    return event_qa_data    


'''
** Save QA pairs to file **
    - (async) save_qa_pairs(event_qa_data, story_num)
'''
async def save_qa_pairs(event_qa_data, story_num):
    """Save event QA data to a JSON file"""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), 'data_qa')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to file with story number
        output_file = os.path.join(output_dir, f'temporal_qa_{story_num}.json')
        
        # If there is an existing file, make a backup
        if os.path.exists(output_file):
            backup_file = os.path.join(output_dir, f'temporal_qa_{story_num}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            os.rename(output_file, backup_file)
            print(f"Backed up existing file to {backup_file}")
        
        # Calculate total QA pairs
        total_qa_pairs = sum(len(event['qa_pairs']) for event in event_qa_data)
        
        # Save new event QA data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'events': event_qa_data,
                'metadata': {
                    'story_num': story_num,
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_pairs': total_qa_pairs
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nSuccessfully saved {len(event_qa_data)} events with {total_qa_pairs} total QA pairs to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error in save_qa_pairs: {e}")
        return False



''' main '''
async def main():
    """Main function to run the QA generation process"""
    try:
        # Load processed events
        dataset = {
            "2016": [25, 218, 227],
            "2025": [1, 2, 3]
        }
        for year in dataset.keys():
            for story_num in dataset[year]:
                print(f"\nProcessing Story #{story_num}")
                
                # Load events for this story
                events = await load_sampled_events(f'processed_events_{story_num}_multistage.json', year)
                if events is None:
                    print(f"Skipping Story #{story_num} due to loading error")
                    continue
                
                # Generate QA pairs
                print("\nGenerating QA pairs...")
                event_qa_data = await generate_multichoice_qa(events)
                
                # Print statistics
                print(f"\nQA Generation Statistics for Story #{story_num}:")
                print(f"Total events processed: {len(event_qa_data) if event_qa_data else 0}")
                total_qa_pairs = sum(len(event['qa_pairs']) for event in event_qa_data) if event_qa_data else 0
                print(f"Total QA pairs generated: {total_qa_pairs}")

                # Save all QA pairs with validation info
                if event_qa_data and len(event_qa_data) > 0: 
                    success = await save_qa_pairs(event_qa_data, story_num)
                    if success:
                        print(f"✓ Successfully saved QA pairs for Story #{story_num}")
                    else:
                        print(f"✗ Failed to save QA pairs for Story #{story_num}")
                else:
                    print(f"No QA pairs generated for Story #{story_num}")
    
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())