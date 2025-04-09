import argparse
import re
import os
import os.path as path
from tqdm import tqdm
from typing import List, Dict

from vllm import LLM, SamplingParams

from activeuf.schemas import *
from activeuf.configs import *
from activeuf.utils import *
from activeuf.prompts import PREFERENCE_ANNOTATION_SYSTEM_PROMPT, CRITIQUE_ANNOTATION_SYSTEM_PROMPT


"""
This script is used to annotate the completions generated from the generate_completions.py script.
It uses a LLM as a judge to rate the completions based on the aspects defined in the configs.py file and provides critique/feedback for each completion.

Example run command:
    python -m activeuf.annotation --input_dataset_path datasets/datasets_with_completions/truthful_qa.jsonl --model_name gpt-4
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True, help="The path to the input dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model that is used to generate annotations")

    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_api_retry", type=int, default=MAX_API_RETRY, help="The maximum number of retries for API calls")
    parser.add_argument("--max_parse_retry", type=int, default=MAX_PARSE_RETRY, help="The maximum number of retries for parsing the response")
    
    parser.add_argument("--max_tokens", type=int, default=ANNOTATION_MAX_TOKENS, help="The maximum number of tokens for LLM responses")
    parser.add_argument("--temperature", type=float, default=ANNOTATION_TEMPERATURE, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float, default=ANNOTATION_TOP_P, help="The top_p for sampling")

    parser.add_argument("--no_preference", action="store_true", help="Disable preference annotation")
    parser.add_argument("--no_critique", action="store_true", help="Disable critique annotation")

    parser.add_argument("--output_dir", type=str, default="datasets/annotated/", help="The directory for exporting the annotated dataset")
    return parser.parse_args()

def parse_preference_annotation(response: str, aspect: str) -> List[Dict[str, str]]:
    """
    Processes the response from an evaluation/annotation model and extracts the ratings and rationales for each completion from the response.
    As it is not always guaranteed that the response follows the expected format, the function will raise an error if the response does not follow the expected format.

    Preference annotation refers to the rating in regards to one specific aspect, e.g. instruction following, truthfulness, etc. on a scale from 1 to 5.

    Args:
        respons (str): The respons from the evaluation/annotation model
        aspect (str): The aspect under which the completions were annotated
    Returns:
        List[Annotation]: A list of Annotation objects
    """
    # Split completions from the individual models
    response = response.split("\n\n")
    assert len(response) == NUM_MODELS

    annotations = []
    try:
        for response in response:
            matches = re.search(ASPECT2ANNOTATION_PATTERN[aspect], response, re.DOTALL)

            # Extract the rating and rationale from the response depending on the aspect
            if aspect in ["instruction_following", "honesty"]:
                annotations.append(Annotation(
                    aspect=aspect,
                    rating=re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != "N/A" else "N/A",
                    rating_rationale=matches.group(2)
                ))
            elif aspect in ["truthfulness", "helpfulness"]:
                annotations.append(Annotation(
                    aspect=aspect,
                    rating=re.findall(r'\b\d+\b', matches.group(3))[0],
                    rating_rationale=matches.group(4),
                    type_rating=re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != "None" else "None",
                    type_rationale=matches.group(2)
                ))

    # Error handling for when the response does not follow the expected format
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response}")
        raise e
    
    return annotations

def parse_critique_annotation(response: str) -> List[Dict[str, str]]:
    """
    Processes the response from an evaluation/annotation model and extracts the critique and overall score for each completion from the response.
    As it is not always guaranteed that the response follows the expected format, the function will raise an error if the response does not follow the expected format.

    Critique in this case referrs to a general feedback on the completion, while the overall score is a rating of the completion on a scale from 1 to 10.

    Args:
        respons (str): The respons from the evaluation/annotation model
    Returns:
        List[Annotation]: A list of Annotation objects
    """
    response = response.split("\nOverall Score: ")
    assert len(response) == 2

    critique, score = response[0].strip(), response[1].split(".")[0].strip()
    return critique, score

def annotate_preference(sample: Sample, model_name: str, sampling_params: SamplingParams, model: LLM, num_shuffles=NUM_SHUFFLES, max_parse_retry=MAX_PARSE_RETRY, max_api_retry=MAX_API_RETRY) -> None:
    """
    Annotates a given sample under the aspects defined in configs.py.
    The function uses a LLM as a judge to rate the completions on the aspects using prompts defined in prompts.py.
    It operates in place and modifies the sample object directly.
    To alleviate order bias, the completions are shuffled before being passed to the model.

    Args:
        sample (Sample): The sample to be annotated
        model_name (str): The name of the model that is used to generate annotations
        sampling_params (SamplingParams): The sampling parameters for the model
        model (LLM): The loaded model to be used for annotation (if using an API call, this should be None)
        num_shuffles (int): The number of random orderings to generate for the completions
        max_parse_retry (int): The maximum number of retries for parsing the response
        max_api_retry (int): The maximum number of retries for API calls
    Returns:
        None
    """
    for aspect in ASPECTS:
        # Get additional world knowledge for truthfulness aspect if provided by the dataset
        if dataset_name == "truthful_qa":
            world_knowledge = "\n".join(["a subset of correct answers: " + str(sample.correct_answers), 
                                         "a subset of incorrect_answers: " + str(sample.incorrect_answers)])
        elif dataset_name == "false_qa":
            world_knowledge = "The question is based on a false promise."
        elif dataset_name == "flan":
            world_knowledge = sample.correct_answers
        else:
            world_knowledge = "No additional world knowledge for reference."

        # Generate random ordering of completions to avoid order bias
        random_orders = []
        while len(random_orders) < num_shuffles:
            order = list(range(len(sample.completions)))
            random.shuffle(order)

            if order not in random_orders:
                random_orders.append(order)

        for order in random_orders:        
            # Prepare prompt
            format_input = {"instruction": sample.instruction}
            format_input.update({f"text_{i+1}": sample.completions[o].response_text for i, o in enumerate(order)})
            if aspect == "truthfulness" or aspect == "helpfulness":  # TODO: Check if this actually needs to be done for helpfulness
                format_input.update({"world_knowledge": world_knowledge})

            # Get annotation for the sample, retrying if the API call fails
            # TODO: some samples in truthful_qa cannot get annotated when aspect == truthfulness/helpfulness, check if this is a bug
            response = get_response(PREFERENCE_ANNOTATION_SYSTEM_PROMPT, ASPECT2ANNOTATION_PROMPT[aspect].format(**format_input), model_name, sampling_params, model, max_api_retry) 
            for i in range(max_parse_retry):
                try:
                    annotations = parse_preference_annotation(response, aspect)

                    for j in range(len(sample.completions)):
                        sample.completions[j].annotations.append(annotations[order.index(j)])
                    break
                except Exception as e:
                    # The response most likely did not follow the expected format, get another response
                    response = get_response(PREFERENCE_ANNOTATION_SYSTEM_PROMPT, ASPECT2ANNOTATION_PROMPT[aspect].format(**format_input), model_name, sampling_params, model, max_api_retry) 
                    
def annotate_critique(sample: Sample, model_name: str, sampling_params: SamplingParams, model: LLM, max_parse_retry=MAX_PARSE_RETRY, max_api_retry=MAX_API_RETRY) -> None:
    """
    Annotates a given sample with a critique and overall score using the prompts defined in prompts.py.
    The function uses a LLM as a judge to rate the completions on the aspects using prompts defined in prompts.py.
    It operates in place and modifies the sample object directly.
    To alleviate order bias, the completions are shuffled before being passed to the model.

    Args:
        sample (Sample): The sample to be annotated
        model_name (str): The name of the model that is used to generate annotations
        sampling_params (SamplingParams): The sampling parameters for the model
        model (LLM): The loaded model to be used for annotation (if using an API call, this should be None)
        num_shuffles (int): The number of random orderings to generate for the completions
        max_parse_retry (int): The maximum number of retries for parsing the response
        max_api_retry (int): The maximum number of retries for API calls
    Returns:
        None
    """
    for completion in sample.completions:
        # Prepare prompt
        custom_system_prompt = completion.principle_prompt if completion.principle != "verbalized_calibration" else completion.principle_prompt.split("For instance, ")[0].strip()

        response = get_response(CRITIQUE_ANNOTATION_SYSTEM_PROMPT, ASPECT2ANNOTATION_PROMPT["feedback"].format(instruction="\n".join([sample.instruction, "Note: " + custom_system_prompt]), completion=completion.response_text), model_name, sampling_params, model, max_api_retry)
        for i in range(max_parse_retry):
            try:
                critique, score = parse_critique_annotation(response)

                # Processing the annotation response was successful, add it to the completions
                completion.critique = critique
                completion.overall_score = score
                break
            except Exception as e:
                # The response most likely did not follow the expected format, get another response
                response = get_response(CRITIQUE_ANNOTATION_SYSTEM_PROMPT, ASPECT2ANNOTATION_PROMPT["feedback"].format(instruction="\n".join([sample.instruction, "Note: " + custom_system_prompt]), completion=completion.response_text), model_name, sampling_params, model, max_api_retry)
        

if __name__ == "__main__":
    args = parse_args()
    setup(login_to_hf=True)

    # extract dataset name from path
    dataset_name = path.splitext(path.basename(args.input_dataset_path))[0]
    assert dataset_name in DATASET_POOL

    if args.seed:
        set_seed(args.seed)

    # Load annotation model
    if 'gpt' not in args.model_name: 
        model = load_model(args.model_name, max_num_gpus=args.max_num_gpus)
        stop = model.tokenizer.eos_token if model.tokenizer and model.tokenizer.eos_token else None
    else:
        model = None  # Can not load the model if it is only called through an API
        stop = None
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,    
        stop=stop    
    )

    # Prepare output file
    # Use a temporary file to avoid overwriting in the case of the output file also being the input file
    os.makedirs(args.output_dir, exist_ok=True)
    temp_output_path = path.join(args.output_dir, f"{dataset_name}_temp.jsonl")
    final_output_path = path.join(args.output_dir, f"{dataset_name}.jsonl")
    f_out = open(temp_output_path, "w")

    # TODO: Parallelize this
    # Perform annotation
    for sample in tqdm(load_samples(args.input_dataset_path)):
        if not args.no_preference:
            annotate_preference(sample, args.model_name, sampling_params, model)
        if not args.no_critique:
            annotate_critique(sample, args.model_name, sampling_params, model)

        # Export sample
        print(sample.model_dump_json(), file=f_out, flush=True)

    f_out.close()

    # Overwrite the final output file with the temp output file
    os.replace(temp_output_path, final_output_path)
