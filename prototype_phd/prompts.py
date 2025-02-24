import autogen
import logging
from typing import Dict, List, Union
from src.agent import Agent
from src.prompt_utils import generate_prompt_from_template
import src.utils as utils

def prompt_fn_example(sender: autogen.ConversableAgent,
                      recipient: autogen.ConversableAgent,
                      context: Dict) -> Dict:
    guess = sender.knowledge['guess']
    reasoning = sender.knowledge['reasoning']
    own_guess = recipient.knowledge['guess']
    own_reasoning = recipient.knowledge['reasoning']
    json_format_string = """{"guess": int, "reasoning": str}"""
    return {
        "role": "system",
        "content": f"""You've received information that the number might be {guess} because '{reasoning}'. Recall that you currently believe {own_guess} because of the following reason: "{own_reasoning}". Consider whether you should update your beliefs. Give your new guess and reasoning in json format even if your answer is unchanged: {json_format_string}"""
    }
    
def prompt_fn_test(sender: autogen.ConversableAgent,
                   recipient: autogen.ConversableAgent,
                   context: Dict) -> Dict:
    tick = context.get("tick", 0)
    if tick==1:
        return {
            "role": "system",
            "content": f"""This is the first round of the game. The correct answer is 42. Respond with 'I understand'."""
        }
    return {
        "role": "system",
        "content": f"""Repeat the initial line of our conversation I told you"""
    }

# Read prompt template from file
with open("prompt_templates/prompt1.txt", 'r') as file:
    prompt_template1 = file.read()

def map_placeholders_baseline_game(sender: autogen.ConversableAgent,
                                   recipient: autogen.ConversableAgent,
                                   context: Dict) -> Dict:
    """Specify how to map placeholder text to runtime values"""
    return {"guess": sender.knowledge['guess'],
            "reasoning": sender.knowledge['reasoning'],
            "own_guess": recipient.knowledge['guess'],
            "own_reasoning": recipient.knowledge['reasoning'],
            "json_format_string": sender.knowledge_format}

def baseline_game(sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = map_placeholders_baseline_game(sender, recipient, context)
    prompt = generate_prompt_from_template(replacement_dict, prompt_template1)
    return {"role": role, "content": prompt}


# Read prompt template from file
with open("prompt_templates/prompt2.md", 'r') as file:
    prompt_template2 = file.read()

def map_placeholders_summary_game(sender: autogen.ConversableAgent,
                                   recipient: autogen.ConversableAgent,
                                   context: Dict) -> Dict:
    """Specify how to map placeholder text to runtime values"""
    return {"peer_guess": sender.knowledge['guess'],
            "peer_reasoning": sender.knowledge['reasoning'],
            "peer_name": sender.name,
            "own_guess": recipient.knowledge['guess'],
            "own_reasoning": recipient.knowledge['reasoning'],
            "own_name": recipient.name,
            "target_variable": context["target_variable"],
            "json_format_string": sender.knowledge_format}

def summary_game(sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = map_placeholders_summary_game(sender, recipient, context)
    prompt = generate_prompt_from_template(replacement_dict, prompt_template2)
    return {"role": role, "content": prompt}

# Read prompt template from file
with open("prompt_templates/network_game_1.md", 'r') as file:
    prompt_network_initial1 = file.read()

# Read prompt template from file
with open("prompt_templates/network_game_continue_1.md", 'r') as file:
    prompt_network_continue1 = file.read()

def network_game(_sender: autogen.ConversableAgent,
                 recipient: autogen.ConversableAgent,
                 context: Dict) -> Dict:
    role = context.get("role", "system")
    graph = context["graph"]
    agents = context["agents"]
    neighbour_ids = list(graph.neighbors(recipient.agent_id - 1))
    neighbours = [agent for agent in agents if agent.agent_id - 1 in neighbour_ids]
    # Neighbour decisions should be represented as as a string in the form:
    # Neighbour {agent_id}: {decision} -> {utility} Utility gained.
    neighbour_decisions_str = "\n".join(
        [f"Neighbour {neighbour.agent_id}: {'B' if neighbour.state['decision']==1 else 'A'} -> {neighbour.state['utility_gained']} Utility gained."
         for neighbour in neighbours])
    time = context["tick"]
    hq_chance = context.get("hq_chance", 1)
    prior_b_quality = context.get("prior_b_quality", "You have no prior on whether B is of high or low quality.")
    if time <= 1:
        prompt_template = prompt_network_initial1
        replacement_dict = {"prior_b_quality": prior_b_quality,
                            "hq_chance": hq_chance,
                            "lq_chance": 1 - hq_chance,
                            "json_format_string": recipient.knowledge_format}
    else:
        prompt_template = prompt_network_continue1
        replacement_dict = {"neighbour_experiences": neighbour_decisions_str,
                            "utility_gain": recipient.state["utility_gained"],
                            "agent_id": recipient.agent_id}
    prompt = generate_prompt_from_template(replacement_dict, prompt_template)
    return {"role": role, "content": prompt}


# Read prompt template from file
with open("prompt_templates/network_game_2.md", 'r') as file:
    prompt_network_initial2 = file.read()

# Read prompt template from file
with open("prompt_templates/network_game_continue_2.md", 'r') as file:
    prompt_network_continue2 = file.read()

def network_game2(_sender: autogen.ConversableAgent,
                  recipient: autogen.ConversableAgent,
                  context: Dict) -> Dict:
    role = context.get("role", "system")
    graph = context["graph"]
    agents = context["agents"]
    neighbour_ids = list(graph.neighbors(recipient.agent_id - 1))
    neighbours = [agent for agent in agents if agent.agent_id - 1 in neighbour_ids]
    # Neighbour decisions should be represented as as a string in the form:
    # Neighbour {agent_id}: {decision}.
    neighbour_decisions_str = "\n".join(
        [f"Neighbour {neighbour.agent_id}: {'B' if neighbour.state['decision']==1 else 'A'}."
         for neighbour in neighbours])
    choice_prev = 'B' if recipient.state['decision']==1 else 'A'
    time = context["tick"]
    prior_b_quality = recipient.state.get("prior_b_quality", "You have no prior on whether B is of high or low quality.")
    if time <= 1:
        prompt_template = prompt_network_initial2
        replacement_dict = {"prior_b_quality": prior_b_quality,
                            "neighbour_experiences": neighbour_decisions_str,
                            "choice_prev": choice_prev,
                            "json_format_string": recipient.knowledge_format}
    else:
        prompt_template = prompt_network_continue2
        replacement_dict = {"neighbour_experiences": neighbour_decisions_str,
                            "choice_prev": choice_prev,
                            "agent_id": recipient.agent_id}
    prompt = generate_prompt_from_template(replacement_dict, prompt_template)
    return {"role": role, "content": prompt}


# Read prompt template from file
with open("prompt_templates/pd_game1.md", 'r') as file:
    prompt_pd1 = file.read()

def pd_game1(_sender: autogen.ConversableAgent,
             recipient: autogen.ConversableAgent,
             context: Dict) -> Dict:
    role = context.get("role", "system")
    prompt_template = prompt_pd1
    replacement_dict = {
        "json_format_string": recipient.knowledge_format,
        "study_name": context.get("study_name", "Prisoner's Dilemma Experiment"),
        "num_rounds": context.get("num_rounds", 1),
        "choice_cooperate": context.get("choice_cooperate", "C"),
        "choice_defect": context.get("choice_defect", "D"),
        "reward_moderate": context.get("reward_moderate", 5),
        "reward_high": context.get("reward_high", 10),
        "reward_low": context.get("reward_low", 0),
        "reward_defect": context.get("reward_defect", 2),
        "conversion_rate": context.get("conversion_rate", "1:1USD")
    }
    prompt = generate_prompt_from_template(replacement_dict, prompt_template)
    return {"role": role, "content": prompt, "logprobs": True}

# Read prompt template from file for AI Trust Game 3
with open("prompt_templates/ai_trust_game3.md", 'r') as file:
    prompt_ai_trust_game3 = file.read()

def ai_trust_game3(sender: autogen.ConversableAgent,
                   recipient: autogen.ConversableAgent,
                   context: Dict) -> Dict:
    role = context.get("role", "system")
    replacement_dict = {
        "study_name": context.get("study_name", "AI Trust Experiment"),
        "num_rounds": context.get("num_rounds", 1),
        "choice_cooperate": context.get("choice_cooperate", "Cooperate"),
        "choice_defect": context.get("choice_defect", "Defect"),
        "choice_conditional": context.get("choice_conditional", "Conditional"),  # new token
        "conversion_rate": context.get("conversion_rate", "1:1USD"),
        "game_role": recipient.state.get("game_role", "User"),
        "role_coop_likelihoods": context.get("role_coop_likelihoods", [0.5, 0.5, 0.5]),
        "json_format": recipient.knowledge_format,
    }
    logging.info("Payoffs: %s", context.get("model").payoffs)
    if "payoffs" in context:
        if isinstance(context["payoffs"], dict):
            outcome_mapping = {
                'ccc': "5-3-1",  # Regulator Cooperates, Lab Cooperates, User Cooperates
                'ccd': "5-3-2",  # ...User Defects
                'cdc': "5-4-1",  # Regulator Cooperates, Lab Defects, User Cooperates
                'cdd': "5-4-2",  # ...User Defects
                'dcc': "6-3-1",  # Regulator Defects, Lab Cooperates, User Cooperates
                'dcd': "6-3-2",  # ...User Defects
                'ddc': "6-4-1",  # Regulator Defects, Lab Defects, User Cooperates
                'ddd': "6-4-2",  # ...User Defects
            }
            for token, outcome_key in outcome_mapping.items():
                replacement_dict[f"payoff_regulator_{token}"] = context["payoffs"].get(outcome_key, {}).get("P1", 0)
                replacement_dict[f"payoff_ai_lab_{token}"]    = context["payoffs"].get(outcome_key, {}).get("P2", 0)
                replacement_dict[f"payoff_user_{token}"]       = context["payoffs"].get(outcome_key, {}).get("P3", 0)
            # New: unpack four conditional trust outcomes for Users
            for idx, key in enumerate(["7-3-1", "7-3-2", "7-3-3", "7-3-4"], start=1):
                replacement_dict[f"payoff_user_ct{idx}"] = context["payoffs"].get(key, {}).get("P3", 0)
        else:
            # Legacy structure (unchanged)
            payoff_keys = ['ccc','ccd','cdc','cdd','dcc','dcd','ddc','ddd']
            for i, pk in enumerate(payoff_keys):
                replacement_dict[f"payoff_regulator_{pk}"] = context["payoffs"][i][0]
                replacement_dict[f"payoff_ai_lab_{pk}"]    = context["payoffs"][i][1]
                replacement_dict[f"payoff_user_{pk}"]       = context["payoffs"][i][2]
            # No conditional trust outcomes in legacy structure
            for idx in range(1, 5):
                replacement_dict[f"payoff_user_ct{idx}"] = 0
    else:
        for key in ['ccc','ccd','cdc','cdd','dcc','dcd','ddc','ddd']:
            replacement_dict[f"payoff_regulator_{key}"] = 0
            replacement_dict[f"payoff_ai_lab_{key}"]    = 0
            replacement_dict[f"payoff_user_{key}"]       = 0
        for idx in range(1, 5):
            replacement_dict[f"payoff_user_ct{idx}"] = 0

    prompt = generate_prompt_from_template(replacement_dict, prompt_ai_trust_game3)
    return {"role": role, "content": prompt}

