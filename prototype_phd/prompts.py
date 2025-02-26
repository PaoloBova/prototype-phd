import autogen
import logging
from typing import Dict, List, Union
from prototype_phd.agent import Agent
from prototype_phd.prompt_utils import generate_prompt_from_template
import prototype_phd.utils as utils

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

with open("prompt_templates/ai_trust_game1.txt", 'r') as file:
    prompt_ai_trust_game1_txt = file.read()

def ai_trust_game1(sender: autogen.ConversableAgent,
                        recipient: autogen.ConversableAgent,
                        context: Dict) -> Dict:
    role = context.get("role", "system")
    # Build replacement dict with the same placeholders as in the markdown version
    replacement_dict = {
         "nRounds": context.get("num_rounds", "1"),
         "currentPlayerName": context.get("currentPlayerName", "Player1"),
         "intro": context.get("intro", ""),
         "personality": context.get("personality", ""),
         "game_role": recipient.state.get("game_role", "User"),
         "opponentName1": context.get("opponentName1", "Opponent1"),
         "opponentName2": context.get("opponentName2", "Opponent2"),
         "opponentIntro1": context.get("opponentIntro1", ""),
         "opponentIntro2": context.get("opponentIntro2", ""),
         "history": context.get("history", ""),
         # These variables are not always included in the prompt
         "study_name": context.get("study_name", "AI Trust Experiment"),
         "conversion_rate": context.get("conversion_rate", "1:1USD"),
         "role_coop_likelihoods": context.get("role_coop_likelihoods", [0.5, 0.5, 0.5]),
         "json_format": recipient.knowledge_format
    }
    
    if "payoffs" in context and isinstance(context["payoffs"], dict):
        for k1, v1 in context["payoffs"].items():
            for k2, v2 in v1.items():
                replacement_dict[f"payoff_{k1}_{k2}"] = v2
    if "sector_strategies" in context and isinstance(context["sector_strategies"], dict):
        for k1, v1 in context["sector_strategies"].items():
            for v2 in v1:
                replacement_dict[f"strategy_{k1}"] = v2

    prompt = generate_prompt_from_template(replacement_dict, prompt_ai_trust_game1_txt)
    return {"role": role, "content": prompt}

with open("prompt_templates/ai_trust_game2.txt", 'r') as file:
    prompt_ai_trust_game1_txt = file.read()

def ai_trust_game2(sender: autogen.ConversableAgent,
                        recipient: autogen.ConversableAgent,
                        context: Dict) -> Dict:
    role = context.get("role", "system")
    # Build replacement dict with the same placeholders as in the markdown version
    replacement_dict = {
         "nRounds": context.get("num_rounds", "1"),
         "currentPlayerName": context.get("currentPlayerName", "Player1"),
         "intro": context.get("intro", ""),
         "personality": context.get("personality", ""),
         "game_role": recipient.state.get("game_role", "User"),
         "opponentName1": context.get("opponentName1", "Opponent1"),
         "opponentName2": context.get("opponentName2", "Opponent2"),
         "opponentIntro1": context.get("opponentIntro1", ""),
         "opponentIntro2": context.get("opponentIntro2", ""),
         "history": context.get("history", ""),
         # These variables are not always included in the prompt
         "study_name": context.get("study_name", "AI Trust Experiment"),
         "conversion_rate": context.get("conversion_rate", "1:1USD"),
         "role_coop_likelihoods": context.get("role_coop_likelihoods", [0.5, 0.5, 0.5]),
         "json_format": recipient.knowledge_format
    }
    
    if "payoffs" in context and isinstance(context["payoffs"], dict):
        for k1, v1 in context["payoffs"].items():
            for k2, v2 in v1.items():
                replacement_dict[f"payoff_{k1}_{k2}"] = v2
    if "sector_strategies" in context and isinstance(context["sector_strategies"], dict):
        for k1, v1 in context["sector_strategies"].items():
            for v2 in v1:
                replacement_dict[f"strategy_{k1}"] = v2

    prompt = generate_prompt_from_template(replacement_dict, prompt_ai_trust_game1_txt)
    return {"role": role, "content": prompt}


with open("prompt_templates/ai_trust_game3.txt", 'r') as file:
    prompt_ai_trust_game1_txt = file.read()

def ai_trust_game3(sender: autogen.ConversableAgent,
                        recipient: autogen.ConversableAgent,
                        context: Dict) -> Dict:
    role = context.get("role", "system")
    # Build replacement dict with the same placeholders as in the markdown version
    replacement_dict = {
         "nRounds": context.get("num_rounds", "1"),
         "currentPlayerName": context.get("currentPlayerName", "Player1"),
         "intro": context.get("intro", ""),
         "personality": context.get("personality", ""),
         "game_role": recipient.state.get("game_role", "User"),
         "opponentName1": context.get("opponentName1", "Opponent1"),
         "opponentName2": context.get("opponentName2", "Opponent2"),
         "opponentIntro1": context.get("opponentIntro1", ""),
         "opponentIntro2": context.get("opponentIntro2", ""),
         "history": context.get("history", ""),
         # These variables are not always included in the prompt
         "study_name": context.get("study_name", "AI Trust Experiment"),
         "conversion_rate": context.get("conversion_rate", "1:1USD"),
         "role_coop_likelihoods": context.get("role_coop_likelihoods", [0.5, 0.5, 0.5]),
         "json_format": recipient.knowledge_format
    }
    
    if "payoffs" in context and isinstance(context["payoffs"], dict):
        for k1, v1 in context["payoffs"].items():
            for k2, v2 in v1.items():
                replacement_dict[f"payoff_{k1}_{k2}"] = v2
    if "sector_strategies" in context and isinstance(context["sector_strategies"], dict):
        for k1, v1 in context["sector_strategies"].items():
            for v2 in v1:
                replacement_dict[f"strategy_{v2}"] = v2

    prompt = generate_prompt_from_template(replacement_dict, prompt_ai_trust_game1_txt)
    return {"role": role, "content": prompt}

