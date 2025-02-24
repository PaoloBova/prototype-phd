from .model_utils import *
from .types import *
from .utils import *
import prototype_phd.data_utils as data_utils
import prototype_phd.payoffs as payoffs

import numpy as np

valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_reg_market(b:valid_dtypes=4, # benefit: The size of the per round benefit of leading the AI development race, b>0
                c:valid_dtypes=1, # cost: The cost of implementing safety recommendations per round, c>0
                s:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                                "stop":5.1,
                                "step":0.1}, 
                p:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                                "stop":1.02,
                                "step":0.02}, 
                B:valid_dtypes=10**4, # prize: The size of the prize from winning the AI development race, B>>b
                W:valid_dtypes=100, # timeline: The anticipated timeline until the development race has a winner if everyone behaves safely, W ∈ [10, 10**6]
                pfo_l:valid_dtypes=0, # detection_risk_lq: The probability that firms who ignore safety precautions are found out by high quality regulators, pfo_h ∈ [0, 1]
                pfo_h:valid_dtypes=0.5, # detection_risk_hq: The probability that firms who ignore safety precautions are found out by low quality regulators, pfo_h ∈ [0, 1]
                λ:valid_dtypes=0, # disaster_penalty: The penalty levied to regulators in case of a disaster
                r_l:valid_dtypes=0, # profit_lq: profits for low quality regulators before including government incentives, r_l ∈ R
                r_h:valid_dtypes=-1, # profit_hq: profits for high quality regulators before including government incentives, r_h ∈ R
                g:valid_dtypes=1, # government budget allocated to regulators per firm regulated, g > 0
                phi_h:valid_dtypes=0, # regulator_impact: how much do regulators punish unsafe firms they detect, by default detected firms always lose the race.
                phi2_h:valid_dtypes=0, # regulator_impact: how much do regulators punish 2 unsafe firms they detect, by default detected firms always lose the race.
                externality:valid_dtypes=0, # externality: damage caused to society when an AI disaster occurs
                decisiveness:valid_dtypes=1, # How decisive the race is after a regulator gets involved
                incentive_mix:valid_dtypes=1, # Composition of incentives, by default government always pays regulator but rescinds payment if unsafe company discovered
                collective_risk:valid_dtypes=0, # The likelihood that a disaster affects all actors
                β:valid_dtypes=1, # learning_rate: the rate at which players imitate each other
                Z:dict={"S1": 50, "S2": 50}, # population_size: the number of firms and regulators
                strategy_set:list[str]=["HQ-AS", "HQ-AU", "HQ-VS",
                                        "LQ-AS", "LQ-AU", "LQ-VS"], # the set of strategy combinations across all sectors
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeRegMarket` and `ModelTypeEGT`
    """Initialise Regulatory Market models for all combinations of the provided
    parameter valules."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models

valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_ai_trust(
    # b:valid_dtypes=4, # benefit: The size of the per round benefit of leading the AI development race, b>0
                # c:valid_dtypes=1, # cost: The cost of implementing safety recommendations per round, c>0
                # s:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                #                 "stop":5.1,
                #                 "step":0.1}, 
                # p:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                #                 "stop":1.02,
                #                 "step":0.02}, 
                bU:valid_dtypes=4, #benefit users get from trust and adopt the AI system
                bP:valid_dtypes=4, # benefit the creator gets from selling the product;
                cP:valid_dtypes=0.5, # cP is the cost of creating the product;
                Eps:valid_dtypes={"start": -5,
                                  "stop": 0.5,
                                  "step": 0.25}, # fraction of user benefit when creators play D (eps in [-infinity, 1))
                u:valid_dtypes=1.5, # v and u are the cost and impact of institutional punishment (we can also consider reward and hybrid of reward/punishment)
                cR:valid_dtypes=5, # cR – cost of developing rules and enforcement tech
                bR:valid_dtypes=4, #  Funding for regulators (perhaps only being generated when users adopt the technologies);
                v:valid_dtypes=0.5, # v and u are the cost and impact of institutional punishment (we can also consider reward and hybrid of reward/punishment)
                b_fo:valid_dtypes={"start": 0,
                                   "stop": 20,
                                   "step": 0.5}, # reward for regulators to find a creator who plays D
                β:valid_dtypes=0.1, # learning_rate: the rate at which players imitate each other
                Z:dict={"S1": 100, "S2": 100, "S3": 100}, # population_size: the number of firms and regulators
                strategy_set:list[str]=["N-D-D", "N-D-C", "N-C-D", "N-C-C",
                                        "T-D-D", "T-D-C", "T-C-D", "T-C-C"], # the set of strategy combinations across all sectors
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeRegMarket` and `ModelTypeEGT`
    """Initialise Regulatory Market models for all combinations of the provided
    parameter valules."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models

valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_multi_race(
                s_1:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations for layer 1, s>1
                                  "stop":5.5,
                                  "step":0.5}, 
                p_1:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster in layer 1, p ∈ [0, 1]
                                  "stop":1.05,
                                  "step":0.05}, 
                s_2:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations for layer 2, s>1
                                  "stop":5.5,
                                  "step":0.5}, 
                p_2:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster in layer 2, p ∈ [0, 1]
                                  "stop":1.05,
                                  "step":0.05}, 
                B_1=1000, # Per round benefits to layer 1 labs of an AI breakthrough
                B_2=1000, # Per round benefits to layer 2 labs of an AI breakthrough
                p_both=0, # Probability of unsafe labs in any layer avoiding disaster if both layers are unsafe
                W_1=100, # Timeline until AI breakthrough if all labs are safe in layer 1
                W_2=100, # Timeline until AI breakthrough if all labs are safe in layer 2
                gamma_1=0, # New AI Market contestability rate in layer 1
                gamma_2=0, # New AI Market contestability rate in layer 2
                delta_1=0.9, # Discount rate for future benefits of labs in layer 1
                delta_2=0.9, # Discount rate for future benefits of labs in layer 2
                alpha_1=2,  # Spillover factor for layer 1 labs of a breakthrough in layer 2
                alpha_2=2, # Spillover factor for layer 2 labs of a breakthrough in layer 1
                β:valid_dtypes=0.1, # learning_rate: the rate at which players imitate each other
                Z:dict={"S1": 100, "S2": 100, "S3": 100}, # population_size: the number of firms and regulators
                strategy_set:list[str]=["AS-AS", "AS-AU", "AS-S1", "AS-S2",
                                        "AU-AS", "AU-AU", "AU-S1", "AU-S2",
                                        "S1-AS", "S1-AU", "S1-S1", "S1-S2",
                                        "S2-AS", "S2-AU", "S2-S1", "S2-S2"], # the set of strategy combinations across all sectors
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeRegMarket` and `ModelTypeEGT`
    """Initialise Multi Race models for all combinations of the provided
    parameter valules."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models
    
class StudySim:
    """Generic simulation for replicating experimental economics lab experiments using LLMs.
    
    Agents are prompted individually (no inter-agent interactions).
    An optional 'collect_stats_fn' can be provided in params.
    """
    def __init__(self, agents, parameters):
        self.agents = agents
        self.tick = 0
        self.num_rounds = parameters.get("num_rounds", 1)
        self.agent_results = []
        self.model_results = []

    def step(self, parameters):
        self.tick += 1
        for agent in self.agents:
            self.ask_agent(agent, parameters)
            agent.state["decision"] = agent.knowledge.get("choice")
    
    def ask_agent(self, agent, parameters):
        construct_prompt_fn = parameters["prompt_functions"]["baseline_game"]
        # The agent is prompted to reflect on the current state of the simulation
        # and update their knowledge or beahviour accordingly.
        args = {**parameters,
                "tick": self.tick,
                "model": self,
                "agents": self.agents}
        adjudicator = parameters["adjudicator_agent"]
        prompt = construct_prompt_fn(adjudicator, agent, args)
        logging.info(f"Prompting agent {agent.name} with: {prompt}")
        chat_result = adjudicator.initiate_chat(
            recipient=agent, 
            message= prompt,
            max_turns=1,
            clear_history=False,
            silent=True,
        )

        # Extract data from the chat message and update the agent's knowledge.
        message = chat_result.chat_history[-1]["content"]
        logging.info(f"Agent {agent.name} received message: {message}")
        if agent.knowledge_format is None:
            agent.state["decision"] = message
        else:
            data_format = agent.knowledge_format if hasattr(agent, "knowledge_format") else {}
            data = data_utils.extract_data(message, data_format)
            if len(data) >= 1:
                agent.update_knowledge(data[0])
                logging.info(f"Agent {agent.name} updated knowledge to: {agent.knowledge}")
    
    def collect_stats(self, parameters):
        collect_stats_fn = parameters.get("collect_stats_fn",
                                          data_utils.collect_stats_default)
        collect_stats_fn(self, parameters)        

class AI_Trust_Sim(StudySim):
    """This class manages the AI trust experiment simulation.
  
    It is a subclass of StudySim and inherits the step and collect_stats methods.
  
    It is the model of the simulation.
    
    Model properties are initialized when instantiated and the
    `step` function specifies what happens when the model is run.
    """
    def __init__(self, agents, parameters):
        super().__init__(agents, parameters)
        self.payoffs = payoffs.build_payoffs(parameters)["payoffs"]
    