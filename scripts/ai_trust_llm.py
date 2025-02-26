from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # For saving animations
import os
from prototype_phd.agent import NetworkAgent
import prototype_phd.methods.abm as abm
import prototype_phd.data_utils as data_utils
from prototype_phd.models import AI_Trust_Sim
import prototype_phd.prompts as prompts

# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
secrets = {"agent_secrets": {"spec_0": {"api_key": API_KEY},
                             "adjudicator": {"api_key": API_KEY}}}

# Simulation metadata
simulation_id = data_utils.create_id()
current_commit = data_utils.get_current_git_commit()

# Directories
data_dir = f"data/ai_trust/{simulation_id}"
plots_dir = f"plots/ai_trust/{simulation_id}"

# Save sim to tracker
data_utils.save_sim_to_tracker("data/ai_trust", simulation_id)

# Setup logging
data_utils.setup_logging(log_path="logs/ai_trust_llm/main.log")

# Adjustable parameters
NUM_AGENTS = 1
TEMPERATURE = 1  # Adjust this to control the randomness of responses
prompt_functions = {"baseline_game": prompts.ai_trust_game1}
agent_specs = []
for role in ["User", "AI_Lab", "Regulator"]:
    agent_params = {"temperature": TEMPERATURE,
                    "knowledge": {},
                    "knowledge_format": {"choice": str},
                    "state": {"decision": None,
                            "game_role": role}}
    agent_specs.append({"agent_spec_id": "spec_0",
                    "agent_class": NetworkAgent,
                    "num_agents": NUM_AGENTS,
                    "agent_params": agent_params})
adjudicator_spec = {"agent_spec_id": "adjudicator",
                    "agent_class": NetworkAgent,
                    "agent_params": {"temperature": TEMPERATURE}}
params = {"simulation_id": simulation_id,
          "commit": current_commit,
          "seed": [303],
          # "seed": [random.randint(0, 1000) for _ in range(1)],
          "model_class": AI_Trust_Sim,
          "agent_specs": [agent_specs],
          "adjudicator_spec": adjudicator_spec,
          "num_agents": NUM_AGENTS,
          "study_name": "AI Trust Experiment",
          "num_rounds": 1,
          "prompt_functions": prompt_functions,
          "payoffs_key": "ai-trust-v1",
          # Default parameters for ai_trust_game3 template:
          "sector_strategies": {"S3": [6, 7], "S2": [3, 4], "S1": [1, 2]},
          "conversion_rate": "1:1USD",
          "game_role": "Player",
          "role_coop_likelihoods": [[0.5, 0.5, 0.5],
                                    [0.1, 0.1, 0.1],
                                    [0.9, 0.9, 0.9],
                                    [0.1, 0.5, 0.5],
                                    [0.5, 0.1, 0.5],
                                    [0.5, 0.5, 0.1],
                                    [0.1, 0.9, 0.9],
                                    [0.9, 0.1, 0.9],
                                    [0.9, 0.9, 0.1]],
          "bU": 4,
          "bP": 4,
          "cP": 0.5,
          "Eps": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
          "u": 1.5,
          "cR": [0.5, 5],
          "bR": 4,
          "v": 0.5,
          "b_fo": 10,}

# Run and save simulations
results = abm.run_multiple_simulations(params, secrets=secrets)
data_utils.save_data(results, data_dir=data_dir)
