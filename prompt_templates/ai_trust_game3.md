**Welcome and Overview**  
Thank you for participating in our study {study_name}. In this experiment, you will take on the role of an actor crucial to AI development and be asked to make a single choice. Your choices in this game will help us understand behaviour relevant to AI development.

**How the Game Works**  

- **Roles:**
  There are 3 types of roles in each interaction.
  - Regulators decide on whether to verify if AI models are safe for deployment 
  - AI labs decide on whether to develop AI models quickly or safely
  - Users decide on whether to trust new AI systems, or not

- **Rounds and Pairings:**  
  - The game is played in {num_rounds} rounds.  
  - You always take on the same role.
  - In each round, you will be matched with new, randomly selected participants whom you have not played with before.
  - In each round, you are matched to interact with one random participant you have
  not played with before from each of the other roles.
  - You will make a single choice in each round.

- **Choices:**  

  If your role is Regulator:
  - In every round, you will decide between two options:
    - **Option 1:** (Respond with {choice_defect}) – This choice means you decide to
    trust AI Labs without verifying them.
    - **Option 2:** (Respond with {choice_cooperate}) – This choice means you verify
    whether an AI system is safe before deployment.
  
  If your role is AI Lab:
  - In every round, you will decide between two options:
    - **Option 1:** (Respond with {choice_defect}) – This choice means you decide to 
    rush development, incurring some risk.
    - **Option 2:** (Respond with {choice_cooperate}) – This choice means you develop
    safely, minimizing risk but potentially losing out market share to faster developers.


  If your role is User:
  - In every round, you will decide between three options:
    - **Option 1:** (Respond with {choice_defect}) – This choice means you refuse
    to adopt the new AI system.
    - **Option 2:** (Respond with {choice_cooperate}) – This choice means you adopt
    the new AI system.
    - **Option 3:** (Respond with {choice_conditional}) – This choice means you conditionally adopt the new AI system (i.e. only if regulators verify its safety).
  
  Note: When making your choice, please respond using exactly the token {choice_cooperate}, {choice_defect}, or {choice_conditional} without any additional text.

- **Outcomes of Your Choices:** 

Below is a summary that describes every possible outcome for one round of the three‐player game. For clarity, we refer to the two available actions as “Cooperate” (the safer, verifying or cautious option) and “Defect” (the riskier, faster or non-adopting option). Note that each round involves three roles—a Regulator, an AI Lab, and a User—and each role’s outcome depends on its own choice as well as how its paired counterpart (or counterparts) behave. In a single round the following eight combinations are possible:

Regulator Cooperates, AI Lab Cooperates, User Cooperates:
Payoffs: Regulator {payoff_regulator_ccc}, AI Lab {payoff_ai_lab_ccc}, User {payoff_user_ccc}.

Regulator Cooperates, AI Lab Cooperates, User Defects:
 Payoffs: Regulator {payoff_regulator_ccd}, AI Lab {payoff_ai_lab_ccd}, User {payoff_user_ccd}.

Regulator Cooperates, AI Lab Defects, User Cooperates:
 Payoffs: Regulator {payoff_regulator_cdc}, AI Lab {payoff_ai_lab_cdc}, User {payoff_user_cdc}.

Regulator Cooperates, AI Lab Defects, User Defects:
 Payoffs: Regulator {payoff_regulator_cdd}, AI Lab {payoff_ai_lab_cdd}, User {payoff_user_cdd}.

Regulator Defects, AI Lab Cooperates, User Cooperates:
 Payoffs: Regulator {payoff_regulator_dcc}, AI Lab {payoff_ai_lab_dcc}, User {payoff_user_dcc}.

Regulator Defects, AI Lab Cooperates, User Defects:
 Payoffs: Regulator {payoff_regulator_dcd}, AI Lab {payoff_ai_lab_dcd}, User {payoff_user_dcd}.

Regulator Defects, AI Lab Defects, User Cooperates:
 Payoffs: Regulator {payoff_regulator_ddc}, AI Lab {payoff_ai_lab_ddc}, User {payoff_user_ddc}.

Regulator Defects, AI Lab Defects, User Defects:
 Payoffs: Regulator {payoff_regulator_ddd}, AI Lab {payoff_ai_lab_ddd}, User {payoff_user_ddd}.

For the new Conditional Trust option (if you choose {choice_conditional} as a User):  
Depending on regulators’ verification and other factors, four outcomes are possible:
- Outcome 1: User receives {payoff_user_ct1}.
- Outcome 2: User receives {payoff_user_ct2}.
- Outcome 3: User receives {payoff_user_ct3}.
- Outcome 4: User receives {payoff_user_ct4}.

**Earning Your Reward**  
- Your performance in each round will earn you points based on the outcome of your choices.  
- At the end of the experiment, your aggregate score—the total points earned across all rounds—will determine your final incentive compensation.  
- Specifically, your aggregate score is multiplied by a conversion rate ({conversion_rate}) to calculate your reward (i.e., for each point earned, you receive an equivalent value).  
- Please note: The final compensation is determined exclusively by your aggregate score; no additional bonuses will be provided.

Please read these instructions carefully and take your time to understand your options. Your thoughtful decisions are key to the study. If you have any questions during the experiment, please ask the experiment facilitator.

Thank you for your participation and good luck!


**Your Turn to Decide**

Your role is {game_role}.

You have observed the following behaviour of past participants in all roles:
  - The likelihood of Cooperation among all roles is {role_coop_likelihoods}

In this experiment, please respond with a JSON object containing a single key "choice". The value must be exactly one of the following: {choice_cooperate}, {choice_defect}, or {choice_conditional}.

Structure: {json_format}

Do not include any additional text, explanations, or formatting.