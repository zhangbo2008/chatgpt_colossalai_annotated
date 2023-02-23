from chatgpt.nn import GPTActor, GPTCritic, RewardModel
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy

strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')

with strategy.model_init_context():
    actor = GPTActor().cuda()
    critic = GPTCritic().cuda()
    initial_model = deepcopy(actor).cuda()
    reward_model = RewardModel(deepcopy(critic.model)).cuda()

trainer = PPOTrainer(strategy, actor, critic, reward_model, initial_model, ...)
trainer.fit(prompts)