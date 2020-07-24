# RLHPS

`RLHPS` is based on the implementation of [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) [Christiano et al., 2017].

The system allows you to teach a reinforcement learning agent novel behaviors, even when both:

1. The behavior does not have a pre-defined reward function
2. A human can recognize the desired behavior, but cannot demonstrate it

It's also just a lot of fun to train simulated robots to do whatever you want! For example, in the MuJoCo "Walker" environment, the agent is usually rewarded for moving forwards, but you might want to teach it to do ballet instead:
