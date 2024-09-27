from agentMET4FOF.agents import AgentNetwork, SineGeneratorAgent, MonitorAgent
from agentMET4FOF.signal_processing import FFT_Agent


def main():
    # start agent network server
    agentNetwork = AgentNetwork(backend="mesa")
    # init agents
    gen_agent = agentNetwork.add_agent(agentType=SineGeneratorAgent)
    fft_agent = agentNetwork.add_agent(agentType=FFT_Agent, buffer_size=50, s_freq=100)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # connect agents : We can connect multiple agents to any particular agent
    agentNetwork.bind_agents(gen_agent, fft_agent)

    # connect
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(fft_agent, monitor_agent, channel=["plot"])

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()