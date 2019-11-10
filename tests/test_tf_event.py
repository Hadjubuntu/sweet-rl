from sweet.agents.agent_log import AgentLog


log = AgentLog(path='./mylogs/')
for i in range(100):
    log.append(metric='test', value=i/100.0, step=i)