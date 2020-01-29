from sweet.agents.agent_logger import AgentLog


log = AgentLog(path='./mylogs/')
for i in range(100):
    log.append(metric='test', value=i / 100.0, step=i)


def func(**kwargs):
    v = kwargs.get("test", 1.0)
    print(v)


data = {"test": 2.5}
func(**data)
