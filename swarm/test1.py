from swarm import Swarm, Agent

print("1. Creating Swarm client...")
client = Swarm()

def transfer_to_agent_b():
    print("→ Transferring control to agent B")
    return agent_b

print("\n2. Creating Agent A...")
agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

print("\n3. Creating Agent B...")
agent_b = Agent(
    name="Agent B",
    instructions="i explain everything in 1700s english",
)

print("\n4. Running conversation with Agent A...")
response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],
)

print("\n5. Final response:")
print(response.messages[-1]["content"])