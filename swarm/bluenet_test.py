from swarm import Swarm, Agent

print("1. Creating Swarm client...")
client = Swarm()

def transfer_to_hunter():
    print("\n→ Transferring to Hunter...")
    return hunter_agent

def transfer_to_cleaner():
    print("\n→ Transferring to Cleaner...")
    return cleaner_agent

def transfer_to_observer():
    print("\n→ Transferring to Observer...")
    return observer_agent

def transfer_to_commander():
    print("\n→ Returning to Commander...")
    return commander_agent

print("\n2. Creating Agents...")
commander_agent = Agent(
    name="Commander",
    instructions="""You are the Commander. Your role is to:
    1. Analyze user commands and delegate to appropriate agents
    2. For surveillance/observation tasks -> delegate to Observer
    3. For capture/pursuit tasks -> delegate to Hunter
    4. For obstacle removal/heavy lifting -> delegate to Cleaner
    5. Always explain which agent you're choosing and why
    """,
    functions=[transfer_to_observer, transfer_to_hunter, transfer_to_cleaner],
)

observer_agent = Agent(
    name="Observer",
    instructions="""You are the Observer agent. Your role is:
    1. Handle surveillance and observation tasks
    2. If the task involves capture or heavy lifting, transfer to appropriate agent
    3. Always explain your approach to the observation task
    4. Return to Commander after task explanation
    """,
    functions=[transfer_to_commander, transfer_to_hunter, transfer_to_cleaner],
)

hunter_agent = Agent(
    name="Hunter",
    instructions="""You are the Hunter agent. Your role is:
    1. Handle capture and pursuit tasks
    2. If the task involves observation or heavy lifting, transfer to appropriate agent
    3. Always explain your approach to the hunting task
    4. Return to Commander after task explanation
    """,
    functions=[transfer_to_commander, transfer_to_observer, transfer_to_cleaner],
)

cleaner_agent = Agent(
    name="Cleaner",
    instructions="""You are the Cleaner agent. Your role is:
    1. Handle obstacle removal and heavy lifting tasks
    2. If the task involves observation or capture, transfer to appropriate agent
    3. Always explain your approach to the cleaning/lifting task
    4. Return to Commander after task explanation
    """,
    functions=[transfer_to_commander, transfer_to_observer, transfer_to_hunter],
)

print("\nCommander is ready. Type 'exit' to end the conversation.")
messages = []

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'exit':
        print("Ending conversation...")
        break
    
    messages.append({"role": "user", "content": user_input})
    
    # Get response from current agent
    response = client.run(
        agent=commander_agent,
        messages=messages
    )
    
    print(f"\n{response.messages[-1]['content']}")
    messages = response.messages
