import threading
import queue
import pygame
from swarm import Swarm, Agent
from simulation_manager import SimulationManager

class BluenetSystem:
    def __init__(self):
        self.client = Swarm()
        self.sim_manager = SimulationManager()
        self.message_queue = queue.Queue()
        self.running = True
        self.messages = []
        
        # 에이전트 생성
        self.commander_agent = self.create_agent("Commander")
        self.observer_agent = self.create_agent("Observer")
        self.hunter_agent = self.create_agent("Hunter")
        self.cleaner_agent = self.create_agent("Cleaner")

    def create_agent(self, role):
        """에이전트 생성 메서드"""
        if role == "Commander":
            return Agent(
                role="Commander",
                goal="Coordinate and manage other agents effectively",
                backstory="I am the Commander, responsible for coordinating the team's actions."
            )
        elif role == "Observer":
            return Agent(
                role="Observer",
                goal="Gather intelligence and monitor situations",
                backstory="I am the Observer, specialized in surveillance and intelligence gathering."
            )
        elif role == "Hunter":
            return Agent(
                role="Hunter",
                goal="Track and capture targets",
                backstory="I am the Hunter, focused on pursuit and capture operations."
            )
        elif role == "Cleaner":
            return Agent(
                role="Cleaner",
                goal="Support and maintain operational integrity",
                backstory="I am the Cleaner, providing support and maintaining team effectiveness."
            )
        else:
            raise ValueError(f"Unknown agent role: {role}")

    def input_handler(self):
        while self.running:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    self.running = False
                    break
                self.message_queue.put(user_input)
            except EOFError:
                self.running = False
                break

    def run(self):
        input_thread = threading.Thread(target=self.input_handler)
        input_thread.daemon = True
        input_thread.start()

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            try:
                while not self.message_queue.empty():
                    user_input = self.message_queue.get_nowait()
                    self.messages.append({"role": "user", "content": user_input})
                    
                    # Observer의 환경 정보 수집
                    environment_info = self.sim_manager.get_environment_info()
                    
                    # Commander가 명령을 해석
                    if "find" in user_input.lower() or "search" in user_input.lower():
                        print("\nCommander: Understood. Deploying Observer to locate the target.")
                        print("\n→ Transferring to Observer...")
                        
                        # Observer에게 환경 정보 전달
                        observer_context = {
                            "role": "system",
                            "content": f"""
                            You are now activated to find the yellow circle target.
                            Current environment information:
                            {environment_info}
                            Begin patrol and search operations.
                            """
                        }
                        
                        response = self.client.run(
                            agent=self.observer_agent,
                            messages=[observer_context]
                        )
                        
                        # Observer 활성화 및 순찰 시작
                        self.sim_manager.handle_agent_response({
                            "role": "assistant",
                            "content": "OBSERVER_ACTIVATED_PATROL"
                        })
                        self.sim_manager.handle_agent_response({
                            "role": "assistant",
                            "content": "HUNTER_ACTIVATED_SEEK"
                        })
                        self.sim_manager.handle_agent_response({
                            "role": "assistant",
                            "content": "CLEANER_ACTIVATED_SUPPORT"
                        })
                    
                    elif "catch" in user_input.lower() or "capture" in user_input.lower():
                        print("\nCommander: Understood. Deploying Hunter to capture the target.")
                        print("\n→ Transferring to Hunter...")
                        
                        # Hunter에게 Observer의 정보 전달
                        hunter_context = {
                            "role": "system",
                            "content": f"""
                            You are now activated to capture the yellow circle target.
                            Observer's findings:
                            {environment_info}
                            Begin pursuit operations.
                            """
                        }
                        
                        response = self.client.run(
                            agent=self.hunter_agent,
                            messages=[hunter_context]
                        )
                        
                        self.sim_manager.handle_agent_response({
                            "role": "assistant",
                            "content": "HUNTER_ACTIVATED_SEEK"
                        })
                    
                    else:
                        # 일반적인 명령은 Commander가 처리하되, 현재 환경 정보 포함
                        commander_context = {
                            "role": "system",
                            "content": f"""
                            Current environment state:
                            {environment_info}
                            Process the command and coordinate agents accordingly.
                            """
                        }
                        
                        self.messages.append(commander_context)
                        response = self.client.run(
                            agent=self.commander_agent,
                            messages=self.messages
                        )
                    
                    print(f"\n{response.messages[-1]['content']}")
                    
            except queue.Empty:
                pass

            # 시뮬레이션 지속적 업데이트
            self.sim_manager.update()
            self.sim_manager.update_agents()
            
            clock.tick(60)

        pygame.quit()

def main():
    print("1. Creating Swarm client...\n")
    print("2. Creating Agents...\n")
    print("Commander is ready. Type 'exit' to end the conversation.\n")

    system = BluenetSystem()
    system.run()

if __name__ == "__main__":
    main()
