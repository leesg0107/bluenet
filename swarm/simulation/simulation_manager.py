import pygame
import random
import numpy as np

class SimulationManager:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Swarm Agent Simulation")
        
        # 폰트 초기화
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 12)
        
        # 에이전트 위치 및 상태
        self.agents = {
            "Commander": {"pos": [400, 300], "color": (255, 255, 255), "score": 0},
            "Observer": {"pos": [300, 200], "color": (0, 255, 0), "score": 0},
            "Hunter": {"pos": [500, 200], "color": (255, 0, 0), "score": 0},
            "Cleaner": {"pos": [400, 400], "color": (0, 0, 255), "score": 0}
        }
        
        # 리워드 객체 초기화
        self.rewards = []
        self.spawn_reward()  # 초기 리워드 생성
        
        # 에이전트별 움직임 상태 추가
        self.agent_states = {
            "Commander": {"velocity": [0, 0], "target": None},
            "Observer": {"velocity": [0, 0], "target": None},
            "Hunter": {"velocity": [0, 0], "target": None},
            "Cleaner": {"velocity": [0, 0], "target": None}
        }
        
        # 에이전트별 현재 임무 상태 추가
        self.agent_missions = {
            "Commander": None,
            "Observer": None,
            "Hunter": None,
            "Cleaner": None
        }
        
        # 관찰 영역 및 장애물 추가
        self.observation_radius = 100  # Observer의 관찰 반경
        self.obstacles = []  # 장애물 리스트
        self.spawn_obstacles(3)  # 초기 장애물 생성
        
        # 에이전트 간 통신을 위한 메시지 큐
        self.agent_messages = {
            "Hunter": [],
            "Cleaner": []
        }

    def get_environment_info(self):
        """현재 환경 상태 정보를 수집하여 반환"""
        environment_info = {
            "agents": {},
            "rewards": []
        }
        
        # 에이전트 정보 수집
        for agent_name, agent_data in self.agents.items():
            environment_info["agents"][agent_name] = {
                "position": agent_data["pos"],
                "color": agent_data["color"],
                "score": agent_data["score"],
                "status": self.agent_missions.get(agent_name, "inactive")
            }
        
        # 리워드(타겟) 정보 수집
        for reward in self.rewards:
            environment_info["rewards"].append({
                "position": reward["pos"],
                "value": reward["value"],
                "color": "yellow"  # 노란색 원 타겟
            })
        
        # 환경 정보를 문자열로 변환
        info_str = (
            f"Environment Status:\n"
            f"- Agents: {len(self.agents)} active\n"
            f"- Rewards: {len(self.rewards)} available\n"
            f"- Target Description: Yellow circle(s) at position(s): "
            f"{[reward['pos'] for reward in self.rewards]}\n"
        )
        
        return info_str

    def handle_agent_response(self, message):
        """에이전트의 응답을 처리하고 시뮬레이션 상태를 업데이트"""
        content = message["content"]
        
        # 특별 명령어 처리
        if content == "OBSERVER_ACTIVATED_PATROL":
            self.agent_missions["Observer"] = "patrol"
            print("DEBUG: Observer activated for patrol")
            return
            
        if content == "HUNTER_ACTIVATED_SEEK":
            self.agent_missions["Hunter"] = "seek_reward"
            print("DEBUG: Hunter activated for seeking target")
            return
            
        if content == "CLEANER_ACTIVATED_SUPPORT":
            self.agent_missions["Cleaner"] = "handle_obstacle"
            print("DEBUG: Cleaner activated for handling obstacles")
            return

        # 일반 메시지 처리
        content = content.lower()
        print("DEBUG: Analyzing message:", content)
        
        # 에이전트 전환 감지
        if "transferring to" in content:
            if "observer" in content:
                print("DEBUG: Activating Observer")
                self.agent_missions["Observer"] = "patrol"
                
            elif "hunter" in content:
                print("DEBUG: Activating Hunter")
                self.agent_missions["Hunter"] = "seek_reward"
                
            elif "cleaner" in content:
                print("DEBUG: Activating Cleaner")
                self.agent_missions["Cleaner"] = "support"
        
        # 명시적인 임무 완료 명령이 있을 때만 비활성화
        elif "mission complete" in content or "mission abort" in content:
            for agent_name in ["Observer", "Hunter", "Cleaner"]:
                if agent_name.lower() in content:
                    print(f"DEBUG: Deactivating {agent_name}")
                    self.agent_missions[agent_name] = None
    
    def update_agents(self):
        """에이전트들의 움직임 업데이트"""
        for agent_name, mission in self.agent_missions.items():
            if mission is None:
                continue
                
            if mission == "patrol" and agent_name == "Observer":
                self.patrol_behavior(agent_name)
                self.check_observation_area()  # 관찰 영역 확인
                
            elif mission == "seek_reward" and agent_name == "Hunter":
                self.seek_target_behavior(agent_name)
                
            elif mission == "handle_obstacle" and agent_name == "Cleaner":
                self.handle_obstacle_behavior(agent_name)

    def patrol_behavior(self, agent_name):
        """Observer의 순찰 행동"""
        agent = self.agents[agent_name]
        state = self.agent_states[agent_name]
        
        # 목표 지점이 없거나 도달했으면 새로운 목표 설정
        if state["target"] is None or self.reached_target(agent["pos"], state["target"]):
            state["target"] = [
                random.randint(50, self.width-50),
                random.randint(50, self.height-50)
            ]
        
        # 목표를 향해 이동
        self.move_towards_target(agent_name, state["target"])

    def seek_target_behavior(self, agent_name):
        """Hunter의 타겟 추적 행동"""
        if self.agent_messages["Hunter"]:
            # Observer가 발견한 타겟 위치로 이동
            target_info = self.agent_messages["Hunter"][0]
            target_pos = target_info["position"]
            
            # 타겟을 향해 이동
            self.move_towards_target(agent_name, target_pos)
            print(f"DEBUG: Hunter moving towards target at {target_pos}")
            
            # 타겟에 도달하면 메시지 제거
            if self.reached_target(self.agents[agent_name]["pos"], target_pos):
                self.agent_messages["Hunter"].pop(0)
                print("DEBUG: Hunter reached target location")

    def handle_obstacle_behavior(self, agent_name):
        """Cleaner의 장애물 처리 행동"""
        if self.agent_messages["Cleaner"]:
            # Observer가 발견한 장애물 위치로 이동
            obstacle_info = self.agent_messages["Cleaner"][0]
            obstacle_pos = obstacle_info["position"]
            
            # 장애물을 향해 이동
            self.move_towards_target(agent_name, obstacle_pos)
            print(f"DEBUG: Cleaner moving towards obstacle at {obstacle_pos}")
            
            # 장애물에 도달하면 처리
            if self.reached_target(self.agents[agent_name]["pos"], obstacle_pos):
                self.remove_obstacle(obstacle_pos)
                self.agent_messages["Cleaner"].pop(0)
                print("DEBUG: Cleaner removed obstacle")

    def remove_obstacle(self, position):
        """장애물 제거"""
        for obstacle in self.obstacles:
            if obstacle["pos"] == position and obstacle["active"]:
                obstacle["active"] = False
                print("DEBUG: Cleaner removed obstacle at", position)

    def move_towards_target(self, agent_name, target, maintain_distance=False):
        """목표를 향해 이동"""
        agent = self.agents[agent_name]
        state = self.agent_states[agent_name]
        
        # 방향 벡터 계산
        dx = target[0] - agent["pos"][0]
        dy = target[1] - agent["pos"][1]
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 0:
            # 속도 정규화 및 적용
            speed = 2.0  # 기본 이동 속도
            if maintain_distance and distance < 100:  # 일정 거리 유지
                speed *= -1
            
            state["velocity"] = [
                dx/distance * speed,
                dy/distance * speed
            ]
            
            # 새로운 위치 계산
            new_x = agent["pos"][0] + state["velocity"][0]
            new_y = agent["pos"][1] + state["velocity"][1]
            
            # 경계 확인
            new_x = max(50, min(self.width-50, new_x))
            new_y = max(50, min(self.height-50, new_y))
            
            agent["pos"] = [new_x, new_y]
    
    def distance(self, pos1, pos2):
        """두 점 사이의 거리 계산"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def reached_target(self, pos, target, threshold=10):
        """목표 지점 도달 여부 확인"""
        return self.distance(pos, target) < threshold

    def spawn_reward(self):
        """새로운 리워드 객체 생성"""
        reward = {
            "pos": [random.randint(50, self.width-50), 
                   random.randint(50, self.height-50)],
            "value": random.randint(10, 50),
            "color": (255, 215, 0)  # 골드 색상
        }
        self.rewards.append(reward)
        
    def spawn_obstacles(self, count):
        """장애물 생성"""
        for _ in range(count):
            obstacle = {
                "pos": [random.randint(50, self.width-50), 
                       random.randint(50, self.height-50)],
                "color": (128, 128, 128),  # 회색
                "active": True
            }
            self.obstacles.append(obstacle)

    def check_observation_area(self):
        """Observer의 관찰 영역 내 대상 확인"""
        observer_pos = self.agents["Observer"]["pos"]
        
        # 리워드(타겟) 확인
        for reward in self.rewards:
            distance = self.distance(observer_pos, reward["pos"])
            if distance <= self.observation_radius:
                # Hunter에게 타겟 위치 전달
                self.agent_messages["Hunter"].append({
                    "type": "target_found",
                    "position": reward["pos"]
                })
                print("DEBUG: Observer found target at", reward["pos"])
        
        # 장애물 확인
        for obstacle in self.obstacles:
            if obstacle["active"]:
                distance = self.distance(observer_pos, obstacle["pos"])
                if distance <= self.observation_radius:
                    # Cleaner에게 장애물 위치 전달
                    self.agent_messages["Cleaner"].append({
                        "type": "obstacle_found",
                        "position": obstacle["pos"]
                    })
                    print("DEBUG: Observer found obstacle at", obstacle["pos"])

    def update(self):
        # 화면 초기화
        self.screen.fill((0, 0, 0))
        
        # Observer의 관찰 영역 표시
        if self.agent_missions.get("Observer") == "patrol":
            observer_pos = self.agents["Observer"]["pos"]
            pygame.draw.circle(self.screen, (50, 50, 50), observer_pos, 
                             self.observation_radius, 1)
        
        # 장애물 그리기
        for obstacle in self.obstacles:
            if obstacle["active"]:
                pygame.draw.rect(self.screen, obstacle["color"], 
                               (obstacle["pos"][0]-10, obstacle["pos"][1]-10, 20, 20))
        
        # 리워드 그리기
        for reward in self.rewards:
            pygame.draw.circle(self.screen, reward["color"], 
                             reward["pos"], 8)
            # 리워드 값 표시
            value_text = self.font.render(str(reward["value"]), True, (255, 255, 255))
            self.screen.blit(value_text, (reward["pos"][0] - 10, reward["pos"][1] - 20))
        
        # 에이전트 그리기
        for agent_name, agent_data in self.agents.items():
            # 에이전트 원 그리기
            pygame.draw.circle(self.screen, agent_data["color"], 
                             agent_data["pos"], 10)
            
            # 에이전트 이름과 점수 표시
            name_text = self.font.render(f"{agent_name}: {agent_data['score']}", 
                                       True, agent_data["color"])
            self.screen.blit(name_text, (agent_data["pos"][0] - 30, 
                                       agent_data["pos"][1] - 25))
            
        # 범 표시 (화면 우측)
        legend_y = 10
        for agent_name, agent_data in self.agents.items():
            legend_text = f"{agent_name}: {agent_data['score']} points"
            text_surface = self.font.render(legend_text, True, agent_data["color"])
            self.screen.blit(text_surface, (self.width - 150, legend_y))
            legend_y += 20
            
        pygame.display.flip()
        
        # 리워드 충돌 체크
        self.check_reward_collisions()

    def check_reward_collisions(self):
        """에이전트와 리워드 간의 충돌 검사"""
        for agent_name, agent_data in self.agents.items():
            for reward in self.rewards[:]:  # [:] 복사본으로 순회
                distance = self.distance(agent_data["pos"], reward["pos"])
                
                if distance < 20:  # 충돌 범위
                    # 점수 추가
                    agent_data["score"] += reward["value"]
                    # 리워드 제거
                    self.rewards.remove(reward)
                    # 새로운 리워드 생성
                    self.spawn_reward()
