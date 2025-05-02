from llm import Configuration, APIModel
from utils import function_to_json

class BaseAgent:
    def __init__(self, config: Configuration):
        self.config = config
        self.model = APIModel(config)
    
    def setting_role(self, role:str, description:list):
        """
        设置角色和描述
        :param role: 角色名称
        :param description: 角色描述
        """
        self.role = role
        self.description = '\n'.join(description)

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt