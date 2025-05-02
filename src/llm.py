import time
import requests
import json
from tqdm import tqdm
import threading

from dotenv import load_dotenv
import os

class Configuration:

    def __init__(self) -> None:
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")
        self.api_url = os.getenv("LLM_API_URL")
        self.para = self.init_params()
        self.tools = False

    def init_params(self, model_name="Qwen/Qwen2.5-72B-Instruct"):
        params_config = {
            "model": model_name,
            "temperature": 0.7,
        }
        return params_config
    
    def set_tools(self, tools=[]):
        return tools

    @staticmethod
    def load_env():
        load_dotenv()


class APIModel:

    def __init__(self, config:Configuration) -> None:
        self.config = config
        self.__api_key = config.api_key 
        self.__api_url = config.api_url 
        self.model = config.para["model"]

    # 发送请求的内部方法
    def __req(self, system:str="", text:str="", max_try=10):
        url = f"{self.__api_url}/chat/completions"  
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.__api_key}', 
        }
        
        messages = []
        if system != "":
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.para["temperature"],
        }

        if self.config.tools:
            payload["tools"] = self.config.set_tools([])

        if "Qwen3" in self.model:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = 4096
        
        for _ in range(max_try):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()  
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                print(f"请求失败: {e}")
                time.sleep(2)
        return None
    
    # 与模型进行单次交互的接口方法
    def chat(self, system:str="", text:str=""):
        response = self.__req(system, text, max_try=5) 
        return response  

    # 用于多线程请求时的内部方法
    def __chat(self, text:str="", res_l=[], idx=0):
        response = self.__req(text)
        res_l[idx] = response  # 将响应存储在指定索引位置
        return response
        
    # 批量发送消息与模型交互的方法，支持多线程
    def batch_chat(self, text_batch):
        max_threads = 12 
        res_l = ['No response'] * len(text_batch)  
        thread_l = []  # 用于存储线程的列表
        # 遍历消息批次并创建线程
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, res_l, i))  # 创建线程
            thread_l.append(thread)  # 将线程添加到列表中
            thread.start()  # 启动线程
            # 控制最大线程数量，防止超出限制
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t.is_alive():  # 检查线程是否已经结束
                        thread_l.remove(t)  # 如果线程结束，则从列表中移除
                time.sleep(2)  # 短暂延迟以避免忙等待

        # 等待所有线程结束
        for thread in thread_l:
            thread.join()  # 等待线程完成
        return res_l  # 返回所有响应  

if __name__ == "__main__":
    config = Configuration()
    
    llm = APIModel(config)

    # 单次交互示例
    response = llm.chat("你好，你是谁？")
    print(f"单次交互的响应: {response}")
    
    # 批量交互示例
    text_batch = ["1+1=?", "34+3=?", "3+4?"]
    responses = llm.batch_chat(text_batch)
    for i, res in enumerate(responses):
        print(f"批量交互的第 {i+1} 个响应: {res}")