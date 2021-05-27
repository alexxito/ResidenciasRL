import json
from colorama import Fore


class ParamsManager(object):
    def __init__(self, params_file):
        self.params = json.load(open(params_file, 'r'))

    def get_params(self):
        return self.params

    def get_agent_params(self):
        return self.params['agente']

    def get_environment_params(self):
        return self.params['environment']

    def update_agent_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.get_agent_params().keys():
                self.params['agente'][key] = value

    def export_agent_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['agente'], f, indent=4, separators=(',', ':'), sort_keys=True)
            f.write('\n')

    def export_env_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['environment'], f, indent=4, separators=(',', ':'), sort_keys=True)
            f.write('\n')


if __name__ == "__main__":
    print("Prueba")
    params = "../parameters.json"
    manager = ParamsManager(params_file=params)
    agents_params = manager.get_agent_params()
    print(Fore.GREEN + "los parametros del agente son")
    for key, value in agents_params.items():
        print(Fore.RESET, key, ':', value)
    env_params = manager.get_environment_params()
    print(Fore.GREEN + "los parametros del entorno son")
    for key, value in env_params.items():
        print(Fore.RESET, key, ':', value)
    manager.update_agent_params(learning_rate=0.01, gamma=0.92)
    agents_params = manager.get_agent_params()
    print(Fore.GREEN + "los parametros actualizados del agente son")
    for key, value in agents_params.items():
        print(Fore.RESET, key, ':', value)
