class HttpStatus():
    def __init__(self, code, message):
        self.code = code
        self.message = message

    @property
    def metodo_example(self):
        return self.append(f'metodo_example with code {self.code}')

    def metodo_example2(self, message:str):
        return self.__private(f'metodo_example2 with code {self.code} and message {message}')

    @staticmethod
    def append(message:str):
        return message + "/n"

    def __private(self, message:str):
        return "private" + message



    