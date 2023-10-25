# API de previsao de temperatura

![GitHub repo size](https://img.shields.io/github/repo-size/iuricode/README-template?style=for-the-badge)
![image](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![image](https://img.shields.io/badge/Kubernetes-326DE6?style=for-the-badge&logo=kubernetes&logoColor=white)
![image](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![image](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

<img src="imgs/imagem.png" alt="API">

> O projeto de API de Ingestao de Dados foi proposto como Checkpoint da Materia de Computacao Neuromorfica, e visa realizar a dockerizacao e Kuberte de uma API que consiga ingerir dados.

### Ajustes e melhorias

O projeto ainda está em desenvolvimento e as próximas atualizações serão voltadas nas seguintes tarefas:

- [x] Criacao da API
- [x] Dockerizacao
- [x] Kubernetes
- [x] Refactoring do Codigo da API
- [x] Organizacao de todo o codigo, afim de se enquadrar nas boas praticas de construcao de APIS Flask

## 💻 Pré-requisitos

Antes de começar, verifique se você atendeu aos seguintes requisitos:

* Você instalou a versão mais recente de `<Python / Docker / Kubernetes (Microk8s)>`
* Você tem uma máquina `<Linux>`. Importante observar que o projeto funciona melhor em maquinas Linux.
* Você instalou as bibliotecas `<Flask / Pandas / Numpy / Sqlite3 / Requests / Datetime / scikit-learn>`.

## 🚀 Docker

Para iniciar a API no docker, siga estas etapas:

Em um ambiente linux, construa a imagem:
```
sudo docker build -t checkpoint2neuro:v1 .
```

Logo apos, teste a imagem com o seguinte comando:
```
sudo docker run checkpoint2neuro:v1
```

Preparando para o kubernete (Importante, verificar o nome da tag inserida no arquivo deployment.yaml)*:
```
sudo docker tag checkpoint2neuro:v1 localhost:32000/checkpoint2neuro:v1
```

Lembre-se de executar o microk8s (microk8s start)
Logo apos, executar e aplicar o comando 

```
sudo microk8s kubectl apply -n staging -f scr_k8s/deployment.yaml
```

Depois disso, executar o push!
```
sudo docker push localhost:32000/checkpoint2neuro:v1
```



## ☕ Kubernete

Para iniciar o microk8s e rodar a API em kubernetes, siga estas etapas:

Aplicar em staging:
```
sudo microk8s kubectl apply -n staging -f scr_k8s/deployment.yaml
```

Pegar o endpoint (Este endpoint sera utilizado posteriormente para as requisicoes*):
```
sudo microk8s kubectl -n staging get endpoints
```

Lembrando que voce deve ter um ambiente docker e kubernete (microk8s) totalmente configurado. Siga as documentacoes de Docker/Kubernetes, para realizar a configuracao correta dos ambientes.

## 📫 Testando com Postman

Para testar a API, siga estas etapas:

1. Inicie o Servidor Flask:

    Certifique-se de que o servidor Flask esteja em execução. Você pode executar o aplicativo Python onde sua API está definida. Se você estiver executando o código do exemplo anterior, execute o arquivo Python em que a aplicação Flask está definida.


2. Abra o Postman:

    Abra o aplicativo Postman em seu computador..


3. Crie uma Nova Requisição: 

    Clique em "New" para criar uma nova requisição.


4. Escolha o Método e a URL:

    Selecione o método da requisição (GET, POST, etc.) e insira a URL correspondente ao endpoint que deseja testar. Por exemplo, se deseja testar a rota de geração de payload, use http://IP_OBTIDO_NO_COMANDO_GET_ENDPOINTS:5000/gerar_payload.


5. Adicione os Dados à Requisição:

    Para uma requisição GET, vá para a aba "Body" no Postman e selecione o formato (geralmente JSON). Insira os dados na seção de corpo da requisição. Por exemplo, para o endpoint /gerar_payload, você pode inserir algo assim no corpo da requisição:
    ```
    {
        
      "date": 2023.0,
      "time": 5.0,
      "local": 1.0,
      "field1": 25.6,
      "field3": 14.0,
      "field4": 22.5,
      "field5": 27.9,
      "field6": 97848.0,
      "field7": 366.0,
      "field8": 415.0
 
    }

    ```


6. Envie a Requisição:

    Clique em "Send" para enviar a requisição.


Você receberá a resposta do servidor Flask no painel de resposta do Postman.

Se você deseja testar o endpoint /return_model_response, siga os mesmos passos, mas use a URL http://IP_OBTIDO_NO_COMANDO_GET_ENDPOINTS:5000/return_model_response e ajuste os parâmetros de consulta na URL conforme necessário. Por exemplo:



## 🤝 Colaboradores

Agradecemos às seguintes pessoas que contribuíram para este projeto:

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/69468384?s=400&u=345cc4cd7eb2af9d149ebdbdfd4b05bb115c17e2&v=4" width="100px;" alt="Foto do Henrico Bela"/><br>
        <sub>
          <b>Henrico Bela</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/104632698?v=4" width="100px;" alt="Foto da Emilly Gabrielly"/><br>
        <sub>
          <b>Emilly Gabrielly</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/101263522?v=4" width="100px;" alt="Foto do Felype Nunes"/><br>
        <sub>
          <b>Felype Nunes</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="imgs/sara.jpg" width="100px;" alt="Foto da Sara Leal"/><br>
        <sub>
          <b>Sara Leal</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="imgs/dani.jpg" width="100px;" alt="Foto do Daniel Faria"/><br>
        <sub>
          <b>Daniel Faria</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## 📝 Licença

Esse projeto está sob licença. Veja o arquivo [LICENÇA](LICENSE.md) para mais detalhes.