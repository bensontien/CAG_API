# CAG_API

Development references come from the following two projects:  
1. [hhhuang](https://github.com/hhhuang/CAG/tree/main)
2. [ronantakizawa](https://github.com/ronantakizawa/cacheaugmentedgeneration)

## Project

```
CAG/
├── cagAPI.py
├── config.ini
├── Data
│   └── YOUR_OWN_DATA.txt
├── docker-compose.yaml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Docker Usage

1. Prepare your data in the format of `YOUR_OWN_DATA_0X.txt`. Support multiple files.
2. Set model path and data name in `config.ini`. Recommend to use `llama` or `mistral` model.
3. Prepare `docker-compose.yaml` and `Dockerfile` for your environment.
4. Run `docker-compose -f docker-compose.yaml up` to build and start the container.
5. Enjoy your own cag api.

## cURL Test

```
curl -X POST "{YOUR_OWN_API_IP}/ask" -H "Content-Type: application/json" -d '{"query":"query","cache_name":"YOUR_OWN_DATA"}' 
```