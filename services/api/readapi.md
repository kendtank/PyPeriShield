### 注意
与java后端和前端客户端的数据交互api, 全部由单独的api_server服务容器去解耦，
用户设置的参数和摄像头参数全部按照摄像头id存放在redis中，
所有生产容器统一去redis服务器获取，避免参数不一致和复杂的心跳线程。
### 这里不在提供与java后端的api，统一又api_server容器来负责