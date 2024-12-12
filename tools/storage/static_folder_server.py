import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import signal
import time
from multiprocessing import Process
import mimetypes
import io


class StaticFileServer:
    def __init__(self, folder_path: str, host: str = "0.0.0.0", port: int = 8000):
        self.folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"The path '{folder_path}' is not a valid directory.")

        self.host = host
        self.port = port
        self.process = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def create_app(self) -> FastAPI:
        app = FastAPI()

        # 启用 Gzip 压缩
        app.add_middleware(GZipMiddleware, minimum_size=1000)

        # 启用 CORS 支持
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有来源，实际使用时应限制为特定域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/{file_path:path}")
        async def serve_path(file_path: str, request: Request):
            """
            提供对文件和文件夹的访问：
            - 如果是文件，则返回文件内容。
            - 如果是文件夹，则返回文件夹内文件的 HTML 列表。
            """
            self.logger.info(f"Handling request for path: {file_path}")

            # 拼接文件或文件夹的完整路径
            full_path = os.path.abspath(os.path.join(self.folder_path, file_path))

            # 确保路径在指定文件夹范围内
            if not os.path.commonpath([full_path, self.folder_path]) == self.folder_path:
                raise HTTPException(status_code=403, detail="Access denied")

            # 路径不存在
            if not os.path.exists(full_path):
                raise HTTPException(status_code=404, detail="File or directory not found")

            # 如果路径是文件，返回文件内容
            if os.path.isfile(full_path):
                return await self.serve_file(full_path, request)

            # 如果路径是文件夹，返回 HTML 格式的文件列表
            if os.path.isdir(full_path):
                return await self.render_directory(full_path, file_path)

        return app

    async def serve_file(self, full_path: str, request: Request) -> FileResponse:
        """处理文件请求，支持 MP4 文件的流式传输和范围请求"""
        mime_type, _ = mimetypes.guess_type(full_path)
        if mime_type is None:
            if full_path.lower().endswith('.mp4'):
                mime_type = 'video/mp4'
            else:
                mime_type = 'application/octet-stream'

        range_header = request.headers.get('Range', None)
        if range_header and mime_type == 'video/mp4':
            return await self.range_file_response(full_path, range_header)
        else:
            return FileResponse(
                full_path,
                headers={
                    "Cache-Control": "public, max-age=86400",  # 缓存一天
                }
            )

    async def range_file_response(self, full_path: str, range_header: str) -> StreamingResponse:
        """处理 MP4 文件的范围请求"""
        file_size = os.path.getsize(full_path)
        start, end = self.parse_range_header(range_header, file_size)
        length = end - start + 1

        headers = {
            'Content-Type': 'video/mp4',
            'Accept-Ranges': 'bytes',
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Content-Length': str(length),
        }

        with open(full_path, 'rb') as file:
            file.seek(start)
            content = file.read(length)

        return StreamingResponse(io.BytesIO(content), status_code=206, headers=headers)

    def parse_range_header(self, range_header: str, file_size: int) -> (int, int):
        """解析 Range 请求头"""
        _, byte_range = range_header.split('=')
        start, end = byte_range.split('-')
        start = int(start) if start else 0
        end = int(end) if end else file_size - 1
        end = min(end, file_size - 1)
        return start, end

    async def render_directory(self, full_path: str, file_path: str) -> HTMLResponse:
        """渲染文件夹内容为 HTML 页面"""
        files = sorted(os.listdir(full_path))
        files_html = "\n".join([
            f'<li><a href="/{os.path.join(file_path, f)}">{f}</a></li>' if os.path.isfile(os.path.join(full_path, f)) else
            f'<li><a href="/{os.path.join(file_path, f)}">{f}/</a></li>'
            for f in files
        ])

        # 添加返回上一级目录的链接（如果不是根目录）
        if file_path:
            parent_dir = os.path.dirname(file_path)
            files_html = f'<li><a href="/{parent_dir if parent_dir != "." else ""}">../</a></li>\n' + files_html

        return HTMLResponse(
            content=f"""
            <html>
            <head><title>Index of /{file_path}</title></head>
            <body>
                <h1>Index of /{file_path}</h1>
                <ul>
                    {files_html}
                </ul>
            </body>
            </html>
            """,
            status_code=200,
        )

    def start_server(self):
        """启动 FastAPI 服务器"""
        self.logger.info(f"Starting static file server at http://{self.host}:{self.port} serving from {self.folder_path}")
        app = self.create_app()
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        server.run()

    def run_in_process(self):
        """在独立进程中启动服务器"""
        self.process = Process(target=self.start_server)
        self.process.start()
        self.logger.info("Static file server process started.")

    def stop_server(self):
        """停止服务器"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.logger.info("Static file server process stopped.")
        else:
            self.logger.warning("Static file server process is not running.")

    def is_running(self) -> bool:
        """检查服务器是否正在运行"""
        return self.process and self.process.is_alive()



# 示例：如何在主流程中使用 StaticFileServer
if __name__ == "__main__":
    # 创建静态文件服务器实例
    static_server = StaticFileServer(folder_path="/home/lyh/temp_images/", host="0.0.0.0", port=8000)

    try:
        # 启动静态文件服务器
        static_server.run_in_process()

        # 主流程继续运行其他任务
        while True:
            # 定期检查静态文件服务器的状态
            if not static_server.is_running():
                static_server.run_in_process()  # 重新启动服务器
            time.sleep(5)  # 每5秒检查一次

    except KeyboardInterrupt:
        # 捕获 Ctrl+C，优雅地关闭服务器
        static_server.stop_server()