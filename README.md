# NLP 分析应用

这是一个使用 React、TypeScript、Ant Design 和 FastAPI 构建的 NLP 分析应用。

## 功能特点

- 用户认证（JWT）
- 文本分析（使用 spaCy）
- 现代化的用户界面
- 响应式设计

## 技术栈

### 前端
- React
- TypeScript
- Ant Design
- Redux Toolkit
- React Router
- Axios

### 后端
- FastAPI
- JWT Authentication
- spaCy
- Python 3.8+

## 安装说明

### 前端设置

1. 进入前端目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
```

3. 启动开发服务器：
```bash
npm start
```

### 后端设置

1. 进入后端目录：
```bash
cd backend
```

2. 创建虚拟环境：
```bash
python -m venv venv
```

3. 激活虚拟环境：
- Windows:
```bash
venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```

4. 安装依赖：
```bash
pip install -r requirements.txt
```

5. 启动后端服务器：
```bash
python main.py
```

## 使用说明

1. 访问 http://localhost:3000
2. 使用以下凭据登录：
   - 用户名：admin
   - 密码：admin123
3. 在文本框中输入要分析的文本
4. 点击"分析文本"按钮查看结果

## 注意事项

- 确保前端和后端服务器都在运行
- 默认情况下，前端运行在 3000 端口，后端运行在 8000 端口
- 在生产环境中，请更改 SECRET_KEY 和其他安全相关的配置 