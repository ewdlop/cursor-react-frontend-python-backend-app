# NLP Application with React Frontend and Python Backend
# NLP 应用程序 - React 前端和 Python 后端

A full-stack application for Chinese text analysis with user authentication, built using React, FastAPI, and MongoDB.
一个使用 React、FastAPI 和 MongoDB 构建的全栈中文文本分析应用程序，包含用户认证功能。

## Features 功能特点

- User authentication (login/register) 用户认证（登录/注册）
- Chinese text analysis using spaCy 使用 spaCy 的中文文本分析
- Entity recognition 实体识别
- Part-of-speech tagging 词性标注
- Tokenization 分词
- MongoDB database integration MongoDB 数据库集成
- Modern React frontend with Redux and RTK Query 使用 Redux 和 RTK Query 的现代 React 前端

## Prerequisites 环境要求

- Python 3.11 or higher Python 3.11 或更高版本
- Node.js 16 or higher Node.js 16 或更高版本
- MongoDB 4.4 or higher MongoDB 4.4 或更高版本
- npm or yarn

## Project Structure 项目结构

```
.
├── backend/             # Python FastAPI backend Python FastAPI 后端
│   ├── main.py         # Main application file 主应用程序文件
│   ├── database.py     # MongoDB configuration MongoDB 配置
│   └── requirements.txt # Python dependencies Python 依赖
└── frontend/           # React frontend React 前端
    ├── src/
    │   ├── components/ # React components React 组件
    │   └── store/      # Redux store and API Redux 存储和 API
    └── package.json    # Node.js dependencies Node.js 依赖
```

## Setup Instructions 安装说明

### 1. Backend Setup 后端设置

1. Navigate to the backend directory 进入后端目录:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended) 创建虚拟环境（可选但推荐）:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies 安装 Python 依赖:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory 在后端目录创建 `.env` 文件:
   ```
   MONGODB_URL=mongodb://localhost:27017
   DATABASE_NAME=nlp_app
   SECRET_KEY=your-secret-key-here
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

5. Start MongoDB service on your system 启动系统上的 MongoDB 服务

6. Run the backend server 运行后端服务器:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000` 服务器将在 `http://localhost:8000` 启动

### 2. Frontend Setup 前端设置

1. Navigate to the frontend directory 进入前端目录:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies 安装 Node.js 依赖:
   ```bash
   npm install
   ```

3. Start the development server 启动开发服务器:
   ```bash
   npm start
   ```
   The application will open at `http://localhost:3000` 应用程序将在 `http://localhost:3000` 打开

## Default Credentials 默认凭据

- Username 用户名: `admin`
- Password 密码: `admin123`

## API Endpoints API 端点

### Authentication 认证
- `POST /api/auth/register` - Register a new user 注册新用户
- `POST /api/auth/login` - Login and get access token 登录并获取访问令牌

### Text Analysis 文本分析
- `POST /api/nlp/analyze` - Analyze Chinese text (requires authentication) 分析中文文本（需要认证）

## Technologies Used 使用的技术

### Backend 后端
- FastAPI
- MongoDB with Motor
- spaCy for NLP
- JWT for authentication JWT 认证
- Python-dotenv for configuration Python-dotenv 配置

### Frontend 前端
- React
- Redux Toolkit
- RTK Query
- React Router
- TypeScript

## Development 开发

### Backend Development 后端开发
- The backend uses FastAPI's automatic API documentation 后端使用 FastAPI 的自动 API 文档
- Access the API docs at `http://localhost:8000/docs` 在 `http://localhost:8000/docs` 访问 API 文档
- MongoDB connection is configured in `database.py` MongoDB 连接在 `database.py` 中配置
- Environment variables are loaded from `.env` 环境变量从 `.env` 加载

### Frontend Development 前端开发
- Components are in `frontend/src/components` 组件在 `frontend/src/components` 中
- Redux store and API configuration in `frontend/src/store` Redux 存储和 API 配置在 `frontend/src/store` 中
- Styling is done with CSS modules 使用 CSS 模块进行样式设计

## Security Notes 安全注意事项

- In production, change the `SECRET_KEY` in `.env` 在生产环境中，更改 `.env` 中的 `SECRET_KEY`
- Use HTTPS in production 在生产环境中使用 HTTPS
- Implement proper password policies 实施适当的密码策略
- Add rate limiting for API endpoints 为 API 端点添加速率限制
- Use environment variables for sensitive data 使用环境变量存储敏感数据

## Contributing 贡献

1. Fork the repository 复刻仓库
2. Create a feature branch 创建特性分支
3. Commit your changes 提交更改
4. Push to the branch 推送到分支
5. Create a Pull Request 创建拉取请求

## License 许可证

This project is licensed under the MIT License.
本项目采用 MIT 许可证。 