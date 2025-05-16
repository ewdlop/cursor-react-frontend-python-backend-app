# NLP Text Analysis Application / NLP文本分析应用

A full-stack application for Chinese text analysis with various NLP features. / 一个具有多种NLP功能的中文文本分析全栈应用。

## Features / 功能特点

### Text Analysis / 文本分析
- Basic Analysis / 基础分析
  - Tokenization / 分词
  - POS Tagging / 词性标注
  - Entity Recognition / 实体识别
- Sentiment Analysis / 情感分析
  - Polarity Score / 情感极性
  - Subjectivity Score / 主观性评分
- Keyword Extraction / 关键词提取
  - TF-IDF based / 基于TF-IDF
  - Weighted Keywords / 带权重的关键词
- Word Frequency Statistics / 词频统计
  - Word Count / 词数统计
  - Frequency Distribution / 频率分布
- Dependency Parsing / 依存句法分析
  - Syntactic Dependencies / 句法依存关系
  - Head-Dependent Relations / 主从关系

### NLTK Features / NLTK功能
- Text Statistics / 文本统计
  - Sentence Count / 句子数量
  - Word Count / 词数统计
  - Unique Words / 唯一词数
  - Average Sentence Length / 平均句子长度
- Sentence Segmentation / 句子分割
- Lemmatization / 词形还原
- POS Tagging / 词性标注
- Stop Word Filtering / 停用词过滤
- N-gram Analysis / N元语法分析
  - Bigrams / 二元语法
  - Trigrams / 三元语法
- Word Frequency Distribution / 词频分布

### User Features / 用户功能
- Authentication / 用户认证
  - Login / 登录
  - Registration / 注册
  - Password Change / 修改密码
- Analysis History / 分析历史
  - View Past Analyses / 查看历史分析
  - User Statistics / 用户统计

## Tech Stack / 技术栈

### Frontend / 前端
- React
- TypeScript
- Redux Toolkit
- RTK Query
- CSS3

### Backend / 后端
- FastAPI
- MongoDB
- spaCy
- TextBlob
- jieba
- NLTK

## Setup / 安装设置

### Prerequisites / 前置要求
- Python 3.8+
- Node.js 14+
- MongoDB

### Backend Setup / 后端设置

1. Create and activate virtual environment / 创建并激活虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies / 安装依赖:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables / 设置环境变量:
Create a `.env` file in the backend directory with:
```env
MONGODB_URL=mongodb://localhost:27017
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

4. Run the backend / 运行后端:
```bash
uvicorn main:app --reload
```

### Frontend Setup / 前端设置

1. Install dependencies / 安装依赖:
```bash
cd frontend
npm install
```

2. Run the frontend / 运行前端:
```bash
npm start
```

## API Documentation / API文档

The API documentation is available at `/docs` when running the backend server. / 运行后端服务器时，API文档可在 `/docs` 路径访问。

### Main Endpoints / 主要端点

- `/api/auth/login` - User login / 用户登录
- `/api/auth/register` - User registration / 用户注册
- `/api/auth/change-password` - Change password / 修改密码
- `/api/nlp/analyze` - Text analysis / 文本分析
- `/api/nlp/history` - Analysis history / 分析历史
- `/api/users/profile` - User profile / 用户资料
- `/api/users/stats` - User statistics / 用户统计

## Development / 开发

### Project Structure / 项目结构

```
.
├── backend/
│   ├── main.py          # FastAPI application / FastAPI应用
│   ├── database.py      # MongoDB configuration / MongoDB配置
│   └── requirements.txt # Python dependencies / Python依赖
└── frontend/
    ├── src/
    │   ├── components/  # React components / React组件
    │   ├── store/       # Redux store and API / Redux存储和API
    │   └── App.tsx      # Main application / 主应用
    └── package.json     # Node.js dependencies / Node.js依赖
```

### Adding New Features / 添加新功能

1. Backend / 后端:
   - Add new endpoints in `main.py`
   - Update models and database schema
   - Add new NLP processing functions

2. Frontend / 前端:
   - Create new components in `src/components`
   - Update API service in `src/store/api.ts`
   - Add new Redux slices if needed

## Contributing / 贡献

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License / 许可证

This project is licensed under the MIT License. / 本项目使用MIT许可证。

## Contact / 联系方式

For any questions or suggestions, please open an issue in the repository. / 如有任何问题或建议，请在仓库中提交issue。 