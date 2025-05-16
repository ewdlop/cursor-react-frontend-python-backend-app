# Text Analysis and Image Processing Application / 文本分析与图像处理应用

A full-stack application based on FastAPI and React that provides various NLP, computer vision, and AI generation features. / 一个基于FastAPI和React的文本分析、图像处理和AI生成应用，提供多种NLP、计算机视觉和AI生成功能。

## Features / 功能特点

### Text Analysis Features / 文本分析功能
- Basic Analysis / 基本分析：Entity Recognition, POS Tagging, Tokenization / 实体识别、词性标注、分词
- Sentiment Analysis / 情感分析：Text Sentiment Polarity Analysis / 文本情感极性分析
- Keyword Extraction / 关键词提取：Using jieba for keyword extraction / 使用jieba提取关键词
- Word Frequency Statistics / 词频统计：Text word frequency analysis / 文本词频分析
- Dependency Parsing / 依存句法分析：Analyzing grammatical relationships between words / 分析词语间的语法关系
- Text Summarization / 文本摘要：Using TF-IDF to extract key sentences / 使用TF-IDF提取关键句子
- Language Detection / 语言检测：Automatic language identification / 自动识别文本语言
- Text Similarity / 文本相似度：Computing similarity between texts / 计算文本间的相似度
- Readability Analysis / 可读性分析：Analyzing text complexity / 分析文本的复杂度
- Entity Relation Analysis / 实体关系分析：Identifying relationships between entities / 识别实体间的关系
- Text Classification / 文本分类：Sentiment-based classification / 基于情感的分类

### Image Processing Features / 图像处理功能
- Image Enhancement / 图像增强：Contrast enhancement, denoising, sharpening / 对比度增强、降噪、锐化
- Object Detection / 目标检测：Using YOLOv5 for object detection / 使用YOLOv5进行物体检测
- Image Segmentation / 图像分割：Automatic segmentation of main regions / 自动分割图像中的主要区域
  - Display segmentation mask / 显示分割掩码
  - Show segmented image / 显示分割后的图像
  - Display top 10 largest segments / 展示前10个最大分割区域
  - Show area and perimeter for each segment / 每个区域显示面积和周长信息
- Style Transfer / 风格迁移：Apply artistic style filters / 应用艺术风格滤镜

### Image Generation Features / 图像生成功能
- AI Image Generation / AI图像生成：Using Stable Diffusion model / 使用Stable Diffusion模型
- Customizable Parameters / 可自定义参数：
  - Number of steps / 步数
  - Guidance scale / 引导系数
  - Image dimensions / 图像尺寸
- Negative Prompt Support / 负面提示词支持：Specify unwanted elements / 指定不需要的元素
- Generation History / 生成历史：Track all generated images / 追踪所有生成的图像

### Text Generation Features / 文本生成功能
- AI Text Generation / AI文本生成：Using GPT-2 model / 使用GPT-2模型
- Customizable Parameters / 可自定义参数：
  - Maximum length / 最大长度
  - Temperature / 温度
  - Top-p sampling / Top-p采样
  - Number of sequences / 序列数量
- Generation History / 生成历史：Track all generated texts / 追踪所有生成的文本

## Tech Stack / 技术栈

### Backend / 后端
- FastAPI
- MongoDB
- PyTorch
- Transformers
- OpenCV
- spaCy
- NLTK
- scikit-learn
- YOLOv5
- Stable Diffusion
- GPT-2

### Frontend / 前端
- React
- Redux Toolkit
- RTK Query
- TypeScript
- Material-UI
- CSS3

## Installation / 安装说明

1. Clone the repository / 克隆仓库
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install backend dependencies / 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

3. Install frontend dependencies / 安装前端依赖
```bash
cd frontend
npm install
```

4. Configure environment variables / 配置环境变量
Create `.env` file in the backend directory / 创建 `.env` 文件在backend目录下：
```
SECRET_KEY=your-secret-key
MONGODB_URL=your-mongodb-url
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

5. Start the application / 启动应用
```bash
# Start backend / 启动后端
cd backend
uvicorn main:app --reload

# Start frontend / 启动前端
cd frontend
npm start
```

## API Documentation / API文档

Visit `http://localhost:8000/docs` after starting the backend server to view the complete API documentation. / 启动后端服务后，访问 `http://localhost:8000/docs` 查看完整的API文档。

### Main API Endpoints / 主要API端点

#### Authentication / 认证
- POST `/api/auth/register` - User registration / 用户注册
- POST `/api/auth/login` - User login / 用户登录
- POST `/api/auth/change-password` - Change password / 修改密码
- GET `/api/users/profile` - Get user profile / 获取用户资料
- GET `/api/users/stats` - Get user statistics / 获取用户统计信息

#### Text Analysis / 文本分析
- POST `/api/nlp/analyze` - Text analysis / 文本分析
- GET `/api/nlp/history` - Get analysis history / 获取分析历史

#### Image Processing / 图像处理
- POST `/api/image/process` - Image processing / 图像处理
- GET `/api/image/history` - Get processing history / 获取处理历史

#### Image Generation / 图像生成
- POST `/api/image/generate` - Generate image / 生成图像
- GET `/api/image/generation-history` - Get generation history / 获取生成历史

#### Text Generation / 文本生成
- POST `/api/text/generate` - Generate text / 生成文本
- GET `/api/text/generation-history` - Get generation history / 获取生成历史

## Usage Guide / 使用说明

### Text Analysis / 文本分析
1. Log in to the system / 登录系统
2. Enter text to analyze in the text analysis page / 在文本分析页面输入要分析的文本
3. Select desired analysis features / 选择需要的分析特征
4. Click "Analyze Text" button / 点击"分析文本"按钮
5. View analysis results and history / 查看分析结果和历史记录

### Image Processing / 图像处理
1. Log in to the system / 登录系统
2. Upload an image in the image processing page / 在图像处理页面上传图片
3. Select desired processing features / 选择需要的处理特征
4. Click "Start Processing" button / 点击"开始处理"按钮
5. View processing results and history / 查看处理结果和历史记录

### Image Generation / 图像生成
1. Log in to the system / 登录系统
2. Enter prompt in the image generation page / 在图像生成页面输入提示词
3. Adjust generation parameters / 调整生成参数
4. Click "Generate Image" button / 点击"生成图像"按钮
5. View generated image and history / 查看生成的图像和历史记录

### Text Generation / 文本生成
1. Log in to the system / 登录系统
2. Enter prompt in the text generation page / 在文本生成页面输入提示词
3. Adjust generation parameters / 调整生成参数
4. Click "Generate Text" button / 点击"生成文本"按钮
5. View generated text and history / 查看生成的文本和历史记录

## Notes / 注意事项

- Image processing features require significant computational resources, processing large images may take longer / 图像处理功能需要较大的计算资源，处理大图片时可能需要较长时间
- Object detection uses YOLOv5 model, supporting detection of 80 common objects / 目标检测功能使用YOLOv5模型，支持80种常见物体的检测
- Image segmentation automatically identifies main regions and sorts them by area / 图像分割功能会自动识别图像中的主要区域，并按面积大小排序
- Image generation uses Stable Diffusion model, which requires GPU for optimal performance / 图像生成功能使用Stable Diffusion模型，需要GPU以获得最佳性能
- Text generation uses GPT-2 model, which provides fast and efficient text generation / 文本生成功能使用GPT-2模型，提供快速高效的文本生成
- All processing results are saved in the database and can be viewed in history / 所有处理结果都会保存在数据库中，可以随时查看历史记录

## Contributing / 贡献指南

1. Fork the project / Fork 项目
2. Create your feature branch / 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. Commit your changes / 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch / 推送到分支 (`git push origin feature/AmazingFeature`)
5. Create a Pull Request / 创建Pull Request

## License / 许可证

[MIT License](LICENSE) 