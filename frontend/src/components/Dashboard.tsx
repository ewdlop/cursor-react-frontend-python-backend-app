import React, { useState } from 'react';
import { Layout, Input, Button, Card, message } from 'antd';
import { LogoutOutlined, SendOutlined } from '@ant-design/icons';
import { useAppDispatch } from '../store/hooks';
import { logout } from '../store/slices/authSlice';
import axios from 'axios';

const { Header, Content } = Layout;
const { TextArea } = Input;

const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const [text, setText] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogout = () => {
    dispatch(logout());
  };

  const handleAnalyze = async () => {
    if (!text.trim()) {
      message.warning('请输入要分析的文本！');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/nlp/analyze', { text });
      setResult(response.data.result);
    } catch (error) {
      message.error('分析失败，请稍后重试！');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ color: 'white', margin: 0 }}>NLP 分析系统</h1>
        <Button type="primary" icon={<LogoutOutlined />} onClick={handleLogout}>
          退出登录
        </Button>
      </Header>
      <Content style={{ margin: '24px 16px', padding: 24, background: '#fff' }}>
        <Card title="文本分析">
          <TextArea
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="请输入要分析的文本..."
            style={{ marginBottom: 16 }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleAnalyze}
            loading={loading}
            block
          >
            分析文本
          </Button>
          {result && (
            <Card title="分析结果" style={{ marginTop: 16 }}>
              <p>{result}</p>
            </Card>
          )}
        </Card>
      </Content>
    </Layout>
  );
};

export default Dashboard; 