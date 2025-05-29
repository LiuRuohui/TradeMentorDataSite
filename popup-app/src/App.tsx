import { useState } from 'react';
import './App.css';
import FloatingIcon from './components/FloatingIcon';
import ServicePopup from './components/ServicePopup';

function App() {
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const togglePopup = () => {
    setIsPopupOpen(!isPopupOpen);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto p-4">
        <h1 className="text-3xl font-bold text-center my-8 text-gray-800">欢迎使用服务弹窗应用</h1>
        <div className="max-w-2xl mx-auto bg-white p-6 rounded-lg shadow-md">
          <p className="text-gray-700 mb-4">
            这是一个示例页面，右下角有一个浮动图标。点击该图标可以打开一个持久弹窗，弹窗中嵌入了来自 localhost:5001 的服务内容。
          </p>
          <p className="text-gray-700 mb-4">
            弹窗可以通过点击右上角的关闭按钮或再次点击浮动图标来关闭。弹窗内容将保持持久显示，直到您手动关闭它。
          </p>
          <p className="text-gray-700">
            请确保您的 localhost:5001 服务已经启动，否则弹窗中将无法显示内容。
          </p>
        </div>
      </div>
      
      {/* 浮动图标组件 */}
      <FloatingIcon onTogglePopup={togglePopup} isPopupOpen={isPopupOpen} />
      
      {/* 服务弹窗组件 */}
      <ServicePopup isOpen={isPopupOpen} onClose={togglePopup} />
    </div>
  );
}

export default App;
