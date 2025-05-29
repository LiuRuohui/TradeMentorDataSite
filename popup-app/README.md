# 浮动图标与弹窗服务项目

## 项目概述

本项目实现了一个网页应用，其中在页面右下角有一个浮动图标，点击该图标后会弹出一个持久的弹窗，弹窗内容嵌入了来自 localhost:5001 端口的服务。

## 快速启动指南

### 1. 启动测试服务（端口5001）

启动trade_chatbot程序

服务将在 http://localhost:5001 上运行，请保持此终端窗口打开。

### 2. 启动前端应用

请在新的终端窗口中执行以下命令：

```bash
# 进入前端项目目录
cd popup-app

# 安装依赖
npm install
# 或者使用 pnpm
# pnpm install

# 启动开发服务器
npm run dev
# 或者使用 pnpm
# pnpm run dev
```

应用将在 http://localhost:5173 上运行，请在浏览器中访问此地址。

## 使用方法

1. 访问 http://localhost:5173 后，您将看到页面右下角有一个蓝色圆形图标
2. 点击该图标，将弹出一个包含 localhost:5001 服务内容的弹窗
3. 弹窗可以通过点击右上角的关闭按钮或再次点击浮动图标来关闭

## 项目结构

```
popup-app/
├── public/              # 静态资源
│   └── vite.svg         # 网站图标
├── src/                 # 源代码
│   ├── components/      # 组件
│   │   ├── FloatingIcon.tsx  # 右下角浮动图标组件
│   │   └── ServicePopup.tsx  # 服务弹窗组件
│   ├── App.css          # 应用样式
│   ├── App.tsx          # 主应用组件
│   ├── index.css        # 全局样式
│   └── main.tsx         # 入口文件
├── index.html           # HTML入口文件
├── package.json         # 项目依赖配置
└── vite.config.ts       # Vite配置文件

test-service/
└── server.py            # 测试服务（端口5001）
```

## 技术实现

1. **前端框架**：React + TypeScript
2. **构建工具**：Vite
3. **样式实现**：CSS
4. **服务嵌入**：使用 iframe 标签嵌入 localhost:5001 服务

## 常见问题解决

1. **404错误**：确保您在正确的目录中运行命令，应该在包含package.json的popup-app目录中运行npm命令

2. **无法看到服务内容**：确保测试服务在5001端口正常运行，可以单独访问 http://localhost:5001 进行验证

3. **依赖安装失败**：尝试使用 `npm install --force` 或清除npm缓存后重新安装

## 注意事项

1. 确保 localhost:5001 服务已启动，否则弹窗中将无法显示内容
2. 在生产环境中，应将 localhost 替换为实际的服务地址
