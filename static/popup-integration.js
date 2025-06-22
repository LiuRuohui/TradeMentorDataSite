// 弹窗和浮动图标功能的JavaScript代码
// 回退为iframe嵌入方式

document.addEventListener('DOMContentLoaded', function() {
  // 创建样式
  const style = document.createElement('style');
  style.textContent = `
    .floating-icon {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      z-index: 9999;
      cursor: pointer;
    }
    .icon-button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 3.5rem;
      height: 3.5rem;
      background-color: #2563eb;
      color: white;
      border-radius: 9999px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      transition: all 0.3s ease;
    }
    .icon-button:hover {
      background-color: #1d4ed8;
      transform: scale(1.05);
    }
    .service-popup {
      position: fixed;
      inset: 0;
      z-index: 9998;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: rgba(0, 0, 0, 0.5);
      visibility: hidden;
      opacity: 0;
      transition: visibility 0s linear 0.25s, opacity 0.25s 0s;
    }
    .service-popup.visible {
      visibility: visible;
      opacity: 1;
      transition-delay: 0s;
    }
    .popup-container {
      position: relative;
      width: 400px;
      height: 80vh;
      max-width: 90vw;
      max-height: 100vh;
      background-color: white;
      border-radius: 0.5rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      overflow: hidden;
    }
    .popup-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
    }
    .popup-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #1f2937;
      margin: 0;
    }
    .close-button {
      padding: 0.25rem;
      border-radius: 9999px;
      background: transparent;
      border: none;
      cursor: pointer;
      transition: background-color 0.2s ease;
    }
    .close-button:hover {
      background-color: #e5e7eb;
    }
    .popup-content {
      width: 100%;
      height: 100%;
      padding: 0;
      margin: 0;
    }
    .embedded-iframe {
      width: 100%;
      height: 100%;
      border: 0;
      display: block;
    }
  `;
  document.head.appendChild(style);

  // 创建浮动图标
  const floatingIcon = document.createElement('div');
  floatingIcon.className = 'floating-icon';
  floatingIcon.innerHTML = `
    <div class="icon-button">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="16" fill="#fff"/>
        <rect x="8" y="12" width="16" height="10" rx="5" fill="#2563eb"/>
        <circle cx="12" cy="17" r="2" fill="#fff"/>
        <circle cx="20" cy="17" r="2" fill="#fff"/>
        <rect x="14" y="8" width="4" height="4" rx="2" fill="#2563eb"/>
      </svg>
    </div>
  `;
  document.body.appendChild(floatingIcon);

  // 创建弹窗（iframe嵌入方式）
  const servicePopup = document.createElement('div');
  servicePopup.className = 'service-popup';
  servicePopup.innerHTML = `
    <div class="popup-container">
      <div class="popup-header">
        <h2 class="popup-title">TradeMentor Assistant</h2>
        <button class="close-button">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      <div class="popup-content">
        <iframe src="http://localhost:5001" class="embedded-iframe" title="Embedded Service"></iframe>
      </div>
    </div>
  `;
  document.body.appendChild(servicePopup);

  // 事件监听
  const closeButton = servicePopup.querySelector('.close-button');
  floatingIcon.addEventListener('click', function() {
    togglePopup();
  });
  closeButton.addEventListener('click', function() {
    togglePopup(false);
  });
  function togglePopup(show) {
    const isVisible = servicePopup.classList.contains('visible');
    if (show === undefined) {
      show = !isVisible;
    }
    if (show) {
      servicePopup.classList.add('visible');
      floatingIcon.querySelector('.icon-button').innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      `;
    } else {
      servicePopup.classList.remove('visible');
      floatingIcon.querySelector('.icon-button').innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 32 32" fill="none">
          <circle cx="16" cy="16" r="16" fill="#fff"/>
          <rect x="8" y="12" width="16" height="10" rx="5" fill="#2563eb"/>
          <circle cx="12" cy="17" r="2" fill="#fff"/>
          <circle cx="20" cy="17" r="2" fill="#fff"/>
          <rect x="14" y="8" width="4" height="4" rx="2" fill="#2563eb"/>
        </svg>
      `;
    }
  }
});
