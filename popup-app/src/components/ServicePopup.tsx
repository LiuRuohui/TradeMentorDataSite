import React from 'react';

interface ServicePopupProps {
  isOpen: boolean;
  onClose: () => void;
}

const ServicePopup: React.FC<ServicePopupProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="service-popup">
      <div className="popup-container">
        <div className="popup-header">
          <h2 className="popup-title">服务内容</h2>
          <button 
            onClick={onClose}
            className="close-button"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        <div className="popup-content">
          <iframe 
            src="http://localhost:5001" 
            className="embedded-iframe" 
            title="Embedded Service"
            sandbox="allow-same-origin allow-scripts allow-forms"
          />
        </div>
      </div>
    </div>
  );
};

export default ServicePopup;
