import { useState } from 'react';

interface FloatingIconProps {
  onTogglePopup: () => void;
  isPopupOpen: boolean;
}

const FloatingIcon: React.FC<FloatingIconProps> = ({ onTogglePopup, isPopupOpen }) => {
  return (
    <div className="floating-icon" onClick={onTogglePopup}>
      <div className="icon-button">
        {isPopupOpen ? (
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M13 10V3L4 14h7v7l9-11h-7z"></path>
          </svg>
        )}
      </div>
    </div>
  );
};

export default FloatingIcon;
